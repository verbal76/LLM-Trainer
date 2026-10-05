#!/usr/bin/env python3
"""Refresh the downloadable-artifact registry (registry/artifacts/<id>.json) from the Hugging Face API.

Runs in CI (``.github/workflows/catalog-refresh.yml``): the authoring sandbox cannot reach huggingface.co, so NOTHING
here may be pre-computed by hand. Standard library only.

For every entry of ``registry/artifacts/sources.json`` the script

1. queries ``/api/models/<repo>?blobs=true`` (falling back to the tree API) for the commit sha (= immutable revision),
   exact file names, sizes and LFS sha256 oids, and the model-card license metadata;
2. resolves the entry's file glob to EXACTLY ONE published file (zero => ``not_published``/error, several => error;
   file names are never guessed);
3. reads the GGUF header with HTTP Range requests: architecture, layers, KV heads, head dim, embedding size, trained
   context, exact parameter count, tokenizer chat-template presence (+ its sha256);
4. fetches the license text and computes ``license_text_sha256``;
5. decides the license state by the LICENSE VERIFICATION RULE (fail closed):
   VERIFIED only if the fetched text is byte-identical to a canonical reference text stored in ``registry/licenses/``
   (Apache-2.0 / MIT) AND the model-card license id agrees; everything else stays UNVERIFIED;
6. writes ``registry/artifacts/<id>.json`` (+ ``index.json``). A failed entry never overwrites a previously refreshed one.

Without network, ``--scaffold`` writes the committed ``unrefreshed`` placeholders (empty hashes; the app treats these as
NOT downloadable).

    python scripts/catalog/refresh_catalog.py              # refresh from Hugging Face (CI)
    python scripts/catalog/refresh_catalog.py --scaffold   # offline: (re)write unrefreshed placeholders
    python scripts/catalog/refresh_catalog.py --check-scaffold   # offline: committed files consistent with sources.json
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
UA = "LLMTrainer-catalog-refresh/1"
RANGE_CHUNK = 8 * 1024 * 1024
GGUF_HEADER_CAP = 192 * 1024 * 1024  # a metadata block larger than this is not a plausible model header
LICENSE_CAP = 1024 * 1024
API_CAP = 64 * 1024 * 1024


class RefreshError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- network (injectable)


class Response:
    def __init__(self, status: int, headers: dict[str, str], body: bytes):
        self.status, self.headers, self.body = status, {k.lower(): v for k, v in headers.items()}, body


Fetcher = Callable[[str, dict, int], Response]


def urllib_fetcher(url: str, headers: dict, max_bytes: int) -> Response:
    """Default fetcher: reads at most ``max_bytes`` of the body (a server that ignores Range must not stream GBs)."""
    h = {"User-Agent": UA, **headers}
    tok = os.environ.get("HF_TOKEN")
    if tok and url.startswith("https://huggingface.co/"):
        h["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=h)
    last: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read(max_bytes + 1)
                if len(body) > max_bytes:
                    body = body[:max_bytes]
                return Response(r.status, dict(r.headers.items()), body)
        except urllib.error.HTTPError as e:
            if e.code in (404, 401, 403):
                return Response(e.code, dict(e.headers.items()), b"")
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise RefreshError(f"network failure for {url}: {last}")


def get_json(fetch: Fetcher, url: str) -> object:
    r = fetch(url, {"Accept": "application/json"}, API_CAP)
    if r.status != 200:
        raise RefreshError(f"HTTP {r.status} for {url}")
    try:
        return json.loads(r.body.decode("utf-8"))
    except ValueError as e:
        raise RefreshError(f"invalid JSON from {url}: {e}") from e


# ----------------------------------------------------------------------------- hashing helpers


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------- repo listing


def list_repo(fetch: Fetcher, base: str, repo: str) -> dict:
    """Return {revision, files: {name: {size, sha256|None}}, card_license: str|None, tags: [...]}.

    Primary: models API with blobs=true (siblings[].lfs.sha256). Fallback for files without LFS data: tree API.
    """
    info = get_json(fetch, f"{base}/api/models/{repo}?blobs=true")
    if not isinstance(info, dict) or not isinstance(info.get("sha"), str) or not re.fullmatch(r"[0-9a-f]{40}", info["sha"]):
        raise RefreshError(f"{repo}: API answer has no 40-hex commit sha")
    rev = info["sha"]
    files: dict[str, dict] = {}
    for s in info.get("siblings") or []:
        name = s.get("rfilename")
        if not isinstance(name, str):
            continue
        lfs = s.get("lfs") or {}
        files[name] = {"size": lfs.get("size", s.get("size")), "sha256": lfs.get("sha256")}
    if any(n.lower().endswith(".gguf") and not v["sha256"] for n, v in files.items()):
        tree = get_json(fetch, f"{base}/api/models/{repo}/tree/{rev}?recursive=true")
        if isinstance(tree, list):
            for t in tree:
                if t.get("type") == "file" and isinstance(t.get("path"), str) and t["path"] in files:
                    lfs = t.get("lfs") or {}
                    if lfs.get("oid") and not files[t["path"]]["sha256"]:
                        files[t["path"]]["sha256"] = lfs["oid"]
                    if lfs.get("size") is not None:
                        files[t["path"]]["size"] = lfs["size"]
    card = info.get("cardData") or {}
    lic = card.get("license")
    if isinstance(lic, list):
        lic = lic[0] if len(lic) == 1 else None
    return {"revision": rev, "files": files, "card_license": lic if isinstance(lic, str) else None, "tags": info.get("tags") or []}


def match_one(files: dict, glob: str, exclude: list[str]) -> tuple[str | None, list[str]]:
    """Case-insensitive glob over *.gguf names. Returns (the single match | None, all matches)."""
    g = glob.lower()
    hits = sorted(
        n for n in files
        if n.lower().endswith(".gguf") and fnmatch.fnmatchcase(n.lower(), g) and not any(fnmatch.fnmatchcase(n.lower(), x.lower()) for x in exclude)
    )
    return (hits[0] if len(hits) == 1 else None), hits


# ----------------------------------------------------------------------------- GGUF header


class _Reader:
    """Sequential reader over a remote file, fetched lazily with Range requests."""

    def __init__(self, fetch: Fetcher, url: str):
        self.fetch, self.url = fetch, url
        self.buf = bytearray()
        self.pos = 0
        self.eof = False

    def _grow(self, need_end: int) -> None:
        while len(self.buf) < need_end and not self.eof:
            if len(self.buf) >= GGUF_HEADER_CAP:
                raise RefreshError("GGUF metadata larger than the allowed header cap")
            start = len(self.buf)
            end = start + RANGE_CHUNK - 1
            r = self.fetch(self.url, {"Range": f"bytes={start}-{end}"}, RANGE_CHUNK)
            if r.status == 200 and start > 0:
                raise RefreshError("server ignored the Range request")
            if r.status not in (200, 206):
                raise RefreshError(f"HTTP {r.status} reading GGUF header from {self.url}")
            if not r.body:
                self.eof = True
                break
            self.buf += r.body
            if len(r.body) < RANGE_CHUNK and r.status == 206:
                total = re.search(r"/(\d+)$", r.headers.get("content-range", ""))
                if total and len(self.buf) >= int(total.group(1)):
                    self.eof = True

    def take(self, n: int) -> bytes:
        self._grow(self.pos + n)
        if self.pos + n > len(self.buf):
            raise RefreshError("unexpected end of GGUF file while reading the header")
        out = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return out

    def skip(self, n: int) -> None:
        self._grow(self.pos + n)
        if self.pos + n > len(self.buf):
            raise RefreshError("unexpected end of GGUF file while skipping")
        self.pos += n

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> bytes:
        n = self.u64()
        if n > 64 * 1024 * 1024:
            raise RefreshError("implausible GGUF string length")
        return self.take(n)


_SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
           10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
_KEEP_MAX = 4096


def _value(rd: _Reader, t: int, keep: bool):
    if t in _SCALAR:
        fmt, size = _SCALAR[t]
        raw = rd.take(size)
        return struct.unpack(fmt, raw)[0]
    if t == 8:
        s = rd.string()
        return s if keep else len(s)
    if t == 9:
        et = rd.u32()
        n = rd.u64()
        if et in _SCALAR:
            rd.skip(n * _SCALAR[et][1])
        elif et == 8:
            for _ in range(n):
                ln = rd.u64()
                rd.skip(ln)
        else:
            raise RefreshError(f"unsupported GGUF array element type {et}")
        return ("array", n)
    raise RefreshError(f"unsupported GGUF value type {t}")


def read_gguf_header(fetch: Fetcher, url: str) -> dict:
    rd = _Reader(fetch, url)
    if rd.take(4) != b"GGUF":
        raise RefreshError("not a GGUF file (bad magic)")
    version = rd.u32()
    if version not in (2, 3):
        raise RefreshError(f"unsupported GGUF version {version}")
    tensor_count, kv_count = rd.u64(), rd.u64()
    meta: dict[str, object] = {}
    template_sha: str | None = None
    for _ in range(kv_count):
        key = rd.string().decode("utf-8", "replace")
        t = rd.u32()
        keep = key == "tokenizer.chat_template" or not key.startswith("tokenizer.")
        v = _value(rd, t, keep)
        if key == "tokenizer.chat_template" and isinstance(v, bytes):
            template_sha = sha256_hex(v)
            meta[key] = len(v)
        elif isinstance(v, bytes):
            if len(v) <= _KEEP_MAX:
                meta[key] = v.decode("utf-8", "replace")
        else:
            meta[key] = v
    params = 0
    for _ in range(tensor_count):
        rd.string()
        nd = rd.u32()
        n = 1
        for _d in range(nd):
            n *= rd.u64()
        rd.u32()
        rd.u64()
        params += n
    arch = meta.get("general.architecture")
    if not isinstance(arch, str):
        raise RefreshError("GGUF has no general.architecture")

    def num(suffix: str):
        v = meta.get(f"{arch}.{suffix}")
        return v if isinstance(v, int) and not isinstance(v, bool) else None

    heads, kv_heads, embd = num("attention.head_count"), num("attention.head_count_kv"), num("embedding_length")
    head_dim = num("attention.key_length")
    if head_dim is None and heads and embd:
        head_dim = embd // heads
    toks = meta.get("tokenizer.ggml.tokens")
    return {
        "gguf_version": version,
        "architecture": arch,
        "layers": num("block_count"),
        "heads": heads,
        "kv_heads": kv_heads if kv_heads is not None else heads,
        "head_dim": head_dim,
        "embd": embd,
        "ffn": num("feed_forward_length"),
        "ctx_train": num("context_length"),
        "vocab": toks[1] if isinstance(toks, tuple) and toks[0] == "array" else None,
        "tensor_count": tensor_count,
        "parameter_count": params,
        "file_type": meta.get("general.file_type") if isinstance(meta.get("general.file_type"), int) else None,
        "chat_template_present": "tokenizer.chat_template" in meta,
        "chat_template_sha256": template_sha,
        "size_label": meta.get("general.size_label") if isinstance(meta.get("general.size_label"), str) else None,
    }


# ----------------------------------------------------------------------------- license rule


def load_canon(root: Path = ROOT) -> dict:
    doc = json.loads((root / "registry/licenses/canonical.json").read_text(encoding="utf-8"))
    out = {}
    for lic in doc["licenses"]:
        text = (root / "registry/licenses" / lic["file"]).read_bytes()
        if sha256_hex(text) != lic["sha256"]:
            raise RefreshError(f"canonical license {lic['spdx_id']} does not match its recorded sha256")
        out[lic["spdx_id"]] = {**lic, "text": text}
    return out


def _mit_normalize(text: bytes, template_line: str) -> bytes | None:
    """Replace the single 'Copyright (c) ...' line with the template line; None unless exactly one such line exists."""
    lines = text.split(b"\n")
    idx = [i for i, ln in enumerate(lines) if re.fullmatch(rb"Copyright \(c\) \S.*", ln)]
    if len(idx) != 1:
        return None
    lines[idx[0]] = template_line.encode("utf-8")
    return b"\n".join(lines)


def match_canonical(text: bytes, canon: dict) -> tuple[str | None, str]:
    """Return (spdx id of the canonical text the bytes match, match mode). (None, 'none') when nothing matches."""
    for spdx, c in canon.items():
        if c["match_mode"] == "exact":
            if text == c["text"]:
                return spdx, "exact"
        elif c["match_mode"] == "mit_copyright_line":
            n = _mit_normalize(text, c["template_copyright_line"])
            if n is not None and n == c["text"]:
                return spdx, "mit_copyright_line"
    return None, "none"


def decide_license(expected: str, text: bytes | None, text_url: str | None, cards: dict[str, str | None], canon: dict, now: str) -> dict:
    """The fail-closed LICENSE VERIFICATION RULE. ``cards`` maps repo -> model-card license id (or None)."""
    ev: dict = {
        "expected_spdx": expected, "license_text_url": text_url, "license_text_sha256": None, "license_text_bytes": None,
        "canonical_spdx": None, "canonical_sha256": None, "canonical_source_url": None, "match": "unavailable",
        "model_card_licenses": cards, "model_card_agrees": False, "fetched_at": now,
    }
    reasons: list[str] = []
    if text is None:
        reasons.append("no license text could be fetched from any configured URL")
    else:
        ev["license_text_sha256"] = "sha256:" + sha256_hex(text)
        ev["license_text_bytes"] = len(text)
        spdx, mode = match_canonical(text, canon)
        if spdx is None:
            ev["match"] = "mismatch"
            reasons.append("fetched license text is not byte-identical to any canonical reference text (Apache-2.0, MIT)")
        else:
            ev["match"] = mode
            ev["canonical_spdx"] = spdx
            ev["canonical_sha256"] = "sha256:" + canon[spdx]["sha256"]
            ev["canonical_source_url"] = canon[spdx]["source_url"]
            if spdx != expected:
                reasons.append(f"license text is {spdx} but the curated list expects {expected}")
    spdx = ev["canonical_spdx"]
    if spdx and spdx in canon:
        allowed = set(canon[spdx]["card_ids"])
        got = {repo: (v or "").strip().lower() for repo, v in cards.items()}
        ev["model_card_agrees"] = bool(got) and all(v in allowed for v in got.values())
        if not ev["model_card_agrees"]:
            reasons.append("model-card license metadata does not agree: " + json.dumps(cards, sort_keys=True))
    verified = text is not None and spdx == expected and ev["model_card_agrees"] and ev["match"] in ("exact", "mit_copyright_line")
    return {"state": "VERIFIED" if verified else "UNVERIFIED", "spdx_id": expected if verified else None, "reasons": [] if verified else reasons, "evidence": ev}


# ----------------------------------------------------------------------------- training class


def full_precision(precision: str) -> bool:
    return precision in ("f32", "f16", "bf16")


def training_block(precision: str, params_b: float, sibling_id: str | None) -> dict:
    """Static training compatibility of this FILE. Device-specific feasibility is computed by the device profiler.

    Basis (until docs/v2/TRAINING_FEASIBILITY.md lands): the on-device trainer is F32-only; memory = F32 weights +
    (4+8) bytes x trainable params + activations. Full tuning of a model costs ~16 bytes/param, so only <= ~0.45B is
    plausible on a phone; <= ~1B can tune the last layers; above that training is external.
    """
    if not full_precision(precision):
        return {
            "class": "inference_only",
            "reasons": ["block-quantized GGUF weights cannot be trained in place; the on-device trainer works on full-precision weights (F32 compute)"],
            "model_class_if_full_precision": model_ceiling(params_b),
            "training_source_artifact_id": sibling_id,
            "assumptions": "F32-only trainer; docs/v2/TRAINING_FEASIBILITY.md not yet available",
        }
    ceiling = model_ceiling(params_b)
    reasons = {
        "local_full": ["small enough that F32 weights + 12 bytes/param optimizer state can fit a high-RAM phone (device profiler decides per device)"],
        "local_partial": ["full tuning needs ~16 bytes/param (too large for a phone); tuning only the last N layers can fit a high-RAM phone"],
        "external_only": ["F32 weights alone exceed what a phone can hold for training; adapter training runs on desktop/GPU (llmtrainer import-job)"],
    }[ceiling]
    return {"class": ceiling, "reasons": reasons, "model_class_if_full_precision": ceiling, "training_source_artifact_id": None,
            "assumptions": "F32-only trainer; docs/v2/TRAINING_FEASIBILITY.md not yet available"}


def model_ceiling(params_b: float) -> str:
    if params_b <= 0.45:
        return "local_full"
    if params_b <= 1.0:
        return "local_partial"
    return "external_only"


# ----------------------------------------------------------------------------- artifact documents


def _runtime_compat(arch: str | None, support: dict) -> dict:
    out = {}
    for rt, table in support.items():
        if rt == "note":
            continue
        row = table.get(arch) if arch else None
        out[rt] = {"architecture": arch, "supported": row["supported"] if row else "unverified",
                   "min_build": row.get("min_build") if row else None}
    out.setdefault("hag-engine", {"architecture": arch, "supported": "unverified", "min_build": None})
    return out


def _sibling_full_precision(repo_cfg: dict, f: dict) -> str | None:
    if full_precision(f["precision"]):
        return None
    for g in repo_cfg["files"]:
        if full_precision(g["precision"]):
            return g["id"]
    return None


def scaffold_artifact(repo_cfg: dict, f: dict, sources: dict) -> dict:
    base = "https://huggingface.co"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": f["id"],
        "refresh_state": "unrefreshed",
        "refreshed_at": None,
        "optional": bool(f.get("optional", False)),
        "published": None,
        "family": repo_cfg["family"],
        "base_repo": repo_cfg["base_repo"],
        "source": {"repo": repo_cfg["repo"], "repo_url": f"{base}/{repo_cfg['repo']}", "file": None, "file_glob": f["glob"],
                   "revision": None, "download_url": None},
        "format": "gguf",
        "quantization": f["quantization"],
        "precision": f["precision"],
        "size_bytes": None,
        "sha256": None,
        "parameter_count_nominal_b": repo_cfg["params_b_nominal"],
        "parameter_count": None,
        "architecture": {"name": None, "layers": None, "kv_heads": None, "heads": None, "head_dim": None, "embd": None, "ffn": None, "ctx_train": None, "vocab": None},
        "chat_template_present": None,
        "chat_template_sha256": None,
        "runtime_compat": _runtime_compat(None, sources.get("runtime_support", {})),
        "training": training_block(f["precision"], repo_cfg["params_b_nominal"], _sibling_full_precision(repo_cfg, f)),
        "license": {
            "state": "UNVERIFIED", "expected_spdx": repo_cfg["expected_license"], "spdx_id": None,
            "reasons": ["unrefreshed: the catalog-refresh CI job has not yet read the license text or the model card"],
            "evidence": None,
        },
        "notes": ["UNREFRESHED placeholder: no hash, size or revision is known. The app treats this artifact as NOT downloadable."],
    }


def build_artifact(repo_cfg: dict, f: dict, listing: dict, file_name: str, header: dict, lic: dict, sources: dict, now: str) -> dict:
    base = "https://huggingface.co"
    entry = listing["files"][file_name]
    rev = listing["revision"]
    sha = entry["sha256"]
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
        raise RefreshError(f"{repo_cfg['repo']}/{file_name}: no LFS sha256 available")
    if not isinstance(entry["size"], int) or entry["size"] <= 0:
        raise RefreshError(f"{repo_cfg['repo']}/{file_name}: no size available")
    doc = scaffold_artifact(repo_cfg, f, sources)
    doc.update(refresh_state="refreshed", refreshed_at=now, published=True, size_bytes=entry["size"], sha256="sha256:" + sha,
               parameter_count=header["parameter_count"], chat_template_present=header["chat_template_present"],
               chat_template_sha256=header["chat_template_sha256"])
    doc["source"].update(file=file_name, revision=rev, download_url=f"{base}/{repo_cfg['repo']}/resolve/{rev}/{file_name}")
    doc["architecture"] = {k: header[k] for k in ("layers", "kv_heads", "heads", "head_dim", "embd", "ffn", "ctx_train", "vocab")}
    doc["architecture"]["name"] = header["architecture"]
    doc["runtime_compat"] = _runtime_compat(header["architecture"], sources.get("runtime_support", {}))
    params_b = header["parameter_count"] / 1e9
    doc["training"] = training_block(f["precision"], params_b, _sibling_full_precision(repo_cfg, f))
    doc["license"] = {"state": lic["state"], "expected_spdx": repo_cfg["expected_license"], "spdx_id": lic["spdx_id"], "reasons": lic["reasons"], "evidence": lic["evidence"]}
    doc["notes"] = []
    return doc


def not_published_artifact(repo_cfg: dict, f: dict, sources: dict, listing: dict, now: str, hits: list[str]) -> dict:
    doc = scaffold_artifact(repo_cfg, f, sources)
    doc.update(refresh_state="refreshed", refreshed_at=now, published=False)
    doc["source"]["revision"] = listing["revision"]
    doc["notes"] = [f"No published file matches '{f['glob']}' at revision {listing['revision']}" + (f" (ambiguous: {hits})" if hits else "") + "; not downloadable."]
    return doc


# ----------------------------------------------------------------------------- driver


def fetch_license_text(fetch: Fetcher, urls: list[str], revision: str) -> tuple[bytes | None, str | None]:
    for tmpl in urls:
        url = tmpl.replace("{revision}", revision)
        r = fetch(url, {}, LICENSE_CAP)
        if r.status == 200 and r.body:
            return r.body, url
    return None, None


def _substantive(doc: dict) -> dict:
    d = json.loads(json.dumps(doc))
    d.pop("refreshed_at", None)
    ev = (d.get("license") or {}).get("evidence")
    if ev:
        ev.pop("fetched_at", None)
    return d


def dump(doc: dict) -> str:
    return json.dumps(doc, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def refresh(sources: dict, out_dir: Path, fetch: Fetcher, canon: dict, now: str, only: set[str] | None = None) -> dict:
    """Refresh every selected artifact. Returns the index document (also written to out_dir/index.json)."""
    base = sources.get("hf_api_base", "https://huggingface.co")
    results: list[dict] = []
    errors: list[str] = []
    for repo_cfg in sources["repos"]:
        wanted = [f for f in repo_cfg["files"] if only is None or f["id"] in only]
        if not wanted:
            continue
        try:
            listing = list_repo(fetch, base, repo_cfg["repo"])
            cards = {repo_cfg["repo"]: listing["card_license"]}
            if repo_cfg.get("base_repo"):
                try:
                    cards[repo_cfg["base_repo"]] = list_repo_card(fetch, base, repo_cfg["base_repo"])
                except RefreshError:
                    cards[repo_cfg["base_repo"]] = None
            text, text_url = fetch_license_text(fetch, repo_cfg["license_text_urls"], listing["revision"])
            lic = decide_license(repo_cfg["expected_license"], text, text_url, cards, canon, now)
        except RefreshError as e:
            for f in wanted:
                errors.append(f"{f['id']}: {e}")
                results.append({"id": f["id"], "outcome": "error", "error": str(e)})
            continue
        for f in wanted:
            path = out_dir / f"{f['id']}.json"
            try:
                name, hits = match_one(listing["files"], f["glob"], f.get("exclude", []))
                if name is None:
                    if f.get("optional") and not hits:
                        doc = not_published_artifact(repo_cfg, f, sources, listing, now, hits)
                        outcome = "not_published"
                    else:
                        raise RefreshError(f"{repo_cfg['repo']}: glob '{f['glob']}' matched {len(hits)} files {hits}; need exactly one")
                else:
                    url = f"{base}/{repo_cfg['repo']}/resolve/{listing['revision']}/{name}"
                    header = read_gguf_header(fetch, url)
                    doc = build_artifact(repo_cfg, f, listing, name, header, lic, sources, now)
                    outcome = "refreshed"
            except RefreshError as e:
                errors.append(f"{f['id']}: {e}")
                results.append({"id": f["id"], "outcome": "error", "error": str(e)})
                continue
            old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            if old is not None and old.get("refresh_state") == "refreshed" and _substantive(old) == _substantive(doc):
                doc = old  # keep timestamps stable when nothing changed
            path.write_text(dump(doc), encoding="utf-8")
            results.append({"id": f["id"], "outcome": outcome, "file": doc["source"]["file"], "revision": doc["source"]["revision"],
                            "license_state": doc["license"]["state"]})
    index = {"schema_version": SCHEMA_VERSION, "refreshed_at": now, "ok": not errors, "errors": errors, "results": results}
    (out_dir / "index.json").write_text(dump(index), encoding="utf-8")
    return index


def list_repo_card(fetch: Fetcher, base: str, repo: str) -> str | None:
    info = get_json(fetch, f"{base}/api/models/{repo}")
    lic = (info.get("cardData") or {}).get("license") if isinstance(info, dict) else None
    if isinstance(lic, list):
        lic = lic[0] if len(lic) == 1 else None
    return lic if isinstance(lic, str) else None


def scaffold(sources: dict, out_dir: Path, overwrite_refreshed: bool = False) -> list[str]:
    written = []
    for repo_cfg in sources["repos"]:
        for f in repo_cfg["files"]:
            path = out_dir / f"{f['id']}.json"
            if path.exists() and not overwrite_refreshed and json.loads(path.read_text(encoding="utf-8")).get("refresh_state") == "refreshed":
                continue
            path.write_text(dump(scaffold_artifact(repo_cfg, f, sources)), encoding="utf-8")
            written.append(f["id"])
    return written


def check_scaffold(sources: dict, out_dir: Path) -> list[str]:
    """Problems: an artifact in sources.json without a file, a stray file, or an unrefreshed file that drifted."""
    problems = []
    ids = set()
    for repo_cfg in sources["repos"]:
        for f in repo_cfg["files"]:
            ids.add(f["id"])
            p = out_dir / f"{f['id']}.json"
            if not p.exists():
                problems.append(f"missing {p.name}")
                continue
            doc = json.loads(p.read_text(encoding="utf-8"))
            if doc.get("refresh_state") == "unrefreshed" and doc != scaffold_artifact(repo_cfg, f, sources):
                problems.append(f"{p.name}: unrefreshed placeholder drifted from sources.json (run --scaffold)")
    for p in out_dir.glob("*.json"):
        if p.name not in ("sources.json", "index.json") and p.stem not in ids:
            problems.append(f"{p.name}: not listed in sources.json")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=str(ROOT / "registry/artifacts/sources.json"))
    ap.add_argument("--out", default=str(ROOT / "registry/artifacts"))
    ap.add_argument("--only", action="append", help="refresh only this artifact id (repeatable)")
    ap.add_argument("--scaffold", action="store_true", help="offline: write unrefreshed placeholders (never overwrites refreshed files)")
    ap.add_argument("--check-scaffold", action="store_true", help="offline: verify committed artifact files against sources.json")
    ap.add_argument("--now", default=None, help="ISO date-time to stamp (tests)")
    args = ap.parse_args(argv)
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.scaffold:
        print("scaffolded:", ", ".join(scaffold(sources, out)) or "(nothing to do)")
        return 0
    if args.check_scaffold:
        problems = check_scaffold(sources, out)
        for p in problems:
            print("PROBLEM:", p, file=sys.stderr)
        return 1 if problems else 0
    now = args.now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    canon = load_canon()
    index = refresh(sources, out, urllib_fetcher, canon, now, set(args.only) if args.only else None)
    for r in index["results"]:
        print(r["id"], r["outcome"], r.get("license_state", ""), r.get("error", ""))
    return 0 if index["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
