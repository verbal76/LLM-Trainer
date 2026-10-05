"""Offline tests for scripts/catalog/refresh_catalog.py. No test touches the network: a fake fetcher serves a synthetic HF."""

import copy
import json
import struct
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import refresh_catalog as rc  # noqa: E402

ROOT = rc.ROOT
FIX = HERE / "fixtures"
APACHE = (ROOT / "registry/licenses/Apache-2.0.txt").read_bytes()
MIT = (ROOT / "registry/licenses/MIT.txt").read_bytes()
NOW = "2026-10-05T00:00:00Z"
REV = "0123456789abcdef0123456789abcdef01234567"
REV2 = "fedcba9876543210fedcba9876543210fedcba98"


# ----------------------------------------------------------------------------- synthetic GGUF


def _s(b: bytes) -> bytes:
    return struct.pack("<Q", len(b)) + b


def kv(key: str, t: int, payload: bytes) -> bytes:
    return _s(key.encode()) + struct.pack("<I", t) + payload


def kv_u32(key, v):
    return kv(key, 4, struct.pack("<I", v))


def kv_str(key, v):
    return kv(key, 8, _s(v.encode()))


def kv_str_array(key, items):
    return kv(key, 9, struct.pack("<IQ", 8, len(items)) + b"".join(_s(i.encode()) for i in items))


def kv_i32_array(key, items):
    return kv(key, 9, struct.pack("<IQ", 5, len(items)) + b"".join(struct.pack("<i", i) for i in items))


def make_gguf(arch="qwen3", layers=28, heads=16, kv_heads=8, embd=1024, ctx=40960, template=True, vocab=300, key_len=128, tensors=((1024, 300), (1024, 1024)), body_pad=4096):
    kvs = [
        kv_str("general.architecture", arch), kv_str("general.size_label", "0.6B"), kv_u32("general.file_type", 7),
        kv_u32(f"{arch}.block_count", layers), kv_u32(f"{arch}.attention.head_count", heads),
        kv_u32(f"{arch}.attention.head_count_kv", kv_heads), kv_u32(f"{arch}.embedding_length", embd),
        kv_u32(f"{arch}.context_length", ctx), kv_u32(f"{arch}.feed_forward_length", 3072),
    ]
    if key_len:
        kvs.append(kv_u32(f"{arch}.attention.key_length", key_len))
    kvs.append(kv_str_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(vocab)]))
    kvs.append(kv_i32_array("tokenizer.ggml.token_type", list(range(vocab))))
    if template:
        kvs.append(kv_str("tokenizer.chat_template", "{% for m in messages %}{{ m.content }}{% endfor %}"))
    infos = b"".join(_s(f"w{i}".encode()) + struct.pack("<I", len(d)) + b"".join(struct.pack("<Q", x) for x in d) + struct.pack("<IQ", 0, 0)
                     for i, d in enumerate(tensors))
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + infos
    return head + b"\x00" * body_pad


# ----------------------------------------------------------------------------- fake Hugging Face


class FakeHF:
    def __init__(self):
        self.repos: dict[str, dict] = {}
        self.texts: dict[str, bytes] = {}
        self.calls: list[tuple[str, dict]] = []
        self.ignore_range = False

    def add_repo(self, repo, files: dict[str, bytes], license_="apache-2.0", rev=REV, lfs=True):
        sib = []
        for n, data in files.items():
            e = {"rfilename": n, "size": len(data)}
            if lfs and n.endswith(".gguf"):
                import hashlib
                e["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "pointerSize": 135}
            sib.append(e)
        self.repos[repo] = {"sha": rev, "cardData": {"license": license_} if license_ else {}, "siblings": sib, "files": files}

    def __call__(self, url, headers, max_bytes):
        self.calls.append((url, dict(headers)))
        m = rc.re.match(r"https://huggingface\.co/api/models/(.+?)(?:\?blobs=true)?$", url)
        if m and "/tree/" not in m.group(1):
            r = self.repos.get(m.group(1))
            if not r:
                return rc.Response(404, {}, b"")
            return rc.Response(200, {}, json.dumps({k: v for k, v in r.items() if k != "files"}).encode())
        m = rc.re.match(r"https://huggingface\.co/api/models/(.+?)/tree/([0-9a-f]+)\?recursive=true$", url)
        if m:
            r = self.repos[m.group(1)]
            import hashlib
            tree = [{"type": "file", "path": n, "size": len(d), "oid": "x", **({"lfs": {"oid": hashlib.sha256(d).hexdigest(), "size": len(d)}} if n.endswith(".gguf") else {})}
                    for n, d in r["files"].items()]
            return rc.Response(200, {}, json.dumps(tree).encode())
        m = rc.re.match(r"https://huggingface\.co/(.+?)/resolve/([^/]+)/(.+)$", url)
        if m:
            repo, rev, name = m.groups()
            if url in self.texts:
                return rc.Response(200, {}, self.texts[url][:max_bytes])
            r = self.repos.get(repo)
            if not r or name not in r["files"]:
                return rc.Response(404, {}, b"")
            data = r["files"][name]
            rng = headers.get("Range")
            if rng and not self.ignore_range:
                a, b = map(int, rng[len("bytes="):].split("-"))
                chunk = data[a:b + 1]
                return rc.Response(206, {"Content-Range": f"bytes {a}-{a + len(chunk) - 1}/{len(data)}"}, chunk[:max_bytes])
            return rc.Response(200, {}, data[:max_bytes])
        return rc.Response(404, {}, b"")


def mini_sources():
    return {
        "hf_api_base": "https://huggingface.co",
        "runtime_support": {"llama.cpp": {"qwen3": {"supported": "yes", "min_build": "b5401"}}, "hag-engine": {}},
        "repos": [{
            "repo": "Example/Tiny-GGUF", "base_repo": "Example/Tiny", "family": "Tiny", "params_b_nominal": 0.0005, "expected_license": "Apache-2.0",
            "license_text_urls": ["https://huggingface.co/Example/Tiny-GGUF/resolve/{revision}/LICENSE", "https://huggingface.co/Example/Tiny/resolve/main/LICENSE"],
            "files": [
                {"id": "tiny-q8_0", "glob": "*q8_0.gguf", "quantization": "Q8_0", "precision": "quantized"},
                {"id": "tiny-f16", "glob": "*f16.gguf", "quantization": "F16", "precision": "f16", "optional": True, "exclude": ["*bf16.gguf"]},
                {"id": "tiny-f32", "glob": "*f32.gguf", "quantization": "F32", "precision": "f32", "optional": True},
            ],
        }],
    }


def world(license_text=APACHE, card="apache-2.0", base_card="apache-2.0", gguf=None, with_bf16=True, lfs=True):
    hf = FakeHF()
    g = gguf or make_gguf()
    files = {"README.md": b"x", "Tiny-Q8_0.gguf": g, "Tiny-F16.gguf": g + b"\x01"}
    if with_bf16:
        files["Tiny-BF16.gguf"] = g + b"\x02"
    hf.add_repo("Example/Tiny-GGUF", files, license_=card, lfs=lfs)
    hf.add_repo("Example/Tiny", {"README.md": b"x"}, license_=base_card)
    if license_text is not None:
        hf.texts[f"https://huggingface.co/Example/Tiny-GGUF/resolve/{REV}/LICENSE"] = license_text
    return hf


@pytest.fixture
def canon():
    return rc.load_canon()


def run(tmp_path, hf, canon, sources=None, only=None):
    return rc.refresh(sources or mini_sources(), tmp_path, hf, canon, NOW, only)


def load(tmp_path, name):
    return json.loads((tmp_path / f"{name}.json").read_text())


# ----------------------------------------------------------------------------- GGUF header


def test_gguf_header_fields():
    data = make_gguf()
    hf = FakeHF()
    hf.add_repo("a/b", {"m.gguf": data})
    h = rc.read_gguf_header(hf, "https://huggingface.co/a/b/resolve/" + REV + "/m.gguf")
    assert h["architecture"] == "qwen3" and h["layers"] == 28 and h["kv_heads"] == 8 and h["heads"] == 16
    assert h["head_dim"] == 128 and h["embd"] == 1024 and h["ctx_train"] == 40960 and h["ffn"] == 3072
    assert h["parameter_count"] == 1024 * 300 + 1024 * 1024
    assert h["chat_template_present"] is True and len(h["chat_template_sha256"]) == 64
    assert h["vocab"] == 300 and h["file_type"] == 7


def test_gguf_header_head_dim_falls_back_to_embd_over_heads_and_no_template():
    data = make_gguf(key_len=0, template=False)
    hf = FakeHF()
    hf.add_repo("a/b", {"m.gguf": data})
    h = rc.read_gguf_header(hf, "https://huggingface.co/a/b/resolve/" + REV + "/m.gguf")
    assert h["head_dim"] == 64 and h["chat_template_present"] is False and h["chat_template_sha256"] is None


def test_gguf_header_spanning_many_range_requests(monkeypatch):
    monkeypatch.setattr(rc, "RANGE_CHUNK", 64)
    data = make_gguf(vocab=500)
    hf = FakeHF()
    hf.add_repo("a/b", {"m.gguf": data})
    h = rc.read_gguf_header(hf, "https://huggingface.co/a/b/resolve/" + REV + "/m.gguf")
    assert h["vocab"] == 500 and h["layers"] == 28
    ranges = [c[1]["Range"] for c in hf.calls]
    assert len(ranges) > 10 and ranges[0] == "bytes=0-63"


def test_gguf_server_ignoring_range_is_an_error(monkeypatch):
    monkeypatch.setattr(rc, "RANGE_CHUNK", 64)
    hf = FakeHF()
    hf.add_repo("a/b", {"m.gguf": make_gguf()})
    hf.ignore_range = True
    with pytest.raises(rc.RefreshError, match="ignored the Range"):
        rc.read_gguf_header(hf, "https://huggingface.co/a/b/resolve/" + REV + "/m.gguf")


def test_gguf_bad_magic_and_truncation():
    hf = FakeHF()
    hf.add_repo("a/b", {"m.gguf": b"NOPE" + b"\x00" * 100, "t.gguf": make_gguf(body_pad=0)[:200]})
    with pytest.raises(rc.RefreshError, match="bad magic"):
        rc.read_gguf_header(hf, "https://huggingface.co/a/b/resolve/" + REV + "/m.gguf")
    with pytest.raises(rc.RefreshError, match="end of GGUF"):
        rc.read_gguf_header(hf, "https://huggingface.co/a/b/resolve/" + REV + "/t.gguf")


# ----------------------------------------------------------------------------- glob + listing


def test_glob_is_case_insensitive_exact_one_and_excludes():
    files = {"Tiny-F16.gguf": {}, "Tiny-BF16.gguf": {}, "Tiny-Q8_0.gguf": {}, "README.md": {}}
    assert rc.match_one(files, "*f16.gguf", [])[0] is None  # ambiguous without exclude (F16 and BF16)
    assert rc.match_one(files, "*f16.gguf", ["*bf16.gguf"])[0] == "Tiny-F16.gguf"
    assert rc.match_one(files, "*q8_0.gguf", [])[0] == "Tiny-Q8_0.gguf"
    assert rc.match_one(files, "*q4_k_m.gguf", []) == (None, [])


def test_list_repo_from_documented_fixture_and_tree_fallback():
    doc = json.loads((FIX / "api_model_blobs.json").read_text())
    tree = json.loads((FIX / "api_tree.json").read_text())

    def fetch(url, headers, mx):
        if "/tree/" in url:
            return rc.Response(200, {}, json.dumps(tree).encode())
        return rc.Response(200, {}, json.dumps(doc).encode())

    L = rc.list_repo(fetch, "https://huggingface.co", "Example/Tiny-GGUF")
    assert L["revision"] == doc["sha"] and L["card_license"] == "apache-2.0"
    assert L["files"]["Tiny-Q8_0.gguf"] == {"size": 123456, "sha256": "a" * 64}
    # a .gguf without lfs data triggers the tree API
    doc2 = copy.deepcopy(doc)
    doc2["siblings"][3].pop("lfs")
    L2 = rc.list_repo(lambda u, h, m: rc.Response(200, {}, json.dumps(tree if "/tree/" in u else doc2).encode()), "https://huggingface.co", "x/y")
    assert L2["files"]["Tiny-Q8_0.gguf"]["sha256"] == "d" * 64


def test_list_repo_rejects_missing_revision_and_http_errors():
    with pytest.raises(rc.RefreshError, match="40-hex"):
        rc.list_repo(lambda u, h, m: rc.Response(200, {}, b'{"sha":"abc","siblings":[]}'), "https://huggingface.co", "x/y")
    with pytest.raises(rc.RefreshError, match="HTTP 403"):
        rc.list_repo(lambda u, h, m: rc.Response(403, {}, b""), "https://huggingface.co", "x/y")


# ----------------------------------------------------------------------------- license rule


def lic(text, cards, expected="Apache-2.0", canon_=None):
    return rc.decide_license(expected, text, "https://x/LICENSE", cards, canon_ or rc.load_canon(), NOW)


def test_verified_requires_byte_identical_text_and_agreeing_card():
    r = lic(APACHE, {"a/b": "apache-2.0", "a/c": "apache-2.0"})
    assert r["state"] == "VERIFIED" and r["spdx_id"] == "Apache-2.0" and r["evidence"]["match"] == "exact"
    assert r["evidence"]["license_text_sha256"] == "sha256:cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


@pytest.mark.parametrize("mutate", [lambda t: t + b"\n", lambda t: t.replace(b"Apache", b"Apach3", 1), lambda t: t.replace(b"\n", b"\r\n"), lambda t: t[:-1], lambda t: b" " + t])
def test_any_byte_difference_stays_unverified(mutate):
    r = lic(mutate(APACHE), {"a/b": "apache-2.0"})
    assert r["state"] == "UNVERIFIED" and r["evidence"]["match"] == "mismatch" and r["spdx_id"] is None


def test_card_disagreement_or_missing_card_stays_unverified():
    assert lic(APACHE, {"a/b": "apache-2.0", "a/c": "other"})["state"] == "UNVERIFIED"
    assert lic(APACHE, {"a/b": "apache-2.0", "a/c": None})["state"] == "UNVERIFIED"
    assert lic(APACHE, {"a/b": "mit"})["state"] == "UNVERIFIED"
    assert lic(APACHE, {})["state"] == "UNVERIFIED"


def test_no_text_unverified_and_expected_mismatch_unverified():
    r = lic(None, {"a/b": "apache-2.0"})
    assert r["state"] == "UNVERIFIED" and r["evidence"]["match"] == "unavailable"
    r = lic(MIT.replace(b"<year> <copyright holders>", b"2025 Someone"), {"a/b": "mit"}, expected="Apache-2.0")
    assert r["state"] == "UNVERIFIED" and any("expects" in x for x in r["reasons"])


def test_mit_template_with_one_copyright_line_verifies_only_for_mit():
    text = MIT.replace(b"<year> <copyright holders>", b"2025 Example Corp")
    r = lic(text, {"a/b": "mit"}, expected="MIT")
    assert r["state"] == "VERIFIED" and r["evidence"]["match"] == "mit_copyright_line"
    two = text.replace(b"\n\nPermission", b"\nCopyright (c) 2024 Other\n\nPermission", 1)
    assert lic(two, {"a/b": "mit"}, expected="MIT")["state"] == "UNVERIFIED"
    altered = text.replace(b"WITHOUT WARRANTY", b"WITH WARRANTY")
    assert lic(altered, {"a/b": "mit"}, expected="MIT")["state"] == "UNVERIFIED"


def test_non_canonical_license_never_verifies():
    llama = b"LLAMA 3.2 COMMUNITY LICENSE AGREEMENT ..."
    r = lic(llama, {"a/b": "llama3.2"}, expected="Llama-3.2")
    assert r["state"] == "UNVERIFIED"


def test_canonical_files_match_their_recorded_hashes():
    c = rc.load_canon()
    assert set(c) == {"Apache-2.0", "MIT"}
    for v in c.values():
        assert rc.sha256_hex(v["text"]) == v["sha256"] and v["source_url"].startswith("https://")


# ----------------------------------------------------------------------------- training class


def test_training_class_rules():
    assert rc.training_block("quantized", 0.135, "x-f16")["class"] == "inference_only"
    assert rc.training_block("quantized", 0.135, "x-f16")["training_source_artifact_id"] == "x-f16"
    assert rc.training_block("f16", 0.135, None)["class"] == "local_full"
    assert rc.training_block("f32", 0.36, None)["class"] == "local_full"
    assert rc.training_block("f16", 0.6, None)["class"] == "local_partial"
    assert rc.training_block("bf16", 1.7, None)["class"] == "external_only"
    assert rc.training_block("quantized", 8.0, None)["model_class_if_full_precision"] == "external_only"


# ----------------------------------------------------------------------------- end-to-end refresh


def test_refresh_writes_exact_immutable_artifacts(tmp_path, canon):
    hf = world()
    idx = run(tmp_path, hf, canon)
    assert idx["ok"] and not idx["errors"]
    a = load(tmp_path, "tiny-q8_0")
    assert a["refresh_state"] == "refreshed" and a["published"] is True
    src = a["source"]
    assert src["revision"] == REV and src["file"] == "Tiny-Q8_0.gguf"
    assert src["download_url"] == f"https://huggingface.co/Example/Tiny-GGUF/resolve/{REV}/Tiny-Q8_0.gguf"
    data = hf.repos["Example/Tiny-GGUF"]["files"]["Tiny-Q8_0.gguf"]
    import hashlib
    assert a["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest() and a["size_bytes"] == len(data)
    assert a["architecture"]["layers"] == 28 and a["architecture"]["name"] == "qwen3" and a["chat_template_present"] is True
    assert a["parameter_count"] == 1024 * 300 + 1024 * 1024
    assert a["runtime_compat"]["llama.cpp"]["supported"] == "yes" and a["runtime_compat"]["hag-engine"]["supported"] == "unverified"
    assert a["license"]["state"] == "VERIFIED" and a["license"]["evidence"]["license_text_url"].endswith(f"{REV}/LICENSE")
    assert a["training"]["class"] == "inference_only" and a["training"]["training_source_artifact_id"] == "tiny-f16"
    f16 = load(tmp_path, "tiny-f16")
    assert f16["source"]["file"] == "Tiny-F16.gguf" and f16["training"]["class"] in ("local_full", "local_partial")


def test_optional_missing_is_not_published_and_required_missing_is_error(tmp_path, canon):
    hf = world()
    idx = run(tmp_path, hf, canon)
    f32 = load(tmp_path, "tiny-f32")
    assert f32["published"] is False and f32["size_bytes"] is None and f32["sha256"] is None and f32["source"]["download_url"] is None
    assert any(r["id"] == "tiny-f32" and r["outcome"] == "not_published" for r in idx["results"])
    src = mini_sources()
    src["repos"][0]["files"][0]["glob"] = "*q5_k_s.gguf"
    out2 = tmp_path / "b"
    out2.mkdir()
    idx = rc.refresh(src, out2, hf, canon, NOW, {"tiny-q8_0"})
    assert not idx["ok"] and "matched 0 files" in idx["errors"][0]


def test_ambiguous_glob_is_an_error_not_a_guess(tmp_path, canon):
    hf = world()
    src = mini_sources()
    src["repos"][0]["files"][1]["exclude"] = []  # F16 glob now also matches BF16
    idx = rc.refresh(src, tmp_path, hf, canon, NOW, {"tiny-f16"})
    assert not idx["ok"] and "need exactly one" in idx["errors"][0]
    assert not (tmp_path / "tiny-f16.json").exists()


def test_unverifiable_license_still_publishes_artifact_as_unverified(tmp_path, canon):
    hf = world(license_text=APACHE + b" ")
    run(tmp_path, hf, canon)
    a = load(tmp_path, "tiny-q8_0")
    assert a["license"]["state"] == "UNVERIFIED" and a["license"]["evidence"]["match"] == "mismatch" and a["sha256"].startswith("sha256:")


def test_license_text_falls_back_to_second_url_and_missing_is_unverified(tmp_path, canon):
    hf = world(license_text=None)
    hf.texts["https://huggingface.co/Example/Tiny/resolve/main/LICENSE"] = APACHE
    run(tmp_path, hf, canon)
    assert load(tmp_path, "tiny-q8_0")["license"]["state"] == "VERIFIED"
    out2 = tmp_path / "b"
    out2.mkdir()
    run(out2, world(license_text=None), canon)
    assert load(out2, "tiny-q8_0")["license"]["state"] == "UNVERIFIED"


def test_failure_never_overwrites_a_refreshed_artifact(tmp_path, canon):
    run(tmp_path, world(), canon)
    before = (tmp_path / "tiny-q8_0.json").read_text()
    broken = FakeHF()  # every request 404s
    idx = rc.refresh(mini_sources(), tmp_path, broken, canon, "2027-01-01T00:00:00Z")
    assert not idx["ok"]
    assert (tmp_path / "tiny-q8_0.json").read_text() == before


def test_refresh_is_idempotent_and_picks_up_new_revision(tmp_path, canon):
    run(tmp_path, world(), canon)
    first = (tmp_path / "tiny-q8_0.json").read_text()
    rc.refresh(mini_sources(), tmp_path, world(), canon, "2027-01-01T00:00:00Z")
    assert (tmp_path / "tiny-q8_0.json").read_text() == first  # timestamps stay stable when nothing changed
    hf = world()
    hf.repos["Example/Tiny-GGUF"]["sha"] = REV2
    hf.texts[f"https://huggingface.co/Example/Tiny-GGUF/resolve/{REV2}/LICENSE"] = APACHE
    rc.refresh(mini_sources(), tmp_path, hf, canon, "2027-01-01T00:00:00Z")
    a = load(tmp_path, "tiny-q8_0")
    assert a["source"]["revision"] == REV2 and a["refreshed_at"] == "2027-01-01T00:00:00Z"


def test_tree_api_used_when_models_api_has_no_lfs(tmp_path, canon):
    hf = world(lfs=False)
    run(tmp_path, hf, canon)
    assert load(tmp_path, "tiny-q8_0")["sha256"].startswith("sha256:")
    assert any("/tree/" in u for u, _ in hf.calls)


def test_refresh_never_downloads_whole_models(tmp_path, canon):
    hf = world()
    run(tmp_path, hf, canon)
    for url, h in hf.calls:
        if url.endswith(".gguf"):
            assert "Range" in h


# ----------------------------------------------------------------------------- committed registry


def test_committed_artifacts_match_sources_and_unrefreshed_have_no_hashes():
    sources = json.loads((ROOT / "registry/artifacts/sources.json").read_text())
    out = ROOT / "registry/artifacts"
    assert rc.check_scaffold(sources, out) == []
    ids = {f["id"] for r in sources["repos"] for f in r["files"]}
    needed = {"smollm2-135m-instruct-q8_0", "smollm2-360m-instruct-q8_0", "qwen3-0.6b-q8_0", "qwen3-1.7b-q8_0", "qwen3-4b-q4_k_m", "qwen3-8b-q4_k_m"}
    assert needed <= ids
    for f in out.glob("*.json"):
        if f.name in ("sources.json", "index.json"):
            continue
        d = json.loads(f.read_text())
        if d["refresh_state"] == "unrefreshed":
            assert d["sha256"] is None and d["size_bytes"] is None and d["source"]["revision"] is None and d["source"]["download_url"] is None
            assert d["license"]["state"] == "UNVERIFIED"
        else:  # a CI-refreshed file must be fully populated
            assert d["sha256"].startswith("sha256:") and d["source"]["revision"] and d["source"]["download_url"].count(d["source"]["revision"]) == 1


def test_cli_scaffold_and_check(tmp_path):
    out = tmp_path / "art"
    src = tmp_path / "sources.json"
    src.write_text(json.dumps(mini_sources()))
    assert rc.main(["--sources", str(src), "--out", str(out), "--scaffold"]) == 0
    assert rc.main(["--sources", str(src), "--out", str(out), "--check-scaffold"]) == 0
    (out / "tiny-q8_0.json").unlink()
    assert rc.main(["--sources", str(src), "--out", str(out), "--check-scaffold"]) == 1
