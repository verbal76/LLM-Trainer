"""Reference consumer for LLM Trainer export packages.

STANDARD LIBRARY ONLY. This file deliberately imports nothing from
``llmtrainer`` so that any application can copy it (or reimplement it from
the package contract) and import a specialist without the factory.

    pkg = Package.open("path/to/package")     # verifies every hash, raises PackageError otherwise
    pkg.identity, pkg.is_stub, pkg.license
    hits = pkg.search("torque", limit=3)       # exact-reference chunks with provenance
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST = "export_package.json"
SUPPORTED_SCHEMA = 1


class PackageError(Exception):
    pass


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _verify_sealed(doc: dict, label: str) -> None:
    body = {k: v for k, v in doc.items() if k != "content_hash"}
    if doc.get("content_hash") != _sha(_canonical(body)):
        raise PackageError(f"{label}: content_hash mismatch")


class Package:
    def __init__(self, root: Path, manifest: dict, model: dict):
        self.root = root
        self.manifest = manifest
        self.model = model
        self._chunks: list[dict] | None = None
        self._index: dict[str, list[str]] | None = None

    @classmethod
    def open(cls, path: str | Path) -> "Package":
        root = Path(path)
        mp = root / MANIFEST
        if not mp.is_file():
            raise PackageError("not an LLM Trainer export package (manifest missing)")
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        if manifest.get("kind") != "export_package" or manifest.get("schema_version") != SUPPORTED_SCHEMA:
            raise PackageError("unsupported package kind or schema_version")
        _verify_sealed(manifest, MANIFEST)
        for f in manifest["files"]:
            rel = PurePosixPath(f["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise PackageError(f"unsafe path in manifest: {f['path']}")
            fp = root / f["path"]
            if not fp.is_file():
                raise PackageError(f"missing file: {f['path']}")
            if _sha(fp.read_bytes()) != f["sha256"]:
                raise PackageError(f"hash mismatch: {f['path']}")
        gate = manifest["license_gate"]
        if not gate["allowed"] or not gate["requested_use"].get("redistribute"):
            raise PackageError("package does not carry an allowed redistribution license gate")
        pkg = cls(root, manifest, {})
        model = json.loads(pkg._role_path("model_manifest").read_text(encoding="utf-8"))
        _verify_sealed(model, "model manifest")
        if model["content_hash"] != manifest["model_manifest_hash"]:
            raise PackageError("model manifest does not match export manifest")
        pkg.model = model
        if not manifest["is_pipeline_validation_stub"]:
            pkg._check_real_specialist()
        return pkg

    def _check_real_specialist(self) -> None:
        m = self.model
        if m.get("local_experiment"):
            raise PackageError("package derives from a local-experiment (unverified license) run")
        if m["base_model"].get("license_state") != "VERIFIED":
            raise PackageError(f"base model license_state is {m['base_model'].get('license_state')!r}, not VERIFIED")
        declared = {a["path"]: a["sha256"] for a in m["artifacts"]}
        for f in self.manifest["files"]:
            if f["role"] == "adapter" and declared.get(f["path"]) != f["sha256"]:
                raise PackageError(f"adapter file not matching model manifest: {f['path']}")
        if not m.get("training_config_hash") or _sha(_canonical(m["training_config"])) != m["training_config_hash"]:
            raise PackageError("training config hash mismatch")

    def _role_path(self, role: str) -> Path:
        for f in self.manifest["files"]:
            if f["role"] == role:
                return self.root / f["path"]
        raise PackageError(f"role not present: {role}")

    def paths(self, role: str) -> list[Path]:
        return [self.root / f["path"] for f in self.manifest["files"] if f["role"] == role]

    @property
    def identity(self) -> str:
        return self.manifest["model_identity"]

    @property
    def is_stub(self) -> bool:
        return bool(self.manifest["is_pipeline_validation_stub"])

    @property
    def license(self) -> dict:
        return json.loads(self._role_path("license_bundle").read_text(encoding="utf-8"))

    @property
    def runtime_formats(self) -> list[str]:
        return list(self.manifest["runtime_formats"])

    @property
    def evaluation(self) -> dict:
        return json.loads(self._role_path("evaluation_report").read_text(encoding="utf-8"))

    @property
    def base_model(self) -> dict:
        """Base-model identity + license state the adapter must be applied to (weights are NOT in the package)."""
        return dict(self.model["base_model"])

    @property
    def license_state(self) -> str | None:
        return self.model["base_model"].get("license_state")

    @property
    def adapters(self) -> list[dict]:
        return [{"path": str(self.root / a["path"]), "sha256": a["sha256"], "size_bytes": a["size_bytes"]}
                for a in self.model["artifacts"] if a["role"] == "adapter"]

    @property
    def tokenizer_files(self) -> dict[str, str]:
        return dict(self.model["tokenizer"]["files"])

    @property
    def training_config_hash(self) -> str | None:
        return self.model.get("training_config_hash")

    @property
    def quantization(self) -> dict | None:
        return self.model.get("quantization")

    @property
    def requirements(self) -> dict:
        return dict(self.model["runtime"])

    def has_reference(self) -> bool:
        return self.manifest.get("reference") is not None

    def _load_reference(self) -> None:
        if self._chunks is not None:
            return
        if not self.has_reference():
            self._chunks, self._index = [], {}
            return
        ref = self.manifest["reference"]
        lines = (self.root / ref["chunks_path"]).read_text(encoding="utf-8").splitlines()
        self._chunks = [json.loads(line) for line in lines if line]
        self._index = json.loads((self.root / ref["index_path"]).read_text(encoding="utf-8"))

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Lexical lookup. Each hit carries source/section/page provenance for citation."""
        self._load_reference()
        assert self._chunks is not None and self._index is not None
        toks = set(re.findall(r"[a-z0-9]+", query.lower()))
        score: dict[str, int] = {}
        for t in toks:
            for ref in self._index.get(t, []):
                score[ref] = score.get(ref, 0) + 1
        by_ref = {c["ref"]: c for c in self._chunks}
        ranked = sorted(score, key=lambda r: (-score[r], r))[:limit]
        return [{**by_ref[r], "score": score[r]} for r in ranked]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: consumer.py PACKAGE_DIR [QUERY]")
        return 2
    try:
        pkg = Package.open(argv[0])
    except PackageError as e:
        print(f"REFUSED: {e}")
        return 1
    print(f"imported {pkg.identity} stub={pkg.is_stub} formats={pkg.runtime_formats}")
    if len(argv) > 1:
        for hit in pkg.search(" ".join(argv[1:])):
            print(f"- {hit['ref']} [{hit['section']} p{hit['page_start']}] {hit['text'][:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
