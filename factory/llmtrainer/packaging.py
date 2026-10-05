"""Export package contract: builder and validator.

Layout (all paths relative, forward slashes)::

    export_package.json            ExportPackage manifest (hash-sealed; lists every other file)
    model/                         weights or adapter
    tokenizer/
    manifests/                     model, source, dataset, runtime manifests
    evaluation/evaluation_run.json
    license/                       license entry, gate result, ATTRIBUTION.md
    reference/chunks.jsonl         exact-reference chunks with provenance (redistributable sources only)
    reference/index.json           token -> chunk ids lexical index
    README.md

A consumer needs only the standard library (see ``consumer.py``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .hashing import hash_file, hash_obj, hash_text, sha256_bytes
from .licenses import RequestedUse, evaluate_gate
from .schemas import (
    BaseModelLicenseEntry,
    DatasetManifest,
    EvaluationRun,
    ExportPackage,
    GateResult,
    ModelManifest,
    PackageFile,
    ReferencePackageInfo,
    SourceManifest,
)

MANIFEST_NAME = "export_package.json"
REQUIRED_ROLES = (
    "model_manifest",
    "evaluation_report",
    "provenance_source_manifest",
    "provenance_dataset_manifest",
    "license_bundle",
    "license_gate",
    "runtime_manifest",
    "tokenizer",
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_reference_store(
    sources: SourceManifest, chunk_texts: dict[tuple[str, str], str]
) -> tuple[bytes, bytes, int, list[str]]:
    """Return (chunks.jsonl, index.json, n_chunks, excluded source ids)."""
    rows, excluded = [], []
    index: dict[str, list[str]] = {}
    for s in sources.active_sources():
        if s.rights.permitted_redistribution != "yes":
            excluded.append(s.source_id)
            continue
        for c in s.chunks:
            text = chunk_texts[(s.source_id, c.chunk_id)]
            key = f"{s.source_id}/{c.chunk_id}"
            rows.append(
                {
                    "ref": key,
                    "source_id": s.source_id,
                    "source_title": s.title,
                    "chunk_id": c.chunk_id,
                    "section": c.section,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "text": text,
                    "text_sha256": hash_text(text),
                }
            )
            for tok in set(tokenize(text)):
                index.setdefault(tok, []).append(key)
    chunks = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    idx = json.dumps({k: sorted(v) for k, v in sorted(index.items())}, sort_keys=True).encode("utf-8")
    return chunks, idx, len(rows), sorted(excluded)


@dataclass
class PackageBuilder:
    out_dir: Path
    files: list[PackageFile] = field(default_factory=list)

    def add(self, role: str, rel: str, data: bytes) -> None:
        p = PurePosixPath(rel)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe package path: {rel}")
        target = self.out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.files.append(PackageFile(path=rel, sha256=sha256_bytes(data), size_bytes=len(data), role=role))  # type: ignore[arg-type]

    def add_file(self, role: str, rel: str, src: Path) -> None:
        self.add(role, rel, src.read_bytes())


def attribution_text(entry: BaseModelLicenseEntry, name: str) -> str:
    lines = [
        f"# Attribution and license notice for {name}",
        "",
        f"Base model: {entry.model_family} {entry.exact_version}",
        f"License: {entry.license_id} ({entry.license_url})",
        f"Attribution required: {entry.attribution_required}",
        "",
        "Restrictions recorded for the base model:",
        *(f"- {r}" for r in entry.restrictions or ["(none recorded)"]),
        "",
        f"License verification: {entry.verification.state} ({entry.verification.evidence_level}), "
        f"license text {entry.verification.license_text_sha256 or 'not hashed'}",
        "License facts last verified: " + str(entry.verification.verified_on),
    ]
    if entry.verification.uncertainties:
        lines += ["", "Known uncertainties:", *(f"- {u}" for u in entry.verification.uncertainties)]
    return "\n".join(lines) + "\n"


def finalize_package(
    builder: PackageBuilder,
    *,
    name: str,
    version: str,
    grade: str,
    model: ModelManifest,
    sources: SourceManifest,
    dataset: DatasetManifest,
    evaluations: list[EvaluationRun],
    gate: GateResult,
    reference: ReferencePackageInfo | None,
    runtime_formats: list[str],
    created_on: str,
) -> ExportPackage:
    pkg = ExportPackage(
        package_name=name,
        package_version=version,
        export_grade=grade,
        model_identity=model.model_identity,
        model_manifest_hash=model.content_hash,
        source_manifest_hash=sources.content_hash,
        dataset_hash=dataset.content_hash,
        evaluation_hashes=[e.content_hash for e in evaluations],
        license_gate=gate,
        files=sorted(builder.files, key=lambda f: f.path),
        reference=reference,
        runtime_formats=runtime_formats,
        is_pipeline_validation_stub=model.is_pipeline_validation_stub,
        created_on=created_on,
    ).seal()
    (builder.out_dir / MANIFEST_NAME).write_text(pkg.to_json(), encoding="utf-8")
    return pkg


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def _load(path: Path, cls, rep: ValidationReport, label: str):
    try:
        obj = cls.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - report any parse/validation failure
        rep.err(f"{label}: cannot load ({str(e).splitlines()[0]})")
        return None
    if hasattr(obj, "verify") and not obj.verify():
        rep.err(f"{label}: content_hash does not match content")
    return obj


def validate_package(root: str | Path) -> ValidationReport:
    root = Path(root)
    rep = ValidationReport()
    mpath = root / MANIFEST_NAME
    if not mpath.is_file():
        rep.err(f"{MANIFEST_NAME} missing")
        return rep
    pkg = _load(mpath, ExportPackage, rep, MANIFEST_NAME)
    if pkg is None:
        return rep

    # 1. file inventory
    declared = {}
    for f in pkg.files:
        p = PurePosixPath(f.path)
        if p.is_absolute() or ".." in p.parts or f.path == MANIFEST_NAME:
            rep.err(f"unsafe or reserved path: {f.path}")
            continue
        if f.path in declared:
            rep.err(f"duplicate path: {f.path}")
        declared[f.path] = f
        fp = root / f.path
        if not fp.is_file():
            rep.err(f"missing file: {f.path}")
            continue
        if fp.stat().st_size != f.size_bytes:
            rep.err(f"size mismatch: {f.path}")
        if hash_file(fp) != f.sha256:
            rep.err(f"hash mismatch: {f.path}")
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {MANIFEST_NAME}
    for extra in sorted(on_disk - set(declared)):
        rep.err(f"undeclared file: {extra}")

    by_role: dict[str, list[PackageFile]] = {}
    for f in pkg.files:
        by_role.setdefault(f.role, []).append(f)
    for role in REQUIRED_ROLES:
        if role not in by_role:
            rep.err(f"required role missing: {role}")
    if "model_weights" not in by_role and "adapter" not in by_role:
        rep.err("package contains neither model_weights nor adapter")
    if pkg.license_gate.attribution_required != "no" and "attribution" not in by_role:
        rep.err("attribution is required (or unverified) but no attribution file is included")
    if (pkg.reference is None) != ("reference_chunks" not in by_role and "reference_index" not in by_role):
        rep.err("reference info and reference files are inconsistent")
    if rep.errors and any(e.startswith(("required role", "missing file")) for e in rep.errors):
        return rep

    def one(role: str) -> Path:
        return root / by_role[role][0].path

    model = _load(one("model_manifest"), ModelManifest, rep, "model manifest")
    sources = _load(one("provenance_source_manifest"), SourceManifest, rep, "source manifest")
    dataset = _load(one("provenance_dataset_manifest"), DatasetManifest, rep, "dataset manifest")
    evalrun = _load(one("evaluation_report"), EvaluationRun, rep, "evaluation run")
    entry = _load(one("license_bundle"), BaseModelLicenseEntry, rep, "license entry")
    if not all((model, sources, dataset, evalrun, entry)):
        return rep

    # 2. identity / hash chain
    if model.content_hash != pkg.model_manifest_hash:
        rep.err("model manifest hash differs from export manifest")
    if model.model_identity != pkg.model_identity:
        rep.err("model identity differs from export manifest")
    if sources.content_hash != pkg.source_manifest_hash or model.source_manifest_hash != sources.content_hash:
        rep.err("source manifest hash chain broken")
    if dataset.content_hash != pkg.dataset_hash or model.dataset_hash != dataset.content_hash:
        rep.err("dataset hash chain broken")
    if dataset.source_manifest_hash != sources.content_hash:
        rep.err("dataset was not built from the packaged source manifest")
    if evalrun.content_hash not in pkg.evaluation_hashes or evalrun.content_hash not in model.evaluation_hashes:
        rep.err("evaluation run hash not referenced by both model manifest and export manifest")
    if evalrun.dataset_hash != dataset.content_hash:
        rep.err("evaluation was run against a different dataset")
    for a in model.artifacts:
        f = declared.get(a.path)
        if f is None or f.sha256 != a.sha256:
            rep.err(f"model manifest artifact not matching package file: {a.path}")

    # 3. license: recompute, never trust the stored verdict
    from .licenses import entry_hash

    if entry_hash(entry) != pkg.license_gate.entry_hash or entry_hash(entry) != model.base_model.license_entry_hash:
        rep.err("license entry hash mismatch")
    use = RequestedUse(fine_tune=True, produce_adapter=True, redistribute=True)
    recomputed = evaluate_gate(entry, use)
    if not recomputed.allowed:
        rep.err("license gate does not permit redistribution: " + "; ".join(recomputed.blocking))
    if not pkg.license_gate.allowed or not pkg.license_gate.requested_use.get("redistribute"):
        rep.err("export manifest license gate is not an allowed redistribution check")

    # 4. evaluation honesty
    if evalrun.train_eval_group_overlap != 0:
        rep.err("evaluation set overlaps training groups")
    if evalrun.evaluator.is_stub != pkg.is_pipeline_validation_stub or model.is_pipeline_validation_stub != pkg.is_pipeline_validation_stub:
        rep.err("stub flags are inconsistent across evaluation, model manifest and package")
    if pkg.is_pipeline_validation_stub:
        rep.warnings.append("PIPELINE-VALIDATION STUB package: not a real model; makes no quality claim")

    # 5. dataset leakage
    lr = dataset.leakage_report
    if lr.group_overlap_pairs or lr.residual_cross_split_near_duplicates:
        rep.err("dataset manifest reports unresolved train/eval leakage")

    # 6. provenance and removal
    active = {s.source_id for s in sources.active_sources()}
    for ex in dataset.examples:
        for ref in ex.derived_from:
            if ref.source_id not in active:
                rep.err(f"example {ex.example_id} derives from removed/unknown source {ref.source_id}")
                break

    # 7. reference store
    if pkg.reference is not None:
        valid_chunks = {(s.source_id, c.chunk_id): c.text_sha256 for s in sources.active_sources() for c in s.chunks}
        n = 0
        refs = set()
        for line in (root / pkg.reference.chunks_path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            n += 1
            refs.add(row["ref"])
            if hash_text(row["text"]) != row["text_sha256"]:
                rep.err(f"reference chunk text hash mismatch: {row['ref']}")
            if valid_chunks.get((row["source_id"], row["chunk_id"])) != row["text_sha256"]:
                rep.err(f"reference chunk not in active source manifest: {row['ref']}")
        if n != pkg.reference.chunk_count:
            rep.err("reference chunk_count mismatch")
        index = json.loads((root / pkg.reference.index_path).read_text(encoding="utf-8"))
        if any(r not in refs for v in index.values() for r in v):
            rep.err("reference index points at unknown chunks")
        rights = {s.source_id: s.rights.permitted_redistribution for s in sources.active_sources()}
        for r in refs:
            if rights.get(r.split("/")[0]) != "yes":
                rep.err(f"reference chunk from source without redistribution rights: {r}")
    return rep


def canonical_package_identity(pkg: ExportPackage) -> str:
    return hash_obj([f.model_dump(mode="json") for f in pkg.files])
