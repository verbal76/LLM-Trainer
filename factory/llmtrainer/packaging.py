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
    ArtifactFile,
    BaseModelLicenseEntry,
    BaseModelRef,
    ContextAssumptions,
    QuantizationInfo,
    RuntimeRequirements,
    TokenizerInfo,
    TrainingRun,
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
# Real specialist package (adapter + tokenizer + training run + evaluation)
# --------------------------------------------------------------------------- #


def build_specialist_package(
    out_dir: Path,
    *,
    run: TrainingRun,
    run_output_dir: Path,
    evalrun: EvaluationRun,
    entry: BaseModelLicenseEntry,
    sources: SourceManifest,
    dataset: DatasetManifest,
    chunk_texts: dict[tuple[str, str], str],
    specialist_name: str,
    specialist_version: str,
    grade: str,
    created_on: str,
    runtime_formats: tuple[str, ...] = ("hf-peft-adapter",),
    runtime_notes: tuple[str, ...] = (),
    quantization: QuantizationInfo | None = None,
    known_limitations: tuple[str, ...] = (),
    commercial: bool = False,
) -> ExportPackage:
    """Refuses unless the base-model license is VERIFIED for redistribution and the run was not a local experiment."""
    from .licenses import entry_hash
    from .specialize.gating import PackagingRefused, redistribution_gate

    if run.local_experiment_unverified_license:
        raise PackagingRefused("run was a local experiment under an unverified license; it can never be packaged")
    if run.status != "completed" or run.is_pipeline_validation_stub or run.executor == "stub":
        raise PackagingRefused("only a completed real training run can be packaged (use the stub exporter for plumbing checks)")
    gate, state = redistribution_gate(entry, commercial=commercial)
    if state != "VERIFIED" or not gate.allowed:
        raise PackagingRefused(f"base-model license is {state} for redistribution: " + "; ".join(gate.blocking))
    if run.base_model_id != entry.entry_id or run.base_model_license_hash != entry_hash(entry):
        raise PackagingRefused("training run was made against a different base-model license entry; re-gate and retrain")
    if run.dataset_hash != dataset.content_hash or run.source_manifest_hash != sources.content_hash:
        raise PackagingRefused("training run does not match the dataset/source manifests being packaged")
    if evalrun.dataset_hash != dataset.content_hash or evalrun.evaluator.is_stub or evalrun.train_eval_group_overlap != 0:
        raise PackagingRefused("evaluation run is stub, leaky, or for a different dataset")
    adapter = {k: v for k, v in run.output_artifacts.items() if k.startswith("adapter/")}
    tok = {k: v for k, v in run.output_artifacts.items() if k.startswith("tokenizer/")}
    if not adapter or not tok:
        raise PackagingRefused("training run has no adapter and/or tokenizer artifacts")
    if out_dir.exists():
        raise PackagingRefused(f"{out_dir} already exists; packages are never silently overwritten")

    blobs = {}
    for name, h in {**adapter, **tok}.items():
        data = (run_output_dir / name).read_bytes()
        if sha256_bytes(data) != h:
            raise PackagingRefused(f"artifact changed since training: {name}")
        blobs[name] = data
    out_dir.mkdir(parents=True)
    b = PackageBuilder(out_dir)
    artifacts, tok_files = [], {}
    for name in sorted(adapter):
        b.add("adapter", f"model/{name}", blobs[name])
        artifacts.append(ArtifactFile(path=f"model/{name}", sha256=adapter[name], size_bytes=len(blobs[name]), role="adapter"))
    for name in sorted(tok):
        b.add("tokenizer", name, blobs[name])
        tok_files[name] = tok[name]
    hp = run.hyperparameters
    ctx = int(hp.get("max_seq_len", 0)) or 1
    runtime = RuntimeRequirements(
        runtimes=["transformers+peft"], formats=list(runtime_formats),
        notes=["Adapter only: requires the base model weights, obtained separately under their own license.", *runtime_notes],
    )
    model = ModelManifest(
        specialist_name=specialist_name, specialist_version=specialist_version,
        base_model=BaseModelRef(model_family=entry.model_family, exact_version=entry.exact_version, license_id=entry.license_id,
                                license_entry_hash=entry_hash(entry), license_state="VERIFIED"),
        adapter_method=run.method, dataset_version=dataset.dataset_version, dataset_hash=dataset.content_hash,
        source_manifest_version=sources.manifest_version, source_manifest_hash=sources.content_hash,
        training_run_hash=run.content_hash, training_config=hp, training_config_hash=run.config_hash,
        tokenizer=TokenizerInfo(name=str(hp.get("base_model_path", entry.entry_id)), files=tok_files),
        context=ContextAssumptions(trained_context_tokens=ctx, max_supported_context_tokens=ctx, recommended_context_tokens=ctx),
        quantization=quantization, runtime=runtime, evaluation_hashes=[evalrun.content_hash], export_date=created_on,
        artifacts=artifacts, target_profile=grade,
        known_limitations=list(known_limitations) or [
            "Evaluation uses deterministic lexical heuristics; see evaluation notes for limits and sample sizes.",
            "Exact facts (torque, clearances, ...) must come from the reference package, not model weights.",
        ],
    ).seal()
    b.add("model_manifest", "manifests/model_manifest.json", model.to_json().encode())
    b.add("training_run", "manifests/training_run.json", run.to_json().encode())
    b.add("provenance_source_manifest", "manifests/source_manifest.json", sources.to_json().encode())
    b.add("provenance_dataset_manifest", "manifests/dataset_manifest.json", dataset.to_json().encode())
    b.add("runtime_manifest", "manifests/runtime_manifest.json", json.dumps(runtime.model_dump(mode="json"), indent=2, sort_keys=True).encode())
    b.add("evaluation_report", "evaluation/evaluation_run.json", evalrun.to_json().encode())
    b.add("license_bundle", "license/base_model_license_entry.json", json.dumps(entry.model_dump(mode="json"), indent=2, sort_keys=True).encode())
    b.add("license_gate", "license/license_gate.json", json.dumps(gate.model_dump(mode="json"), indent=2, sort_keys=True).encode())
    b.add("attribution", "license/ATTRIBUTION.md", attribution_text(entry, specialist_name).encode())
    chunks, index, n, excluded = build_reference_store(sources, chunk_texts)
    ref = None
    if n:
        b.add("reference_chunks", "reference/chunks.jsonl", chunks)
        b.add("reference_index", "reference/index.json", index)
        ref = ReferencePackageInfo(chunk_count=n, chunks_path="reference/chunks.jsonl", index_path="reference/index.json", excluded_source_ids=excluded)
    b.add("readme", "README.md", (f"# {model.model_identity}\n\nLoRA/QLoRA adapter specialist. Needs the base model ({entry.entry_id}) "
                                  "separately. Verify with `python consumer.py <dir>` (stdlib only).\n").encode())
    return finalize_package(b, name=specialist_name, version=specialist_version, grade=grade, model=model, sources=sources,
                            dataset=dataset, evaluations=[evalrun], gate=gate, reference=ref, runtime_formats=list(runtime_formats),
                            created_on=created_on)


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

    # 3b. real-specialist requirements
    if not pkg.is_pipeline_validation_stub:
        _validate_real_specialist(root, model, evalrun, by_role, declared, rep)

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


def _validate_real_specialist(root, model, evalrun, by_role, declared, rep: ValidationReport) -> None:
    if model.local_experiment:
        rep.err("package derives from a local-experiment (unverified license) run")
    if model.base_model.license_state != "VERIFIED":
        rep.err(f"base model license_state must be VERIFIED for redistribution, got {model.base_model.license_state}")
    if "training_run" not in by_role or "adapter" not in by_role:
        rep.err("real specialist package needs training_run and adapter files")
        return
    run = _load(root / by_role["training_run"][0].path, TrainingRun, rep, "training run")
    if run is None:
        return
    if run.content_hash != model.training_run_hash:
        rep.err("training run hash differs from model manifest")
    if run.local_experiment_unverified_license:
        rep.err("training run was a local experiment under an unverified license")
    if run.is_pipeline_validation_stub or run.status != "completed":
        rep.err("training run is a stub or did not complete")
    if run.dataset_hash != model.dataset_hash:
        rep.err("training run dataset differs from model manifest")
    if not model.training_config_hash or hash_obj(model.training_config) != model.training_config_hash or run.config_hash != model.training_config_hash:
        rep.err("training config hash does not match training config")
    manifest_paths = {a.path for a in model.artifacts}
    for f in by_role["adapter"]:
        if f.path not in manifest_paths:
            rep.err(f"adapter file not listed with hash in model manifest: {f.path}")
    for name, h in model.tokenizer.files.items():
        f = declared.get(name)
        if f is None or f.sha256 != h:
            rep.err(f"tokenizer file mismatch: {name}")
    if evalrun.evaluator.is_stub:
        rep.err("real specialist package carries a stub evaluation")
    if evalrun.improvement_claim_allowed and (evalrun.n_eval_examples < 50 or any("forced to false" in n for n in evalrun.notes)):
        rep.err("evaluation allows an improvement claim it cannot support")


def canonical_package_identity(pkg: ExportPackage) -> str:
    return hash_obj([f.model_dump(mode="json") for f in pkg.files])
