"""Workspace operations used by the CLI (and any future UI): project, sources, datasets, experiments, export.

Directory layout of a project workspace::

    project.json
    registry/                 base-model license entries (*.json)
    sources/raw/<id>/<file>   original bytes, preserved unmodified
    sources/source_manifest.json
    corpus/<id>.jsonl         derived chunk text
    datasets/<dataset_id>/    split files + dataset_manifest.json
    runs/<run_id>/            training run + adapter outputs
    evals/<eval_id>.json
    models/<identity>/model_manifest.json
    exports/<name>/           export packages
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from . import evaluation as ev
from . import training as tr
from .clock import now_iso, today
from .datasets import build_dataset, load_split, verify_no_leakage
from .hashing import hash_file, hash_obj, short
from .ingest import ingest_bytes
from .licenses import LicenseRegistry, RequestedUse, enforce_gate, entry_hash, gate_sources_for_training
from .packaging import PackageBuilder, attribution_text, build_reference_store, finalize_package
from .provenance import plan_removal
from .schemas import (
    ArtifactFile,
    BaseModelRef,
    ContextAssumptions,
    DatasetManifest,
    EvalSubject,
    EvaluationRun,
    ExportPackage,
    ModelManifest,
    QuantizationInfo,
    ReferencePackageInfo,
    RemovalRecord,
    RightsInfo,
    RuntimeRequirements,
    SourceManifest,
    SpecialistProject,
    TokenizerInfo,
    TrainingRun,
)

STUB_MODEL_ID = "pipeline-validation-stub@1"
SYNTHETIC_RIGHTS = RightsInfo(
    status="synthetic",
    license_id="LLM-Trainer-synthetic-fixture",
    permitted_training="yes",
    permitted_commercial="yes",
    permitted_redistribution="yes",
    evidence="Authored in-repo as a synthetic fixture; contains no third-party material.",
)


class WorkspaceError(RuntimeError):
    pass


@dataclass
class Workspace:
    root: Path

    @property
    def project_file(self) -> Path:
        return self.root / "project.json"

    @property
    def manifest_file(self) -> Path:
        return self.root / "sources" / "source_manifest.json"

    def project(self) -> SpecialistProject:
        if not self.project_file.is_file():
            raise WorkspaceError(f"not a project workspace: {self.root}")
        p = SpecialistProject.model_validate_json(self.project_file.read_text(encoding="utf-8"))
        if not p.verify():
            raise WorkspaceError("project.json content_hash mismatch (edited by hand?)")
        return p

    def sources(self) -> SourceManifest:
        m = SourceManifest.model_validate_json(self.manifest_file.read_text(encoding="utf-8"))
        if not m.verify():
            raise WorkspaceError("source manifest content_hash mismatch")
        return m

    def save_sources(self, m: SourceManifest) -> SourceManifest:
        m = m.seal()
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_file.write_text(m.to_json(), encoding="utf-8")
        return m

    def datasets(self) -> list[DatasetManifest]:
        out = []
        for f in sorted((self.root / "datasets").glob("*/dataset_manifest.json")):
            out.append(DatasetManifest.model_validate_json(f.read_text(encoding="utf-8")))
        return out

    def model_manifests(self) -> list[ModelManifest]:
        return [
            ModelManifest.model_validate_json(f.read_text(encoding="utf-8"))
            for f in sorted((self.root / "models").glob("*/model_manifest.json"))
        ]

    def registry(self) -> LicenseRegistry:
        return LicenseRegistry.load(self.root / "registry")


def init_project(root: Path, name: str, domain: str, *, seed: int = 1234, description: str = "") -> SpecialistProject:
    root = Path(root)
    if (root / "project.json").exists():
        raise WorkspaceError(f"project already exists at {root}")
    for d in ("registry", "sources/raw", "corpus", "datasets", "runs", "evals", "models", "exports"):
        (root / d).mkdir(parents=True, exist_ok=True)
    slug = "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")
    project = SpecialistProject(
        project_id=f"proj-{slug}", name=name, domain=domain, description=description, created_on=today(), seed=seed
    ).seal()
    (root / "project.json").write_text(project.to_json(), encoding="utf-8")
    stub = resources.files("llmtrainer").joinpath("data/registry/pipeline-validation-stub.json").read_text(encoding="utf-8")
    (root / "registry" / "pipeline-validation-stub.json").write_text(stub, encoding="utf-8")
    ws = Workspace(root)
    ws.save_sources(SourceManifest(project_id=project.project_id))
    return project


def add_source(
    ws: Workspace, file: Path, *, title: str, origin: str, rights: RightsInfo, source_version: str | None = None
) -> SourceManifest:
    raw = Path(file).read_bytes()
    record, chunks = ingest_bytes(
        raw, filename=Path(file).name, title=title, origin=origin, rights=rights, ingested_on=today(), source_version=source_version
    )
    manifest = ws.sources()
    if any(s.source_id == record.source_id for s in manifest.sources):
        existing = manifest.get(record.source_id)
        if existing.status == "active":
            raise WorkspaceError(f"source already present: {record.source_id}")
        raise WorkspaceError(f"source {record.source_id} was removed; re-adding removed material needs an explicit new decision")
    raw_dir = ws.root / "sources" / "raw" / record.source_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / record.original_filename).write_bytes(raw)
    with (ws.root / "corpus" / f"{record.source_id}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for c in chunks:
            fh.write(json.dumps({"chunk_id": c.record.chunk_id, "text": c.text}, sort_keys=True, ensure_ascii=False) + "\n")
    return ws.save_sources(
        manifest.model_copy(update={"sources": [*manifest.sources, record], "manifest_version": manifest.manifest_version + 1})
    )


def _removal_manifest(manifest: SourceManifest, source_ids: list[str], reason: str) -> SourceManifest:
    new = []
    for s in manifest.sources:
        if s.source_id in source_ids and s.status == "active":
            s = s.model_copy(
                update={
                    "status": "removed",
                    "chunks": [],
                    "removal": RemovalRecord(removed_on=today(), reason=reason, removed_chunk_ids=[c.chunk_id for c in s.chunks]),
                }
            )
        new.append(s)
    return manifest.model_copy(update={"sources": new, "manifest_version": manifest.manifest_version + 1})


def removal_plan(ws: Workspace, source_ids: list[str]):
    packages = [
        ExportPackage.model_validate_json(f.read_text(encoding="utf-8")) for f in sorted((ws.root / "exports").glob("*/export_package.json"))
    ]
    models = []
    ds_by_hash = {d.content_hash: d for d in ws.datasets()}
    for m in ws.model_manifests():
        models.append((m, ds_by_hash.get(m.dataset_hash, m.dataset_hash)))
    return plan_removal(ws.sources(), source_ids, list(ds_by_hash.values()), models, packages)


def remove_source(ws: Workspace, source_ids: list[str], reason: str):
    """Compute the rebuild plan, then tombstone sources and delete raw + derived corpus files."""
    plan = removal_plan(ws, source_ids)
    manifest = _removal_manifest(ws.sources(), source_ids, reason)
    for sid in source_ids:
        shutil.rmtree(ws.root / "sources" / "raw" / sid, ignore_errors=True)
        (ws.root / "corpus" / f"{sid}.jsonl").unlink(missing_ok=True)
    ws.save_sources(manifest)
    return plan


def build_dataset_for_project(ws: Workspace, *, allow_blocked_sources: bool = False) -> DatasetManifest:
    project = ws.project()
    manifest = ws.sources()
    allowed, blocked = gate_sources_for_training(manifest)
    if blocked and not allow_blocked_sources:
        raise WorkspaceError("sources not cleared for training (rights gate):\n  " + "\n  ".join(blocked))
    if not allowed:
        raise WorkspaceError("no sources cleared for training")
    return build_dataset(
        manifest,
        ws.root / "corpus",
        ws.root / "datasets",
        project_id=project.project_id,
        seed=project.seed,
        config=project.split_config,
        allowed_source_ids=allowed,
        excluded=[b.split(":")[0] for b in blocked],
    )


def latest_dataset(ws: Workspace) -> DatasetManifest:
    ds = [d for d in ws.datasets() if d.source_manifest_hash == ws.sources().content_hash]
    if not ds:
        raise WorkspaceError("no dataset built from the current source manifest; run build-dataset first")
    return ds[-1]


def run_experiment(ws: Workspace, cfg: tr.ExperimentConfig, *, export: bool = True) -> dict:
    """End to end with the STUB trainer/evaluator: gate -> train -> evaluate -> model manifest -> export."""
    if cfg.method != "stub":
        raise WorkspaceError("only the pipeline-validation stub executes locally; real training is an external executor")
    project = ws.project()
    sources = ws.sources()
    dataset = latest_dataset(ws)
    problems = verify_no_leakage(dataset, ws.root / "datasets")
    if problems:
        raise WorkspaceError("dataset leakage re-check failed: " + "; ".join(problems))

    entry = ws.registry().get(cfg.base_model_id)
    use = RequestedUse(
        fine_tune=True,
        produce_adapter=True,
        commercial=project.intended_use.commercial,
        redistribute=True,  # an export package redistributes derivative weights
        export_format="stub-json",
    )
    gate = enforce_gate(entry, use)  # raises LicenseGateError

    ds_root = ws.root / "datasets"
    train_rows = load_split(ds_root, dataset, "train")
    val_rows = load_split(ds_root, dataset, "validation")
    test_rows = load_split(ds_root, dataset, "test")

    run_id = tr.run_id_for(cfg, dataset.content_hash)
    run_dir = ws.root / "runs" / run_id
    trainer = tr.StubTrainer()
    started = now_iso()
    result = trainer.train(cfg, train_rows, val_rows, run_dir / "output")
    estimate = tr.estimate_resources(cfg, train_tokens=int(result.metrics["stub_train_tokens"]))
    run = TrainingRun(
        run_id=run_id,
        project_id=project.project_id,
        dataset_hash=dataset.content_hash,
        source_manifest_hash=sources.content_hash,
        base_model_id=cfg.base_model_id,
        base_model_license_hash=entry_hash(entry),
        license_gate=gate,
        method="stub",
        executor="stub",
        is_pipeline_validation_stub=True,
        seed=cfg.seed,
        hyperparameters=cfg.model_dump(mode="json"),
        environment=tr.environment_info(),
        resource_estimate=estimate,
        status="completed",
        started_on=started,
        finished_on=now_iso(),
        train_metrics=result.metrics,
        output_artifacts=tr.hash_artifacts(result.artifacts),
    ).seal()
    (run_dir / "training_run.json").write_text(run.to_json(), encoding="utf-8")

    base = ev.base_subject()
    specialist = result.subject
    quantized = ev.quantize_subject(specialist)
    evaluator = ev.StubEvaluator()
    subjects = [base, specialist, quantized]
    metrics = ev.run_evaluation(evaluator, subjects, test_rows)
    comparisons = ev.compare(metrics, evaluator.metrics)
    train_groups = {r["group_id"] for r in train_rows} | {r["group_id"] for r in val_rows}
    overlap = len(train_groups & {r["group_id"] for r in test_rows})
    adapter_hash = run.output_artifacts["adapter"]
    evalrun = EvaluationRun(
        eval_id="eval-" + short(hash_obj([run.content_hash, evaluator.info.model_dump()])),
        project_id=project.project_id,
        dataset_hash=dataset.content_hash,
        eval_split="test",
        n_eval_examples=len(test_rows),
        train_eval_group_overlap=overlap,
        evaluator=evaluator.info,
        subjects=[
            EvalSubject(role="base", model_ref=cfg.base_model_id),
            EvalSubject(role="specialist", model_ref=specialist.ref, artifact_hash=adapter_hash),
            EvalSubject(role="quantized", model_ref=quantized.ref),
        ],
        metrics=metrics,
        comparisons=comparisons,
        improvement_claim_allowed=ev.improvement_claim_allowed(evaluator.info, comparisons, len(test_rows)),
        notes=[
            "PIPELINE-VALIDATION STUB: toy unigram-coverage metrics. These numbers demonstrate plumbing only and "
            "make no claim about real model quality or domain improvement."
        ],
        evaluated_on=today(),
    ).seal()
    (ws.root / "evals").mkdir(exist_ok=True)
    (ws.root / "evals" / f"{evalrun.eval_id}.json").write_text(evalrun.to_json(), encoding="utf-8")

    adapter_bytes = result.artifacts["adapter"].read_bytes()
    tok_bytes = json.dumps({"stub_tokenizer": "lowercase-alnum-split", "version": 1}, sort_keys=True).encode()
    model = ModelManifest(
        specialist_name=project.project_id,
        specialist_version="0.1.0-stub",
        base_model=BaseModelRef(
            model_family=entry.model_family,
            exact_version=entry.exact_version,
            license_id=entry.license_id,
            license_entry_hash=entry_hash(entry),
        ),
        adapter_method="stub",
        dataset_version=dataset.dataset_version,
        dataset_hash=dataset.content_hash,
        source_manifest_version=sources.manifest_version,
        source_manifest_hash=sources.content_hash,
        training_run_hash=run.content_hash,
        training_config=run.hyperparameters,
        tokenizer=TokenizerInfo(name="stub-lowercase-alnum", files={"tokenizer/tokenizer.json": _hash_bytes(tok_bytes)}),
        context=ContextAssumptions(trained_context_tokens=cfg.context_tokens, max_supported_context_tokens=cfg.context_tokens, recommended_context_tokens=cfg.context_tokens),
        quantization=QuantizationInfo(method="stub-prune-rare", bits_per_weight=0.0, format="stub-json"),
        runtime=RuntimeRequirements(runtimes=["none (stub)"], formats=["stub-json"], notes=["Not loadable by any real inference runtime."]),
        evaluation_hashes=[evalrun.content_hash],
        export_date=today(),
        artifacts=[ArtifactFile(path="model/stub_adapter.json", sha256=adapter_hash, size_bytes=len(adapter_bytes), role="adapter")],
        target_profile="pipeline-validation",
        known_limitations=[
            "Pipeline-validation stub: not a language model.",
            "Evaluation metrics are toy vocabulary coverage; no domain-improvement claim is supported.",
        ],
        is_pipeline_validation_stub=True,
    ).seal()
    mdir = ws.root / "models" / model.model_identity.replace("@", "_").replace("+", "_")
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "model_manifest.json").write_text(model.to_json(), encoding="utf-8")

    out = {"run": run, "evaluation": evalrun, "model": model, "package": None, "package_dir": None}
    if export:
        pkg_dir = ws.root / "exports" / model.model_identity.replace("@", "_").replace("+", "_")
        out["package"] = export_package(ws, model, dataset, sources, evalrun, entry, gate, adapter_bytes, tok_bytes, pkg_dir)
        out["package_dir"] = pkg_dir
    return out


def _hash_bytes(b: bytes) -> str:
    from .hashing import sha256_bytes

    return sha256_bytes(b)


def export_package(ws, model, dataset, sources, evalrun, entry, gate, adapter_bytes, tok_bytes, pkg_dir: Path) -> ExportPackage:
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True)
    b = PackageBuilder(pkg_dir)
    b.add("adapter", "model/stub_adapter.json", adapter_bytes)
    b.add("tokenizer", "tokenizer/tokenizer.json", tok_bytes)
    b.add("model_manifest", "manifests/model_manifest.json", model.to_json().encode())
    b.add("provenance_source_manifest", "manifests/source_manifest.json", sources.to_json().encode())
    b.add("provenance_dataset_manifest", "manifests/dataset_manifest.json", dataset.to_json().encode())
    b.add("runtime_manifest", "manifests/runtime_manifest.json", json.dumps(model.runtime.model_dump(mode="json"), indent=2, sort_keys=True).encode())
    b.add("evaluation_report", "evaluation/evaluation_run.json", evalrun.to_json().encode())
    b.add("license_bundle", "license/base_model_license_entry.json", json.dumps(entry.model_dump(mode="json"), indent=2, sort_keys=True).encode())
    b.add("license_gate", "license/license_gate.json", json.dumps(gate.model_dump(mode="json"), indent=2, sort_keys=True).encode())
    b.add("attribution", "license/ATTRIBUTION.md", attribution_text(entry, model.specialist_name).encode())
    chunk_texts = {}
    for s in sources.active_sources():
        for line in (ws.root / "corpus" / f"{s.source_id}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            chunk_texts[(s.source_id, row["chunk_id"])] = row["text"]
    chunks, index, n, excluded = build_reference_store(sources, chunk_texts)
    ref = None
    if n:
        b.add("reference_chunks", "reference/chunks.jsonl", chunks)
        b.add("reference_index", "reference/index.json", index)
        ref = ReferencePackageInfo(chunk_count=n, chunks_path="reference/chunks.jsonl", index_path="reference/index.json", excluded_source_ids=excluded)
    b.add(
        "readme",
        "README.md",
        (
            f"# {model.model_identity}\n\nPIPELINE-VALIDATION STUB package. Not a real model; no quality claim.\n\n"
            "Verify with `python consumer.py <this directory>` (standard library only) or any implementation of the "
            "export package contract.\n"
        ).encode(),
    )
    return finalize_package(
        b,
        name=model.specialist_name,
        version=model.specialist_version,
        grade="pipeline-validation",
        model=model,
        sources=sources,
        dataset=dataset,
        evaluations=[evalrun],
        gate=gate,
        reference=ref,
        runtime_formats=model.runtime.formats,
        created_on=today(),
    )


def package_specialist(ws: Workspace, run_id: str, eval_id: str, *, name: str | None = None, version: str = "0.1.0", grade: str = "desktop") -> tuple[ExportPackage, Path]:
    """Package a real (non-stub) completed run + its evaluation. Raises PackagingRefused on license/local-experiment problems."""
    from .packaging import build_specialist_package

    run = TrainingRun.model_validate_json((ws.root / "runs" / run_id / "training_run.json").read_text(encoding="utf-8"))
    evalrun = EvaluationRun.model_validate_json((ws.root / "evals" / f"{eval_id}.json").read_text(encoding="utf-8"))
    if not run.verify() or not evalrun.verify():
        raise WorkspaceError("training run or evaluation run content_hash mismatch")
    sources = ws.sources()
    dataset = next((d for d in ws.datasets() if d.content_hash == run.dataset_hash), None)
    if dataset is None:
        raise WorkspaceError("dataset used by this run is not in the workspace")
    entry = ws.registry().get(run.base_model_id)
    chunk_texts = {}
    for s in sources.active_sources():
        for line in (ws.root / "corpus" / f"{s.source_id}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            chunk_texts[(s.source_id, row["chunk_id"])] = row["text"]
    project = ws.project()
    spec_name = name or project.project_id
    out = ws.root / "exports" / f"{spec_name}_{version}_{run_id}"
    pkg = build_specialist_package(
        out, run=run, run_output_dir=ws.root / "runs" / run_id / "output", evalrun=evalrun, entry=entry, sources=sources,
        dataset=dataset, chunk_texts=chunk_texts, specialist_name=spec_name, specialist_version=version, grade=grade,
        created_on=today(), commercial=project.intended_use.commercial,
    )
    return pkg, out
