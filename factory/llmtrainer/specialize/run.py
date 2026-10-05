"""Plan (dry-run) and execute a real specialization run."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .. import pipeline as pl
from .. import training as tr
from ..clock import now_iso
from ..hashing import hash_file, hash_obj
from ..licenses import LicenseRegistry, entry_hash
from ..schemas import BaseModelLicenseEntry, DatasetManifest, ResourceEstimate, TrainingRun
from . import hf_lora
from .config import SpecializeConfig
from .data import approx_tokens, load_training_data
from .gating import TrainingLicenseDecision, decide_training_license


class TrainingRefused(RuntimeError):
    pass


@dataclass
class TrainPlan:
    config: SpecializeConfig
    config_path: str
    ws: pl.Workspace
    dataset: DatasetManifest
    entry: BaseModelLicenseEntry
    decision: TrainingLicenseDecision
    estimate: ResourceEstimate
    n_train: int
    n_val: int
    run_id: str
    allow_unverified: bool
    allow_download: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def command(self) -> str:
        parts = ["llmtrainer", "train", "--config", shlex.quote(self.config_path), "--execute"]
        if self.allow_unverified:
            parts.append("--allow-unverified-license-for-local-experiment")
        if self.allow_download:
            parts.append("--allow-download")
        return " ".join(parts)

    def as_dict(self) -> dict:
        return {
            "mode": "dry-run (nothing launched, nothing downloaded)",
            "run_id": self.run_id,
            "config_hash": self.config.config_hash(),
            "dataset_id": self.dataset.dataset_id,
            "dataset_hash": self.dataset.content_hash,
            "source_manifest_hash": self.dataset.source_manifest_hash,
            "base_model": {"registry_id": self.entry.entry_id, "license_entry_hash": entry_hash(self.entry),
                           "path_or_repo": self.config.base_model_path},
            "license_state": self.decision.license_state,
            "local_experiment_unverified_license": self.decision.local_experiment,
            "n_train_examples": self.n_train,
            "n_validation_examples": self.n_val,
            "test_split_read": False,
            "resource_estimate": self.estimate.model_dump(),
            "missing_training_dependencies": hf_lora.missing_dependencies(self.config.method),
            "reproducible_command": self.command,
            "warnings": self.warnings,
        }


def _registry(ws: pl.Workspace, cfg: SpecializeConfig) -> LicenseRegistry:
    reg = ws.registry()
    if cfg.registry_path:
        for e in LicenseRegistry.load(cfg.registry_path).entries.values():
            reg.add(e)
    return reg


def _check_sources_cleared(ws: pl.Workspace, dataset: DatasetManifest) -> None:
    sources = ws.sources()
    active = {s.source_id: s for s in sources.active_sources()}
    bad = set()
    for ex in dataset.examples:
        if ex.split == "test":
            continue
        for ref in ex.derived_from:
            s = active.get(ref.source_id)
            if s is None or s.rights.permitted_training != "yes":
                bad.add(ref.source_id)
    if bad:
        raise TrainingRefused(f"dataset derives from removed or not-cleared sources {sorted(bad)}; rebuild the dataset")


def plan_training(
    config_path: str | Path, *, allow_unverified: bool = False, allow_download: bool = False
) -> TrainPlan:
    cfg = SpecializeConfig.load(config_path)
    ws = pl.Workspace(Path(cfg.project))
    project = ws.project()
    sources = ws.sources()
    if cfg.dataset_id:
        matches = [d for d in ws.datasets() if d.dataset_id == cfg.dataset_id]
        if not matches:
            raise TrainingRefused(f"dataset {cfg.dataset_id} not found in workspace")
        dataset = matches[0]
        if dataset.source_manifest_hash != sources.content_hash:
            raise TrainingRefused("dataset was built from an older source manifest; rebuild it")
    else:
        dataset = pl.latest_dataset(ws)
    _check_sources_cleared(ws, dataset)

    entry = _registry(ws, cfg).get(cfg.base_model_id)
    decision = decide_training_license(
        entry, commercial=project.intended_use.commercial, allow_unverified_local=allow_unverified
    )
    train_rows, val_rows = load_training_data(ws.root / "datasets", dataset)  # train/validation ONLY
    est = tr.estimate_resources(cfg.to_experiment(), approx_tokens(train_rows))
    warnings = list(decision.warnings)
    if not Path(cfg.base_model_path).is_dir():
        warnings.append(
            "base_model_path is not a local directory: a real run will refuse to download unless --allow-download is given"
        )
        if not allow_download:
            warnings.append("(offline mode: local_files_only=True)")
    run_id = "run-" + hash_obj([cfg.config_hash(), dataset.content_hash, decision.local_experiment]).removeprefix("sha256:")[:12]
    return TrainPlan(cfg, str(config_path), ws, dataset, entry, decision, est, len(train_rows), len(val_rows), run_id,
                     allow_unverified, allow_download, warnings)


def execute_training(plan: TrainPlan, trainer=None) -> TrainingRun:
    """Run the training. ``trainer`` is injectable (tests use a fake); default is the real HF backend."""
    cfg = plan.config
    run_dir = plan.ws.root / "runs" / plan.run_id
    run_file = run_dir / "training_run.json"
    if run_file.exists() and TrainingRun.model_validate_json(run_file.read_text(encoding="utf-8")).status != "failed":
        raise TrainingRefused(f"{run_file} already exists; training artifacts are never silently overwritten")
    if trainer is None:
        hf_lora.require_dependencies(cfg.method)
        trainer = hf_lora.HFLoraTrainer(cfg.base_model_path, allow_download=plan.allow_download)
    train_rows, val_rows = load_training_data(plan.ws.root / "datasets", plan.dataset)
    started = now_iso()
    notes = list(plan.decision.warnings)
    env = hf_lora.environment_snapshot()
    base_cfg_json = Path(cfg.base_model_path) / "config.json"
    if base_cfg_json.is_file():
        env["base_model_config_sha256"] = hash_file(base_cfg_json)

    def record(status, metrics=None, artifacts=None, extra_notes=()):
        timings = {k.removeprefix("timing_"): v for k, v in (metrics or {}).items() if k.startswith("timing_")}
        run = TrainingRun(
            run_id=plan.run_id, project_id=plan.ws.project().project_id, dataset_hash=plan.dataset.content_hash,
            source_manifest_hash=plan.dataset.source_manifest_hash, base_model_id=plan.entry.entry_id,
            base_model_license_hash=entry_hash(plan.entry), license_gate=plan.decision.gate, method=cfg.method,
            executor="hf_lora", is_pipeline_validation_stub=False, seed=cfg.seed,
            hyperparameters=cfg.model_dump(mode="json", exclude={"project", "registry_path"}), environment=env,
            resource_estimate=plan.estimate, status=status, started_on=started, finished_on=now_iso(),
            train_metrics={k: v for k, v in (metrics or {}).items() if not k.startswith("timing_")},
            output_artifacts=tr.hash_artifacts(artifacts or {}), config_hash=cfg.config_hash(),
            trainer=f"{getattr(trainer, 'name', '?')}/{getattr(trainer, 'version', '?')}",
            license_state=plan.decision.license_state,
            local_experiment_unverified_license=plan.decision.local_experiment, timings_s=timings,
            notes=[*notes, *extra_notes],
        ).seal()
        run_dir.mkdir(parents=True, exist_ok=True)
        run_file.write_text(run.to_json(), encoding="utf-8")
        return run

    try:
        res = trainer.train(cfg.to_experiment(), train_rows, val_rows, run_dir / "output")
    except Exception as e:  # noqa: BLE001 - record the failure, then re-raise
        record("failed", extra_notes=[f"{type(e).__name__}: {e}"])
        raise
    return record("completed", res.metrics, res.artifacts)
