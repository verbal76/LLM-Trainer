"""Real-specialization path: everything testable without torch/transformers/peft/trl."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmtrainer import cli
from llmtrainer import datasets as ds
from llmtrainer import pipeline as pl
from llmtrainer.licenses import LicenseGateError
from llmtrainer.schemas import BaseModelLicenseEntry, TrainingRun
from llmtrainer.specialize import hf_lora
from llmtrainer.specialize.config import SpecializeConfig
from llmtrainer.specialize.data import TestSplitAccessError, load_training_split
from llmtrainer.specialize.gating import decide_training_license
from llmtrainer.specialize.run import TrainingRefused, execute_training, plan_training
from llmtrainer.training import TrainerResult

ENTRY = {
    "model_family": "test-llm", "exact_version": "1", "license_id": "Apache-2.0", "license_url": "https://example.invalid/license",
    "commercial_use": "yes", "fine_tuning_permitted": "yes", "derivative_adapter_permitted": "yes",
    "redistribution_permitted": "yes", "attribution_required": "yes", "restrictions": [],
    "supported_formats": ["safetensors"],
    "schema_version": 2,
    "verification": {
        "state": "VERIFIED", "evidence_level": "primary_license_text_read", "verified_scope": "test fixture", "license_text_sha256": "sha256:" + "a" * 64,
        "source_urls": ["https://example.invalid/license"], "verified_on": "2026-10-05", "uncertainties": [],
    },
}


def add_entry(ws, **kw):
    e = {**ENTRY, **kw}
    (ws.root / "registry" / f"{e['model_family']}-{e['exact_version']}.json").write_text(json.dumps(e))
    return f"{e['model_family']}@{e['exact_version']}"


def write_cfg(ws, tmp_path, **kw):
    cfg = {"project": str(ws.root), "base_model_id": "test-llm@1", "base_model_path": str(tmp_path / "no-such-model"),
           "base_params_b": 1.7, "method": "lora", **kw}
    p = tmp_path / "train.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def setup(built, tmp_path):
    ws, d = built
    add_entry(ws)
    return ws, d, tmp_path


class FakeTrainer:
    name, version, is_stub = "fake", "0", False

    def __init__(self, fail=False):
        self.seen = None
        self.fail = fail

    def train(self, cfg, train_rows, val_rows, out_dir):
        if self.fail:
            raise RuntimeError("cuda oom")
        self.seen = (train_rows, val_rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        f = out_dir / "adapter_model.safetensors"
        f.write_bytes(b"fake-adapter")
        return TrainerResult(metrics={"train_loss": 1.0, "timing_total_s": 2.0}, artifacts={"adapter/adapter_model.safetensors": f})


# ------------------------------- config ------------------------------------

def test_config_validation(tmp_path):
    base = dict(project="p", base_model_id="a@1", base_model_path="m", base_params_b=1)
    SpecializeConfig(**base)
    for bad in ({"method": "full"}, {"target_modules": []}, {"lora_rank": 0}, {"base_params_b": 0}, {"bogus": 1}, {"method": "qlora", "bf16": False}):
        with pytest.raises(ValidationError):
            SpecializeConfig(**{**base, **bad})


def test_config_hash_ignores_paths_but_not_hyperparameters():
    base = dict(project="p", base_model_id="a@1", base_model_path="m", base_params_b=1)
    a = SpecializeConfig(**base)
    assert a.config_hash() == SpecializeConfig(**{**base, "project": "elsewhere"}).config_hash()
    assert a.config_hash() != SpecializeConfig(**{**base, "seed": 5}).config_hash()
    assert a.to_experiment().method == "lora"


# ------------------------------- gating ------------------------------------

def entry(**kw):
    return BaseModelLicenseEntry.model_validate({**ENTRY, **kw})


def test_verified_entry_trains():
    d = decide_training_license(entry(), commercial=True, allow_unverified_local=False)
    assert d.license_state == "VERIFIED" and not d.local_experiment


def test_unverified_refused_unless_local_flag():
    e = entry(fine_tuning_permitted="unverified")
    with pytest.raises(LicenseGateError):
        decide_training_license(e, commercial=False, allow_unverified_local=False)
    d = decide_training_license(e, commercial=False, allow_unverified_local=True)
    assert d.license_state == "UNVERIFIED" and d.local_experiment and "LOCAL EXPERIMENT" in d.warnings[0]


def test_unverified_state_fails_closed_even_if_permissions_claim_yes():
    e = entry(verification={"state": "UNVERIFIED", "evidence_level": "secondary_source_only",
                            "source_urls": ["https://example.invalid/x"], "verified_on": "2026-10-05", "uncertainties": []})
    with pytest.raises(LicenseGateError):
        decide_training_license(e, commercial=False, allow_unverified_local=False)
    d = decide_training_license(e, commercial=False, allow_unverified_local=True)
    assert d.license_state == "UNVERIFIED" and d.local_experiment


def test_disallowed_state_is_never_overridable():
    e = entry(verification={"state": "DISALLOWED", "evidence_level": "primary_license_text_read", "disallowed_reason": "test",
                            "source_urls": ["https://example.invalid/x"], "verified_on": "2026-10-05", "uncertainties": []})
    with pytest.raises(LicenseGateError):
        decide_training_license(e, commercial=False, allow_unverified_local=True)


def test_conditional_needs_flag_and_explicit_no_never_overridable():
    c = entry(derivative_adapter_permitted="conditional")
    with pytest.raises(LicenseGateError):
        decide_training_license(c, commercial=False, allow_unverified_local=False)
    assert decide_training_license(c, commercial=False, allow_unverified_local=True).license_state == "CONDITIONAL"
    with pytest.raises(LicenseGateError):
        decide_training_license(entry(fine_tuning_permitted="no"), commercial=False, allow_unverified_local=True)


def test_missing_evidence_is_not_verified():
    e = entry(commercial_use="unverified", fine_tuning_permitted="unverified", derivative_adapter_permitted="unverified",
              redistribution_permitted="unverified", attribution_required="unverified",
              verification={"state": "UNVERIFIED", "evidence_level": "none", "source_urls": [], "verified_on": None, "uncertainties": []})
    with pytest.raises(LicenseGateError):
        decide_training_license(e, commercial=False, allow_unverified_local=False)


# ------------------------------- dry run -----------------------------------

def test_dry_run_plans_without_side_effects_or_test_split(setup, monkeypatch, capsys):
    ws, d, tmp = setup
    cfg = write_cfg(ws, tmp)
    (ws.root / "datasets" / d.dataset_id / "test.jsonl").unlink()  # the test split must not be needed at all
    loaded = []
    real = ds.load_split
    monkeypatch.setattr("llmtrainer.specialize.data.load_split", lambda root, m, split: (loaded.append(split), real(root, m, split))[1])
    before = sorted(p.relative_to(ws.root).as_posix() for p in ws.root.rglob("*"))
    assert cli.main(["train", "--config", str(cfg)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert sorted(p.relative_to(ws.root).as_posix() for p in ws.root.rglob("*")) == before  # nothing written
    assert loaded == ["train", "validation"] and out["test_split_read"] is False
    assert out["license_state"] == "VERIFIED" and out["dataset_hash"] == d.content_hash
    assert out["reproducible_command"].startswith("llmtrainer train --config ") and "--execute" in out["reproducible_command"]
    assert out["resource_estimate"]["gpu_memory_gib"] > 0 and out["config_hash"].startswith("sha256:")
    assert "--allow-download" not in out["reproducible_command"]
    assert any("not a local directory" in w for w in out["warnings"])


def test_dry_run_flag_and_execute_are_mutually_exclusive(setup):
    ws, _, tmp = setup
    with pytest.raises(SystemExit):
        cli.main(["train", "--config", str(write_cfg(ws, tmp)), "--dry-run", "--execute"])


def test_dry_run_refuses_unverified_model_and_records_flag_in_command(setup, capsys):
    ws, _, tmp = setup
    add_entry(ws, exact_version="2", fine_tuning_permitted="unverified")
    cfg = write_cfg(ws, tmp, base_model_id="test-llm@2")
    assert cli.main(["train", "--config", str(cfg)]) == 1
    assert "license gate refused" in capsys.readouterr().err
    assert cli.main(["train", "--config", str(cfg), "--allow-unverified-license-for-local-experiment"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["local_experiment_unverified_license"] is True
    assert "--allow-unverified-license-for-local-experiment" in out["reproducible_command"]


def test_unknown_base_model_and_missing_project(setup, tmp_path):
    ws, _, tmp = setup
    with pytest.raises(KeyError):
        plan_training(write_cfg(ws, tmp, base_model_id="nope@1"))
    with pytest.raises(pl.WorkspaceError):
        plan_training(write_cfg(ws, tmp, project=str(tmp_path / "missing")))


def test_commercial_project_requires_commercial_permission(setup):
    from llmtrainer.schemas import IntendedUse

    ws, _, tmp = setup
    add_entry(ws, exact_version="3", commercial_use="no")
    p = ws.project()
    ws.project_file.write_text(p.model_copy(update={"intended_use": IntendedUse(commercial=True)}).seal().to_json())
    with pytest.raises(LicenseGateError, match="commercial"):
        plan_training(write_cfg(ws, tmp, base_model_id="test-llm@3"), allow_unverified=True)


def test_removed_source_blocks_stale_dataset(setup):
    ws, d, tmp = setup
    sid = d.examples[0].derived_from[0].source_id
    pl.remove_source(ws, [sid], "rights withdrawn")
    with pytest.raises(pl.WorkspaceError, match="no dataset built"):
        plan_training(write_cfg(ws, tmp))
    with pytest.raises(TrainingRefused, match="older source manifest"):
        plan_training(write_cfg(ws, tmp, dataset_id=d.dataset_id))


# ------------------------------- test split --------------------------------

def test_test_split_cannot_be_loaded_for_training(setup):
    ws, d, _ = setup
    with pytest.raises(TestSplitAccessError):
        load_training_split(ws.root / "datasets", d, "test")
    assert load_training_split(ws.root / "datasets", d, "train")


def test_trainer_receives_only_train_and_validation_rows(setup):
    ws, d, tmp = setup
    t = FakeTrainer()
    execute_training(plan_training(write_cfg(ws, tmp)), trainer=t)
    seen = {r["example_id"] for rows in t.seen for r in rows}
    by_split = {s: {e.example_id for e in d.examples if e.split == s} for s in ("train", "validation", "test")}
    assert seen == by_split["train"] | by_split["validation"] and not seen & by_split["test"]


def test_leaky_dataset_refused(setup):
    ws, d, tmp = setup
    plan = plan_training(write_cfg(ws, tmp))
    bad = plan.dataset.model_copy(update={"leakage_report": plan.dataset.leakage_report.model_copy(update={"group_overlap_pairs": 1})})
    plan.dataset = bad
    with pytest.raises(ValueError, match="leakage"):
        execute_training(plan, trainer=FakeTrainer())


# ------------------------------- execute (fake trainer) --------------------

def test_execute_writes_complete_reproducible_manifest(setup):
    ws, d, tmp = setup
    plan = plan_training(write_cfg(ws, tmp, seed=77))
    run = execute_training(plan, trainer=FakeTrainer())
    saved = TrainingRun.model_validate_json((ws.root / "runs" / run.run_id / "training_run.json").read_text())
    assert saved == run and run.verify()
    assert run.executor == "hf_lora" and not run.is_pipeline_validation_stub and run.status == "completed"
    assert run.seed == 77 and run.dataset_hash == d.content_hash and run.config_hash == plan.config.config_hash()
    assert run.base_model_id == "test-llm@1" and run.license_state == "VERIFIED" and not run.local_experiment_unverified_license
    assert run.hyperparameters["lora_rank"] == 16 and "project" not in run.hyperparameters
    assert {"python", "platform", "gpu", "lib:torch", "lib:peft"} <= set(run.environment)
    assert run.timings_s == {"total_s": 2.0} and run.train_metrics == {"train_loss": 1.0}
    assert run.output_artifacts["adapter/adapter_model.safetensors"].startswith("sha256:")
    assert run.resource_estimate.gpu_memory_gib > 0
    # deterministic identity: same config + dataset => same run id; no silent overwrite
    assert plan_training(write_cfg(ws, tmp, seed=77)).run_id == run.run_id
    with pytest.raises(TrainingRefused, match="never silently overwritten"):
        execute_training(plan, trainer=FakeTrainer())
    assert plan_training(write_cfg(ws, tmp, seed=78)).run_id != run.run_id


def test_local_experiment_flag_recorded_in_run(setup):
    ws, _, tmp = setup
    add_entry(ws, exact_version="2", derivative_adapter_permitted="conditional")
    plan = plan_training(write_cfg(ws, tmp, base_model_id="test-llm@2"), allow_unverified=True)
    run = execute_training(plan, trainer=FakeTrainer())
    assert run.local_experiment_unverified_license and run.license_state == "CONDITIONAL" and not run.license_gate.allowed
    assert any("LOCAL EXPERIMENT" in n for n in run.notes)


def test_failed_run_is_recorded_and_retryable(setup):
    ws, _, tmp = setup
    plan = plan_training(write_cfg(ws, tmp))
    with pytest.raises(RuntimeError, match="oom"):
        execute_training(plan, trainer=FakeTrainer(fail=True))
    f = ws.root / "runs" / plan.run_id / "training_run.json"
    failed = TrainingRun.model_validate_json(f.read_text())
    assert failed.status == "failed" and "cuda oom" in failed.notes[-1]
    assert execute_training(plan, trainer=FakeTrainer()).status == "completed"


# ------------------------------- heavy deps --------------------------------

def test_execute_without_libraries_fails_clearly_and_early(setup):
    if not hf_lora.missing_dependencies("qlora"):
        pytest.skip("training libraries are installed here")
    ws, _, tmp = setup
    plan = plan_training(write_cfg(ws, tmp, method="qlora"))
    with pytest.raises(hf_lora.MissingTrainingDependencies, match="bitsandbytes"):
        execute_training(plan)
    assert not (ws.root / "runs").exists() or not list((ws.root / "runs").iterdir())


def test_importing_backend_does_not_import_heavy_libs():
    import subprocess
    import sys

    code = "import sys, llmtrainer.specialize.hf_lora, llmtrainer.specialize.run; assert not {'torch','transformers','peft','trl'} & set(sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stderr


def test_lora_target_module_translation():
    from llmtrainer.training import ExperimentConfig

    assert hf_lora.lora_target_modules(ExperimentConfig(base_model_id="x")) == "all-linear"
    assert hf_lora.lora_target_modules(ExperimentConfig(base_model_id="x", target_modules=["q", "v"])) == ["q", "v"]
