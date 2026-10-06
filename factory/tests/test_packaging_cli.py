import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from llmtrainer import cli, consumer
from llmtrainer import pipeline as pl
from llmtrainer.licenses import LicenseGateError
from llmtrainer.packaging import MANIFEST_NAME, validate_package
from llmtrainer.schemas import ExportPackage
from llmtrainer.training import ExperimentConfig

CFG = ExperimentConfig(base_model_id=pl.STUB_MODEL_ID)


@pytest.fixture
def exp(built):
    ws, d = built
    return ws, d, pl.run_experiment(ws, CFG)


def test_end_to_end_pipeline_artifacts(exp):
    ws, d, out = exp
    run, ev, model, pkg = out["run"], out["evaluation"], out["model"], out["package"]
    assert all(a.verify() for a in (run, ev, model, pkg))
    assert run.is_pipeline_validation_stub and run.license_gate.allowed
    assert ev.evaluator.is_stub and not ev.improvement_claim_allowed
    assert {s.role for s in ev.subjects} == {"base", "specialist", "quantized"}
    assert ev.train_eval_group_overlap == 0
    assert len(ev.comparisons) == 2  # per metric, no aggregate
    assert model.is_pipeline_validation_stub and model.dataset_hash == d.content_hash
    assert model.model_identity.startswith("proj-synthetic-fixture@0.1.0-stub+")
    assert pkg.model_identity == model.model_identity and pkg.reference.chunk_count > 0


def test_experiment_is_reproducible(built, tmp_path):
    ws, _ = built
    a = pl.run_experiment(ws, CFG)
    b = pl.run_experiment(ws, CFG)
    assert a["model"].content_hash == b["model"].content_hash
    assert a["package"].content_hash == b["package"].content_hash
    c = pl.run_experiment(ws, ExperimentConfig(base_model_id=pl.STUB_MODEL_ID, seed=99))
    assert c["model"].model_identity != a["model"].model_identity  # materially different -> different identity


def test_package_validates(exp):
    _, _, out = exp
    rep = validate_package(out["package_dir"])
    assert rep.ok, rep.errors
    assert any("STUB" in w for w in rep.warnings)


def test_license_gate_refuses_commercial_project(built):
    ws, _ = built
    p = ws.project()
    from llmtrainer.schemas import IntendedUse

    ws.project_file.write_text(p.model_copy(update={"intended_use": IntendedUse(commercial=True)}).seal().to_json())
    with pytest.raises(LicenseGateError, match="commercial"):
        pl.run_experiment(ws, CFG)


def test_unknown_base_model_refused(built):
    ws, _ = built
    with pytest.raises(KeyError, match="not in the license registry"):
        pl.run_experiment(ws, ExperimentConfig(base_model_id="llama-unlisted@1"))


def test_unverified_registry_entry_refused(built):
    ws, _ = built
    e = json.loads((ws.root / "registry" / "pipeline-validation-stub.json").read_text())
    e.update(exact_version="2", redistribution_permitted="unverified")
    (ws.root / "registry" / "unv.json").write_text(json.dumps(e))
    with pytest.raises(LicenseGateError, match="unverified"):
        pl.run_experiment(ws, ExperimentConfig(base_model_id="pipeline-validation-stub@2"))


def test_real_training_method_not_executed_locally(built):
    ws, _ = built
    with pytest.raises(pl.WorkspaceError):
        pl.run_experiment(ws, ExperimentConfig(base_model_id=pl.STUB_MODEL_ID, method="qlora"))


def _copy(pkg_dir, tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(pkg_dir, dst)
    return dst


def test_validator_detects_tampered_file(exp, tmp_path):
    p = _copy(exp[2]["package_dir"], tmp_path)
    (p / "model" / "stub_adapter.json").write_text("{}")
    rep = validate_package(p)
    assert not rep.ok and any("hash mismatch" in e or "size mismatch" in e for e in rep.errors)


def test_validator_detects_undeclared_and_missing(exp, tmp_path):
    p = _copy(exp[2]["package_dir"], tmp_path)
    (p / "extra.bin").write_bytes(b"x")
    assert any("undeclared" in e for e in validate_package(p).errors)
    (p / "extra.bin").unlink()
    (p / "tokenizer" / "tokenizer.json").unlink()
    assert any("missing file" in e for e in validate_package(p).errors)


def test_validator_detects_edited_manifest(exp, tmp_path):
    p = _copy(exp[2]["package_dir"], tmp_path)
    m = json.loads((p / MANIFEST_NAME).read_text())
    m["package_version"] = "9.9.9"
    (p / MANIFEST_NAME).write_text(json.dumps(m))
    assert any("content_hash" in e for e in validate_package(p).errors)


def test_validator_requires_license_gate_to_hold(exp, tmp_path):
    """Even a re-sealed package is rejected if the packaged license entry no longer permits redistribution."""
    p = _copy(exp[2]["package_dir"], tmp_path)
    lic = p / "license" / "base_model_license_entry.json"
    d = json.loads(lic.read_text())
    d["redistribution_permitted"] = "unverified"
    lic.write_text(json.dumps(d))
    rep = validate_package(p)
    assert not rep.ok  # hash mismatch and recomputed gate both fail


def test_validator_flags_missing_roles(exp, tmp_path):
    p = _copy(exp[2]["package_dir"], tmp_path)
    m = json.loads((p / MANIFEST_NAME).read_text())
    m["files"] = [f for f in m["files"] if f["role"] != "evaluation_report"]
    pkg = ExportPackage.model_validate({**m, "content_hash": ""}).seal()
    (p / MANIFEST_NAME).write_text(pkg.to_json())
    assert any("evaluation_report" in e for e in validate_package(p).errors)


def test_validator_missing_manifest(tmp_path):
    assert validate_package(tmp_path).errors == [f"{MANIFEST_NAME} missing"]


def test_reference_store_excludes_sources_without_redistribution_rights(built):
    from conftest import SYNTH

    ws, _ = built
    f = ws.root / "restricted.md"
    f.write_text("# Restricted\n\nThis text may train a model but must never be shipped inside a package.\n")
    pl.add_source(ws, f, title="r", origin="o", rights=SYNTH.model_copy(update={"permitted_redistribution": "no"}))
    pl.build_dataset_for_project(ws)
    out = pl.run_experiment(ws, CFG)
    pkg = out["package"]
    assert len(pkg.reference.excluded_source_ids) == 1
    chunks = (out["package_dir"] / pkg.reference.chunks_path).read_text()
    assert "must never be shipped" not in chunks
    assert validate_package(out["package_dir"]).ok


# ---------------------------- reference consumer -----------------------------

def test_consumer_imports_package_and_searches(exp):
    pkg = consumer.Package.open(exp[2]["package_dir"])
    assert pkg.identity == exp[2]["model"].model_identity and pkg.is_stub
    assert pkg.license["license_id"] == "LLM-Trainer-internal-test-fixture"
    hits = pkg.search("shaft seal lubricant", limit=2)
    assert hits and hits[0]["source_title"] == "pump_maintenance" and hits[0]["section"] == "Seal Replacement"
    assert hits[0]["page_start"] == 1 and hits[0]["text_sha256"].startswith("sha256:")
    assert pkg.evaluation["evaluator"]["is_stub"] is True
    assert pkg.paths("adapter")[0].is_file()


def test_consumer_refuses_tampered_package(exp, tmp_path):
    p = _copy(exp[2]["package_dir"], tmp_path)
    (p / "reference" / "chunks.jsonl").write_text("")
    with pytest.raises(consumer.PackageError, match="hash mismatch"):
        consumer.Package.open(p)


def test_consumer_has_no_factory_dependency():
    tree = ast.parse(Path(consumer.__file__).read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            assert n.level == 0, "consumer must not use relative imports"
            mods.add((n.module or "").split(".")[0])
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods


def test_consumer_runs_standalone_outside_the_package(exp, tmp_path):
    """Copy only consumer.py, run it with isolated mode so llmtrainer cannot be imported."""
    solo = tmp_path / "solo"
    solo.mkdir()
    shutil.copy(consumer.__file__, solo / "consumer.py")
    r = subprocess.run([sys.executable, "-I", str(solo / "consumer.py"), str(exp[2]["package_dir"]), "vibration", "amber"],
                       capture_output=True, text=True, cwd=solo)
    assert r.returncode == 0, r.stderr
    assert "imported proj-synthetic-fixture@0.1.0-stub" in r.stdout
    assert "First Rotation" in r.stdout
    bad = _copy(exp[2]["package_dir"], tmp_path)
    (bad / "README.md").write_text("tampered")
    r2 = subprocess.run([sys.executable, "-I", str(solo / "consumer.py"), str(bad)], capture_output=True, text=True)
    assert r2.returncode == 1 and "REFUSED" in r2.stdout


# ---------------------------- CLI --------------------------------------------

def run_cli(*args):
    return cli.main([str(a) for a in args])


def test_cli_full_flow(tmp_path, capsys):
    from conftest import fixture_files

    proj = tmp_path / "p"
    assert run_cli("init", proj, "--name", "CLI Fixture", "--domain", "test") == 0
    for f in fixture_files():
        assert run_cli("add-source", proj, f, "--title", f.stem, "--origin", "fixture", "--rights-status", "synthetic",
                       "--permit-training", "yes", "--permit-redistribution", "yes", "--permit-commercial", "yes") == 0
    assert run_cli("build-dataset", proj) == 0
    assert run_cli("run-experiment", proj) == 0
    out = capsys.readouterr().out
    assert "PIPELINE-VALIDATION STUB" in out and '"improvement_claim_allowed": false' in out
    pkg = next((proj / "exports").iterdir())
    assert run_cli("validate-package", pkg) == 0
    (pkg / "README.md").write_text("x")
    assert run_cli("validate-package", pkg) == 1


def test_cli_refuses_unverified_source_and_reports_error(tmp_path, capsys):
    proj = tmp_path / "p"
    run_cli("init", proj, "--name", "N", "--domain", "d")
    f = tmp_path / "a.md"
    f.write_text("# T\n\nA source whose rights were never recorded by anyone at all.\n")
    run_cli("add-source", proj, f, "--title", "t", "--origin", "o")
    assert run_cli("build-dataset", proj) == 1
    assert "rights gate" in capsys.readouterr().err


def test_cli_check_license_exit_codes(tmp_path, capsys):
    reg = Path(cli.__file__).parent / "data" / "registry"
    assert run_cli("check-license", reg, "pipeline-validation-stub@1", "--redistribute", "--format", "stub-json") == 0
    assert run_cli("check-license", reg, "pipeline-validation-stub@1", "--commercial") == 1


def test_cli_qualify_device(tmp_path, capsys):
    from conftest import flagship12

    f = tmp_path / "dev.json"
    f.write_text(flagship12().to_json())
    assert run_cli("qualify-device", f, "--out", tmp_path / "rec.json") == 0
    out = json.loads(capsys.readouterr().out)
    assert [p["tier"] for p in out["picks"]] == ["provisional"] * 3
    assert (tmp_path / "rec.json").exists()


def test_cli_plan_removal_and_remove(exp, capsys):
    ws, d, _ = exp
    sid = d.examples[0].derived_from[0].source_id
    assert run_cli("plan-removal", ws.root, sid) == 0
    assert d.content_hash in capsys.readouterr().out
    assert run_cli("remove-source", ws.root, sid, "--reason", "test") == 0
    assert ws.sources().get(sid).status == "removed"


def test_cli_estimate_resources_and_schema_export(tmp_path, capsys):
    assert run_cli("estimate-resources", "--params-b", "7", "--tokens", "5000000") == 0
    assert "gpu_memory_gib" in capsys.readouterr().out
    assert run_cli("export-schemas", tmp_path) == 0
    assert (tmp_path / "v1" / "model_manifest.schema.json").exists()


def test_python_dash_m_entrypoint(tmp_path):
    r = subprocess.run([sys.executable, "-m", "llmtrainer", "estimate-resources", "--params-b", "1", "--tokens", "1000"],
                       capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0 and "gpu_hours" in r.stdout


def test_init_twice_fails(tmp_path):
    pl.init_project(tmp_path / "p", "A", "d")
    with pytest.raises(pl.WorkspaceError):
        pl.init_project(tmp_path / "p", "A", "d")
