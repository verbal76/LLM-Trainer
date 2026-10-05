"""Export of a real specialist package: gates, validator and the stdlib reference consumer."""

import json
import shutil

import pytest

from llmtrainer import cli, consumer
from llmtrainer import pipeline as pl
from llmtrainer.evalsuite import CallableSubject, LexicalRetriever, build_items, load_test_chunks, run_suite, to_evaluation_run
from llmtrainer.packaging import MANIFEST_NAME, validate_package
from llmtrainer.specialize.gating import PackagingRefused
from llmtrainer.specialize.run import execute_training, plan_training
from llmtrainer.training import TrainerResult
from test_specialize import add_entry, write_cfg


class AdapterTrainer:
    name, version, is_stub = "fake-hf", "0", False

    def train(self, cfg, train_rows, val_rows, out_dir):
        (out_dir / "adapter").mkdir(parents=True, exist_ok=True)
        (out_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
        a = out_dir / "adapter" / "adapter_model.safetensors"
        c = out_dir / "adapter" / "adapter_config.json"
        t = out_dir / "tokenizer" / "tokenizer.json"
        a.write_bytes(b"adapter-bytes")
        c.write_text('{"r": 16}')
        t.write_text('{"tok": 1}')
        return TrainerResult(metrics={"train_loss": 0.5}, artifacts={"adapter/adapter_model.safetensors": a, "adapter/adapter_config.json": c, "tokenizer/tokenizer.json": t})


def make_eval(ws, d, run_id):
    chunks = load_test_chunks(d, ws.root / "corpus", ws.sources())
    items = build_items(chunks)
    gen_b = lambda p: "something"  # noqa: E731
    gen_s = lambda p: "pump seal lubricant"  # noqa: E731
    subs = [CallableSubject("base", "fake-base", gen_b), CallableSubject("specialist", "fake-spec", gen_s)]
    r = LexicalRetriever([{"ref": c["ref"], "text": c["text"]} for c in chunks])
    ev = to_evaluation_run(run_suite(subs, items, retrieve=r), project_id=ws.project().project_id, dataset=d,
                           train_eval_group_overlap=0, evaluated_on="2026-10-05", subjects=subs)
    (ws.root / "evals").mkdir(exist_ok=True)
    (ws.root / "evals" / f"{ev.eval_id}.json").write_text(ev.to_json())
    return ev


@pytest.fixture
def trained(built, tmp_path):
    ws, d = built
    add_entry(ws)
    run = execute_training(plan_training(write_cfg(ws, tmp_path)), trainer=AdapterTrainer())
    return ws, d, run, make_eval(ws, d, run.run_id)


def test_package_real_specialist_validates_and_consumer_imports(trained):
    ws, d, run, ev = trained
    pkg, out = pl.package_specialist(ws, run.run_id, ev.eval_id)
    assert not pkg.is_pipeline_validation_stub
    rep = validate_package(out)
    assert rep.ok, rep.errors
    roles = {f.role for f in pkg.files}
    assert {"adapter", "tokenizer", "training_run", "evaluation_report", "license_bundle", "reference_chunks"} <= roles
    p = consumer.Package.open(out)
    assert p.license_state == "VERIFIED" and p.base_model["model_family"] == "test-llm"
    assert len(p.adapters) == 2 and all(a["sha256"].startswith("sha256:") for a in p.adapters)
    assert p.training_config_hash == run.config_hash and "tokenizer/tokenizer.json" in p.tokenizer_files
    assert p.evaluation["improvement_claim_allowed"] is False  # synthetic fixture corpus
    assert p.requirements["formats"] == ["hf-peft-adapter"] and p.search("shaft seal")


def test_local_experiment_run_cannot_be_packaged(built, tmp_path):
    ws, d = built
    add_entry(ws, exact_version="2", derivative_adapter_permitted="conditional")
    run = execute_training(plan_training(write_cfg(ws, tmp_path, base_model_id="test-llm@2"), allow_unverified=True), trainer=AdapterTrainer())
    ev = make_eval(ws, d, run.run_id)
    with pytest.raises(PackagingRefused, match="local experiment"):
        pl.package_specialist(ws, run.run_id, ev.eval_id)
    assert not any((ws.root / "exports").iterdir())


def test_unverified_redistribution_refused(trained):
    ws, d, run, ev = trained
    # same entry id, but redistribution is no longer verified at packaging time
    e = json.loads((ws.root / "registry" / "test-llm-1.json").read_text())
    e["redistribution_permitted"] = "unverified"
    (ws.root / "registry" / "test-llm-1.json").write_text(json.dumps(e))
    with pytest.raises(Exception, match="redistribution|hash|license"):
        pl.package_specialist(ws, run.run_id, ev.eval_id)


def test_stub_run_and_stub_eval_refused(built):
    ws, d = built
    out = pl.run_experiment(ws, pl.tr.ExperimentConfig(base_model_id=pl.STUB_MODEL_ID), export=False)
    with pytest.raises(PackagingRefused, match="real training run"):
        pl.package_specialist(ws, out["run"].run_id, out["evaluation"].eval_id)


def test_package_never_overwrites(trained):
    ws, d, run, ev = trained
    pl.package_specialist(ws, run.run_id, ev.eval_id)
    with pytest.raises(PackagingRefused, match="already exists"):
        pl.package_specialist(ws, run.run_id, ev.eval_id)


def _copy(out, tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(out, dst)
    return dst


def test_validator_and_consumer_catch_tampering(trained, tmp_path):
    ws, d, run, ev = trained
    _, out = pl.package_specialist(ws, run.run_id, ev.eval_id)
    p = _copy(out, tmp_path)
    (p / "model" / "adapter" / "adapter_model.safetensors").write_bytes(b"evil")
    assert not validate_package(p).ok
    with pytest.raises(consumer.PackageError, match="hash mismatch"):
        consumer.Package.open(p)


def test_consumer_refuses_resealed_local_experiment_or_unverified_manifest(trained, tmp_path):
    """Even a correctly re-sealed manifest chain that declares a non-VERIFIED license is refused by the consumer."""
    ws, d, run, ev = trained
    _, out = pl.package_specialist(ws, run.run_id, ev.eval_id)
    p = _copy(out, tmp_path)
    from llmtrainer.schemas import ExportPackage, ModelManifest

    mm = ModelManifest.model_validate_json((p / "manifests" / "model_manifest.json").read_text())
    bad = mm.model_copy(update={"local_experiment": True, "content_hash": ""}).seal()
    (p / "manifests" / "model_manifest.json").write_text(bad.to_json())
    pk = ExportPackage.model_validate_json((p / MANIFEST_NAME).read_text())
    from llmtrainer.hashing import hash_file

    files = [f.model_copy(update={"sha256": hash_file(p / f.path), "size_bytes": (p / f.path).stat().st_size}) if f.path == "manifests/model_manifest.json" else f for f in pk.files]
    (p / MANIFEST_NAME).write_text(pk.model_copy(update={"files": files, "model_manifest_hash": bad.content_hash, "content_hash": ""}).seal().to_json())
    with pytest.raises(consumer.PackageError, match="local-experiment"):
        consumer.Package.open(p)
    assert not validate_package(p).ok


def test_cli_package_specialist_refusal_exit_code(built, tmp_path, capsys):
    ws, d = built
    add_entry(ws, exact_version="2", derivative_adapter_permitted="conditional")
    run = execute_training(plan_training(write_cfg(ws, tmp_path, base_model_id="test-llm@2"), allow_unverified=True), trainer=AdapterTrainer())
    ev = make_eval(ws, d, run.run_id)
    assert cli.main(["package-specialist", str(ws.root), "--run", run.run_id, "--eval", ev.eval_id]) == 1
    assert "local experiment" in capsys.readouterr().err
