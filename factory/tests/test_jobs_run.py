"""evaluate / export-results / run-job: honest labelling, dry-run defaults, license gates, results package contract."""

import json
import sys
import zipfile

import pytest
from jobs_helpers import OWNED_RIGHTS, make_job, write_job

from llmtrainer import cli
from llmtrainer import pipeline as pl
from llmtrainer.jobs import hf_backend
from llmtrainer.jobs.evaluate import evaluate_workspace
from llmtrainer.jobs.importer import import_job
from llmtrainer.jobs.results import export_results, validate_results_zip
from llmtrainer.jobs.run import run_job
from llmtrainer.jobs.zipio import JobRejected, read_zip
from llmtrainer.packaging import validate_package
from llmtrainer.training import TrainerResult


class AdapterTrainer:
    name, version, is_stub = "fake-hf", "0", False

    def __init__(self):
        self.calls = 0

    def train(self, cfg, train_rows, val_rows, out_dir):
        self.calls += 1
        (out_dir / "adapter").mkdir(parents=True, exist_ok=True)
        (out_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
        a, c, t = out_dir / "adapter" / "adapter_model.safetensors", out_dir / "adapter" / "adapter_config.json", out_dir / "tokenizer" / "tokenizer.json"
        a.write_bytes(b"adapter-bytes")
        c.write_text('{"r": 16}')
        t.write_text('{"tok": 1}')
        return TrainerResult(metrics={"train_loss": 0.5}, artifacts={"adapter/adapter_model.safetensors": a, "adapter/adapter_config.json": c,
                                                                     "tokenizer/tokenizer.json": t})


def results_of(path):
    files = read_zip(path)
    return json.loads(files["manifest.json"]), json.loads(files["evaluation_report.json"]), files


def oracle_generators(job):
    """Base never knows; the 'specialist' answers from the job's own held-out items (test oracle, NOT a model)."""
    by_q = {h["question"]: h for h in job["heldout"]}

    def base(prompt):
        return "I do not know."

    def spec(prompt):
        if "Sources:" in prompt:
            for ln in prompt.split("Sources:", 1)[1].splitlines():
                if ln.startswith("[") and "]" in ln:
                    ref, text = ln[1:].split("]", 1)
                    return f"{text.strip()} [{ref}]"
        q = prompt.split("Question: ", 1)[-1]
        h = by_q.get(q)
        if h is None:
            return "I do not know."
        return f"{h['expected']['value']:g} {h['expected']['unit']}" if h["kind"] == "fact" else " ".join(h["required_terms"])

    return base, spec


@pytest.fixture
def imported(tmp_path):
    job = make_job()
    z = write_job(tmp_path / "j.zip", job)
    import_job(z, tmp_path / "ws")
    return pl.Workspace(tmp_path / "ws"), job, z


# ------------------------------------------------------------------ evaluate

def test_stub_backend_is_labelled_and_never_allows_a_claim(imported):
    ws, _, _ = imported
    o = evaluate_workspace(ws, backend="stub")
    ev = o.evalrun
    assert o.executed and ev.evaluator.is_stub and not ev.improvement_claim_allowed and ev.verify()
    assert ev.notes[0].startswith("STUB BACKEND") and ev.eval_split == "test" and ev.train_eval_group_overlap == 0
    assert {s.role for s in ev.subjects} == {"base", "specialist"} and ev.n_eval_examples == o.sidecar["items"]["total"]
    assert "delta_ci" in o.sidecar and (ws.root / "evals" / f"{ev.eval_id}.report.json").is_file()
    again = evaluate_workspace(ws, backend="stub")  # deterministic -> identical artifact, no overwrite conflict
    assert again.evalrun.content_hash == ev.content_hash


def test_stub_forbids_claim_even_with_many_real_items(tmp_path):
    job = make_job(n_docs=30, rights=OWNED_RIGHTS)
    import_job(write_job(tmp_path / "j.zip", job), tmp_path / "ws")
    ws = pl.Workspace(tmp_path / "ws")
    o = evaluate_workspace(ws, backend="stub")
    assert o.evalrun.n_eval_examples >= 50 and o.evalrun.evaluator.is_stub and o.evalrun.improvement_claim_allowed is False


def test_hf_backend_dry_run_plans_only_and_imports_nothing_heavy(imported):
    ws, _, _ = imported
    before = sorted(p.relative_to(ws.root).as_posix() for p in ws.root.rglob("*"))
    heavy = {m for m in ("torch", "transformers", "peft") if m in sys.modules}
    o = evaluate_workspace(ws, backend="hf")
    assert not o.executed and o.evalrun is None and "dry-run" in o.plan["mode"]
    assert o.plan["download_allowed"] is False and "reproducible_command" in o.plan
    assert sorted(p.relative_to(ws.root).as_posix() for p in ws.root.rglob("*")) == before, "dry-run must not write"
    assert {m for m in ("torch", "transformers", "peft") if m in sys.modules} == heavy
    assert set(hf_backend.missing_dependencies()) <= {"torch", "transformers", "peft"}


def test_hf_execute_without_adapter_or_base_path_is_an_error(imported):
    ws, _, _ = imported
    with pytest.raises(pl.WorkspaceError, match="base-model-path"):
        evaluate_workspace(ws, backend="hf", execute=True)
    with pytest.raises(pl.WorkspaceError, match="adapter"):
        evaluate_workspace(ws, backend="hf", execute=True, base_model_path="/nonexistent")


def test_hf_execute_refuses_non_local_base_without_allow_download(imported, tmp_path):
    ws, _, _ = imported
    ad = tmp_path / "adapter"
    ad.mkdir()
    with pytest.raises(pl.WorkspaceError, match="allow-download"):
        evaluate_workspace(ws, backend="hf", execute=True, base_model_path="Qwen/Qwen3-1.7B", adapter_path=str(ad))


def test_evaluation_refuses_a_workspace_whose_data_was_tampered(imported):
    ws, _, _ = imported
    ds = pl.latest_dataset(ws)
    f = ws.root / "datasets" / ds.splits["test"].file
    f.write_text(f.read_text() + "\n")
    with pytest.raises(pl.WorkspaceError, match="verification failed"):
        evaluate_workspace(ws, backend="stub")


def test_synthetic_or_small_n_never_allows_claim_even_with_a_perfect_specialist(tmp_path):
    job = make_job()  # synthetic sources, few items
    import_job(write_job(tmp_path / "j.zip", job), tmp_path / "ws")
    ws = pl.Workspace(tmp_path / "ws")
    o = evaluate_workspace(ws, backend="hf", generators=oracle_generators(job))
    assert not o.evalrun.evaluator.is_stub and o.executed
    assert o.sidecar["synthetic_data"] is True and o.evalrun.improvement_claim_allowed is False
    big = make_job(n_docs=30, rights=OWNED_RIGHTS)
    import_job(write_job(tmp_path / "b.zip", big), tmp_path / "wsb")
    ob = evaluate_workspace(pl.Workspace(tmp_path / "wsb"), backend="hf", generators=oracle_generators(big))
    assert ob.evalrun.n_eval_examples >= 50 and ob.evalrun.improvement_claim_allowed is True  # the only way to a claim: real data, n>=50, CI>0


# ------------------------------------------------------------------ run-job

def test_run_job_dry_run_reports_planned_vs_ran_and_trains_nothing(tmp_path):
    z = write_job(tmp_path / "j.zip", make_job())
    res = run_job(z, tmp_path / "ws")
    states = {s["stage"]: s["state"] for s in res["stages"]}
    assert states == {"import": "executed", "verify_dataset": "executed", "train": "planned", "evaluate": "planned", "package_results": "executed"}
    assert any("train" in x for x in res["only_planned"]) and not res["failed"] and not res["refused"]
    assert not list((tmp_path / "ws" / "runs").iterdir()), "dry-run must not create a training run"
    m, r, files = results_of(res["results"]["results"])
    assert validate_results_zip(res["results"]["results"]) == []
    assert m["status"] == "planned_only" and m["specialist"]["kind"] == "none" and m["specialist"]["artifact_refs"] == []
    assert r["improvement_claim_allowed"] is False and r["rows"] == [] and r["held_out_only"] is True
    assert m["license_state"]["owner_attested"] is True and m["license_state"]["state"] == "VERIFIED"
    assert [s["state"] for s in m["stages"]].count("planned") == 2 and m["job_id"] == "job-sample-0001"
    assert m["packaging"]["exportable"] is False


def test_run_job_stub_eval_is_labelled_stub(tmp_path):
    z = write_job(tmp_path / "j.zip", make_job())
    res = run_job(z, tmp_path / "ws", eval_backend="stub")
    m, r, _ = results_of(res["results"]["results"])
    assert m["status"] == "stub" and r["generated_by"]["stub"] is True and r["improvement_claim_allowed"] is False
    assert "STUB" in r["improvement_claim_reason"] and r["rows"] and all(set(x) >= {"ci_low", "ci_high", "n", "delta"} for x in r["rows"])
    assert any(x["ci_low"] is not None for x in r["rows"]) and m["specialist"]["kind"] == "none"
    assert validate_results_zip(res["results"]["results"]) == []


def test_run_job_rerun_reuses_import_and_overwrites_results(tmp_path):
    z = write_job(tmp_path / "j.zip", make_job())
    a = run_job(z, tmp_path / "ws")
    b = run_job(z, tmp_path / "ws", eval_backend="stub")
    assert {s["stage"]: s["state"] for s in b["stages"]}["import"] == "skipped" and a["results"]["results"] == b["results"]["results"]


@pytest.mark.parametrize("method", ["rag_only", "prompt_only", "full-finetune"])
def test_run_job_without_adapter_method_skips_training(tmp_path, method):
    z = write_job(tmp_path / "j.zip", make_job(method=method))
    res = run_job(z, tmp_path / "ws", execute=True, trainer=AdapterTrainer())
    st = {s["stage"]: s["state"] for s in res["stages"]}
    assert st["train"] == "skipped" and st["evaluate"] == "skipped"
    m, _, _ = results_of(res["results"]["results"])
    assert m["specialist"]["kind"] == "none" and (m["status"] == "no_training") == (method != "full-finetune")


def test_unverified_license_refuses_execute_and_never_calls_trainer(tmp_path, capsys):
    job = make_job()
    job["manifest"].pop("license_evidence")
    z = write_job(tmp_path / "j.zip", job)
    tr = AdapterTrainer()
    res = run_job(z, tmp_path / "ws", execute=True, trainer=tr)
    assert tr.calls == 0 and res["refused"] and "LICENSE_UNVERIFIED" in res["refused"][0] and not list((tmp_path / "ws" / "runs").iterdir())
    m, _, _ = results_of(res["results"]["results"])
    assert m["license_state"]["state"] == "UNVERIFIED" and m["license_state"]["owner_attested"] is False and m["status"] == "planned_only"
    z2 = write_job(tmp_path / "k.zip", job)
    rc = cli.main(["run-job", str(z2), "--workspace", str(tmp_path / "ws2"), "--execute"])
    assert rc == 2  # refused while --execute was requested


def test_local_experiment_is_flagged_and_can_never_be_exported(tmp_path):
    job = make_job()
    job["manifest"].pop("license_evidence")
    z = write_job(tmp_path / "j.zip", job)
    res = run_job(z, tmp_path / "ws", execute=True, allow_unverified_local=True, trainer=AdapterTrainer(), eval_backend="stub")
    m, r, _ = results_of(res["results"]["results"])
    assert m["specialist"]["kind"] == "adapter" and m["specialist"]["local_experiment"] is True
    assert m["packaging"]["exportable"] is False and any("local experiment" in b for b in m["packaging"]["blockers"])
    assert r["improvement_claim_allowed"] is False and any("LOCAL EXPERIMENT" in c for c in r["caveats"])
    assert m["license_state"]["state"] == "UNVERIFIED" and validate_results_zip(res["results"]["results"]) == []
    with pytest.raises(pl.WorkspaceError, match="refusing --include-artifacts"):
        export_results(pl.Workspace(tmp_path / "ws"), tmp_path / "x.zip", include_artifacts=True)


def test_full_attested_path_trains_evaluates_claims_and_packages(tmp_path):
    job = make_job(n_docs=30, rights=OWNED_RIGHTS)
    z = write_job(tmp_path / "j.zip", job)
    tr = AdapterTrainer()
    res = run_job(z, tmp_path / "ws", execute=True, trainer=tr, generators=oracle_generators(job), base_model_path=str(tmp_path))
    assert tr.calls == 1 and not res["failed"] and not res["refused"]
    out = res["results"]["results"]
    m, r, files = results_of(out)
    assert validate_results_zip(out) == []
    assert m["status"] == "completed" and m["specialist"]["kind"] == "adapter" and m["specialist"]["local_experiment"] is False
    assert m["specialist"]["training_run_id"] and set(m["specialist"]["artifact_refs"]) >= {"adapter/adapter_model.safetensors"}
    assert r["improvement_claim_allowed"] is True and r["generated_by"]["stub"] is False and r["sample_sizes"]["heldout_items"] >= 50
    assert m["packaging"] == {"exportable": True, "blockers": []} and m["license_state"]["owner_attested"] is True
    assert all(x["ci_low"] is not None for x in r["rows"] if x["n"])
    # --include-artifacts embeds exactly the hashed bytes
    ws = pl.Workspace(tmp_path / "ws")
    inc = export_results(ws, tmp_path / "inc.zip", include_artifacts=True)
    assert validate_results_zip(tmp_path / "inc.zip") == [] and "artifacts/adapter/adapter_model.safetensors" in inc["included_artifacts"]
    assert read_zip(tmp_path / "inc.zip")["artifacts/adapter/adapter_model.safetensors"] == b"adapter-bytes"
    with pytest.raises(pl.WorkspaceError, match="already exists"):
        export_results(ws, tmp_path / "inc.zip")
    # the owner-attested entry flows into a real specialist package, which validates and records the attestation
    run_id, eval_id = m["specialist"]["training_run_id"], m["evaluation"]["eval_id"]
    pkg, pdir = pl.package_specialist(ws, run_id, eval_id)
    assert validate_package(pdir).ok
    lic = (pdir / "license" / "base_model_license_entry.json").read_text()
    assert "OWNER-ATTESTED" in lic and pkg.model_identity


def test_tampered_results_zip_is_detected(tmp_path):
    z = write_job(tmp_path / "j.zip", make_job())
    out = run_job(z, tmp_path / "ws")["results"]["results"]
    files = read_zip(out)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        for n, b in files.items():
            zf.writestr(n, b.replace(b'"held_out_only": true', b'"held_out_only": false') if n == "evaluation_report.json" else b)
    assert any("checksum mismatch" in p for p in validate_results_zip(bad))


def test_hostile_job_never_reaches_training(tmp_path):
    job = make_job()
    job["examples"]["test"][0]["origin"] = "synthetic"
    z = write_job(tmp_path / "j.zip", job)
    tr = AdapterTrainer()
    with pytest.raises(JobRejected):
        run_job(z, tmp_path / "ws", execute=True, trainer=tr)
    assert tr.calls == 0 and not (tmp_path / "ws").exists()


# ------------------------------------------------------------------ standalone export-results + CLI

def test_export_results_needs_job_id_for_foreign_workspace(built, tmp_path):
    ws, _ = built
    with pytest.raises(pl.WorkspaceError, match="job-id"):
        export_results(ws, tmp_path / "r.zip")
    export_results(ws, tmp_path / "r.zip", job_id="job-manual")
    m, r, _ = results_of(tmp_path / "r.zip")
    assert m["job_id"] == "job-manual" and m["status"] == "planned_only" and r["rows"] == []
    assert validate_results_zip(tmp_path / "r.zip") == []


def test_cli_evaluate_export_results_run_job(tmp_path, capsys):
    z = write_job(tmp_path / "j.zip", make_job())
    assert cli.main(["import-job", str(z), "--workspace", str(tmp_path / "ws")]) == 0
    capsys.readouterr()
    assert cli.main(["evaluate", str(tmp_path / "ws"), "--backend", "stub"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["is_stub"] is True and out["improvement_claim_allowed"] is False
    assert cli.main(["evaluate", str(tmp_path / "ws"), "--backend", "hf"]) == 0
    assert "dry-run" in json.loads(capsys.readouterr().out)["mode"]
    assert cli.main(["export-results", str(tmp_path / "ws"), "--out", str(tmp_path / "r.zip")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "stub"
    assert cli.main(["export-results", str(tmp_path / "ws"), "--out", str(tmp_path / "r.zip")]) == 1  # no silent overwrite
    capsys.readouterr()
    assert cli.main(["export-results", str(tmp_path / "ws"), "--out", str(tmp_path / "r.zip"), "--force"]) == 0
    capsys.readouterr()
    assert cli.main(["run-job", str(z), "--workspace", str(tmp_path / "ws3")]) == 0
    cap = capsys.readouterr()
    assert "ONLY PLANNED (not run)" in cap.err and "RAN" in cap.err
    assert validate_results_zip(next((tmp_path / "ws3" / "results").glob("*.zip"))) == []
