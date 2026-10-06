"""``llmtrainer export-results``: build the results zip the phone imports (docs/studio/PACKAGE_FORMATS.md section 2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import __version__
from .. import pipeline as pl
from ..clock import now_iso
from ..hashing import sha256_bytes
from ..licenses import entry_hash
from ..pipeline import WorkspaceError
from ..schemas import DatasetManifest, EvaluationRun, TrainingRun
from ..specialize.gating import redistribution_gate
from . import FORMAT_VERSION, REPORT_FORMAT, RESULTS_FORMAT
from .evaluate import latest_run
from .importer import read_job_state
from .license_override import is_owner_attested
from .zipio import jdump, write_zip

MIN_CLAIM_ITEMS = 50
STAGES = ("import", "verify_dataset", "train", "evaluate", "package_results")


def list_evals(ws: pl.Workspace, dataset: DatasetManifest) -> list[tuple[EvaluationRun, Path]]:
    out = []
    for f in sorted((ws.root / "evals").glob("eval-*.json")):
        if f.name.endswith(".report.json"):
            continue
        ev = EvaluationRun.model_validate_json(f.read_text(encoding="utf-8"))
        if not ev.verify():
            raise WorkspaceError(f"{f} content_hash mismatch")
        if ev.dataset_hash == dataset.content_hash:
            out.append((ev, f))
    return out


def latest_eval(ws: pl.Workspace, dataset: DatasetManifest, eval_id: str | None = None) -> tuple[EvaluationRun, Path] | None:
    evs = list_evals(ws, dataset)
    if eval_id:
        for e in evs:
            if e[0].eval_id == eval_id:
                return e
        raise WorkspaceError(f"evaluation {eval_id} not found for the current dataset")
    return max(evs, key=lambda e: (e[0].evaluated_on, e[1].stat().st_mtime_ns, e[0].eval_id)) if evs else None


def _matches_run(ev: EvaluationRun, run: TrainingRun) -> bool:
    spec = next((s for s in ev.subjects if s.role == "specialist"), None)
    if spec is None:
        return False
    adapter_hash = run.output_artifacts.get("adapter/adapter_model.safetensors")
    return (run.run_id in spec.model_ref) or bool(adapter_hash and spec.artifact_hash == adapter_hash)


def claim_decision(ev: EvaluationRun | None, sidecar: dict, run: TrainingRun | None, kind: str) -> tuple[bool, str]:
    """Package-level improvement claim: never looser than the sealed EvaluationRun."""
    if ev is None:
        return False, "No evaluation has been run; nothing can be claimed."
    why = []
    if ev.evaluator.is_stub or sidecar.get("stub"):
        why.append("STUB evaluation: the evaluated subjects were toy responders, not models")
    if kind == "none" or run is None:
        why.append("no trained specialist exists (no parameter training was executed)")
    elif run.local_experiment_unverified_license:
        why.append("the run was a local experiment under a non-VERIFIED base-model license")
    elif not _matches_run(ev, run):
        why.append("the evaluated specialist subject is not this training run's adapter")
    if sidecar.get("synthetic_data"):
        why.append("held-out data comes from synthetic sources")
    if ev.n_eval_examples < MIN_CLAIM_ITEMS:
        why.append(f"only {ev.n_eval_examples} held-out items (< {MIN_CLAIM_ITEMS})")
    if not ev.improvement_claim_allowed and not why:
        why.append("the evaluation did not meet the claim rules (every domain metric must improve with a 95% CI excluding 0, no regression)")
    if why:
        return False, "Improvement claim NOT allowed: " + "; ".join(why) + "."
    return True, (f"Allowed: non-stub evaluation on {ev.n_eval_examples} held-out items, every domain metric improved with a 95% CI excluding 0, "
                  "no regression, trained specialist evaluated.")


def _rows(ev: EvaluationRun, sidecar: dict) -> list[dict]:
    ci = sidecar.get("delta_ci", {})
    ns = sidecar.get("metric_n") or {r.metric: r.n for r in ev.metrics if r.subject_role == "specialist"}
    rows = []
    for c in ev.comparisons:
        lo_hi = ci.get(c.metric)
        notes = [f"category: {c.category}", "higher is better" if c.higher_is_better else "lower is better"]
        if c.regression:
            notes.append("REGRESSION beyond tolerance")
        if c.quantization_regression:
            notes.append("quantization regression")
        if lo_hi is None:
            notes.append("no confidence interval recorded")
        rows.append({"metric": c.metric, "category": c.category, "higher_is_better": c.higher_is_better, "base": c.base,
                     "specialist": c.specialist, "delta": c.specialist_minus_base, "ci_low": lo_hi[0] if lo_hi else None,
                     "ci_high": lo_hi[1] if lo_hi else None, "n": ns.get(c.metric), "notes": "; ".join(notes), "quantized": c.quantized})
    return rows


def derive_status(ev: EvaluationRun | None, run: TrainingRun | None, stages: list[dict]) -> str:
    if any(s["state"] == "failed" for s in stages):
        return "failed" if not any(s["state"] == "executed" and s["stage"] in ("train", "evaluate") for s in stages) else "partial"
    if ev is not None and (ev.evaluator.is_stub):
        return "stub"
    if ev is not None and run is not None and run.status == "completed" and _matches_run(ev, run):
        return "completed"
    if ev is not None or run is not None:
        return "partial"
    return "planned_only"


def export_results(
    ws: pl.Workspace, out: str | Path, *, include_artifacts: bool = False, job_id: str | None = None, eval_id: str | None = None,
    run_id: str | None = None, stages: list[dict] | None = None, status: str | None = None, overwrite: bool = False,
    notes: list[str] | None = None, ignore_evals: bool = False, ignore_runs: bool = False,
) -> dict[str, Any]:
    out = Path(out)
    if out.exists() and not overwrite:
        raise WorkspaceError(f"{out} already exists (pass --force to overwrite)")
    project = ws.project()
    dataset = pl.latest_dataset(ws)
    state = read_job_state(ws.root)
    job_id = job_id or (state or {}).get("job_id")
    if not job_id:
        raise WorkspaceError("workspace was not created by import-job: pass --job-id to name the job these results answer")
    got = None if ignore_evals else latest_eval(ws, dataset, eval_id)
    ev, ev_path = got if got else (None, None)
    sidecar: dict = {}
    if ev_path is not None:
        sc = ev_path.with_name(ev_path.stem + ".report.json")
        if sc.is_file():
            sidecar = json.loads(sc.read_text(encoding="utf-8"))
    sc_run = sidecar.get("training_run_id")
    run = None if ignore_runs else (latest_run(ws, dataset, run_id or sc_run) if (run_id or sc_run) else latest_run(ws, dataset))
    base_id = (run.base_model_id if run else None) or ((state or {}).get("base_model") or {}).get("registry_id")
    entry = None
    try:
        entry = ws.registry().get(base_id) if base_id else None
    except KeyError:
        entry = None
    kind = "adapter" if (run is not None and run.status == "completed" and not run.is_pipeline_validation_stub) else "none"

    # ---- license state (redistribution-grade, fail closed)
    if entry is not None:
        gate, lic_state = redistribution_gate(entry, commercial=project.intended_use.commercial)
        v = entry.verification
        license_state = {
            "state": lic_state, "owner_attested": is_owner_attested(entry), "attested_by": "owner" if is_owner_attested(entry) else None,
            "attested_on": v.verified_on if is_owner_attested(entry) else None, "evidence_level": v.evidence_level,
            "license_url": v.license_text_url or (v.source_urls[0] if v.source_urls else None), "license_text_sha256": v.license_text_sha256,
            "scope_note": v.verified_scope, "export_allowed": lic_state == "VERIFIED", "reasons": list(gate.blocking)}
        lic_hash = entry_hash(entry)
    else:
        license_state = {"state": "UNVERIFIED", "owner_attested": False, "attested_by": None, "attested_on": None, "evidence_level": "none",
                         "license_url": None, "license_text_sha256": None, "scope_note": None, "export_allowed": False,
                         "reasons": ["base model has no license registry entry in this workspace"]}
        lic_hash = None
        lic_state = "UNVERIFIED"

    # ---- specialist
    refs = sorted(run.output_artifacts) if (run and kind == "adapter") else []
    local_exp = bool(run and run.local_experiment_unverified_license)
    blockers: list[str] = []
    if kind != "adapter":
        blockers.append("no trained adapter exists")
    if local_exp:
        blockers.append("run was a local experiment under a non-VERIFIED license; it can never be exported")
    if lic_state != "VERIFIED":
        blockers.append(f"base-model license is {lic_state} for redistribution")
    if ev is None:
        blockers.append("no evaluation")
    elif ev.evaluator.is_stub:
        blockers.append("evaluation is a stub")
    elif ev.train_eval_group_overlap != 0:
        blockers.append("evaluation reports train/eval group overlap")
    elif run is not None and not _matches_run(ev, run):
        blockers.append("evaluation is not of this run's adapter")
    exportable = not blockers
    files: dict[str, bytes] = {}
    included: list[str] = []
    if include_artifacts:
        if not exportable:
            raise WorkspaceError("refusing --include-artifacts: " + "; ".join(blockers))
        assert run is not None
        for ref in refs:
            data = (ws.root / "runs" / run.run_id / "output" / ref).read_bytes()
            if sha256_bytes(data) != run.output_artifacts[ref]:
                raise WorkspaceError(f"artifact changed since training: {ref}")
            files[f"artifacts/{ref}"] = data
            included.append(f"artifacts/{ref}")

    allowed, reason = claim_decision(ev, sidecar, run, kind)
    stages = list(stages) if stages is not None else [
        {"stage": "package_results", "state": "executed",
         "detail": "exported from existing workspace state; per-stage history was not recorded by this command"}]
    created = now_iso()
    ds = dataset
    caveats: list[str] = []
    if ev is not None:
        caveats = [n for n in ev.notes if not n.startswith("performance[") and " delta=" not in n]
        if not sidecar:
            caveats.append("No sidecar report found: confidence intervals and per-metric n are unavailable.")
        if run is not None and not _matches_run(ev, run):
            caveats.append("The evaluated specialist subject is not this workspace's trained adapter.")
    else:
        caveats.append("No evaluation was run: there are no metrics to display.")
    if license_state["owner_attested"]:
        caveats.append("The base-model license state rests on the owner's attestation of the license text; LLM Trainer did not interpret it.")
    if local_exp:
        caveats.append("LOCAL EXPERIMENT: trained under a non-VERIFIED license; the artifacts must never be exported or shared.")
    caveats += list(notes or [])

    report = {
        "format": REPORT_FORMAT, "version": FORMAT_VERSION, "project_id": project.project_id, "job_id": job_id,
        "eval_id": ev.eval_id if ev else None, "evaluated_on": ev.evaluated_on if ev else None, "held_out_only": True, "split": "test",
        "generated_by": {"stub": bool(ev is None or ev.evaluator.is_stub), "backend": sidecar.get("backend", "none" if ev is None else "external"),
                         "evaluator": ev.evaluator.model_dump() if ev else None, "tool": "llmtrainer", "tool_version": __version__,
                         "subjects": sidecar.get("subject_models") or ({s.role: s.model_ref for s in ev.subjects} if ev else {})},
        "rows": _rows(ev, sidecar) if ev else [],
        "regressions": sorted(c.metric for c in ev.comparisons if c.regression or c.quantization_regression) if ev else [],
        "caveats": caveats,
        "sample_sizes": {
            "heldout_items": ev.n_eval_examples if ev else 0, "fact_items": (sidecar.get("items") or {}).get("fact"),
            "concept_items": (sidecar.get("items") or {}).get("concept"), "train_examples": ds.splits["train"].n_examples,
            "validation_examples": ds.splits["validation"].n_examples, "test_examples": ds.splits["test"].n_examples,
            "per_metric_n": sidecar.get("metric_n") or ({r.metric: r.n for r in ev.metrics if r.subject_role == "specialist"} if ev else {})},
        "improvement_claim_allowed": allowed, "improvement_claim_reason": reason,
        "performance": sidecar.get("performance") or {},
    }
    status = status or derive_status(ev, run, stages)
    manifest = {
        "format": RESULTS_FORMAT, "version": FORMAT_VERSION, "project_id": project.project_id, "job_id": job_id,
        "job_content_hash": (state or {}).get("job_content_hash"), "created_at": created, "status": status,
        "base_model": {"registry_id": entry.entry_id if entry else base_id, "family": entry.model_family if entry else None,
                       "exact_version": entry.exact_version if entry else None,
                       "variant": ((state or {}).get("base_model") or {}).get("variant"), "license_entry_hash": lic_hash},
        "license_state": license_state,
        "specialist": {
            "kind": kind, "artifact_refs": refs, "sha256s": {r: run.output_artifacts[r] for r in refs} if run else {},
            "included_artifacts": included, "quantization": None, "training_run_id": run.run_id if run else None,
            "training_run_hash": run.content_hash if run else None, "config_hash": run.config_hash if run else None,
            "method": run.method if run else None, "trainer": run.trainer if run else None, "local_experiment": local_exp,
            "is_pipeline_validation_stub": bool(run and run.is_pipeline_validation_stub)},
        "dataset": {"dataset_id": ds.dataset_id, "dataset_hash": ds.content_hash, "source_manifest_hash": ds.source_manifest_hash,
                    "n_train": ds.splits["train"].n_examples, "n_validation": ds.splits["validation"].n_examples,
                    "n_test": ds.splits["test"].n_examples},
        "evaluation": {"eval_id": ev.eval_id if ev else None, "evaluation_hash": ev.content_hash if ev else None,
                       "is_stub": bool(ev.evaluator.is_stub) if ev else None, "improvement_claim_allowed": allowed},
        "stages": stages, "packaging": {"exportable": exportable, "blockers": blockers},
        "generated_by": {"tool": "llmtrainer", "tool_version": __version__}, "notes": list(notes or []),
    }
    files["manifest.json"] = jdump(manifest)
    files["evaluation_report.json"] = jdump(report)
    if ev is not None:
        files["python/evaluation_run.json"] = ev.to_json().encode("utf-8")
    if run is not None:
        files["python/training_run.json"] = run.to_json().encode("utf-8")
    files["python/dataset_manifest.json"] = ds.to_json().encode("utf-8")
    zip_sha = write_zip(out, files)
    return {"results": str(out), "zip_sha256": zip_sha, "status": status, "specialist_kind": kind, "license_state": license_state["state"],
            "owner_attested": license_state["owner_attested"], "exportable": exportable, "blockers": blockers,
            "improvement_claim_allowed": allowed, "improvement_claim_reason": reason, "eval_id": ev.eval_id if ev else None,
            "included_artifacts": included, "stages": stages}


# --------------------------------------------------------------------------- #
# Reference validator (what a consumer such as the phone app must enforce)
# --------------------------------------------------------------------------- #

_STATUS = {"completed", "stub", "planned_only", "partial", "failed", "no_training"}
_KIND = {"adapter", "merged", "none"}
_LIC = {"VERIFIED", "UNVERIFIED", "CONDITIONAL", "RESTRICTED"}
_STAGE_STATE = {"executed", "planned", "skipped", "refused", "failed"}
_ROW_KEYS = {"metric", "category", "higher_is_better", "base", "specialist", "delta", "ci_low", "ci_high", "n", "notes", "quantized"}


def validate_results_zip(path: str | Path) -> list[str]:
    """Problems found in a results package (empty == valid v1). Checks hygiene, checksums, required fields and honesty invariants."""
    from .zipio import JobRejected, read_zip, verify_checksums

    try:
        files = read_zip(path)
    except JobRejected as e:
        return e.problems
    p, _ = verify_checksums(files)
    if p:
        return p
    try:
        m = json.loads(files["manifest.json"])
        r = json.loads(files["evaluation_report.json"])
    except (KeyError, ValueError) as e:
        return [f"manifest.json/evaluation_report.json missing or invalid: {e}"]
    p = []
    if m.get("format") != RESULTS_FORMAT or m.get("version") != FORMAT_VERSION:
        p.append("manifest format/version")
    if r.get("format") != REPORT_FORMAT or r.get("version") != FORMAT_VERSION:
        p.append("report format/version")
    for k in ("project_id", "job_id", "created_at", "status", "base_model", "license_state", "specialist", "dataset", "evaluation", "stages", "packaging"):
        if k not in m:
            p.append(f"manifest missing {k}")
    if m.get("status") not in _STATUS:
        p.append(f"bad status {m.get('status')!r}")
    sp_ = m.get("specialist", {})
    if sp_.get("kind") not in _KIND:
        p.append(f"bad specialist.kind {sp_.get('kind')!r}")
    if sp_.get("kind") == "none" and (sp_.get("artifact_refs") or sp_.get("included_artifacts")):
        p.append("specialist.kind none must have no artifacts")
    if set(sp_.get("sha256s", {})) != set(sp_.get("artifact_refs", [])):
        p.append("sha256s keys must equal artifact_refs")
    for inc in sp_.get("included_artifacts", []):
        ref = inc.removeprefix("artifacts/")
        if inc not in files or sha256_bytes(files[inc]) != sp_.get("sha256s", {}).get(ref):
            p.append(f"included artifact {inc} missing or hash differs")
    if m.get("license_state", {}).get("state") not in _LIC:
        p.append("bad license_state.state")
    if any(s.get("state") not in _STAGE_STATE or s.get("stage") not in STAGES for s in m.get("stages", [])):
        p.append("bad stage entry")
    if r.get("held_out_only") is not True or r.get("split") != "test":
        p.append("report must be held_out_only on the test split")
    for row in r.get("rows", []):
        if set(row) != _ROW_KEYS:
            p.append(f"row keys for {row.get('metric')}: {sorted(set(row) ^ _ROW_KEYS)}")
    claim = r.get("improvement_claim_allowed")
    if claim != m.get("evaluation", {}).get("improvement_claim_allowed"):
        p.append("improvement_claim_allowed differs between manifest and report")
    if not r.get("improvement_claim_reason"):
        p.append("improvement_claim_reason missing")
    if claim and (r.get("generated_by", {}).get("stub") or sp_.get("kind") == "none" or sp_.get("local_experiment")
                  or r.get("sample_sizes", {}).get("heldout_items", 0) < MIN_CLAIM_ITEMS):
        p.append("improvement claim allowed although stub / no specialist / local experiment / too few items")
    if m.get("status") == "completed" and (r.get("generated_by", {}).get("stub") or sp_.get("kind") == "none"):
        p.append("status completed with a stub evaluation or no specialist")
    return p
