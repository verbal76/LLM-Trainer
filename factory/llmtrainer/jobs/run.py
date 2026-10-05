"""``llmtrainer run-job``: import -> dataset verify -> train -> evaluate -> results package, saying exactly what ran.

Dry-run is the default. Training only happens with ``execute=True`` (and then honours the existing license gate and
``allow_download`` rules); real-model evaluation likewise. Every stage is reported as one of
``executed | planned | skipped | refused | failed``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from .. import pipeline as pl
from ..licenses import LicenseGateError
from ..specialize.config import SpecializeConfig
from ..specialize.run import TrainingRefused, execute_training, plan_training
from .evaluate import evaluate_workspace
from .importer import import_job, read_job_state
from .results import export_results
from .verify import verify_workspace_dataset
from .zipio import JobRejected, ZipLimits

NO_TRAIN_METHODS = ("rag_only", "prompt_only")
TRAIN_METHODS = ("lora", "qlora")


def parse_params_b(parameter_count: str | None) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*B", parameter_count or "")
    return float(m.group(1)) if m else None


class _Stages:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, stage: str, state: str, detail: str, **data: Any) -> dict:
        row = {"stage": stage, "state": state, "detail": detail}
        self.items.append(row | ({"data": data} if data else {}))
        return row

    def public(self) -> list[dict]:
        return [{k: v for k, v in s.items() if k != "data"} for s in self.items]

    def by_state(self, state: str) -> list[str]:
        return [f"{s['stage']}: {s['detail']}" for s in self.items if s["state"] == state]


def run_job(
    job_zip: str | Path, workspace: str | Path, *, execute: bool = False, eval_backend: str = "hf", allow_download: bool = False,
    allow_unverified_local: bool = False, base_model_path: str | None = None, base_params_b: float | None = None,
    out: str | Path | None = None, trainer: Any = None, generators: tuple[Callable, Callable] | None = None,
    limits: ZipLimits = ZipLimits(), catalog_dir: Path | None = None,
) -> dict[str, Any]:
    st = _Stages()
    # 1. import (hostile-input validation happens here; a rejected job raises JobRejected and nothing else runs)
    rep = import_job(job_zip, workspace, limits=limits, catalog_dir=catalog_dir, reuse_existing=True)
    st.add("import", "skipped" if rep.reused else "executed",
           "workspace already holds this exact job (same job_content_hash); revalidated the zip, re-verified the data" if rep.reused
           else f"validated and rebuilt workspace from job {rep.job_id}", license=rep.license["state"], owner_attested=rep.license["owner_attested"])
    ws = pl.Workspace(Path(workspace))
    state = read_job_state(ws.root) or {}
    method = (state.get("method") or {}).get("requested", "")

    # 2. dataset verification (again, from the files, regardless of import)
    problems = verify_workspace_dataset(ws)
    if problems:
        raise JobRejected(problems)
    st.add("verify_dataset", "executed", "group/chunk/near-duplicate/containment leakage, synthetic-in-test and rights checks clean; test split not read by training")

    # 3. train
    run = None
    eval_id = None
    train_state = "skipped"
    if method in NO_TRAIN_METHODS:
        st.add("train", "skipped", f"method {method!r} requests no parameter training; nothing to train (reference/prompt specialization is assembled by the app)")
    elif method not in TRAIN_METHODS:
        st.add("train", "skipped", f"requested method {method!r} is not supported on the desktop (only {', '.join(TRAIN_METHODS)}); planned only")
    else:
        entry = ws.registry().get(state["base_model"]["registry_id"])
        params = base_params_b or parse_params_b(entry.parameter_count)
        path = base_model_path or entry.hf_repo_or_source
        if params is None or not path:
            st.add("train", "skipped", "cannot plan training: pass --base-params-b and --base-model-path (the catalog has no "
                                       f"parameter_count/source for {entry.entry_id})")
        else:
            cfg = SpecializeConfig(project=str(ws.root), base_model_id=entry.entry_id, base_model_path=str(path), base_params_b=params,
                                   method=method, seed=ws.project().seed)
            cfg_path = ws.root / "job" / "train_config.json"
            cfg_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
            try:
                plan = plan_training(cfg_path, allow_unverified=allow_unverified_local, allow_download=allow_download)
            except LicenseGateError as e:
                st.add("train", "refused", "license gate: " + "; ".join(e.result.blocking) +
                       " | to run as a LOCAL EXPERIMENT only (never exportable) add --allow-unverified-license-for-local-experiment",
                       reason_codes=e.result.reason_codes)
                train_state = "refused"
            except (TrainingRefused, pl.WorkspaceError, ValueError) as e:
                st.add("train", "failed", f"cannot plan training: {e}")
                train_state = "failed"
            else:
                if not execute:
                    flags = (" --allow-download" if allow_download else "") + (" --allow-unverified-license-for-local-experiment" if allow_unverified_local else "")
                    st.add("train", "planned", f"dry-run only: would run `llmtrainer train --config job/train_config.json --execute{flags}`",
                           plan=plan.as_dict())
                    train_state = "planned"
                else:
                    try:
                        run = execute_training(plan, trainer=trainer)
                        st.add("train", "executed", f"{run.method} run {run.run_id} {run.status}; adapter artifacts: {sorted(run.output_artifacts)}",
                               run_id=run.run_id, local_experiment=run.local_experiment_unverified_license)
                        train_state = "executed"
                    except Exception as e:  # noqa: BLE001 - recorded, reported, exit code non-zero
                        st.add("train", "failed", f"{type(e).__name__}: {e}")
                        train_state = "failed"

    # 4. evaluate
    if method not in TRAIN_METHODS:
        st.add("evaluate", "skipped", "no specialist to compare against the base model")
    elif eval_backend == "stub":
        o = evaluate_workspace(ws, backend="stub")
        eval_id = o.evalrun.eval_id
        st.add("evaluate", "executed", "STUB backend: toy responders, plumbing only; no claim possible" +
               ("; no real training happened" if run is None else ""), eval_id=o.evalrun.eval_id)
    elif run is not None:
        try:
            o = evaluate_workspace(ws, backend="hf", run_id=run.run_id, execute=True, allow_download=allow_download,
                                   base_model_path=base_model_path, generators=generators)
            eval_id = o.evalrun.eval_id
            st.add("evaluate", "executed", f"hf backend on {o.evalrun.n_eval_examples} held-out items", eval_id=eval_id)
        except Exception as e:  # noqa: BLE001
            st.add("evaluate", "failed", f"{type(e).__name__}: {e}")
    else:
        o = evaluate_workspace(ws, backend="hf", execute=False, base_model_path=base_model_path, allow_download=allow_download)
        st.add("evaluate", "planned" if train_state in ("planned", "skipped") else "skipped",
               "dry-run only: real-model evaluation needs a trained adapter" + (" (training was not executed)" if train_state == "planned" else ""),
               plan=o.plan)

    # 5. results package
    out = Path(out) if out else ws.root / "results" / f"{ws.project().project_id}-{state.get('job_id')}.llmtrainer-results.zip"
    status = "no_training" if method in NO_TRAIN_METHODS else None
    st.add("package_results", "executed", "wrote the results package (checksummed zip)")
    res = export_results(ws, out, stages=st.public(), status=status, overwrite=True, eval_id=eval_id, ignore_evals=eval_id is None,
                         run_id=run.run_id if run else None, ignore_runs=run is None,
                         notes=["Dry-run: nothing was trained or really evaluated; stages above show what was only planned."] if not execute else None)
    return {
        "job_id": state.get("job_id"), "workspace": str(ws.root), "execute": execute,
        "ran": st.by_state("executed"), "only_planned": st.by_state("planned"), "skipped": st.by_state("skipped"),
        "refused": st.by_state("refused"), "failed": st.by_state("failed"), "stages": st.items,
        "license": rep.license, "results": res,
    }
