"""CLI registration for the studio job commands (kept out of cli.py to limit coupling)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

COMMANDS = ("import-job", "evaluate", "export-results", "run-job")


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def register(sub) -> None:
    s = sub.add_parser("import-job", help="validate a studio training-job zip and rebuild a project workspace (leakage re-verified)")
    s.add_argument("job_zip")
    s.add_argument("--workspace", required=True, help="new (or empty) directory that becomes the project workspace")
    s.add_argument("--catalog", help="base-model catalog registry dir (default: repository registry/base-models)")

    s = sub.add_parser("evaluate", help="base vs specialist on the held-out items (stub backend = plumbing only; hf = real, dry-run by default)")
    s.add_argument("project")
    s.add_argument("--backend", choices=["stub", "hf"], default="stub")
    s.add_argument("--run", help="training run id (hf: default is the latest completed run)")
    s.add_argument("--execute", action="store_true", help="hf only: really load the models and generate (default: print the plan)")
    s.add_argument("--allow-download", action="store_true", help="permit downloading base-model weights (default: local files only)")
    s.add_argument("--base-model-path")
    s.add_argument("--adapter-path")
    s.add_argument("--seed", type=int, default=0, help="bootstrap seed for confidence intervals")

    s = sub.add_parser("export-results", help="build the results zip the phone imports from the latest evaluation/training state")
    s.add_argument("project")
    s.add_argument("--out", required=True)
    s.add_argument("--include-artifacts", action="store_true", help="embed adapter files (only when the run is exportable)")
    s.add_argument("--job-id", help="needed only if the workspace was not created by import-job")
    s.add_argument("--eval", dest="eval_id")
    s.add_argument("--run", dest="run_id")
    s.add_argument("--force", action="store_true", help="overwrite an existing results zip")

    s = sub.add_parser("run-job", help="import -> verify -> train -> evaluate -> results; dry-run unless --execute")
    s.add_argument("job_zip")
    s.add_argument("--workspace", required=True)
    s.add_argument("--execute", action="store_true", help="really train (GPU + 'train' extra) and evaluate with real models")
    s.add_argument("--eval-backend", choices=["stub", "hf"], default="hf")
    s.add_argument("--allow-download", action="store_true", help="permit downloading base-model weights (may be many GB)")
    s.add_argument("--allow-unverified-license-for-local-experiment", action="store_true",
                   help="allow a non-VERIFIED (never an explicit 'no') license for a LOCAL experiment; blocks export forever")
    s.add_argument("--base-model-path", help="local base-model directory or HF repo id (default: catalog source)")
    s.add_argument("--base-params-b", type=float, help="parameter count in billions (default: catalog parameter_count)")
    s.add_argument("--out", help="results zip (default: <workspace>/results/<project>-<job>.llmtrainer-results.zip)")
    s.add_argument("--catalog", help="base-model catalog registry dir")


def dispatch(a) -> int:
    from .. import pipeline as pl
    from .zipio import JobRejected

    try:
        if a.cmd == "import-job":
            from .importer import import_job

            _print(import_job(a.job_zip, a.workspace, catalog_dir=Path(a.catalog) if a.catalog else None).as_dict())
            return 0
        if a.cmd == "evaluate":
            from .evaluate import evaluate_workspace

            o = evaluate_workspace(pl.Workspace(Path(a.project)), backend=a.backend, run_id=a.run, execute=a.execute,
                                   allow_download=a.allow_download, base_model_path=a.base_model_path, adapter_path=a.adapter_path, seed=a.seed)
            _print(o.as_dict())
            return 0
        if a.cmd == "export-results":
            from .results import export_results

            _print(export_results(pl.Workspace(Path(a.project)), a.out, include_artifacts=a.include_artifacts, job_id=a.job_id,
                                  eval_id=a.eval_id, run_id=a.run_id, overwrite=a.force))
            return 0
        if a.cmd == "run-job":
            from .run import run_job

            res = run_job(a.job_zip, a.workspace, execute=a.execute, eval_backend=a.eval_backend, allow_download=a.allow_download,
                          allow_unverified_local=a.allow_unverified_license_for_local_experiment, base_model_path=a.base_model_path,
                          base_params_b=a.base_params_b, out=a.out, catalog_dir=Path(a.catalog) if a.catalog else None)
            _print(res)
            for key, label in (("ran", "RAN"), ("only_planned", "ONLY PLANNED (not run)"), ("skipped", "SKIPPED"),
                               ("refused", "REFUSED"), ("failed", "FAILED")):
                for line in res[key]:
                    print(f"{label:<24} {line}", file=sys.stderr)
            if res["failed"]:
                return 1
            return 2 if (res["refused"] and a.execute) else 0
    except JobRejected as e:
        print(f"REJECTED {getattr(a, 'job_zip', '')}", file=sys.stderr)
        for p in e.problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    raise AssertionError(a.cmd)
