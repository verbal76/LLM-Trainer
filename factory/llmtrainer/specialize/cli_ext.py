"""CLI registration for the real-specialization commands (kept out of cli.py to limit coupling)."""

from __future__ import annotations

import json
import sys


def register(sub) -> None:
    s = sub.add_parser(
        "train",
        help="real LoRA/QLoRA specialization (default and --dry-run only plan; --execute launches compute)",
    )
    s.add_argument("--config", required=True, help="SpecializeConfig JSON")
    mode = s.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate, estimate, print the command; launch nothing")
    mode.add_argument("--execute", action="store_true", help="really train (needs GPU + the 'train' extra)")
    s.add_argument("--allow-unverified-license-for-local-experiment", action="store_true",
                   help="allow a non-VERIFIED (never an explicit 'no') base-model license for a LOCAL experiment; "
                        "recorded in the TrainingRun and permanently blocks packaging/export")
    s.add_argument("--allow-download", action="store_true",
                   help="permit downloading base-model weights (may be many GB); default is offline/local only")


def dispatch(a) -> int:
    from .run import execute_training, plan_training

    plan = plan_training(a.config, allow_unverified=a.allow_unverified_license_for_local_experiment,
                         allow_download=a.allow_download)
    if not a.execute:
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
        return 0
    run = execute_training(plan)
    print(json.dumps({"run_id": run.run_id, "status": run.status, "config_hash": run.config_hash,
                      "local_experiment_unverified_license": run.local_experiment_unverified_license,
                      "output_artifacts": run.output_artifacts, "train_metrics": run.train_metrics}, indent=2, sort_keys=True))
    return 0


def exit_hint(exc: Exception) -> None:
    print(f"error: {exc}", file=sys.stderr)
