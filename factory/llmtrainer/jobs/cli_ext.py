"""CLI registration for the studio job commands (kept out of cli.py to limit coupling)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

COMMANDS = ("import-job",)


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def register(sub) -> None:
    s = sub.add_parser("import-job", help="validate a studio training-job zip and rebuild a project workspace (leakage re-verified)")
    s.add_argument("job_zip")
    s.add_argument("--workspace", required=True, help="new (or empty) directory that becomes the project workspace")
    s.add_argument("--catalog", help="base-model catalog registry dir (default: repository registry/base-models)")


def dispatch(a) -> int:
    from .importer import import_job
    from .zipio import JobRejected

    if a.cmd == "import-job":
        try:
            rep = import_job(a.job_zip, a.workspace, catalog_dir=Path(a.catalog) if a.catalog else None)
        except JobRejected as e:
            print(f"REJECTED {a.job_zip}", file=sys.stderr)
            for p in e.problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        _print(rep.as_dict())
        return 0
    raise AssertionError(a.cmd)
