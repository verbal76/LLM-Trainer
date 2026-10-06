"""Time source. Set LLMTRAINER_FIXED_NOW (ISO-8601) for reproducible output."""

from __future__ import annotations

import datetime as _dt
import os


def now_iso() -> str:
    fixed = os.environ.get("LLMTRAINER_FIXED_NOW")
    if fixed:
        return fixed
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def today() -> str:
    return now_iso()[:10]
