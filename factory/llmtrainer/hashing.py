"""Canonical JSON and content hashing.

Canonical form: UTF-8, keys sorted, no insignificant whitespace, NaN/Infinity
rejected. All hashes are rendered as ``sha256:<64 hex>``.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

HASH_PREFIX = "sha256:"


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return HASH_PREFIX + hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def hash_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hash_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk_size):
            h.update(block)
    return HASH_PREFIX + h.hexdigest()


def short(h: str, n: int = 12) -> str:
    return h.removeprefix(HASH_PREFIX)[:n]
