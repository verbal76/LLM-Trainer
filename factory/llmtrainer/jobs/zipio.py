"""Hostile-input-safe zip reading, deterministic zip writing and the ``checksums.json`` contract."""

from __future__ import annotations

import json
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..hashing import canonical_json, hash_obj, sha256_bytes

CHECKSUMS_NAME = "checksums.json"
_FIXED_DATE = (2026, 10, 5, 0, 0, 0)  # deterministic archives
_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class JobRejected(ValueError):
    """A package was refused. ``problems`` lists every reason found (nothing is written on rejection)."""

    def __init__(self, problems: list[str] | str):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("package rejected:\n  - " + "\n  - ".join(self.problems))


@dataclass(frozen=True)
class ZipLimits:
    max_zip_bytes: int = 256 * 1024 * 1024
    max_members: int = 10_000
    max_member_bytes: int = 128 * 1024 * 1024
    max_total_bytes: int = 256 * 1024 * 1024
    max_ratio: int = 200  # uncompressed/compressed, only enforced for members > 1 MiB


def check_member_name(name: str) -> str | None:
    """Reason a member name is unsafe, or None."""
    if not name or "\x00" in name:
        return "empty or NUL name"
    if "\\" in name:
        return "backslash in name"
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return "absolute path"
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return "empty, '.' or '..' path segment"
    if any(not _SEGMENT.match(p) for p in parts):
        return "unsupported characters in path segment"
    return None


def read_zip(path: str | os.PathLike[str], limits: ZipLimits = ZipLimits()) -> dict[str, bytes]:
    """Read every member into memory under hard limits. Raises JobRejected on any hygiene violation."""
    p = Path(path)
    if not p.is_file():
        raise JobRejected(f"not a file: {p}")
    if p.stat().st_size > limits.max_zip_bytes:
        raise JobRejected(f"zip is {p.stat().st_size} bytes, over the {limits.max_zip_bytes} limit")
    problems: list[str] = []
    out: dict[str, bytes] = {}
    try:
        zf = zipfile.ZipFile(p)
    except zipfile.BadZipFile as e:
        raise JobRejected(f"not a valid zip: {e}") from e
    with zf:
        infos = zf.infolist()
        if len(infos) > limits.max_members:
            raise JobRejected(f"{len(infos)} members, over the {limits.max_members} limit")
        total = 0
        seen: set[str] = set()
        for info in infos:
            name = info.filename
            if name.endswith("/") and not check_member_name(name.rstrip("/")):
                continue  # directory entry: ignored, never extracted
            why = check_member_name(name)
            if why:
                problems.append(f"unsafe member name {name!r}: {why}")
                continue
            if name in seen:
                problems.append(f"duplicate member {name!r}")
                continue
            seen.add(name)
            mode = (info.external_attr >> 16) & 0o170000
            if mode and not stat.S_ISREG(mode):
                problems.append(f"member {name!r} is not a regular file (symlink/device)")
                continue
            if info.flag_bits & 0x1:
                problems.append(f"member {name!r} is encrypted")
                continue
            if info.file_size > limits.max_member_bytes:
                problems.append(f"member {name!r} is {info.file_size} bytes, over the {limits.max_member_bytes} limit")
                continue
            if info.file_size > (1 << 20) and info.compress_size and info.file_size / info.compress_size > limits.max_ratio:
                problems.append(f"member {name!r} has suspicious compression ratio")
                continue
            total += info.file_size
            if total > limits.max_total_bytes:
                problems.append(f"total uncompressed size over the {limits.max_total_bytes} limit")
                break
            with zf.open(info) as fh:  # headers can lie: read with a hard cap
                data = fh.read(limits.max_member_bytes + 1)
            if len(data) > limits.max_member_bytes or len(data) != info.file_size:
                problems.append(f"member {name!r} size does not match its header")
                continue
            out[name] = data
    if problems:
        raise JobRejected(problems)
    return out


def checksums_bytes(files: dict[str, bytes]) -> bytes:
    """Serialised ``checksums.json`` for ``files`` (which must not include checksums.json)."""
    doc = {"algorithm": "sha256", "files": {n: sha256_bytes(b) for n, b in sorted(files.items())}}
    return (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def verify_checksums(files: dict[str, bytes]) -> tuple[list[str], dict[str, str]]:
    """Problems found + the declared map. Exact set equality and every hash."""
    raw = files.get(CHECKSUMS_NAME)
    if raw is None:
        return [f"missing {CHECKSUMS_NAME}"], {}
    try:
        doc = json.loads(raw.decode("utf-8"))
        declared = doc["files"]
        if doc.get("algorithm") != "sha256" or not isinstance(declared, dict):
            raise ValueError("algorithm must be sha256 and files an object")
    except (ValueError, KeyError, TypeError, UnicodeDecodeError, AttributeError) as e:
        return [f"invalid {CHECKSUMS_NAME}: {e}"], {}
    problems = []
    actual = {n for n in files if n != CHECKSUMS_NAME}
    for n in sorted(set(declared) - actual):
        problems.append(f"{CHECKSUMS_NAME} lists {n!r} which is not in the zip")
    for n in sorted(actual - set(declared)):
        problems.append(f"{n!r} is in the zip but not listed in {CHECKSUMS_NAME}")
    for n in sorted(actual & set(declared)):
        if declared[n] != sha256_bytes(files[n]):
            problems.append(f"checksum mismatch for {n!r}")
    return problems, declared


def job_content_hash(declared: dict[str, str]) -> str:
    return hash_obj(declared)


def write_zip(path: str | os.PathLike[str], files: dict[str, bytes], *, with_checksums: bool = True) -> str:
    """Write a deterministic (sorted, fixed timestamps, STORED) zip; returns the sha256 of the zip file."""
    members = dict(files)
    if with_checksums:
        members.pop(CHECKSUMS_NAME, None)
        members[CHECKSUMS_NAME] = checksums_bytes(members)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_STORED) as z:
        for name in sorted(members):
            zi = zipfile.ZipInfo(name, date_time=_FIXED_DATE)
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o100644 << 16
            z.writestr(zi, members[name])
    return sha256_bytes(p.read_bytes())


def jdump(obj) -> bytes:
    """Human-readable, key-sorted JSON file body (not the canonical hashing form)."""
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def jsonl(rows: list[dict]) -> bytes:
    return b"".join(canonical_json(r) + b"\n" for r in rows)
