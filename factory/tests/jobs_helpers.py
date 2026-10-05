"""Helpers to build (and corrupt) studio job zips in tests."""

from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

from llmtrainer.hashing import hash_text
from llmtrainer.jobs.sample import SAMPLE_BASE_MODEL, build_job_dict, job_files, sample_license_evidence
from llmtrainer.jobs.zipio import checksums_bytes, jdump, jsonl, write_zip

OWNED_RIGHTS = {"status": "owned", "license_id": None, "permitted_training": "yes", "permitted_commercial": "yes",
                "permitted_redistribution": "yes", "evidence": "test", "notes": None}


def make_job(**kw) -> dict:
    kw.setdefault("license_evidence", sample_license_evidence())
    return build_job_dict(**kw)


def write_job(path: Path, job: dict, *, fix_counts: bool = True) -> Path:
    if fix_counts:
        job["dataset_manifest"]["counts"] = {k: len(v) for k, v in job["examples"].items()}
    write_zip(path, job_files(job))
    return path


def write_members(path: Path, members: dict[str, bytes], *, checksums: str = "correct") -> Path:
    """Write a zip exactly as given. checksums: 'correct' (computed from members), 'none', or 'as-given'."""
    members = dict(members)
    if checksums == "correct":
        members.pop("checksums.json", None)
        members["checksums.json"] = checksums_bytes(members)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for n, b in members.items():
            z.writestr(n, b)
    return path


def reserialise(job: dict) -> dict[str, bytes]:
    return job_files(job)


def first_test_ref(job: dict) -> tuple[str, str]:
    d = job["examples"]["test"][0]["derived_from"][0]
    return d["source_id"], d["chunk_id"]


def chunk_text_of(job: dict, sid: str, cid: str) -> str:
    return next(c["text"] for c in job["chunks"] if c["source_id"] == sid and c["id"] == cid)


def other_model(job: dict, registry_id: str = "Nonexistent@1") -> dict:
    job = copy.deepcopy(job)
    fam, ver = registry_id.split("@", 1)
    job["manifest"]["base_model"] = {"registry_id": registry_id, "family": fam, "exact_version": ver, "variant": None, "source_url": None}
    if job["manifest"].get("license_evidence"):
        job["manifest"]["license_evidence"]["model_id"] = registry_id
    return job


__all__ = ["make_job", "write_job", "write_members", "reserialise", "first_test_ref", "chunk_text_of", "other_model", "hash_text",
           "jdump", "jsonl", "json", "SAMPLE_BASE_MODEL", "OWNED_RIGHTS"]
