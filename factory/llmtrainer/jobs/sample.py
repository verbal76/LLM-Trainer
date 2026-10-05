"""Deterministic sample/fixture job packages.

Emulates what the phone app produces, using the factory's own chunker/splitter so the result is internally consistent.
The text is INVENTED synthetic fixture text (no third-party material). Used by tests and by ``tools/make_studio_fixtures.py``.
"""

from __future__ import annotations

import copy
from typing import Any

from .. import splits as sp
from ..datasets import generate_examples
from ..evalsuite.items import build_items
from ..hashing import hash_obj, hash_text, sha256_bytes, short
from ..ingest import chunk_text
from .zipio import jdump, jsonl

SAMPLE_JOB_ID = "job-sample-0001"
SAMPLE_PROJECT_ID = "proj-pump-service"
SAMPLE_BASE_MODEL = {
    "registry_id": "Qwen3@Qwen3-1.7B (2504 generation, hybrid thinking)",
    "family": "Qwen3", "exact_version": "Qwen3-1.7B (2504 generation, hybrid thinking)",
    "variant": "hf-weights", "source_url": "https://huggingface.co/Qwen/Qwen3-1.7B",
}
OWNED = {"status": "owned", "license_id": None, "permitted_training": "yes", "permitted_commercial": "yes",
         "permitted_redistribution": "yes", "evidence": "invented sample text authored for this fixture", "notes": None}
SYNTHETIC = {**OWNED, "status": "synthetic"}


def sample_license_evidence(model_id: str = SAMPLE_BASE_MODEL["registry_id"], text: bytes = b"SAMPLE LICENSE TEXT (fixture, not a real license)\n") -> dict:
    return {
        "model_id": model_id, "license_url": "https://example.invalid/sample-license", "fetched_at": "2026-10-05T11:00:00Z",
        "text_sha256": sha256_bytes(text), "evidence_level": "primary_license_text_read", "attested_by": "owner",
        "attested_on": "2026-10-05",
        "permissions": {"commercial_use": "yes", "fine_tuning_permitted": "yes", "derivative_adapter_permitted": "yes",
                        "redistribution_permitted": "yes", "attribution_required": "yes"},
        "scope_note": "FIXTURE ATTESTATION: sample data for format tests only.",
    }


def _doc(i: int) -> str:
    parts = [f"# Pump manual {i}\n"]
    for j in range(6):
        u = lambda n: f"z{i}{j}q{n}"  # noqa: E731  unique tokens: no 5-gram is shared across documents
        parts.append(
            f"## Procedure {i}-{j}\n\nTighten {u(1)} the flange{i}{j} {u(2)} bolt {u(3)} to {10 + i + j} N·m {u(4)} using {u(5)} the calibrated {u(6)} wrench "
            f"before {u(7)} commissioning the {u(8)} assembly. Maintain {u(9)} a clearance {u(10)} of 0.{i}{j + 1} mm {u(11)} between the {u(12)} rotor{i}{j} "
            f"{u(13)} housing and {u(14)} the guard {u(15)} during inspection. The {u(16)} technician must {u(17)} record the {u(18)} actuator{i}{j} "
            f"{u(19)} reading in {u(20)} the logbook {u(21)} after every {u(22)} shift change.\n"
        )
    return "\n".join(parts)


def build_job_dict(*, n_docs: int = 6, rights: dict | None = None, license_evidence: dict | None = None,
                   base_model: dict | None = None, method: str = "qlora", job_id: str = SAMPLE_JOB_ID, seed: int = 1234,
                   with_license_text: bool = True, commercial: bool = False) -> dict[str, Any]:
    """Return the logical job as plain Python data: manifest, chunks, dataset_manifest, examples{split}, heldout (all mutable)."""
    rights = copy.deepcopy(rights if rights is not None else SYNTHETIC)
    sources, chunks = [], []
    for i in range(n_docs):
        sid = f"src-sample-{i:02d}"
        raw = _doc(i).encode("utf-8")
        sources.append({
            "id": sid, "file_name": f"pump_manual_{i}.md", "title": f"Pump manual {i}", "sha256": sha256_bytes(raw),
            "size": len(raw), "mime": "text/markdown", "ingested_at": "2026-10-05T10:00:00Z", "extractor": "plain-text", "extractor_version": "1",
            "origin": "sample fixture", "rights": copy.deepcopy(rights), "issues": [],
        })
        for c in chunk_text(sid, raw.decode("utf-8")):
            chunks.append({
                "id": c.record.chunk_id, "source_id": sid, "page": c.record.page_start, "section_path": [c.record.section],
                "char_start": c.record.char_start, "char_end": c.record.char_end, "role": "train", "exclude_reason": None,
                "text": c.text, "sha256": hash_text(c.text), "origin": "extracted",
            })
    # examples: deterministic template examples per chunk, whole documents assigned to one split
    cand = []
    for c in chunks:
        for ex in generate_examples(c["section_path"][0], c["text"]):
            if ex["quality"] < 0.4:
                continue
            cand.append({"example_id": "ex-" + short(hash_obj([c["source_id"], c["id"], ex["task"], ex["idx"]])), "group_id": c["source_id"],
                         "prompt": ex["prompt"], "response": ex["response"], "origin": "source_derived",
                         "derived_from": [{"source_id": c["source_id"], "chunk_id": c["id"]}], "task": ex["task"], "quality": round(ex["quality"], 4)})
    weights: dict[str, int] = {}
    for e in cand:
        weights[e["group_id"]] = weights.get(e["group_id"], 0) + 1
    ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}
    assign = sp.assign_groups(weights, ratios, seed)
    examples = {s: sorted((e for e in cand if assign[e["group_id"]] == s), key=lambda e: e["example_id"]) for s in ("train", "validation", "test")}
    # held-out items: only chunks that back test examples
    test_refs = {(e["derived_from"][0]["source_id"], e["derived_from"][0]["chunk_id"]) for e in examples["test"]}
    test_chunks = [{"ref": f"{c['source_id']}/{c['id']}", "section": c["section_path"][0], "text": c["text"],
                    "synthetic": rights["status"] == "synthetic"} for c in chunks if (c["source_id"], c["id"]) in test_refs]
    heldout = []
    for it in build_items(test_chunks, domain="pump service"):
        heldout.append({"item_id": it.item_id, "kind": it.kind, "question": it.question, "gold_refs": list(it.gold_refs),
                        "expected": None if it.expected is None else {"value": it.expected.value, "unit": it.expected.unit},
                        "required_terms": list(it.required_terms), "origin": "source_derived"})
    manifest = {
        "format": "llmtrainer-training-job", "version": 1, "job_id": job_id, "created_at": "2026-10-05T12:00:00Z",
        "created_by": {"app": "llmtrainer-studio", "app_version": "0.1.0-fixture"},
        "project": {"id": SAMPLE_PROJECT_ID, "name": "Pump Service", "domain": "pump service",
                    "purpose": "Sample project for format tests", "created_at": "2026-10-01T09:00:00Z",
                    "intended_use": {"commercial": commercial, "redistribute_model": False}},
        "device": {"name": "sample-device", "total_ram_mb": 8192},
        "base_model": copy.deepcopy(base_model or SAMPLE_BASE_MODEL),
        "method": {"requested": method, "rationale": "Sample: domain vocabulary via adapter training; exact specs stay in retrieval."},
        "sources": sources, "extensions": {},
    }
    if license_evidence is not None:
        manifest["license_evidence"] = license_evidence
    dsm = {
        "dataset_id": "ds-phone-sample", "dataset_version": 1, "seed": seed,
        "split_config": {"ratios": ratios, "group_by": "document", "near_duplicate_threshold": 0.8, "shingle_size": 5},
        "builder": {"name": "studio-template-builder", "version": "1", "type": "template"},
        "split_assignment": {"group_by": "document", "groups": dict(sorted(assign.items()))},
        "counts": {s: len(v) for s, v in examples.items()},
    }
    return {"manifest": manifest, "chunks": chunks, "dataset_manifest": dsm, "examples": examples, "heldout": heldout,
            "license_text": b"SAMPLE LICENSE TEXT (fixture, not a real license)\n" if with_license_text else None}


def job_files(job: dict[str, Any]) -> dict[str, bytes]:
    """Serialise a logical job into zip members (without checksums.json)."""
    files = {
        "manifest.json": jdump(job["manifest"]),
        "chunks.jsonl": jsonl(job["chunks"]),
        "dataset_manifest.json": jdump(job["dataset_manifest"]),
        "eval/heldout.jsonl": jsonl(job["heldout"]),
    }
    for s, rows in job["examples"].items():
        files[f"dataset/{s}.jsonl"] = jsonl(rows)
    if job.get("license_text") is not None:
        files["license/license_text.txt"] = job["license_text"]
    return files


def sample_job_files(**kw) -> dict[str, bytes]:
    kw.setdefault("license_evidence", sample_license_evidence())
    return job_files(build_job_dict(**kw))

