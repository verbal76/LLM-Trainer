"""``llmtrainer import-job``: validate a phone-built training-job zip and rebuild a project workspace from it.

Nothing is written unless every check passes (the workspace is assembled in a sibling temporary directory and renamed).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .. import pipeline as pl
from .. import splits as sp
from ..clock import now_iso
from ..datasets import leak_text
from ..hashing import hash_obj, hash_text, sha256_bytes, short
from ..licenses import entry_hash
from ..schemas import (
    ChunkRecord,
    ChunkRef,
    DatasetManifest,
    ExampleRecord,
    GeneratorInfo,
    IntendedUse,
    LeakageReport,
    SourceManifest,
    SourceRecord,
    SpecialistProject,
    SplitConfig,
    SplitInfo,
    Transformation,
)
from . import FORMAT_VERSION, JOB_FORMAT
from .license_override import LicenseOutcome, resolve_license
from .schema import ChunkRow, DatasetManifestJob, ExampleRow, JobManifest
from .verify import verify_workspace_dataset
from .zipio import JobRejected, ZipLimits, job_content_hash, jdump, read_zip, verify_checksums

LICENSE_TEXT = "license/license_text.txt"
REQUIRED = ("manifest.json", "chunks.jsonl", "dataset_manifest.json", "dataset/train.jsonl", "dataset/validation.jsonl",
            "dataset/test.jsonl", "eval/heldout.jsonl")
ALLOWED = frozenset(REQUIRED) | {LICENSE_TEXT, "checksums.json"}
SPLITS = ("train", "validation", "test")
JOB_STATE = "job/job_state.json"


def _fmt(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in x['loc'])}: {x['msg']}" for x in e.errors()[:5])


def _rows(name: str, data: bytes, model: type[BaseModel], problems: list[str]) -> list[Any]:
    out: list[Any] = []
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as e:
        problems.append(f"{name}: not UTF-8 ({e})")
        return out
    bad = 0
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            out.append(model.model_validate_json(line))
        except ValidationError as e:
            problems.append(f"{name} line {n}: {_fmt(e)}")
        except ValueError as e:
            problems.append(f"{name} line {n}: {e}")
        else:
            continue
        bad += 1
        if bad >= 10:
            problems.append(f"{name}: too many invalid rows, stopped checking")
            break
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "untitled"


@dataclass
class ImportReport:
    job_id: str
    job_content_hash: str
    zip_sha256: str
    project_id: str
    workspace: str
    sources: int
    chunks: dict[str, int]
    examples: dict[str, int]
    heldout_items: int
    dataset_id: str
    dataset_hash: str
    license: dict[str, Any]
    method: dict[str, Any]
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reused: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _parse(files: dict[str, bytes]) -> tuple[JobManifest, list[ChunkRow], DatasetManifestJob, dict[str, list[ExampleRow]], list[dict]]:
    problems: list[str] = []
    unexpected = sorted(set(files) - ALLOWED)
    if unexpected:
        problems.append(f"unexpected members: {unexpected}")
    for r in REQUIRED:
        if r not in files:
            problems.append(f"missing required member {r}")
    if problems:
        raise JobRejected(problems)
    try:
        manifest = JobManifest.model_validate_json(files["manifest.json"])
    except ValidationError as e:
        errs = e.errors()
        for x in errs:
            if x["loc"] in (("format",), ("version",)):
                raise JobRejected(f"unsupported format/version in manifest.json: {_fmt(e)} (expected {JOB_FORMAT} v{FORMAT_VERSION})") from e
        raise JobRejected(f"manifest.json: {_fmt(e)}") from e
    except ValueError as e:
        raise JobRejected(f"manifest.json: {e}") from e
    chunks = _rows("chunks.jsonl", files["chunks.jsonl"], ChunkRow, problems)
    try:
        dsm = DatasetManifestJob.model_validate_json(files["dataset_manifest.json"])
    except (ValidationError, ValueError) as e:
        raise JobRejected(f"dataset_manifest.json: {_fmt(e) if isinstance(e, ValidationError) else e}") from e
    examples = {s: _rows(f"dataset/{s}.jsonl", files[f"dataset/{s}.jsonl"], ExampleRow, problems) for s in SPLITS}
    held: list[dict] = []
    for n, line in enumerate(files["eval/heldout.jsonl"].decode("utf-8", "replace").splitlines(), 1):
        if line.strip():
            try:
                held.append(json.loads(line))
            except ValueError as e:
                problems.append(f"eval/heldout.jsonl line {n}: {e}")
    if problems:
        raise JobRejected(problems)
    return manifest, chunks, dsm, examples, held


def _semantic_checks(m: JobManifest, chunks: list[ChunkRow], dsm: DatasetManifestJob, ex: dict[str, list[ExampleRow]]) -> None:
    p: list[str] = []
    src_ids = {s.id for s in m.sources}
    seen: set[tuple[str, str]] = set()
    for c in chunks:
        if c.source_id not in src_ids:
            p.append(f"chunk {c.id}: unknown source_id {c.source_id}")
        if (c.source_id, c.id) in seen:
            p.append(f"duplicate chunk {c.source_id}/{c.id}")
        seen.add((c.source_id, c.id))
        if hash_text(c.text) != c.sha256:
            p.append(f"chunk {c.source_id}/{c.id}: sha256 does not match text")
    try:
        SplitConfig.model_validate(dsm.split_config.model_dump())
    except ValidationError as e:
        p.append(f"dataset_manifest.split_config: {_fmt(e)}")
    ids: set[str] = set()
    for split, rows in ex.items():
        for r in rows:
            if r.example_id in ids:
                p.append(f"duplicate example_id {r.example_id}")
            ids.add(r.example_id)
            assigned = dsm.split_assignment.groups.get(r.group_id)
            if assigned is None:
                p.append(f"example {r.example_id}: group {r.group_id!r} missing from split_assignment")
            elif assigned != split:
                p.append(f"example {r.example_id}: in {split}.jsonl but its group {r.group_id!r} is assigned to {assigned}")
            if split == "test" and r.origin == "synthetic":
                p.append(f"example {r.example_id}: synthetic example in the test split")
    if dsm.counts is not None:
        for s in SPLITS:
            if dsm.counts.get(s) != len(ex[s]):
                p.append(f"dataset_manifest.counts[{s}]={dsm.counts.get(s)} but {len(ex[s])} rows")
    if p:
        raise JobRejected(p)


def _write_workspace(root: Path, m: JobManifest, chunks: list[ChunkRow], dsm: DatasetManifestJob,
                     ex: dict[str, list[ExampleRow]], held: list[dict], outcome: LicenseOutcome,
                     files: dict[str, bytes], jhash: str, zip_sha: str) -> tuple[pl.Workspace, DatasetManifest, SourceManifest]:
    cfg = SplitConfig.model_validate(dsm.split_config.model_dump())
    pr = m.project
    pl.init_project(root, pr.name, pr.domain, seed=dsm.seed, description=pr.purpose)
    project = SpecialistProject(
        project_id=pr.id, name=pr.name, domain=pr.domain, description=pr.purpose, created_on=pr.created_at[:10], seed=dsm.seed,
        intended_use=IntendedUse(commercial=pr.intended_use.commercial, redistribute_model=pr.intended_use.redistribute_model),
        split_config=cfg, base_model_candidates=[m.base_model.registry_id],
    ).seal()
    (root / "project.json").write_text(project.to_json(), encoding="utf-8")
    ws = pl.Workspace(root)

    by_source: dict[str, list[ChunkRow]] = {s.id: [] for s in m.sources}
    for c in chunks:
        by_source[c.source_id].append(c)
    records: list[SourceRecord] = []
    for s in m.sources:
        crecs, rows = [], []
        for c in by_source[s.id]:
            section = " > ".join(c.section_path) or "Document"
            gid = c.group_id or f"{s.id}#{_slug(section)}"
            crecs.append(ChunkRecord(chunk_id=c.id, section=section, group_id=gid, page_start=c.page, page_end=c.page,
                                     char_start=c.char_start, char_end=c.char_end, text_sha256=c.sha256))
            rows.append({"chunk_id": c.id, "text": c.text, "page_start": c.page, "page_end": c.page, "char_start": c.char_start,
                         "char_end": c.char_end, "section": section, "section_path": c.section_path, "role": c.role,
                         "excluded_reason": c.exclude_reason, "origin": c.origin, "group_id": gid})
        with (root / "corpus" / f"{s.id}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
        issues = [i.model_dump() for i in s.issues]
        (root / "corpus" / f"{s.id}.provenance.json").write_text(json.dumps({
            "extractor": f"{s.extractor}/{s.extractor_version}", "pages_total": None,
            "needs_ocr_pages": sorted({i.page for i in s.issues if i.code == "needs_ocr" and i.page is not None}),
            "cleaning": {"transformations": []}, "issues": issues, "imported_from_job": m.job_id,
        }, indent=2, sort_keys=True), encoding="utf-8")
        records.append(SourceRecord(
            source_id=s.id, title=s.title or s.file_name, source_version=s.source_version, origin=s.origin or "studio-app",
            media_type=s.mime, original_filename=s.file_name, size_bytes=s.size, sha256=s.sha256, ingested_on=s.ingested_at[:10],
            rights=s.rights, transformations=[Transformation(name=s.extractor, version=s.extractor_version, performed_on=s.ingested_at[:10])],
            chunks=crecs))
    sources = ws.save_sources(SourceManifest(project_id=pr.id, manifest_version=1, sources=records))

    builder = GeneratorInfo(**dsm.builder.model_dump())
    recs: list[ExampleRecord] = []
    for split in SPLITS:
        for r in ex[split]:
            recs.append(ExampleRecord(
                example_id=r.example_id, split=split, group_id=r.group_id, task=r.task,
                derived_from=[ChunkRef(source_id=d.source_id, chunk_id=d.chunk_id) for d in r.derived_from], generator=builder,
                text_sha256=hash_obj({"prompt": r.prompt, "response": r.response}), quality_score=round(r.quality, 4), origin=r.origin))
    groups_by_split = {s: {r.group_id for r in ex[s]} for s in SPLITS}
    overlap = sum(len(groups_by_split[a] & groups_by_split[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    split_of = {r.example_id: s for s in SPLITS for r in ex[s]}
    texts = {r.example_id: leak_text(r.prompt, r.response) for s in SPLITS for r in ex[s]}
    cross = sp.cross_split_pairs(sp.find_near_duplicates(texts, cfg.near_duplicate_threshold, cfg.shingle_size), split_of)
    bodies: dict[str, bytes] = {}
    for split in SPLITS:
        lines = []
        for r in sorted(ex[split], key=lambda r: r.example_id):
            lines.append(json.dumps({"example_id": r.example_id, "group_id": r.group_id, "prompt": r.prompt, "response": r.response,
                                     "origin": r.origin, "derived_from": [d.model_dump() for d in r.derived_from]},
                                    sort_keys=True, ensure_ascii=False) + "\n")
        bodies[split] = "".join(lines).encode("utf-8")
    ds_id = "ds-" + short(hash_obj([sources.content_hash, dsm.seed, cfg.model_dump(mode="json"), builder.model_dump(),
                                    {s: sha256_bytes(b) for s, b in bodies.items()}]))
    out_dir = root / "datasets" / ds_id
    out_dir.mkdir(parents=True)
    split_info = {}
    for split in SPLITS:
        (out_dir / f"{split}.jsonl").write_bytes(bodies[split])
        split_info[split] = SplitInfo(file=f"{ds_id}/{split}.jsonl", sha256=sha256_bytes(bodies[split]), n_examples=len(ex[split]),
                                      n_groups=len(groups_by_split[split]))
    reference = [{"source_id": c.source_id, "chunk_id": c.id, "section": " > ".join(c.section_path) or "Document", "page_start": c.page,
                  "text_sha256": c.sha256, "table": None} for c in sorted(chunks, key=lambda c: (c.source_id, c.id)) if c.role == "reference"]
    if reference:
        with (out_dir / "reference_chunks.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in reference:
                fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    blocked = sorted(s.id for s in m.sources if s.rights.permitted_training != "yes")
    dataset = DatasetManifest(
        dataset_id=ds_id, dataset_version=dsm.dataset_version, project_id=pr.id, source_manifest_hash=sources.content_hash,
        source_manifest_version=sources.manifest_version, seed=dsm.seed, split_config=cfg, builder=builder, examples=recs,
        splits=split_info,
        leakage_report=LeakageReport(
            group_overlap_pairs=overlap, near_duplicate_pairs_found=len(cross), dropped_example_ids=[],
            residual_cross_split_near_duplicates=len(cross), effective_group_by=dsm.split_assignment.group_by,
            notes=[f"imported from studio job {m.job_id} (phone dataset_id {dsm.dataset_id!r}); split assignment taken from the job, "
                   "leakage re-verified by the desktop (leak-free or rejected)"]),
        excluded_sources=blocked,
    ).seal()
    (out_dir / "dataset_manifest.json").write_text(dataset.to_json(), encoding="utf-8")

    (root / "evals").mkdir(exist_ok=True)
    (root / "evals" / "heldout.jsonl").write_bytes(files["eval/heldout.jsonl"])
    entry = outcome.entry
    (root / "registry" / f"{_slug(entry.entry_id)}.json").write_text(json.dumps(entry.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    job = root / "job"
    job.mkdir()
    (job / "job_manifest.json").write_bytes(files["manifest.json"])
    (job / "job_checksums.json").write_bytes(files["checksums.json"])
    if LICENSE_TEXT in files:
        (job / "license_text.txt").write_bytes(files[LICENSE_TEXT])
    (job / "license_attestation.json").write_bytes(jdump({
        "outcome": outcome.as_dict(), "evidence_as_received": m.license_evidence.model_dump() if m.license_evidence else None,
        "note": "owner attestation: the app does not interpret legal text; permissions are as attested by the owner"}))
    (job / "job_state.json").write_bytes(jdump({
        "format": "llmtrainer-job-state", "version": 1, "job_id": m.job_id, "job_content_hash": jhash, "zip_sha256": zip_sha,
        "project_id": pr.id, "imported_at": now_iso(), "base_model": m.base_model.model_dump(), "method": m.method.model_dump(),
        "license_entry_id": entry.entry_id, "license_entry_hash": outcome.entry_hash, "dataset_id": ds_id}))
    return ws, dataset, sources


def _report(ws: pl.Workspace, m: JobManifest, jhash: str, zip_sha: str, outcome: LicenseOutcome, dataset: DatasetManifest,
            chunks: list[ChunkRow], n_held: int, reused: bool) -> ImportReport:
    roles = {r: sum(1 for c in chunks if c.role == r) for r in ("train", "reference", "excluded")}
    warnings = []
    if outcome.state != "VERIFIED":
        warnings.append(f"base model license is {outcome.state}: training only as an explicit local experiment; never exportable")
    warnings += outcome.reasons
    if any(c.role == "reference" for c in chunks):
        warnings.append(f"{roles['reference']} reference chunk(s) are retrieval-only (never training data)")
    ex = {s: dataset.splits[s].n_examples for s in SPLITS}
    return ImportReport(
        job_id=m.job_id, job_content_hash=jhash, zip_sha256=zip_sha, project_id=m.project.id, workspace=str(ws.root),
        sources=len(m.sources), chunks=roles, examples=ex, heldout_items=n_held, dataset_id=dataset.dataset_id,
        dataset_hash=dataset.content_hash, license=outcome.as_dict(), method=m.method.model_dump(),
        checks=["zip hygiene (names, sizes, ratio, symlinks)", "checksums.json exact set and every sha256", "schema of every file",
                "chunk text hashes", "split assignment consistency", "group/chunk/near-duplicate/containment leakage re-run",
                "synthetic-in-test, role and rights gates", "held-out items derive only from test chunks and do not overlap training text"],
        warnings=warnings, reused=reused)


def read_job_state(ws_root: Path) -> dict | None:
    f = Path(ws_root) / JOB_STATE
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def import_job(zip_path: str | os.PathLike[str], workspace: str | os.PathLike[str], *, limits: ZipLimits = ZipLimits(),
               catalog_dir: Path | None = None, reuse_existing: bool = False) -> ImportReport:
    """Validate ``zip_path`` and create the workspace at ``workspace``. Raises JobRejected; writes nothing on rejection."""
    zip_path, root = Path(zip_path), Path(workspace)
    files = read_zip(zip_path, limits)
    problems, declared = verify_checksums(files)
    if problems:
        raise JobRejected(problems)
    manifest, chunks, dsm, examples, held = _parse(files)
    _semantic_checks(manifest, chunks, dsm, examples)
    jhash = job_content_hash(declared)
    zip_sha = sha256_bytes(zip_path.read_bytes())
    lic_text = files.get(LICENSE_TEXT)
    outcome = resolve_license(manifest.base_model, manifest.license_evidence, lic_text, catalog_dir)

    if root.exists() and any(root.iterdir()):
        state = read_job_state(root)
        if reuse_existing and state and state.get("job_content_hash") == jhash:
            ws = pl.Workspace(root)
            dataset = pl.latest_dataset(ws)
            probs = verify_workspace_dataset(ws, dataset)
            if probs:
                raise JobRejected(probs)
            return _report(ws, manifest, jhash, zip_sha, outcome, dataset, chunks, len(held), reused=True)
        raise JobRejected(f"workspace {root} already exists and is not empty; refusing to overwrite (use a new directory)")
    root.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{root.name}.importing-", dir=root.parent))
    try:
        ws, dataset, _sources = _write_workspace(tmp / "ws", manifest, chunks, dsm, examples, held, outcome, files, jhash, zip_sha)
        probs = verify_workspace_dataset(ws, dataset)
        if probs:
            raise JobRejected(probs)
        if root.exists():
            root.rmdir()
        os.rename(tmp / "ws", root)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    final = pl.Workspace(root)
    report = _report(final, manifest, jhash, zip_sha, outcome, dataset, chunks, len(held), reused=False)
    (root / "job" / "import_report.json").write_bytes(jdump(report.as_dict()))
    return report
