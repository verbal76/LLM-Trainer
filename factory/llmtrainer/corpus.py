"""Workspace-level corpus operations: batch ingest, corpus reading, near-duplicate and terminology inspection.

Layout additions (all derived data lives under corpus/, originals under sources/raw/):

    sources/raw/<id>/<file>        original bytes, byte-identical, hash recorded in the manifest
    corpus/<id>.jsonl              chunk rows: text + role + provenance (page, section path, offsets, ...)
    corpus/<id>.provenance.json    extractor, cleaning transformations, issues, OCR-needed pages
    corpus/<id>.extracted.txt      extraction stream for PDF/DOCX (chunk char offsets index this)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import splits as sp
from .clock import today
from .extract import SUPPORTED, IngestIssue
from .hashing import hash_file, sha256_bytes
from .ingest import ROLE_REFERENCE, ROLE_TRAIN, ingest_file, source_id_for
from .pipeline import Workspace, WorkspaceError
from .schemas import RightsInfo


@dataclass
class BatchReport:
    ingested: list[dict[str, Any]] = field(default_factory=list)
    issues: list[IngestIssue] = field(default_factory=list)
    duplicates: list[dict[str, str]] = field(default_factory=list)
    needs_ocr: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"ingested": self.ingested, "issues": [i.as_dict() for i in self.issues], "duplicates": self.duplicates,
                "needs_ocr": self.needs_ocr}


def expand_paths(paths: list[Path]) -> tuple[list[Path], list[IngestIssue]]:
    files: list[Path] = []
    issues: list[IngestIssue] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.is_file() and not any(part.startswith(".") for part in f.relative_to(p).parts)))
        elif p.is_file():
            files.append(p)
        else:
            issues.append(IngestIssue("not_found", str(p), "path does not exist"))
    return files, issues


def ingest_paths(ws: Workspace, paths: list[Path], *, origin: str, rights: RightsInfo, title: str | None = None,
                 source_version: str | None = None) -> BatchReport:
    """Ingest many files. A bad file yields an IngestIssue; the batch always completes."""
    rep = BatchReport()
    files, issues = expand_paths(list(paths))
    rep.issues.extend(issues)
    manifest = ws.sources()
    known = {s.source_id: s for s in manifest.sources}
    new_records = []
    seen_batch: dict[str, str] = {}
    for f in files:
        name = str(f)
        try:
            raw = f.read_bytes()
        except OSError as e:
            rep.issues.append(IngestIssue("unreadable", name, str(e)))
            continue
        sid = source_id_for(raw)
        if sid in seen_batch or sid in known:
            prev = seen_batch.get(sid) or known[sid].original_filename
            if sid in known and known[sid].status == "removed":
                rep.issues.append(IngestIssue("previously_removed", name, f"{sid} was removed earlier; re-adding needs an explicit decision"))
            else:
                rep.duplicates.append({"file": name, "duplicate_of": prev, "source_id": sid})
            continue
        out = ingest_file(raw, filename=f.name, origin=origin, rights=rights, ingested_on=today(),
                          title=title if len(files) == 1 else None, source_version=source_version)
        for i in out.issues:
            rep.issues.append(IngestIssue(i.code, name, i.detail, i.severity, i.page))
        if out.needs_ocr_pages:
            rep.needs_ocr.append({"file": name, "pages": out.needs_ocr_pages})
        if not out.ok:
            continue
        seen_batch[sid] = name
        raw_dir = ws.root / "sources" / "raw" / sid
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f.name).write_bytes(raw)
        if hash_file(raw_dir / f.name) != sha256_bytes(raw):
            raise WorkspaceError(f"raw copy hash mismatch for {name}")
        corpus = ws.root / "corpus"
        with (corpus / f"{sid}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for c in out.chunks:
                r = c.record
                row = {"chunk_id": r.chunk_id, "text": c.text, "page_start": r.page_start, "page_end": r.page_end,
                       "char_start": r.char_start, "char_end": r.char_end, "section": r.section, **c.meta}
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        (corpus / f"{sid}.provenance.json").write_text(json.dumps(out.provenance, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        if out.stream is not None:
            (corpus / f"{sid}.extracted.txt").write_text(out.stream, encoding="utf-8", newline="")
        new_records.append(out.record)
        rep.ingested.append({"source_id": sid, "file": name, "chunks": len(out.chunks),
                             "roles": {r: sum(1 for c in out.chunks if c.meta["role"] == r) for r in ("train", "reference", "excluded")}})
    if new_records:
        ws.save_sources(manifest.model_copy(update={"sources": [*manifest.sources, *new_records],
                                                    "manifest_version": manifest.manifest_version + 1}))
    return rep


def load_chunk_rows(ws: Workspace) -> list[dict[str, Any]]:
    """All chunk rows of active sources, with source_id attached, in deterministic order."""
    rows = []
    for s in ws.sources().active_sources():
        f = ws.root / "corpus" / f"{s.source_id}.jsonl"
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                r["source_id"] = s.source_id
                r.setdefault("role", ROLE_TRAIN)
                rows.append(r)
    return rows


def corpus_near_duplicates(rows: list[dict[str, Any]], threshold: float = 0.8) -> list[dict[str, Any]]:
    texts = {f"{r['source_id']}/{r['chunk_id']}": r["text"] for r in rows if r["role"] == ROLE_TRAIN}
    return [{"a": a, "b": b, "jaccard": j} for a, b, j in sp.find_near_duplicates(texts, threshold)]


def inspect_corpus(ws: Workspace, *, top_terms: int = 20, threshold: float = 0.8) -> dict[str, Any]:
    from .terms import extract_terms

    rows = load_chunk_rows(ws)
    per_source = []
    for s in ws.sources().active_sources():
        prov_f = ws.root / "corpus" / f"{s.source_id}.provenance.json"
        prov = json.loads(prov_f.read_text(encoding="utf-8")) if prov_f.is_file() else {}
        mine = [r for r in rows if r["source_id"] == s.source_id]
        per_source.append({
            "source_id": s.source_id, "file": s.original_filename, "rights": s.rights.status,
            "extractor": prov.get("extractor"), "pages_total": prov.get("pages_total"), "needs_ocr_pages": prov.get("needs_ocr_pages", []),
            "chunks": {r: sum(1 for x in mine if x["role"] == r) for r in ("train", "reference", "excluded")},
            "cleaning": [{"name": t["name"], "count": t.get("count")} for t in prov.get("cleaning", {}).get("transformations", [])],
            "issues": prov.get("issues", []),
        })
    train_rows = [r for r in rows if r["role"] == ROLE_TRAIN]
    return {
        "sources": per_source,
        "totals": {r: sum(1 for x in rows if x["role"] == r) for r in ("train", "reference", "excluded")},
        "near_duplicate_chunks": corpus_near_duplicates(rows, threshold),
        "reference_tables_flagged_for_retrieval": [f"{r['source_id']}/{r['chunk_id']}" for r in rows if r["role"] == ROLE_REFERENCE],
        "top_terms": extract_terms(train_rows, top_k=top_terms),
    }


__all__ = ["BatchReport", "ingest_paths", "inspect_corpus", "load_chunk_rows", "SUPPORTED", "ROLE_REFERENCE"]
