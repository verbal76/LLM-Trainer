"""Ingestion and section-aware chunking.

Two entry points:

* ``ingest_bytes``: the original plain-text/Markdown path (kept for compatibility).
* ``ingest_file``: the full path. Extractors (PDF via optional pypdf, DOCX, TXT, Markdown,
  CSV/TSV/JSON) produce page/section-addressed blocks, ``extract.clean`` normalises them
  (every transformation recorded), ``chunk_blocks`` cuts chunks with provenance. Per-file
  problems become ``IngestIssue`` records, never exceptions. No OCR is performed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .extract import Block, Extraction, IngestIssue, extract
from .extract.clean import clean
from .hashing import hash_text, sha256_bytes, short
from .schemas import ChunkRecord, RightsInfo, SourceRecord, Transformation

CHUNKER_NAME = "section_paragraph_chunker"
CHUNKER_VERSION = "1"
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass(frozen=True)
class Chunk:
    record: ChunkRecord
    text: str
    meta: dict[str, Any] = field(default_factory=dict)  # role, kind, section_path, byte offsets, table, ...


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "untitled"


def source_id_for(raw: bytes) -> str:
    return "src-" + short(sha256_bytes(raw))


def chunk_text(source_id: str, text: str, max_chars: int = 600, min_chars: int = 20) -> list[Chunk]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks: list[Chunk] = []
    section = "Document"
    pos = 0
    buf: list[tuple[int, int, str]] = []  # (start, end, paragraph)

    def page_at(offset: int) -> int:
        return text.count("\f", 0, offset) + 1

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        body = "\n\n".join(p for _, _, p in buf).strip()
        start, end = buf[0][0], buf[-1][1]
        buf = []
        if len(body) < min_chars:
            return
        cid = f"c{len(chunks) + 1:04d}"
        chunks.append(
            Chunk(
                ChunkRecord(
                    chunk_id=cid,
                    section=section,
                    group_id=f"{source_id}#{slugify(section)}",
                    page_start=page_at(start),
                    page_end=page_at(end),
                    char_start=start,
                    char_end=end,
                    text_sha256=hash_text(body),
                ),
                body,
            )
        )

    for block in re.split(r"(\n\s*\n)", text):
        start, end = pos, pos + len(block)
        pos = end
        stripped = block.strip().replace("\f", "")
        if not stripped:
            continue
        # a block may begin with a heading line followed by body lines
        lines = stripped.split("\n")
        m = _HEADING.match(lines[0])
        if m:
            flush()
            section = m.group(2)
            stripped = "\n".join(lines[1:]).strip()
            if not stripped:
                continue
        cur = sum(len(p) + 2 for _, _, p in buf)
        if buf and cur + len(stripped) > max_chars:
            flush()
        buf.append((start, end, stripped))
    flush()
    return chunks


def ingest_bytes(
    raw: bytes,
    *,
    filename: str,
    title: str,
    origin: str,
    rights: RightsInfo,
    ingested_on: str,
    media_type: str = "text/plain",
    source_version: str | None = None,
    source_date: str | None = None,
) -> tuple[SourceRecord, list[Chunk]]:
    text = raw.decode("utf-8")
    sid = source_id_for(raw)
    chunks = chunk_text(sid, text)
    record = SourceRecord(
        source_id=sid,
        title=title,
        source_version=source_version,
        source_date=source_date,
        origin=origin,
        media_type=media_type,
        original_filename=filename,
        size_bytes=len(raw),
        sha256=sha256_bytes(raw),
        ingested_on=ingested_on,
        rights=rights,
        transformations=[
            Transformation(
                name=CHUNKER_NAME,
                version=CHUNKER_VERSION,
                params={"max_chars": 600, "min_chars": 20, "newline_normalization": True},
                performed_on=ingested_on,
            )
        ],
        chunks=[c.record for c in chunks],
    )
    return record, chunks


# --------------------------------------------------------------------------- #
# Full ingestion: extract -> clean -> chunk
# --------------------------------------------------------------------------- #

ROLE_TRAIN = "train"  # prose usable as source for parameter-adaptation examples
ROLE_REFERENCE = "reference"  # tables / exact-fact records: retrieval only, never training prose
ROLE_EXCLUDED = "excluded"  # kept for provenance, used nowhere (duplicate, bad extraction)
MIN_LETTER_RATIO = 0.5
TABLE_ROWS_PER_CHUNK = 25
_SENT = re.compile(r"(?<=[.!?])\s+")


def _split_oversize(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    cur = ""
    for sent in _SENT.split(text):
        while len(sent) > max_chars:  # no sentence boundary: hard split on whitespace
            cut = sent.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if cur and len(cur) + 1 + len(sent) > max_chars:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return [p for p in parts if p]


def _letter_ratio(t: str) -> float:
    chars = [c for c in t if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


def chunk_blocks(source_id: str, blocks: list[Block], max_chars: int = 600, min_chars: int = 20) -> tuple[list[Chunk], dict[str, Any]]:
    """Cut cleaned blocks into chunks. Chunks never cross a section boundary; tables are their own chunks."""
    chunks: list[Chunk] = []
    stats = {"dropped_short": 0, "excluded_low_quality": 0, "excluded_duplicate": 0, "split_oversize_paragraphs": 0}
    seen_text: dict[str, str] = {}
    buf: list[Block] = []
    buf_section: tuple[str, ...] = ()

    def emit(text: str, bl: list[Block], role: str, kind: str, extra: dict[str, Any] | None = None) -> None:
        text = text.strip()
        if len(text) < min_chars:
            stats["dropped_short"] += 1
            return
        section_path = bl[0].section_path
        section = " > ".join(section_path) or "Document"
        pages = [b.page for b in bl if b.page is not None]
        bs = [b.byte_start for b in bl]
        be = [b.byte_end for b in bl]
        cid = f"c{len(chunks) + 1:04d}"
        meta: dict[str, Any] = {
            "role": role, "kind": kind, "section_path": list(section_path),
            "byte_start": bs[0] if bs and None not in bs else None, "byte_end": be[-1] if be and None not in be else None,
            "transforms": sorted({t for b in bl for t in b.transforms}),
        }
        digest = hash_text(re.sub(r"\s+", " ", text).lower())
        if role == ROLE_TRAIN:
            if _letter_ratio(text) < MIN_LETTER_RATIO:
                meta.update(role=ROLE_EXCLUDED, excluded_reason="low_extraction_quality")
                stats["excluded_low_quality"] += 1
            elif digest in seen_text:
                meta.update(role=ROLE_EXCLUDED, excluded_reason="duplicate_chunk", duplicate_of=seen_text[digest])
                stats["excluded_duplicate"] += 1
            else:
                seen_text[digest] = cid
        if extra:
            meta.update(extra)
        chunks.append(Chunk(ChunkRecord(
            chunk_id=cid, section=section, group_id=f"{source_id}#{slugify(section)}",
            page_start=min(pages) if pages else None, page_end=max(pages) if pages else None,
            char_start=bl[0].start, char_end=bl[-1].end, text_sha256=hash_text(text)), text, meta))

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        paras, buf = buf, []
        cur: list[Block] = []
        cur_len = 0
        for b in paras:
            pieces = _split_oversize(b.text, max_chars) if len(b.text) > max_chars else [b.text]
            if len(pieces) > 1:
                stats["split_oversize_paragraphs"] += 1
                if cur:
                    emit("\n\n".join(x.text for x in cur), cur, ROLE_TRAIN, "prose")
                    cur, cur_len = [], 0
                for pc in pieces:
                    emit(pc, [b], ROLE_TRAIN, "prose", {"split_from_block": True})
                continue
            if cur and cur_len + len(b.text) + 2 > max_chars:
                emit("\n\n".join(x.text for x in cur), cur, ROLE_TRAIN, "prose")
                cur, cur_len = [], 0
            cur.append(b)
            cur_len += len(b.text) + 2
        if cur:
            emit("\n\n".join(x.text for x in cur), cur, ROLE_TRAIN, "prose")

    for b in blocks:
        if b.kind == "table":
            flush()
            rows = b.rows or []
            header = b.header or []
            for i in range(0, max(len(rows), 1), TABLE_ROWS_PER_CHUNK):
                part = rows[i : i + TABLE_ROWS_PER_CHUNK]
                text = "\n".join(" | ".join(r) for r in ([header] if header else []) + part)
                emit(text, [b], ROLE_REFERENCE, "table",
                     {"table": {"header": header, "rows": part, "row_offset": i + b.meta.get("row_start", 1) - 1}})
            continue
        if b.kind == "heading":
            flush()
            buf_section = b.section_path
            continue
        if buf and b.section_path != buf_section:
            flush()
        buf_section = b.section_path
        buf.append(b)
    flush()
    return chunks, stats


@dataclass
class IngestOutcome:
    file: str
    record: SourceRecord | None = None
    chunks: list[Chunk] = field(default_factory=list)
    issues: list[IngestIssue] = field(default_factory=list)
    needs_ocr_pages: list[int] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    stream: str | None = None  # extraction stream to store when it is not the original text
    extracted_blocks: list[dict[str, Any]] = field(default_factory=list)  # pre-cleaning audit trail

    @property
    def ok(self) -> bool:
        return self.record is not None


def ingest_file(
    raw: bytes,
    *,
    filename: str,
    origin: str,
    rights: RightsInfo,
    ingested_on: str,
    title: str | None = None,
    source_version: str | None = None,
    source_date: str | None = None,
    max_chars: int = 600,
    min_chars: int = 20,
) -> IngestOutcome:
    out = IngestOutcome(file=filename)
    ex: Extraction = extract(filename, raw)
    out.issues.extend(ex.issues)
    out.needs_ocr_pages = list(ex.needs_ocr_pages)
    if not ex.blocks:
        if not out.issues:
            out.issues.append(IngestIssue("empty_content", filename, "no extractable content"))
        return out
    sid = source_id_for(raw)
    out.extracted_blocks = [
        {"kind": b.kind, "page": b.page, "section_path": list(b.section_path), "char_start": b.start, "char_end": b.end, "text": b.text[:2000]}
        for b in ex.blocks
    ]
    cleaned, rep = clean(ex)
    chunks, cstats = chunk_blocks(sid, cleaned, max_chars, min_chars)
    if not chunks:
        out.issues.append(IngestIssue("no_usable_chunks", filename, f"content too short or unusable after cleaning ({cstats})"))
        return out
    for c in chunks:
        c.meta.update(extractor=ex.extractor, extractor_version=ex.version)
    stream_sha = hash_text(ex.stream)
    transforms = [
        Transformation(name=f"extract:{ex.extractor}", version=ex.version, performed_on=ingested_on, output_sha256=stream_sha,
                       params={"media_type": ex.media_type, "pages_total": ex.pages_total, "needs_ocr_pages": ex.needs_ocr_pages,
                               "stream_is_original_text": ex.stream_is_original, "bom_bytes": ex.bom_bytes}),
        *[Transformation(name=f"clean:{t['name']}", version=t["version"], performed_on=ingested_on,
                         params={k: v for k, v in t.items() if k not in ("name", "version")}) for t in rep.transformations],
        Transformation(name=CHUNKER_NAME, version=CHUNKER_VERSION, performed_on=ingested_on,
                       params={"max_chars": max_chars, "min_chars": min_chars, **cstats}),
    ]
    out.record = SourceRecord(
        source_id=sid, title=title or filename.rsplit(".", 1)[0], source_version=source_version, source_date=source_date,
        origin=origin, media_type=ex.media_type, original_filename=filename, size_bytes=len(raw), sha256=sha256_bytes(raw),
        ingested_on=ingested_on, rights=rights, transformations=transforms, chunks=[c.record for c in chunks],
    )
    out.chunks = chunks
    out.stream = None if ex.stream_is_original else ex.stream
    out.provenance = {
        "source_id": sid, "original_filename": filename, "sha256": sha256_bytes(raw), "size_bytes": len(raw),
        "extractor": {"name": ex.extractor, "version": ex.version}, "pages_total": ex.pages_total,
        "needs_ocr_pages": ex.needs_ocr_pages, "stream_sha256": stream_sha,
        "stream_file": None if ex.stream_is_original else f"{sid}.extracted.txt",
        "offset_note": ("char offsets index the decoded original file" if ex.stream_is_original
                        else "char offsets index the stored extraction stream (<id>.extracted.txt), not the original binary"),
        "issues": [i.as_dict() for i in out.issues],
        "cleaning": {"transformations": rep.transformations, "removed_noise": rep.removed_noise},
        "chunking": {"max_chars": max_chars, "min_chars": min_chars, **cstats},
    }
    return out
