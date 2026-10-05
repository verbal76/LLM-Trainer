"""Plain-text / Markdown ingestion and section-aware chunking.

PDF/OCR/HTML extraction is intentionally out of scope for this foundation
round; extractors must emit text plus page markers (form feed, ``\\f``) and
record themselves as a Transformation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .hashing import hash_text, sha256_bytes, short
from .schemas import ChunkRecord, RightsInfo, SourceRecord, Transformation

CHUNKER_NAME = "section_paragraph_chunker"
CHUNKER_VERSION = "1"
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass(frozen=True)
class Chunk:
    record: ChunkRecord
    text: str


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
