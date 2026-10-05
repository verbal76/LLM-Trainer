"""Shared extraction data model: blocks with provenance, issues, results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IngestIssue:
    """A structured, non-fatal problem found while ingesting one file. Never raised; collected."""

    code: str
    file: str
    detail: str
    severity: str = "error"  # error: file not ingested; warning: ingested with caveat
    page: int | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "file": self.file, "detail": self.detail, "severity": self.severity}
        if self.page is not None:
            d["page"] = self.page
        return d


@dataclass(frozen=True)
class Line:
    text: str
    start: int  # char offset in the extraction stream
    end: int


@dataclass
class Block:
    """One addressable unit of extracted content.

    ``start``/``end`` are character offsets into the extractor's *stream* (the decoded
    file for text formats, the stored ``<id>.extracted.txt`` for PDF/DOCX).
    ``byte_start``/``byte_end`` are exact byte offsets in the original file when known.
    Line-based blocks (text/PDF) carry their ``lines`` so cleaning can edit at line level
    without losing offsets.
    """

    kind: str  # heading | paragraph | table
    text: str
    start: int
    end: int
    page: int | None = None
    level: int = 0
    section_path: tuple[str, ...] = ()
    lines: list[Line] | None = None
    byte_start: int | None = None
    byte_end: int | None = None
    header: list[str] | None = None
    rows: list[list[str]] | None = None
    transforms: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Extraction:
    extractor: str
    version: str
    media_type: str
    stream: str  # text the offsets refer to
    stream_is_original: bool  # True: stream == decoded original file (no separate copy needed)
    blocks: list[Block] = field(default_factory=list)
    issues: list[IngestIssue] = field(default_factory=list)
    pages_total: int | None = None
    needs_ocr_pages: list[int] = field(default_factory=list)
    line_based: bool = False  # cleaning may reflow/heading-detect at line level
    detect_headings: bool = False  # heuristic heading detection (no structural headings available)
    bom_bytes: int = 0
