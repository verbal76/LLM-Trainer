"""Extractors: raw bytes -> page/section-addressed blocks with provenance. Never raises per-file."""

from __future__ import annotations

from .models import Block, Extraction, IngestIssue, Line

SUPPORTED = {
    ".txt": "text", ".text": "text", ".md": "markdown", ".markdown": "markdown", ".docx": "docx", ".pdf": "pdf",
    ".csv": "csv", ".tsv": "tsv", ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl",
}


def kind_for(name: str, raw: bytes) -> str | None:
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    if raw[:5] == b"%PDF-":
        return "pdf"
    return SUPPORTED.get(ext)


def extract(name: str, raw: bytes) -> Extraction:
    """Dispatch on file type. Any extractor failure becomes an IngestIssue, never an exception."""
    from . import docx, pdf, structured, text

    kind = kind_for(name, raw)
    if kind is None:
        ex = Extraction("none", "0", "application/octet-stream", "", False)
        ex.issues.append(IngestIssue("unsupported_format", name, f"unsupported file type (supported: {', '.join(sorted(SUPPORTED))})"))
        return ex
    if not raw.strip():
        ex = Extraction(kind, "0", "", "", False)
        ex.issues.append(IngestIssue("empty_file", name, "file is empty"))
        return ex
    try:
        if kind == "text":
            ex = text.extract_text(raw, name, markdown=False)
        elif kind == "markdown":
            ex = text.extract_text(raw, name, markdown=True)
        elif kind == "docx":
            ex = docx.extract_docx(raw, name)
        elif kind == "pdf":
            ex = pdf.extract_pdf(raw, name)
        elif kind in ("csv", "tsv"):
            ex = structured.extract_delimited(raw, name, "," if kind == "csv" else "\t")
        else:
            ex = structured.extract_json(raw, name, lines_format=(kind == "jsonl"))
    except Exception as e:  # defensive: one bad file must not kill the batch
        ex = Extraction(kind, "0", "", "", False)
        ex.issues.append(IngestIssue("extractor_error", name, f"{type(e).__name__}: {e}"))
    return ex


__all__ = ["Block", "Extraction", "IngestIssue", "Line", "extract", "kind_for", "SUPPORTED"]
