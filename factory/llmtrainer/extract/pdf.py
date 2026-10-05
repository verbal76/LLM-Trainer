"""PDF text extraction via the OPTIONAL BSD-licensed ``pypdf``. No OCR is ever performed.

Pages with no extractable text layer are reported as ``needs_ocr`` and left out; the user decides
whether to run an OCR tool and re-ingest its output. (PyMuPDF/fitz is AGPL and deliberately not used.)
"""

from __future__ import annotations

import io

from .models import Block, Extraction, IngestIssue, Line

VERSION = "1"
MIN_CHARS_FOR_TEXT_PAGE = 5


def pypdf_available() -> bool:
    try:
        import pypdf  # noqa: F401
        return True
    except ImportError:
        return False


def extract_pdf(raw: bytes, name: str) -> Extraction:
    ex = Extraction("pypdf", VERSION, "application/pdf", "", False, line_based=True, detect_headings=True)
    try:
        import pypdf
    except ImportError:
        ex.issues.append(IngestIssue("missing_dependency", name, "PDF support needs the optional dependency 'pypdf' (BSD): pip install pypdf"))
        return ex
    ex.version = f"{VERSION}+pypdf-{pypdf.__version__}"
    try:
        reader = pypdf.PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:  # missing crypto backend, unsupported algorithm
                ok = 0
            if not ok:
                ex.issues.append(IngestIssue("encrypted", name, "PDF is password-protected (or needs the 'cryptography' package); decrypt it first"))
                return ex
        n = len(reader.pages)
    except Exception as e:
        ex.issues.append(IngestIssue("corrupt_file", name, f"cannot read PDF: {type(e).__name__}: {e}"))
        return ex
    ex.pages_total = n
    parts: list[str] = []
    pos = 0
    for pno in range(1, n + 1):
        try:
            text = reader.pages[pno - 1].extract_text() or ""
        except Exception as e:
            ex.issues.append(IngestIssue("page_extract_failed", name, f"{type(e).__name__}: {e}", "warning", page=pno))
            text = ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if len("".join(text.split())) < MIN_CHARS_FOR_TEXT_PAGE:
            ex.needs_ocr_pages.append(pno)
            ex.issues.append(IngestIssue("needs_ocr", name, "page has no extractable text layer (scanned/image-only?); NOT OCRed", "warning", page=pno))
            parts.append("\f")
            pos += 1
            continue
        buf: list[Line] = []

        def flush() -> None:
            if buf:
                ex.blocks.append(Block("paragraph", "\n".join(x.text for x in buf), buf[0].start, buf[-1].end, page=pno, lines=list(buf)))
                buf.clear()

        for ln in text.split("\n"):
            start = pos
            pos += len(ln) + 1
            if ln.strip():
                buf.append(Line(ln, start, start + len(ln)))
            else:
                flush()
        flush()
        parts.append(text + "\n\f")
        pos += 1
    ex.stream = "".join(parts)
    if not ex.blocks:
        ex.issues.append(IngestIssue(
            "needs_ocr", name,
            f"none of the {n} page(s) has extractable text; the file looks scanned. Not OCRed: run an OCR tool and ingest its text output",
        ))
    return ex
