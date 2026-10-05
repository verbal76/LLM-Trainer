"""DOCX extraction with the standard library only (zipfile + xml). Headings, paragraphs, tables as rows."""

from __future__ import annotations

import io
import re
import zipfile
import xml.etree.ElementTree as ET

from .models import Block, Extraction, IngestIssue

VERSION = "1"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MAX_XML = 64 * 1024 * 1024  # zip-bomb guard


def _read(z: zipfile.ZipFile, name: str) -> bytes | None:
    try:
        info = z.getinfo(name)
    except KeyError:
        return None
    if info.file_size > MAX_XML:
        raise ValueError(f"{name} too large ({info.file_size} bytes)")
    return z.read(name)


def _para_text(p: ET.Element) -> str:
    out = []
    for el in p.iter():
        if el.tag == W + "t":
            out.append(el.text or "")
        elif el.tag == W + "tab":
            out.append("\t")
        elif el.tag in (W + "br", W + "cr"):
            out.append("\n")
    return "".join(out)


def _style_names(z: zipfile.ZipFile) -> dict[str, str]:
    data = _read(z, "word/styles.xml")
    names: dict[str, str] = {}
    if data:
        for st in ET.fromstring(data).iter(W + "style"):
            nm = st.find(W + "name")
            if nm is not None:
                names[st.get(W + "styleId", "")] = nm.get(W + "val", "")
    return names


def _heading_level(style_id: str, names: dict[str, str]) -> int:
    for cand in (names.get(style_id, ""), style_id):
        m = re.fullmatch(r"(?i)heading\s*([1-9])", cand.strip())
        if m:
            return int(m.group(1))
        if cand.strip().lower() == "title":
            return 1
    return 0


def extract_docx(raw: bytes, name: str) -> Extraction:
    ex = Extraction("docx_stdlib", VERSION, "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "", False)
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        if raw.startswith(b"\xd0\xcf\x11\xe0"):
            ex.issues.append(IngestIssue("encrypted_or_legacy", name, "OLE container: an encrypted .docx or a legacy .doc, not an OOXML zip"))
        else:
            ex.issues.append(IngestIssue("corrupt_file", name, "not a valid zip/OOXML container"))
        return ex
    try:
        body = _read(z, "word/document.xml")
        if body is None:
            ex.issues.append(IngestIssue("corrupt_file", name, "word/document.xml missing"))
            return ex
        names = _style_names(z)
        root = ET.fromstring(body).find(W + "body")
    except (ET.ParseError, ValueError, zipfile.BadZipFile) as e:
        ex.issues.append(IngestIssue("corrupt_file", name, f"cannot parse document XML: {e}"))
        return ex
    if root is None:
        ex.issues.append(IngestIssue("corrupt_file", name, "no w:body"))
        return ex

    parts: list[str] = []
    pos = 0
    path: list[tuple[int, str]] = []

    def emit(text: str) -> tuple[int, int]:
        nonlocal pos
        start = pos
        parts.append(text + "\n")
        pos += len(text) + 1
        return start, start + len(text)

    def spath() -> tuple[str, ...]:
        return tuple(t for _, t in path)

    for child in root:
        if child.tag == W + "p":
            text = _para_text(child).strip()
            if not text:
                continue
            ppr = child.find(W + "pPr")
            sid = ""
            if ppr is not None and ppr.find(W + "pStyle") is not None:
                sid = ppr.find(W + "pStyle").get(W + "val", "")
            lvl = _heading_level(sid, names)
            s, e = emit(text)
            if lvl:
                while path and path[-1][0] >= lvl:
                    path.pop()
                path.append((lvl, text))
                ex.blocks.append(Block("heading", text, s, e, level=lvl, section_path=spath()))
            else:
                ex.blocks.append(Block("paragraph", text, s, e, section_path=spath()))
        elif child.tag == W + "tbl":
            rows = []
            for tr in child.iter(W + "tr"):
                cells = []
                for tc in tr.findall(W + "tc"):
                    cells.append(" ".join(t for t in (_para_text(p).strip() for p in tc.iter(W + "p")) if t))
                if any(cells):
                    rows.append(cells)
            if not rows:
                continue
            s0 = pos
            for r in rows:
                emit(" | ".join(r))
            ex.blocks.append(Block("table", "\n".join(" | ".join(r) for r in rows), s0, pos - 1, section_path=spath(),
                                   header=rows[0], rows=rows[1:]))
    ex.stream = "".join(parts)
    if not ex.blocks:
        ex.issues.append(IngestIssue("empty_content", name, "document contains no text"))
    return ex
