"""TXT and Markdown extraction (exact char and byte offsets, heading hierarchy)."""

from __future__ import annotations

import re

from .models import Block, Extraction, IngestIssue, Line

VERSION = "1"
_ATX = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^ {0,3}(```|~~~)")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)\s*$")
_PIPE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def decode_text(raw: bytes, name: str) -> tuple[str | None, int, IngestIssue | None]:
    bom = 3 if raw.startswith(b"\xef\xbb\xbf") else 0
    try:
        return raw[bom:].decode("utf-8"), bom, None
    except UnicodeDecodeError as e:
        return None, bom, IngestIssue(
            "undecodable_text", name,
            f"not valid UTF-8 ({e.reason} at byte {e.start}); re-encode the file, the extractor does not guess encodings",
        )


def _scan_lines(text: str, bom: int) -> list[tuple[str, int, int, int, int]]:
    """(line_without_eol, char_start, char_end, byte_start, byte_end)."""
    out = []
    pos = bpos = 0
    for raw_line in text.splitlines(keepends=True):
        body = raw_line.rstrip("\r\n")
        bstart = bpos + bom
        bpos += len(raw_line.encode("utf-8"))
        out.append((body, pos, pos + len(body), bstart, bstart + len(body.encode("utf-8"))))
        pos += len(raw_line)
    return out


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def extract_text(raw: bytes, name: str, markdown: bool) -> Extraction:
    text, bom, issue = decode_text(raw, name)
    ex = Extraction(
        "markdown" if markdown else "plain_text", VERSION, "text/markdown" if markdown else "text/plain",
        text or "", True, line_based=True, detect_headings=not markdown, bom_bytes=bom,
    )
    if issue:
        ex.issues.append(issue)
        return ex
    lines = _scan_lines(text, bom)
    path: list[tuple[int, str]] = []  # (level, title)
    buf: list[tuple[str, int, int, int, int]] = []
    in_fence = False

    def spath() -> tuple[str, ...]:
        return tuple(t for _, t in path)

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        ls = [Line(b[0], b[1], b[2]) for b in buf]
        ex.blocks.append(Block("paragraph", "\n".join(x.text for x in ls), buf[0][1], buf[-1][2], section_path=spath(),
                               lines=ls, byte_start=buf[0][3], byte_end=buf[-1][4]))
        buf = []

    def push_heading(level: int, title: str, ln) -> None:
        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, title))
        ex.blocks.append(Block("heading", title, ln[1], ln[2], level=level, section_path=spath(), byte_start=ln[3], byte_end=ln[4]))

    i = 0
    while i < len(lines):
        ln = lines[i]
        body = ln[0]
        if markdown and _FENCE.match(body):
            in_fence = not in_fence
            buf.append(ln)
            i += 1
            continue
        if in_fence:
            buf.append(ln)
            i += 1
            continue
        if not body.strip():
            flush()
            i += 1
            continue
        if markdown:
            m = _ATX.match(body)
            if m and m.group(2):
                flush()
                push_heading(len(m.group(1)), m.group(2), ln)
                i += 1
                continue
            if (i + 1 < len(lines) and not buf and _SETEXT.match(lines[i + 1][0])
                    and not body.lstrip().startswith(("-", "*", ">", "|"))):
                nxt = lines[i + 1]
                push_heading(1 if nxt[0].strip().startswith("=") else 2, body.strip(), (body, ln[1], nxt[2], ln[3], nxt[4]))
                i += 2
                continue
            if "|" in body and i + 1 < len(lines) and _PIPE_SEP.match(lines[i + 1][0]) and not buf:
                header = _split_row(body)
                j = i + 2
                rows = []
                while j < len(lines) and "|" in lines[j][0] and lines[j][0].strip():
                    rows.append(_split_row(lines[j][0]))
                    j += 1
                last = lines[j - 1]
                ex.blocks.append(Block("table", "\n".join(" | ".join(r) for r in [header, *rows]), ln[1], last[2], section_path=spath(),
                                       byte_start=ln[3], byte_end=last[4], header=header, rows=rows))
                i = j
                continue
        buf.append(ln)
        i += 1
    flush()
    if not ex.blocks:
        ex.issues.append(IngestIssue("empty_content", name, "no text content"))
    return ex
