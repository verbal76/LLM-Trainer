"""CSV / TSV / JSON / JSONL record extraction. Records are tables: reference data, not training prose."""

from __future__ import annotations

import csv
import io
import json

from .models import Block, Extraction, IngestIssue
from .text import decode_text

VERSION = "1"
ROWS_PER_TABLE = 25


def _tables(ex: Extraction, header: list[str], rows: list[list[str]], spans: list[tuple[int, int]], title: str) -> None:
    for i in range(0, len(rows), ROWS_PER_TABLE):
        part = rows[i : i + ROWS_PER_TABLE]
        sp = spans[i : i + ROWS_PER_TABLE]
        ex.blocks.append(
            Block("table", "\n".join(" | ".join(r) for r in [header, *part]), sp[0][0], sp[-1][1], section_path=(title,),
                  header=header, rows=part, meta={"row_start": i + 1, "row_end": i + len(part)})
        )


def extract_delimited(raw: bytes, name: str, delimiter: str) -> Extraction:
    text, bom, issue = decode_text(raw, name)
    tsv = delimiter == "\t"
    ex = Extraction("tsv" if tsv else "csv", VERSION, "text/tab-separated-values" if tsv else "text/csv", text or "", True, bom_bytes=bom)
    if issue:
        ex.issues.append(issue)
        return ex
    starts = [0]
    for ln in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(ln))
    rd = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    recs: list[list[str]] = []
    spans: list[tuple[int, int]] = []
    prev_line = 0
    try:
        for row in rd:
            end_line = rd.line_num
            if any(c.strip() for c in row):
                recs.append([c.strip() for c in row])
                spans.append((starts[prev_line], starts[min(end_line, len(starts) - 1)]))
            prev_line = end_line
    except csv.Error as e:
        ex.issues.append(IngestIssue("corrupt_structured", name, f"CSV/TSV parse error: {e}"))
        return ex
    if len(recs) < 2:
        ex.issues.append(IngestIssue("empty_content", name, "no data rows (header only or empty)"))
        return ex
    header, rows = recs[0], recs[1:]
    bad = sum(1 for r in rows if len(r) != len(header))
    if bad:
        ex.issues.append(IngestIssue("ragged_rows", name, f"{bad} row(s) have a different column count than the header", "warning"))
    _tables(ex, header, rows, spans[1:], name)
    return ex


def _flat(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)


def extract_json(raw: bytes, name: str, lines_format: bool) -> Extraction:
    text, bom, issue = decode_text(raw, name)
    ex = Extraction("jsonl" if lines_format else "json", VERSION,
                    "application/x-ndjson" if lines_format else "application/json", text or "", True, bom_bytes=bom)
    if issue:
        ex.issues.append(issue)
        return ex
    records: list = []
    spans: list[tuple[int, int]] = []
    coarse = False
    try:
        if lines_format:
            pos = 0
            for ln in text.splitlines(keepends=True):
                if ln.strip():
                    records.append(json.loads(ln))
                    spans.append((pos, pos + len(ln.rstrip("\r\n"))))
                pos += len(ln)
        else:
            data = json.loads(text)
            if isinstance(data, dict):
                lists = [v for v in data.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
                data = lists[0] if len(lists) == 1 else [data]
            if not isinstance(data, list):
                raise ValueError("top level must be an object or an array of objects")
            records = data
            spans = [(0, len(text))] * len(records)  # records are not located individually
            coarse = True
    except ValueError as e:  # includes json.JSONDecodeError
        ex.issues.append(IngestIssue("corrupt_structured", name, f"JSON parse error: {e}"))
        return ex
    objs = [(r, s) for r, s in zip(records, spans) if isinstance(r, dict) and r]
    if not objs:
        ex.issues.append(IngestIssue("empty_content", name, "no non-empty JSON object records"))
        return ex
    if coarse:
        ex.issues.append(IngestIssue("coarse_offsets", name, "JSON record offsets are document-level (whole file)", "warning"))
    header: list[str] = []
    for r, _ in objs:
        for k in r:
            if k not in header:
                header.append(k)
    rows = [[_flat(r.get(k, "")) for k in header] for r, _ in objs]
    _tables(ex, header, rows, [s for _, s in objs], name)
    return ex
