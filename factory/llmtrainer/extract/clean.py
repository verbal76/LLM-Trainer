"""Corpus cleaning. Every transformation is counted and returned for the provenance record.

Cleaning never touches the original file: it works on extracted blocks, and each cleaned block
keeps the stream offsets of the text it came from.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .models import Block, Extraction, Line

VERSION = "1"
LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​‌‍⁠﻿]")
_PAGE_NUM = re.compile(r"^(page\s+)?([0-9]{1,4}|[ivxlc]{1,6})(\s*(of|/)\s*[0-9]{1,4})?$", re.I)
_NUM_HEADING = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,3})[.)]?\s+[A-Z][^.!?;]{2,70}$")
_LIST_START = re.compile(r"^\s*([-*•●]|\d{1,3}[.)])\s+")
_HYPH_END = re.compile(r"[A-Za-z]{2,}-$")
_MAX_SAMPLES = 8


@dataclass
class CleanReport:
    transformations: list[dict[str, Any]] = field(default_factory=list)
    removed_noise: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, count: int, **params: Any) -> None:
        if count:
            self.transformations.append({"name": name, "version": VERSION, "count": count, **params})


def norm_noise_key(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def _detect_noise(blocks: list[Block], pages_total: int) -> tuple[set[str], int]:
    """Repeated edge lines across pages (running headers/footers). Returns (noise keys, threshold)."""
    by_page: dict[int, list[Line]] = defaultdict(list)
    for b in blocks:
        if b.lines and b.page is not None:
            by_page[b.page].extend(b.lines)
    pages = [p for p in by_page if by_page[p]]
    if len(pages) < 3:
        return set(), 0
    need = max(3, math.ceil(0.5 * len(pages)))
    seen: dict[str, set[int]] = defaultdict(set)
    for p, ls in by_page.items():
        edge = ls[:2] + ls[-2:]
        for ln in edge:
            if ln.text.strip():
                seen[norm_noise_key(ln.text)].add(p)
    return {k for k, ps in seen.items() if len(ps) >= need}, need


def _is_heading_line(line: str) -> int:
    """Heuristic heading level (0 = not a heading) for formats with no structural headings."""
    s = line.strip()
    if not s or len(s) > 80 or s.endswith((".", ",", ";", ":")):
        return 0
    m = _NUM_HEADING.match(s)
    if m:
        return m.group(1).count(".") + 1
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 4 and len(s.split()) <= 8 and all(c.isupper() for c in letters):
        return 1
    return 0


def clean(ex: Extraction) -> tuple[list[Block], CleanReport]:
    rep = CleanReport()
    lig = ctrl = nbsp = spaces = 0

    def norm(s: str) -> str:
        nonlocal lig, ctrl, nbsp, spaces
        for k, v in LIGATURES.items():
            if k in s:
                lig += s.count(k)
                s = s.replace(k, v)
        c = len(_CTRL.findall(s))
        if c:
            ctrl += c
            s = _CTRL.sub("", s)
        if " " in s:
            nbsp += s.count(" ")
            s = s.replace(" ", " ")
        collapsed = re.sub(r"[ \t]{2,}", " ", s)
        if collapsed != s:
            spaces += 1
        return collapsed.strip()

    out: list[Block] = []
    if ex.line_based:
        noise_keys, need = _detect_noise(ex.blocks, ex.pages_total or 0)
        removed: Counter[str] = Counter()
        removed_pages: dict[str, set[int]] = defaultdict(set)
        pagenum_removed = 0
        # vocabulary for hyphenation decisions
        vocab = Counter(w.lower() for b in ex.blocks if b.lines for l in b.lines for w in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", l.text))
        hyph_fixed: list[str] = []
        hyph_kept = 0
        heading_lines = 0

        for b in ex.blocks:
            if not b.lines:
                out.append(b)
                continue
            kept: list[Line] = []
            edge = set(id(x) for x in (b.lines[:2] + b.lines[-2:]))
            for ln in b.lines:
                key = norm_noise_key(ln.text)
                if b.page is not None and ex.pages_total and ex.pages_total >= 3:
                    if key in noise_keys:
                        removed[ln.text.strip()] += 1
                        removed_pages[ln.text.strip()].add(b.page)
                        continue
                    if id(ln) in edge and _PAGE_NUM.match(ln.text.strip()):
                        pagenum_removed += 1
                        continue
                kept.append(ln)
            if not kept:
                continue
            # hyphenation repair across line breaks inside the paragraph
            merged: list[Line] = []
            for ln in kept:
                if merged and _HYPH_END.search(merged[-1].text.rstrip()) and ln.text[:1].islower():
                    prev = merged[-1]
                    stem = prev.text.rstrip()[:-1]
                    tail = re.match(r"[A-Za-z]+(?:-[A-Za-z]+)*", ln.text)
                    left = re.search(r"[A-Za-z]+(?:-[A-Za-z]+)*$", stem)
                    if tail and left:
                        hyphenated = (left.group(0) + "-" + tail.group(0)).lower()
                        if vocab.get(hyphenated, 0) > 0:
                            hyph_kept += 1  # compound seen intact elsewhere: keep the hyphen
                            merged.append(ln)
                            continue
                        hyph_fixed.append(f"{left.group(0)}-|{tail.group(0)}")
                        joined = stem + ln.text
                        merged[-1] = Line(joined, prev.start, ln.end)
                        continue
                merged.append(ln)
            # heuristic headings (PDF / plain text only)
            groups: list[tuple[str, list[Line], int]] = []  # (kind, lines, level)
            cur: list[Line] = []
            for ln in merged:
                lvl = _is_heading_line(ln.text) if ex.detect_headings else 0
                if lvl:
                    if cur:
                        groups.append(("paragraph", cur, 0))
                        cur = []
                    groups.append(("heading", [ln], lvl))
                    heading_lines += 1
                else:
                    cur.append(ln)
            if cur:
                groups.append(("paragraph", cur, 0))
            for kind, ls, lvl in groups:
                if kind == "heading":
                    text = norm(ls[0].text)
                    tr = list(b.transforms)
                    if ex.detect_headings:
                        tr.append("heading_heuristic")
                    out.append(Block("heading", text, ls[0].start, ls[0].end, page=b.page, level=lvl,
                                     section_path=b.section_path, byte_start=None, byte_end=None, transforms=tr))
                    continue
                parts: list[str] = []
                reflowed = 0
                if any(l.text.lstrip().startswith(("```", "~~~")) for l in ls):  # fenced code: keep layout
                    out.append(Block("paragraph", "\n".join(l.text.rstrip() for l in ls), ls[0].start, ls[-1].end, page=b.page,
                                     section_path=b.section_path, byte_start=b.byte_start, byte_end=b.byte_end,
                                     transforms=list(b.transforms), meta={"preformatted": True}))
                    continue
                for ln in ls:
                    t = norm(ln.text)
                    if not t:
                        continue
                    if parts and not _LIST_START.match(t):
                        parts[-1] += " " + t
                        reflowed += 1
                    else:
                        parts.append(t)
                if not parts:
                    continue
                text = "\n".join(parts)
                tr = list(b.transforms)
                if reflowed:
                    tr.append("reflow")
                exact = len(ls) == len(b.lines) == len(kept)
                out.append(Block("paragraph", text, ls[0].start, ls[-1].end, page=b.page, section_path=b.section_path,
                                 byte_start=b.byte_start if exact else None, byte_end=b.byte_end if exact else None,
                                 transforms=tr, meta={"reflowed_lines": reflowed}))
        for text, n in sorted(removed.items(), key=lambda kv: (-kv[1], kv[0]))[:_MAX_SAMPLES]:
            rep.removed_noise.append({"text": text, "occurrences": n, "pages": sorted(removed_pages[text])})
        rep.add("remove_repeated_header_footer", sum(removed.values()), min_pages=need, distinct_lines=len(removed))
        rep.add("remove_page_number_lines", pagenum_removed)
        rep.add("repair_hyphenation", len(hyph_fixed), examples=hyph_fixed[:_MAX_SAMPLES], kept_compound_hyphens=hyph_kept)
        rep.add("heuristic_heading_detection", heading_lines)
        rep.add("reflow_soft_wrapped_lines", sum(b.meta.get("reflowed_lines", 0) for b in out))
        if ex.detect_headings:
            path: list[tuple[int, str]] = []
            for b in out:
                if b.kind == "heading":
                    while path and path[-1][0] >= b.level:
                        path.pop()
                    path.append((b.level, b.text))
                b.section_path = tuple(t for _, t in path) if path else b.section_path
    else:
        for b in ex.blocks:
            if b.kind == "table":
                b.header = [norm(c) for c in b.header] if b.header else b.header
                b.rows = [[norm(c) for c in r] for r in b.rows] if b.rows is not None else None
                b.text = "\n".join(" | ".join(r) for r in ([b.header] if b.header else []) + (b.rows or []))
            else:
                b.text = "\n".join(norm(x) for x in b.text.split("\n"))
            if b.text.strip():
                out.append(b)
        rep.add("reflow_soft_wrapped_lines", 0)
    rep.add("normalize_ligatures", lig)
    rep.add("strip_control_and_zero_width_chars", ctrl)
    rep.add("normalize_nbsp", nbsp)
    rep.add("collapse_whitespace", spaces)
    return out, rep
