"""Held-out evaluation items (test-split chunks only) and the leakage guard."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .. import splits as sp
from ..hashing import hash_obj, short
from ..schemas import DatasetManifest
from .metrics import STOPWORDS, Quantity, content_words, extract_quantities


class LeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    kind: str  # "fact" | "concept"
    question: str
    gold_refs: tuple[str, ...]  # "source_id/chunk_id"
    gold_text: str  # text of the gold chunk(s)
    content_text: str  # text whose overlap with training data is checked (masked sentence or chunk)
    expected: Quantity | None = None
    required_terms: tuple[str, ...] = ()
    synthetic: bool = False


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip()) if s]


def _key_terms(text: str, section: str, idf: dict[str, float], k: int = 5) -> tuple[str, ...]:
    tf = Counter(w for w in content_words(text + " " + section) if not w.isdigit() and len(w) >= 5 and w not in STOPWORDS)
    ranked = sorted(tf, key=lambda w: (-(tf[w] * idf.get(w, 1.0)), w))
    return tuple(ranked[:k])


def build_items(chunks: list[dict], *, domain: str = "this domain", max_facts_per_chunk: int = 2) -> list[EvalItem]:
    """chunks: dicts with ref, section, text, synthetic. Deterministic; sorted by ref."""
    chunks = sorted(chunks, key=lambda c: c["ref"])
    df = Counter(w for c in chunks for w in set(content_words(c["text"])))
    idf = {w: 1.0 / n for w, n in df.items()}
    items: list[EvalItem] = []
    for c in chunks:
        ref, sec, text, syn = c["ref"], c["section"], c["text"], bool(c.get("synthetic"))
        terms = _key_terms(text, sec, idf)
        if len(terms) >= 2:
            items.append(EvalItem(
                "it-" + short(hash_obj([ref, "concept"])), "concept",
                f'Describe, as a {domain} specialist would, what the section "{sec}" covers.',
                (ref,), text, text, None, terms, syn))
        n = 0
        for sent in _sentences(text):
            qs = extract_quantities(sent)
            if len(qs) != 1 or n >= max_facts_per_chunk:
                continue
            q = qs[0]
            masked = re.sub(r"([-−]?\d+(?:[.,]\d+)?|\.\d+)\s*[^\s,;.]+", "____", sent, count=1)
            if "____" not in masked:
                continue
            items.append(EvalItem(
                "it-" + short(hash_obj([ref, "fact", sent])), "fact",
                f'In the {domain} documentation, section "{sec}": "{masked}" What is the missing value (number and unit)?',
                (ref,), text, sent, q, (), syn))
            n += 1
    return items


def load_test_chunks(dataset: DatasetManifest, corpus_dir: Path, sources) -> list[dict]:
    """Chunks referenced by TEST examples only (never train/validation-only chunks)."""
    test_refs = {(r.source_id, r.chunk_id) for e in dataset.examples if e.split == "test" for r in e.derived_from}
    out = []
    for sid in sorted({s for s, _ in test_refs}):
        src = sources.get(sid)
        texts = {}
        for line in (corpus_dir / f"{sid}.jsonl").read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                texts[row["chunk_id"]] = row["text"]
        meta = {c.chunk_id: c for c in src.chunks}
        for (s, cid) in sorted(test_refs):
            if s == sid and cid in texts:
                out.append({"ref": f"{sid}/{cid}", "section": meta[cid].section, "text": texts[cid],
                            "synthetic": src.rights.status == "synthetic"})
    return out


def train_refs(dataset: DatasetManifest) -> set[str]:
    return {f"{r.source_id}/{r.chunk_id}" for e in dataset.examples if e.split in ("train", "validation") for r in e.derived_from}


def check_leakage(items: list[EvalItem], dataset: DatasetManifest, train_rows: list[dict], *, threshold: float = 0.6, k: int = 5) -> dict:
    """Fail (LeakageError) if an eval item reuses a train/validation chunk or its content is largely contained in
    train/validation text. Returns a small report on success."""
    seen = train_refs(dataset)
    test_refs = {f"{r.source_id}/{r.chunk_id}" for e in dataset.examples if e.split == "test" for r in e.derived_from}
    problems = []
    for it in items:
        for r in it.gold_refs:
            if r in seen:
                problems.append(f"{it.item_id}: gold chunk {r} is also used by train/validation examples")
            if r not in test_refs:
                problems.append(f"{it.item_id}: gold chunk {r} is not a test-split chunk")
    train_sh: set[int] = set()
    for r in train_rows:
        train_sh |= sp.shingles(r["prompt"] + " " + r["response"], k)
    for it in items:
        sh = sp.shingles(it.content_text, k)
        if sh and len(sh & train_sh) / len(sh) >= threshold:
            problems.append(f"{it.item_id}: content overlaps training text (containment >= {threshold})")
    if problems:
        raise LeakageError("evaluation leakage guard failed:\n  " + "\n  ".join(problems))
    return {"items_checked": len(items), "train_rows_compared": len(train_rows), "containment_threshold": threshold}
