"""Python-authoritative re-verification of an imported (or any workspace) dataset and its held-out items.

Run at import time and again before training/evaluation: the phone's claim that its split is leak-free is never trusted.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .. import pipeline as pl
from .. import splits as sp
from ..datasets import leak_text, load_split, verify_no_leakage
from ..evalsuite.items import EvalItem, check_leakage
from ..evalsuite.metrics import extract_quantities
from ..schemas import DatasetManifest
from .schema import HeldoutItem

CONTAINMENT = 0.8
HELDOUT_FILE = "evals/heldout.jsonl"


def chunk_table(ws: pl.Workspace) -> dict[tuple[str, str], dict]:
    """(source_id, chunk_id) -> {text, role, group, origin, section, synthetic_source} for active sources."""
    out: dict[tuple[str, str], dict] = {}
    for s in ws.sources().active_sources():
        f = ws.root / "corpus" / f"{s.source_id}.jsonl"
        if not f.is_file():
            continue
        meta = {c.chunk_id: c for c in s.chunks}
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            r = json.loads(line)
            c = meta.get(r["chunk_id"])
            out[(s.source_id, r["chunk_id"])] = {
                "text": r["text"], "role": r.get("role", "train"), "group": c.group_id if c else None,
                "origin": r.get("origin", "extracted"), "section": c.section if c else "",
                "synthetic_source": s.rights.status == "synthetic",
            }
    return out


def _group_of(sid: str, chunk: dict, group_by: str) -> str:
    return sid if group_by == "document" else chunk["group"]


def verify_workspace_dataset(ws: pl.Workspace, dataset: DatasetManifest | None = None) -> list[str]:
    """Return every leakage / protection problem found (empty == clean)."""
    dataset = dataset or pl.latest_dataset(ws)
    ds_root = ws.root / "datasets"
    if not dataset.verify():
        return ["dataset manifest content_hash mismatch"]
    try:
        problems = list(verify_no_leakage(dataset, ds_root))
        rows = {s: load_split(ds_root, dataset, s) for s in ("train", "validation", "test")}
    except (ValueError, OSError) as e:
        return [f"dataset files unreadable or altered: {e}"]
    lr = dataset.leakage_report
    if lr.group_overlap_pairs or lr.residual_cross_split_near_duplicates:
        problems.append("dataset manifest records unresolved leakage")
    for s, r in rows.items():
        if not r:
            problems.append(f"{s} split is empty")
    chunks = chunk_table(ws)
    sources = {s.source_id: s for s in ws.sources().active_sources()}
    group_by = lr.effective_group_by
    chunk_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    group_split: dict[str, set[str]] = defaultdict(set)
    row_by_id = {r["example_id"]: r for s in rows.values() for r in s}
    for ex in dataset.examples:
        for ref in ex.derived_from:
            key = (ref.source_id, ref.chunk_id)
            ch = chunks.get(key)
            if ch is None:
                problems.append(f"example {ex.example_id}: unknown chunk {ref.source_id}/{ref.chunk_id}")
                continue
            if ch["role"] != "train":
                problems.append(f"example {ex.example_id}: chunk {ref.source_id}/{ref.chunk_id} has role {ch['role']}, only train chunks may feed examples")
            if _group_of(ref.source_id, ch, group_by) != ex.group_id:
                problems.append(f"example {ex.example_id}: group_id {ex.group_id!r} does not match its chunk's group")
            chunk_split[key].add(ex.split)
            if ex.split == "test" and ch["origin"] == "synthetic":
                problems.append(f"test example {ex.example_id} derives from a synthetic chunk")
            src = sources.get(ref.source_id)
            if src is None:
                problems.append(f"example {ex.example_id}: source {ref.source_id} removed or unknown")
            elif ex.split != "test" and src.rights.permitted_training != "yes":
                problems.append(f"example {ex.example_id}: source {ref.source_id} not cleared for training (permitted_training={src.rights.permitted_training})")
        group_split[ex.group_id].add(ex.split)
    for key, sp_set in sorted(chunk_split.items()):
        if len(sp_set) > 1:
            problems.append(f"chunk {key[0]}/{key[1]} backs examples in several splits {sorted(sp_set)}")
    for g, sp_set in sorted(group_split.items()):
        if len(sp_set) > 1:
            problems.append(f"group {g} appears in several splits {sorted(sp_set)}")

    # containment: text of lower-priority examples must not be (largely) contained in held-out chunk text
    split_of_group = {g: next(iter(v)) for g, v in group_split.items() if len(v) == 1}
    chunk_text_split: dict[str, str] = {}
    for (sid, cid), ch in chunks.items():
        if ch["role"] != "train":
            continue
        g = _group_of(sid, ch, group_by)
        if g in split_of_group:
            chunk_text_split[f"{sid}/{cid}"] = split_of_group[g]
    protected = {ref: chunks[tuple(ref.split("/", 1))]["text"] for ref, s in chunk_text_split.items() if s in ("validation", "test")}
    cfg = dataset.split_config
    ex_split = {e.example_id: e.split for e in dataset.examples}
    cand = {eid: leak_text(r["prompt"], r["response"]) for eid, r in row_by_id.items() if ex_split.get(eid) != "test"}
    for eid, pid, c in sp.containment_leaks(cand, protected, CONTAINMENT, cfg.shingle_size):
        if sp.SPLIT_PRIORITY[chunk_text_split[pid]] > sp.SPLIT_PRIORITY[ex_split[eid]]:
            problems.append(f"example {eid} ({ex_split[eid]}) is {c:.0%} contained in {chunk_text_split[pid]} chunk {pid}")
    # chunk-level near duplicates across splits (a copied document in train hides a test document)
    texts = {ref: chunks[tuple(ref.split("/", 1))]["text"] for ref in chunk_text_split}
    for a, b, j in sp.cross_split_pairs(sp.find_near_duplicates(texts, cfg.near_duplicate_threshold, cfg.shingle_size), chunk_text_split):
        problems.append(f"chunks {a} ({chunk_text_split[a]}) and {b} ({chunk_text_split[b]}) are near-duplicates (jaccard={j})")

    try:
        items = load_heldout(ws)
    except ValueError as e:
        problems.append(f"held-out items invalid: {e}")
        items = None
    if items:
        test_chunk_refs = {r for r, s in chunk_text_split.items() if s == "test"}
        for it in items:
            for ref in it.gold_refs:
                if ref not in test_chunk_refs:
                    problems.append(f"held-out item {it.item_id}: gold chunk {ref} is not a test-split chunk")
        try:
            check_leakage(items, dataset, rows["train"] + rows["validation"])
        except Exception as e:  # LeakageError carries the full list  # noqa: BLE001
            problems.append(str(e))
    return sorted(set(problems))


def load_heldout(ws: pl.Workspace) -> list[EvalItem] | None:
    """Imported held-out items (evals/heldout.jsonl) as EvalItems, or None when the workspace has none."""
    f = ws.root / HELDOUT_FILE
    if not f.is_file():
        return None
    chunks = chunk_table(ws)
    items: list[EvalItem] = []
    for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            h = HeldoutItem.model_validate_json(line)
        except ValueError as e:
            raise ValueError(f"{HELDOUT_FILE} line {n}: {e}") from e
        refs = tuple(h.gold_refs)
        texts, syn = [], False
        for ref in refs:
            sid, _, cid = ref.partition("/")
            ch = chunks.get((sid, cid))
            if ch is None:
                raise ValueError(f"{HELDOUT_FILE} item {h.item_id}: unknown chunk {ref}")
            texts.append(ch["text"])
            syn = syn or ch["synthetic_source"] or ch["origin"] == "synthetic"
        expected = None
        if h.expected is not None:
            qs = extract_quantities(f"{h.expected.value:g} {h.expected.unit}")
            if len(qs) != 1:
                raise ValueError(f"{HELDOUT_FILE} item {h.item_id}: expected {h.expected.value} {h.expected.unit!r} is not a scorable quantity")
            expected = qs[0]
        text = "\n".join(texts)
        items.append(EvalItem(h.item_id, h.kind, h.question, refs, text, text, expected, tuple(h.required_terms), syn))
    ids = [i.item_id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate held-out item ids")
    return items


def heldout_path(ws: pl.Workspace) -> Path:
    return ws.root / HELDOUT_FILE
