"""Dataset builder: chunks -> examples -> leakage-protected splits -> manifest + JSONL files."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from . import splits as sp
from .hashing import hash_file, hash_obj, short
from .schemas import (
    ChunkRef,
    DatasetManifest,
    ExampleRecord,
    GeneratorInfo,
    LeakageReport,
    SourceManifest,
    SplitConfig,
    SplitInfo,
)

BUILDER = GeneratorInfo(name="template_cloze_builder", version="2", type="template")
SYNTHETIC_DEFAULT = GeneratorInfo(name="unspecified_synthetic", version="0", type="llm")
CONTAINMENT_THRESHOLD = 0.8
MIN_QUALITY = 0.4
SPLITS = ("train", "validation", "test")


def _sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text).strip()
    return [s for s in re.split(r"(?<=[.!?])\s+", flat) if s]


def generate_examples(section: str, text: str) -> list[dict]:
    """Deterministic template examples from one chunk. Pipeline fixture generator, not a quality generator."""
    out = []
    for idx, sent in enumerate(_sentences(text)):
        w = sent.split()
        if len(w) < 8:
            continue
        cut = len(w) // 2
        out.append(
            {
                "task": "cloze_continuation",
                "idx": idx,
                "prompt": f'Complete the statement from section "{section}": ' + " ".join(w[:cut]),
                "response": " ".join(w[cut:]),
                "quality": min(1.0, len(w) / 20),
            }
        )
    return out


_PREFIX = re.compile(r'^Complete the statement from section ".*?": ')


def leak_text(prompt: str, response: str) -> str:
    """Text compared for near-duplicate detection: the underlying content without template boilerplate."""
    return _PREFIX.sub("", prompt) + " " + response


def _load_chunk_rows(corpus_dir: Path, source_id: str) -> dict[str, dict]:
    """chunk_id -> corpus row. Rows without a role (legacy corpora) are treated as role=train."""
    f = corpus_dir / f"{source_id}.jsonl"
    rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line]
    return {r["chunk_id"]: {**r, "role": r.get("role", "train")} for r in rows}


def build_dataset(
    sources: SourceManifest,
    corpus_dir: Path,
    out_root: Path,
    *,
    project_id: str,
    seed: int,
    config: SplitConfig,
    allowed_source_ids: list[str],
    excluded: list[str] = (),
    synthetic_examples: list[dict] = (),
) -> DatasetManifest:
    """Build train/validation/test.

    Three data roles are kept apart: ``train`` chunks feed parameter-adaptation examples;
    ``reference`` chunks (tables, exact facts) are only listed in ``reference_chunks.jsonl`` for
    retrieval; the ``test`` split is the held-out evaluation set. ``synthetic_examples`` (dicts with
    prompt, response, source_id, chunk_id, optional task/quality/generator) are labelled
    ``origin=synthetic``, must cite a train chunk, are never placed in test and are dropped when the
    chunk they cite belongs to a validation/test group (that would leak held-out content).
    """
    candidates: list[dict] = []
    reference: list[dict] = []
    chunk_rows: dict[tuple[str, str], dict] = {}
    for sid in sorted(allowed_source_ids):
        src = sources.get(sid)
        rows = _load_chunk_rows(corpus_dir, sid)
        texts = {k: v["text"] for k, v in rows.items()}
        for ch in src.chunks:
            row = rows[ch.chunk_id]
            if row["role"] == "reference":
                reference.append({"source_id": sid, "chunk_id": ch.chunk_id, "section": ch.section, "page_start": ch.page_start,
                                  "text_sha256": ch.text_sha256, "table": row.get("table")})
            if row["role"] != "train":
                continue
            chunk_rows[(sid, ch.chunk_id)] = {"text": row["text"], "group": ch.group_id}
            for ex in generate_examples(ch.section, texts[ch.chunk_id]):
                if ex["quality"] < MIN_QUALITY:
                    continue
                eid = "ex-" + short(hash_obj([sid, ch.chunk_id, ex["task"], ex["idx"]]))
                ex.update(example_id=eid, source_id=sid, chunk_id=ch.chunk_id, section_group=ch.group_id, origin="source_derived", generator=BUILDER)
                candidates.append(ex)
    for sx in synthetic_examples:
        key = (sx["source_id"], sx["chunk_id"])
        if key not in chunk_rows:
            raise ValueError(f"synthetic example must cite an existing train-role chunk, got {key}")
        gen = sx.get("generator", SYNTHETIC_DEFAULT)
        if sx.get("split") == "test":
            raise ValueError("synthetic examples are forbidden in the held-out test split")
        candidates.append({
            "task": sx.get("task", "synthetic"), "idx": 0, "prompt": sx["prompt"], "response": sx["response"], "quality": sx.get("quality", 0.5),
            "example_id": "ex-" + short(hash_obj(["synthetic", key, sx["prompt"], sx["response"]])), "source_id": key[0], "chunk_id": key[1],
            "section_group": chunk_rows[key]["group"], "origin": "synthetic", "generator": gen,
        })
    candidates = [c for c in candidates if c["quality"] >= MIN_QUALITY]
    if not candidates:
        raise ValueError("no examples could be generated from the permitted sources")

    n_docs = len({c["source_id"] for c in candidates})
    effective = config.group_by
    notes: list[str] = []
    if effective == "document" and n_docs < 3:
        effective = "section"
        notes.append(f"only {n_docs} source documents; fell back to section-level grouping")
    for c in candidates:
        c["group_id"] = c["source_id"] if effective == "document" else c["section_group"]
    weights: dict[str, int] = defaultdict(int)
    for c in candidates:
        weights[c["group_id"]] += 1
    group_split = sp.assign_groups(dict(weights), config.ratios, seed)
    split_of = {c["example_id"]: group_split[c["group_id"]] for c in candidates}

    # synthetic examples may only live in train, and only if the chunk they cite is a train-group chunk
    syn_dropped = [c["example_id"] for c in candidates if c["origin"] == "synthetic" and split_of[c["example_id"]] != "train"]
    for eid in syn_dropped:
        split_of.pop(eid)
    if syn_dropped:
        notes.append(f"dropped {len(syn_dropped)} synthetic example(s) citing validation/test-group chunks (held-out contamination)")
    cand_by_id = {c["example_id"]: c for c in candidates}
    texts = {c["example_id"]: leak_text(c["prompt"], c["response"]) for c in candidates if c["example_id"] in split_of}
    # exact duplicates within the same split carry no information: keep the first by id
    seen: dict[tuple[str, str], str] = {}
    dup_dropped: list[str] = []
    for eid in sorted(texts):
        key = (split_of[eid], " ".join(sp.words(texts[eid])))
        if key in seen:
            dup_dropped.append(eid)
        else:
            seen[key] = eid
    for eid in dup_dropped:
        split_of.pop(eid)
        texts.pop(eid)
    if dup_dropped:
        notes.append(f"dropped {len(dup_dropped)} exact in-split duplicate examples")

    # containment check against the CHUNKS of held-out groups (not only their examples)
    group_split = {g: s for g, s in group_split.items()}
    held = {f"{k[0]}/{k[1]}": v["text"] for k, v in chunk_rows.items() if group_split.get(v["group"] if effective == "section" else k[0]) in ("validation", "test")}
    chunk_split = {f"{k[0]}/{k[1]}": group_split.get(v["group"] if effective == "section" else k[0]) for k, v in chunk_rows.items()}
    contained: list[str] = []
    for eid, pid, _ in sp.containment_leaks(
        {e: t for e, t in texts.items() if split_of[e] != "test"}, held, CONTAINMENT_THRESHOLD, config.shingle_size
    ):
        if sp.SPLIT_PRIORITY[chunk_split[pid]] > sp.SPLIT_PRIORITY[split_of[eid]]:
            contained.append(eid)
    for eid in contained:
        split_of.pop(eid)
        texts.pop(eid)
    if contained:
        notes.append(f"dropped {len(contained)} example(s) largely contained in held-out chunk text")
    remaining, dropped, found = sp.resolve_leakage(
        texts, split_of, config.near_duplicate_threshold, config.shingle_size
    )
    dropped = sorted(set(dropped) | set(contained) | set(syn_dropped))
    kept_texts = {i: texts[i] for i in remaining}
    residual = len(
        sp.cross_split_pairs(
            sp.find_near_duplicates(kept_texts, config.near_duplicate_threshold, config.shingle_size),
            remaining,
        )
    )
    by_split_groups: dict[str, set[str]] = defaultdict(set)
    for c in candidates:
        if c["example_id"] in remaining:
            by_split_groups[remaining[c["example_id"]]].add(c["group_id"])
    overlap = sum(
        len(by_split_groups[a] & by_split_groups[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )

    ds_id = "ds-" + short(hash_obj([sources.content_hash, seed, config.model_dump(mode="json"), BUILDER.model_dump(),
                                      {"min_quality": MIN_QUALITY, "containment": CONTAINMENT_THRESHOLD, "roles": "train-only-v1"},
                                      sorted(hash_obj([s["source_id"], s["chunk_id"], s["prompt"], s["response"]]) for s in synthetic_examples)]))
    out_dir = out_root / ds_id
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[ExampleRecord] = []
    split_info: dict[str, SplitInfo] = {}
    for split in SPLITS:
        rows = sorted((c for c in candidates if remaining.get(c["example_id"]) == split), key=lambda c: c["example_id"])
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for c in rows:
                fh.write(
                    json.dumps(
                        {
                            "example_id": c["example_id"],
                            "group_id": c["group_id"],
                            "prompt": c["prompt"],
                            "response": c["response"],
                            "origin": c["origin"],
                            "derived_from": [{"source_id": c["source_id"], "chunk_id": c["chunk_id"]}],
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                records.append(
                    ExampleRecord(
                        example_id=c["example_id"],
                        split=split,
                        group_id=c["group_id"],
                        task=c["task"],
                        derived_from=[ChunkRef(source_id=c["source_id"], chunk_id=c["chunk_id"])],
                        generator=c["generator"],
                        origin=c["origin"],
                        text_sha256=hash_obj({"prompt": c["prompt"], "response": c["response"]}),
                        quality_score=round(c["quality"], 4),
                    )
                )
        split_info[split] = SplitInfo(
            file=f"{ds_id}/{split}.jsonl",
            sha256=hash_file(path),
            n_examples=len(rows),
            n_groups=len({c["group_id"] for c in rows}),
        )
    if reference:  # retrieval-only set: separate from training and evaluation data
        with (out_dir / "reference_chunks.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in sorted(reference, key=lambda r: (r["source_id"], r["chunk_id"])):
                fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
        notes.append(f"{len(reference)} reference chunk(s) written to reference_chunks.jsonl (retrieval only; excluded from training and evaluation)")
    if any(si.n_examples == 0 for si in split_info.values()):
        raise ValueError(f"a split is empty after leakage protection: { {k: v.n_examples for k, v in split_info.items()} }")

    manifest = DatasetManifest(
        dataset_id=ds_id,
        project_id=project_id,
        source_manifest_hash=sources.content_hash,
        source_manifest_version=sources.manifest_version,
        seed=seed,
        split_config=config,
        builder=BUILDER,
        examples=records,
        splits=split_info,
        leakage_report=LeakageReport(
            group_overlap_pairs=overlap,
            near_duplicate_pairs_found=found,
            dropped_example_ids=dropped,
            residual_cross_split_near_duplicates=residual,
            effective_group_by=effective,
            notes=notes,
        ),
        excluded_sources=sorted(excluded),
    ).seal()
    (out_dir / "dataset_manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def load_split(dataset_root: Path, manifest: DatasetManifest, split: str) -> list[dict]:
    info = manifest.splits[split]
    path = dataset_root / info.file
    if hash_file(path) != info.sha256:
        raise ValueError(f"split file hash mismatch: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_no_leakage(manifest: DatasetManifest, dataset_root: Path) -> list[str]:
    """Re-check a built dataset from its files. Returns a list of problems (empty == clean)."""
    problems: list[str] = []
    rows = {s: load_split(dataset_root, manifest, s) for s in SPLITS}
    groups = {s: {r["group_id"] for r in rows[s]} for s in SPLITS}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        both = groups[a] & groups[b]
        if both:
            problems.append(f"group(s) {sorted(both)} appear in both {a} and {b}")
    texts, split_of = {}, {}
    for s in SPLITS:
        for r in rows[s]:
            texts[r["example_id"]] = leak_text(r["prompt"], r["response"])
            split_of[r["example_id"]] = s
    for ex in manifest.examples:
        if ex.origin == "synthetic" and ex.split == "test":
            problems.append(f"synthetic example {ex.example_id} in test split")
    for r in rows["test"]:
        if r.get("origin") == "synthetic":
            problems.append(f"synthetic example {r['example_id']} in test split file")
    cfg = manifest.split_config
    for a, b, j in sp.cross_split_pairs(sp.find_near_duplicates(texts, cfg.near_duplicate_threshold, cfg.shingle_size), split_of):
        problems.append(f"near-duplicate across splits: {a} ({split_of[a]}) ~ {b} ({split_of[b]}) jaccard={j}")
    return problems
