"""Deterministic group-level splitting and near-duplicate leakage detection."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from itertools import combinations

_P = (1 << 61) - 1
NUM_PERM = 64
BANDS = 16  # 16 bands x 4 rows: high recall for Jaccard >= ~0.7
ROWS = NUM_PERM // BANDS
SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}  # higher = more protected


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


_PERMS = [(_h64(f"a{i}") % _P | 1, _h64(f"b{i}") % _P) for i in range(NUM_PERM)]


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(text: str, k: int = 5) -> set[int]:
    w = words(text)
    if len(w) <= k:
        return {_h64(" ".join(w))}
    return {_h64(" ".join(w[i : i + k])) for i in range(len(w) - k + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def minhash(sh: set[int]) -> tuple[int, ...]:
    return tuple(min((a * x + b) % _P for x in sh) for a, b in _PERMS)


def find_near_duplicates(
    texts: dict[str, str], threshold: float = 0.8, k: int = 5
) -> list[tuple[str, str, float]]:
    """All id pairs with exact shingle-Jaccard >= threshold (MinHash-LSH candidates, exactly verified)."""
    sh = {i: shingles(t, k) for i, t in texts.items()}
    sig = {i: minhash(s) for i, s in sh.items()}
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for i, s in sig.items():
        for b in range(BANDS):
            buckets[(b, s[b * ROWS : (b + 1) * ROWS])].append(i)
    cands: set[tuple[str, str]] = set()
    for ids in buckets.values():
        if len(ids) > 1:
            cands.update(combinations(sorted(ids), 2))
    out = []
    for a, b in sorted(cands):
        j = jaccard(sh[a], sh[b])
        if j >= threshold:
            out.append((a, b, round(j, 6)))
    return out


def assign_groups(
    group_weights: dict[str, int], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """Assign whole groups to splits.

    Groups are ordered by hash(seed, group) so the result is deterministic and
    independent of input order. The first groups seed test, then validation
    (so every split is non-empty when there are >= 3 groups); the rest go to
    whichever split is furthest below its target weight.
    """
    if len(group_weights) < 3:
        raise ValueError(f"need at least 3 groups to form train/validation/test, got {len(group_weights)}")
    order = sorted(group_weights, key=lambda g: (_h64(f"{seed}|{g}"), g))
    total = sum(group_weights.values())
    target = {s: r * total for s, r in ratios.items()}
    got = {s: 0 for s in ratios}
    assign: dict[str, str] = {}
    seeds = ["test", "validation", "train"]
    for g, s in zip(order[:3], seeds):
        assign[g] = s
        got[s] += group_weights[g]
    for g in order[3:]:
        s = max(ratios, key=lambda x: (target[x] - got[x], SPLIT_PRIORITY[x] == 0))
        assign[g] = s
        got[s] += group_weights[g]
    return assign


def cross_split_pairs(
    pairs: list[tuple[str, str, float]], split_of: dict[str, str]
) -> list[tuple[str, str, float]]:
    return [(a, b, j) for a, b, j in pairs if split_of[a] != split_of[b]]


def resolve_leakage(
    texts: dict[str, str], split_of: dict[str, str], threshold: float, k: int
) -> tuple[dict[str, str], list[str], int]:
    """Drop the less-protected side of every cross-split near-duplicate pair.

    Returns (remaining split_of, dropped ids, number of pairs found). Priority:
    test is never dropped in favour of validation/train; validation is never
    dropped in favour of train.
    """
    pairs = find_near_duplicates(texts, threshold, k)
    dropped: set[str] = set()
    found = 0
    for a, b, _ in cross_split_pairs(pairs, split_of):
        found += 1
        if a in dropped or b in dropped:
            continue
        victim = a if SPLIT_PRIORITY[split_of[a]] < SPLIT_PRIORITY[split_of[b]] else b
        dropped.add(victim)
    remaining = {i: s for i, s in split_of.items() if i not in dropped}
    return remaining, sorted(dropped), found


def containment_leaks(
    candidates: dict[str, str], protected: dict[str, str], threshold: float, k: int
) -> list[tuple[str, str, float]]:
    """Candidates whose shingles are (almost) contained in a protected text.

    Jaccard misses a short example copied out of a long held-out chunk; containment
    (|cand & prot| / |cand|) catches it. Returns (candidate_id, protected_id, containment).
    """
    index: dict[int, set[str]] = defaultdict(set)
    for pid, t in protected.items():
        for s in shingles(t, k):
            index[s].add(pid)
    out = []
    for cid in sorted(candidates):
        sh = shingles(candidates[cid], k)
        hits: dict[str, int] = defaultdict(int)
        for s in sh:
            for pid in index.get(s, ()):
                hits[pid] += 1
        if hits:
            pid = max(sorted(hits), key=lambda p: hits[p])
            c = hits[pid] / len(sh)
            if c >= threshold:
                out.append((cid, pid, round(c, 6)))
    return out
