"""Deterministic domain-term candidates (TF-IDF-like scoring, no model) with term -> chunk provenance."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

STOP = frozenset("""a about above after again all also an and any are as at be because been before being below between both but by can could did do
does doing down during each few for from further had has have having he her here hers him his how i if in into is it its just may me more most
must my no nor not now of off on once only or other our out over own same she should so some such than that the their them then there these they
this those through to too under until up use used using very was we were what when where which while who whom why will with would you your
shall per via etc""".split())
_TOKEN = re.compile(r"[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _ok(t: str) -> bool:
    return len(t) >= 3 and t not in STOP


def extract_terms(rows: list[dict[str, Any]], *, top_k: int = 50, min_df: int = 1, min_tf: int = 2, max_chunks: int = 10) -> list[dict[str, Any]]:
    """rows need source_id, chunk_id, text. Unigrams and bigrams; score = tf * idf, ties broken alphabetically."""
    tf: Counter[str] = Counter()
    df: dict[str, set[str]] = defaultdict(set)
    n = 0
    for r in rows:
        n += 1
        ref = f"{r['source_id']}/{r['chunk_id']}"
        toks = _tokens(r["text"])
        terms = [t for t in toks if _ok(t)]
        terms += [f"{a} {b}" for a, b in zip(toks, toks[1:]) if _ok(a) and _ok(b)]
        for t in terms:
            tf[t] += 1
            df[t].add(ref)
    out = []
    for t, c in tf.items():
        d = len(df[t])
        if c < min_tf or d < min_df or t.replace(" ", "").replace("-", "").isdigit():
            continue
        idf = math.log((1 + n) / (1 + d)) + 1
        bonus = 1.5 if " " in t else 1.0  # multi-word candidates are more domain-specific
        out.append({"term": t, "tf": c, "df": d, "score": round(c * idf * bonus, 4), "chunks": sorted(df[t])[:max_chunks]})
    out.sort(key=lambda x: (-x["score"], x["term"]))
    return out[:top_k]
