# ADR 0010: Reference package = single SQLite file with FTS5 (+ optional sqlite-vec vectors)

- Status: proposed
- Date: 2026-10-05

## Decision
The companion exact-reference package is one SQLite database (`reference.sqlite`) plus a manifest. Tables: `documents`, `chunks` (text, page/section anchor, source hash, rights flags), `tables_structured` (rows/cols for specs tables), `fts_chunks` (FTS5), optional `vec_chunks` (vector index). Retrieval is hybrid (FTS5 BM25 + vectors + reciprocal-rank fusion) when vectors are present, FTS5-only otherwise. Citations resolve to `chunk_id -> source hash -> page`. Built in Python; read from Kotlin (Android bundles SQLite; FTS5 availability on the system SQLite varies by device/API level and is **unverified here**, so the app ships its own SQLite build) or any language.

## Evidence
- FTS5 is part of SQLite; hybrid FTS5+vector patterns with RRF are in common use (several third-party write-ups in search results).
- sqlite-vec (https://github.com/asg017/sqlite-vec): MIT/Apache-2.0 dual, pre-v1 ("breaking changes expected"), PyPI 0.1.9 observed; after a hiatus the project was reported revived in 2026 with open PRs for Android support and Android 16 KB pages (search results; PRs not verified merged).

## Alternatives evaluated
- Dedicated vector DBs (Qdrant, Chroma, FAISS): server/native deps and index files that are awkward on Android; not self-describing with citations/rights.
- FAISS/HNSW file + separate metadata: two files to keep consistent; no SQL join to rights/provenance.
- sqlite-vss: superseded by sqlite-vec by the same author (repo notice, not re-fetched).
- Brute-force vectors in a plain BLOB column: acceptable fallback; at expected chunk counts (tens of thousands, 384-1024 dims) it is cheap; a vector extension is optional, not required for v1 correctness.
- LLM-only memory (no retrieval): contradicts `CLAUDE.md` principle 4.

## Embedding model
Needs an embedding model runnable on both the factory and the phone; model choice, license check (same registry gate) and runtime (llama.cpp can serve embedding GGUFs - unverified here) are open items. Vector index memory counts in the device budget (ADR 0011). The manifest records embedding model id/hash so a package never mixes embedding spaces.

## Consequences
Because FTS5 works without any embedding model, an exact-fact lookup (torque value) never depends on vector quality; tables of specs are stored structured, not only as chunk text.
