# ADR 0006: Dataset handling and provenance storage

- Status: proposed
- Date: 2026-10-05

## Decision
- On-disk dataset format: UTF-8 JSONL for examples (human-diffable, line-hashable) with Parquet allowed for large corpora; content-addressed by sha256 of canonical form. `datasets` (Apache-2.0, 5.0.1) is used for loading/training, not as the system of record.
- Provenance system of record: append-only `events.jsonl` + SQLite lineage index (source -> extraction -> chunk -> example -> dataset -> training run -> model -> export). SQLite chosen because it is a single file, transactional, queryable from Python and Android.
- Every example row carries `source_id`, `chunk_id` (page/section/offset span), generator id + version, prompt hash, confidence, split, and transform list.
- Splits assigned by source section before generation; MinHash near-duplicate clusters (datasketch, MIT, 2.0.0) assigned atomically; frozen test set; leakage check is a blocking gate (n-gram overlap + embedding proximity).
- Dedupe: exact hash + MinHash LSH; table and OCR-confidence flags retained rather than silently dropped.

## Alternatives rejected
- Hugging Face Dataset repos/hub as system of record: pushes private corpora to a service (security rule) and provides no rights/lineage model.
- Spreadsheet/CSV: no nested provenance.
- Graph database (Neo4j etc.): heavy dependency for a graph that fits SQLite recursive CTEs.
- Vector DB as primary store: retrieval index is a derived artifact (ADR 0010), not provenance.

## Consequences
Source removal is a graph query; rebuild is deterministic. Large corpora cost: lineage rows scale with examples; acceptable at expected sizes (not measured).
