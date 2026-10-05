# LLM Trainer factory core

Python backend foundation: artifact contracts, provenance, license gates, leakage-safe datasets,
evaluation/training interfaces, device qualification, export packages and a CLI. No UI dependency.
Python 3.11+, runtime dependency: `pydantic>=2.6` only. The reference consumer is stdlib-only.

## Run

    cd factory
    pip install -e '.[test]'        # or: pip install pydantic pytest
    python -m pytest                # full suite, offline, a few seconds
    python -m llmtrainer --help     # or the `llmtrainer` entry point

Quick end-to-end (pipeline-validation STUB, no GPU, no network, no real training):

    llmtrainer init /tmp/demo --name "Demo" --domain test
    for f in llmtrainer/fixtures/synthetic_corpus/*.md; do
      llmtrainer add-source /tmp/demo $f --title $(basename $f .md) --origin in-repo-fixture \
        --rights-status synthetic --permit-training yes --permit-redistribution yes --permit-commercial yes
    done
    llmtrainer build-dataset /tmp/demo
    llmtrainer run-experiment /tmp/demo
    llmtrainer validate-package /tmp/demo/exports/*
    python llmtrainer/consumer.py /tmp/demo/exports/* "shaft seal"   # stdlib-only reference consumer

Other commands: `remove-source`, `plan-removal`, `check-license`, `qualify-device`,
`estimate-resources`, `export-schemas ../schemas`.

## Modules (`llmtrainer/`)

| Module | Role |
| --- | --- |
| `hashing.py` | Canonical JSON (sorted keys, compact, UTF-8) and `sha256:` hashing. |
| `schemas.py` | v1 artifact contracts (pydantic, `extra=forbid`), `seal()`/`verify()` content hashes. |
| `licenses.py` | Base-model license registry + gate. Only `"yes"` passes; `unverified` is blocked. Source rights gate. |
| `ingest.py` | Text/Markdown ingestion, section/page-aware chunking with provenance. |
| `splits.py` | Deterministic group-level split, MinHash-LSH near-duplicate detection, leakage resolution. |
| `datasets.py` | Template example generation, split files, `DatasetManifest`, leakage re-verification. |
| `provenance.py` | source -> chunk -> example -> dataset -> model -> package graph; removal -> rebuild plan. |
| `evaluation.py` | `Evaluator`/`Subject` protocols, per-metric comparison, claim rules, STUB evaluator. |
| `training.py` | `Trainer` protocol, `ExperimentConfig`, resource estimator, STUB trainer. |
| `device.py` | Pure-function device qualification (storage / RAM / sustained verdicts, profiles). |
| `packaging.py` | Export package builder and validator. |
| `consumer.py` | Reference consumer; imports nothing from the factory. |
| `pipeline.py` | Workspace operations behind the CLI. |
| `schema_export.py` | Generates `../schemas/v1/*.schema.json`. |
| `data/registry/` | Bundled registry entry for the in-repo stub "model" only. |
| `fixtures/synthetic_corpus/` | Invented, clearly synthetic documents (no third-party material). |

## Honesty notes

* `run-experiment` uses a unigram-table stub trainer/evaluator. It validates plumbing only and can never
  set `improvement_claim_allowed`. Real training is an external executor behind the `Trainer` protocol.
* No real base-model license entries are shipped here; they must be verified against authoritative sources
  and added to the registry (`registry/`), with `verification.source_urls` and `verified_on`.
* `device.py` reference model geometries and quantization quality penalties are illustrative heuristics.
  Estimates are inflated and never yield the `recommended` tier; only a complete passing on-device
  benchmark does.
