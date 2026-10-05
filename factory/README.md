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

Studio phone <-> desktop jobs (formats: `../docs/studio/PACKAGE_FORMATS.md`; sample packages: `tests/fixtures/studio/`):

    llmtrainer import-job job.zip --workspace /tmp/ws         # validate + rebuild; leakage re-verified, hostile zips rejected
    llmtrainer run-job job.zip --workspace /tmp/ws2           # DRY-RUN: import, verify, plan training/eval, results zip
    llmtrainer run-job job.zip --workspace /tmp/ws2 --execute # really train (GPU + 'train' extra) and evaluate; license gate applies
    llmtrainer evaluate /tmp/ws --backend stub|hf [--execute] # held-out items only; stub = plumbing, can never allow a claim
    llmtrainer export-results /tmp/ws --out results.zip

Other commands: `remove-source`, `plan-removal`, `check-license`, `qualify-device`,
`estimate-resources`, `export-schemas ../schemas`.

## Modules (`llmtrainer/`)

| Module | Role |
| --- | --- |
| `hashing.py` | Canonical JSON (sorted keys, compact, UTF-8) and `sha256:` hashing. |
| `schemas.py` | v1 artifact contracts (pydantic, `extra=forbid`), `seal()`/`verify()` content hashes. |
| `licenses.py` | Base-model license registry + fail-closed gate (entry schema v2). Permission fields are claims; they count only when `verification.state == VERIFIED`. UNVERIFIED/DISALLOWED block with reason codes. Source rights gate. |
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
| `jobs/` | Studio job packages: hostile-input-safe zip IO, job schema, `import-job` (owner license attestation -> `OWNER-ATTESTED` registry entry), Python-authoritative leakage re-verification, `evaluate` (stub/hf), `export-results`, `run-job`. |
| `schema_export.py` | Generates `../schemas/v1/*.schema.json`. |
| `data/registry/` | Bundled registry entry for the in-repo stub "model" only. |
| `fixtures/synthetic_corpus/` | Invented, clearly synthetic documents (no third-party material). |

## Honesty notes

* `run-experiment` uses a unigram-table stub trainer/evaluator. It validates plumbing only and can never
  set `improvement_claim_allowed`. Real training is an external executor behind the `Trainer` protocol.
* Real base-model entries live in `../registry/base-models/`. All are `UNVERIFIED` (secondary sources only)
  except the two Llama community licenses, whose official text was read and hashed (their permissions are
  still `conditional`, so the gate blocks). To unblock a model: read the LICENSE from the official repo, record
  its sha256 as `verification.license_text_sha256`, set a `primary_*` `evidence_level`, and set `state: VERIFIED`.
* `device.py` reference model geometries and quantization quality penalties are illustrative heuristics.
  Estimates are inflated and never yield the `recommended` tier; only a complete passing on-device
  benchmark does.
