# Studio package fixtures (format v1)

Fixtures for the Kotlin/Android side to test against. Formats: `docs/studio/PACKAGE_FORMATS.md`. All text is invented synthetic
fixture text (pump-service manuals, 6 documents); nothing here is a real model, a real license, or a real evaluation.

Regenerate: `cd factory && python tools/make_studio_fixtures.py` (the Python test-suite asserts the job zips are byte-reproducible and that
every fixture passes the reference validators, so the files cannot drift silently).

## Job packages (phone → desktop)

| File | What it is | Desktop `import-job` result |
| --- | --- | --- |
| `sample_job.zip` | complete job: 6 sources, 36 train chunks, 72/18/18 train/validation/test examples, 18 held-out items, `license_evidence` attested by "owner" (fixture URL `https://example.invalid/sample-license`, `license/license_text.txt` present), method `qlora`, base model `Qwen3@Qwen3-1.7B (2504 generation, hybrid thinking)` | accepted; model becomes VERIFIED, **owner-attested** |
| `sample_job_unattested.zip` | same data, no `license_evidence` | accepted; model stays **UNVERIFIED** (training only as local experiment, never exportable) |
| `sample_job_bad_checksum.zip` | `chunks.jsonl` altered after `checksums.json` was computed | **rejected** (`checksum mismatch for 'chunks.jsonl'`) |

Use `sample_job.zip` to check your own *producer*: the Kotlin exporter should be able to emit byte-compatible-in-structure packages (same
members, same JSON shapes, `checksums.json` = `{"algorithm":"sha256","files":{path: "sha256:<hex>"}}`).

## Results packages (desktop → phone)

All produced by the real `run-job` code path. None contains a positive improvement claim (that needs real training + real evaluation of
≥ 50 unseen, non-synthetic items); to unit-test the "claim allowed" display, edit a copy of `evaluation_report.json`/`manifest.json`
in your test and recompute `checksums.json`.

| File | `manifest.status` | `specialist.kind` | Notes |
| --- | --- | --- | --- |
| `sample_results_planned.zip` | `planned_only` | `none` | dry-run: `train` and `evaluate` stages are `planned`; `rows` is empty; `license_state` VERIFIED + `owner_attested: true` |
| `sample_results_stub.zip` | `stub` | `none` | stub evaluation executed (`generated_by.stub: true`); 6 metric rows with `ci_low/ci_high/n`; `improvement_claim_allowed: false` with reason |
| `sample_results_adapter_local_experiment.zip` | `stub` | `adapter` | fake trainer under an UNVERIFIED license: `local_experiment: true`, `packaging.exportable: false` with blockers, `artifact_refs` listed but no artifact bytes included |

What the app must show next to numbers: `status`, `generated_by.stub`, `caveats`, `improvement_claim_allowed` + `improvement_claim_reason`,
`held_out_only`. Reject results whose `job_id`/`project_id` do not match the job that was exported.
