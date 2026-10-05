# studio-core (`:studio-core`) — implementation notes

Real `Studio` implementation behind `StudioFactory.create(rootDir, deviceSnapshotJson)`. Pure JVM; runtime deps are kotlin-stdlib, platform
`org.json` (compileOnly) and the `qualify` / `extract` / `studio-api` modules. Only API-26-safe `java.*` calls are used (no `JSONObject.keySet`,
no Java 9+ library methods) so it dexes into the OTA bundle. Contract: `FACADE.md`; product rules: `PRODUCT.md`; zip formats: `PACKAGE_FORMATS.md`.

## Seams (all faked in tests)
`Http` (HTTPS only; `JavaHttp` is the real one), `Clock`, `StorageProbe`, `IdSource`, `TaskRunner` (downloads/imports run on 2 daemon threads).
`StudioCore` takes them as optional constructor arguments; `StudioFactory` uses the real ones.

## On-disk layout (`rootDir`)
```
workspace/licenses.json            owner attestations + the last fetched/imported license text per model (hash-anchored)
workspace/license_texts/<sha>.txt  exact bytes the owner read
workspace/catalog_overrides.json   extra downloadable variants (the shipped registry has repository pages only; see addVariantOverride)
workspace/operations/<op>.json     downloads / imports; RUNNING|QUEUED reload as PAUSED (downloads) or FAILED (imports: the picked stream is gone)
workspace/models/<variant>/        <file>.part while downloading, <file> + acquired.json when done
projects/<id>/project.json         identity, selections, export flags, job ids          (atomic write, one .bak generation, schema number)
projects/<id>/ingest_report.json   per-file status, rewritten after every file (finished=false while a batch runs)
projects/<id>/sources/<src>/       raw/<file> (byte-identical, hash-verified), blocks.jsonl, stream.txt, provenance.json, source.json (written last)
projects/<id>/dataset/             build.json, chunks.jsonl, synthetic.jsonl, review.json (swapped in as a directory; see reconcile)
projects/<id>/evaluation/          view.json + the evaluation_report.json exactly as imported
```
Crash handling: `.tmp-*` source dirs and `dataset.tmp` are discarded; `dataset.old` is restored if the swap was interrupted; a dataset whose recorded
source snapshot (ids, sha256, rights) differs from the sources on disk is marked STALE and rows of vanished sources are purged; unreadable
project files are set aside (`.corrupt-*`), reported in `StudioCore.startupProblems`, and never deleted.

## Honest semantics worth knowing
* **License**: registry baseline is read-only. `fetchLicenseText` hashes the exact bytes (whole file; display is truncated at 200k chars) and records `fetched_at`.
  `attestLicense` -> VERIFIED only if the sha matches that fetch/import AND fine-tune/adapter are YES AND all four permissions are explicitly given.
  Explicit NO on fine-tune/adapter -> DISALLOWED (owner-attested, owner may re-attest after re-reading). CONDITIONAL is refused (nothing recorded) rather than
  stored as VERIFIED; registry DISALLOWED is never overridable. Gate = `licenses.py`: not VERIFIED => everything blocked, permission claims ignored.
* **Catalog**: variants are exactly the registry's (HF repository pages: no sizes, hashes or file URLs). Sizes shown are labelled ESTIMATE from the
  parameter count; "effective" parameter counts are not guessed (verdict UNKNOWN). Hence a download needs a workspace variant override or a model-file import.
* **Recommendations**: `qualify` over registry models (nominal params; geometry from the nearest generic size class when the registry has none, flagged),
  snapshot-only so `benchmarked=false`, confidence LOW, plus the "no inference runtime" warning. Larger models that were passed over are explained.
* **Dataset**: groups (whole document, or section when < 3 documents) are assigned with the Python algorithm (BLAKE2b/MinHash port, golden-tested).
  Examples are deterministic *templates over source text* (builder `studio-template-builder/1`, type `template`): cloze continuation = `source_derived`;
  opt-in question templates = `synthetic`, excluded by default, train-only. Near-duplicates / containment across splits are flagged `POSSIBLE_LEAKAGE` and excluded by
  default; export re-runs the Python leakage checks on the final rows and refuses on any finding. Fewer than 3 groups => no split, no training/held-out export.
* **Not detected** (flags exist in the API): `CONTRADICTION`, `OUTDATED_REVISION`. `OCR_NOISE` is a token-garbling heuristic; no OCR is performed.

## Studio-defined packages (not in PACKAGE_FORMATS.md; all carry `checksums.json` like the job format)
* held-out set `*.llmtrainer-heldout.zip`: `manifest.json` (`llmtrainer-heldout-set` v1), `eval/heldout.jsonl` (same item shape as the job).
* reference package `*-reference.zip`: `reference_manifest.json` (`llmtrainer-reference-package` v1, lists held-out chunk refs), `chunks.jsonl` (provenance + rights per chunk), `sources.json`, `terms.json`.
* specialist package `*.specialist.zip`: `specialist_manifest.json` (`llmtrainer-specialist-package` v1: base model identity, license state + evidence, dataset hash, artifact refs +
  sha256 from the desktop results, evaluation summary with the claim flag/reason), the shipped `evaluation_report.json`, license evidence/text, `reference/chunks.jsonl`, `provenance.json`.
  Versions are `1.0.<n>` and `n` increments on every export, so two different artifacts never share an identity. The phone holds no adapter bytes; it references them by hash.

## Tests
`./gradlew --no-daemon :studio-core:test` (JVM). Includes Python golden vectors (`factory/tools/gen_golden_splits.py`), Python-built fixtures
(`factory/tests/fixtures/studio`), an independent re-implementation of the desktop import rules (`JobValidator`), FakeStudio parity, a real `JavaHttp`
against a loopback server, and `SampleEmitTest`, which writes `build/sample/job.zip` for `factory/tests/test_kotlin_job_interop.py`
(`LLMT_KOTLIN_JOB_ZIP=studio-core/build/sample/job.zip pytest factory/tests/test_kotlin_job_interop.py` runs the real Python `import-job` on it).

## v2 (phone-first) services
`LocalStudio` (+ `TrainingService`, `EvalService`, `SpecialistRegistry`, `EngineCore`) implement chat, local training, specialists, local evaluation and A/B behind
`InferenceBackend` / `TrainingBackend` (studio-api `Backends.kt`). Contract and honesty rules: `V2_API.md`. Layout additions under `projects/<id>/`:
`training/<run>/{run.json,sequences.jsonl,work/}`, `specialists/<id>/{specialist.json,patch.hagpatch}` + `specialists/registry.json`, `chats/<id>.json`, `local_eval/<id>.json`, `ab/<id>.json`;
`workspace/local/file_hashes.json` caches base-model hashes (a hash recorded by the downloader in `acquired.json` is used when present).
Tests use scripted backends (`ScriptedEngine.kt`, `LocalRig.kt`); real-engine proofs belong to the engine/emulator tests.
