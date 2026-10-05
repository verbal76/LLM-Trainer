# Studio package formats (v1)

Two zip packages connect the Android Studio app (the OTA bundle, see `PRODUCT.md`) to the desktop Python factory
(`factory/llmtrainer`). Both are **format version 1**. A reader MUST reject any other `version` or `format` value.

```
phone ──(training-job.zip)──▶ desktop:  llmtrainer import-job / run-job
phone ◀──(results.zip)─────── desktop:  llmtrainer export-results / run-job
```

Python is authoritative. The desktop never trusts a job package: it re-validates checksums and schema, re-derives every
hash it records, and re-runs leakage and test-split protection on the imported data (rejecting the job on any failure).
The phone never claims training happened: it only displays what a results package says, including its `status`,
`stages`, `generated_by.stub` and `improvement_claim_allowed`.

## 0. Common rules

| Rule | Definition |
| --- | --- |
| Hash string | `sha256:` + 64 lowercase hex digits (the same rendering as `factory/llmtrainer/hashing.py`). Raw file hashes are SHA-256 of the exact bytes. |
| Canonical JSON | UTF-8, object keys sorted (by Unicode code point), separators `,` and `:` (no whitespace), non-ASCII emitted literally (not `\uXXXX`), NaN/Infinity forbidden. Identical to `hashing.canonical_json`. Used only for *derived* hashes (see below) which are computed by Python; the phone never has to reproduce a canonical-JSON hash. |
| Text files | UTF-8, no BOM, `\n` line endings. `*.jsonl` = one JSON object per line, no blank lines required. |
| Timestamps | ISO-8601 with offset, e.g. `2026-10-05T12:00:00Z` (`+00:00` also accepted). Dates: `YYYY-MM-DD`. |
| Identifiers | `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` for `project.id`, `job_id`, `sources[].id`, `chunks[].id`, `example_id`, `item_id`. Used in file names on desktop, so nothing else is accepted. A chunk is globally addressed as `<source_id>/<chunk_id>`. |
| Permissions | `yes` \| `no` \| `conditional` (the job/results formats never carry `unverified`; absence of attestation means UNVERIFIED). Source `rights` additionally allow `unverified` (see 1.4). |
| Unknown fields | Rejected in every object defined here except `device` and `extensions` (free-form objects, ignored by Python). Add new data under `extensions` or bump `version`. |
| `checksums.json` | `{"algorithm":"sha256","files":{"<path>":"sha256:<hex>", ...}}`. MUST list every member of the zip except itself, and nothing else. Paths are the zip member names. |
| Zip hygiene | Members are stored/deflated regular files. No directories required, no absolute paths, no `..` segments, no backslashes, no drive letters, no NUL, no duplicate names, no symlinks, no members outside the layouts below. Desktop limits (configurable): 10 000 members, 128 MiB per member, 256 MiB total uncompressed, compression ratio <= 200:1 for members > 1 MiB. Violations reject the whole package. |
| `job_content_hash` | `hash_obj(checksums.files)` (canonical JSON of the `files` map). Computed by Python; identifies the exact job contents. |

## 1. Training-job package (phone → desktop)

File name convention: `<project.id>-<job_id>.llmtrainer-job.zip`. Layout (all paths relative to the zip root):

```
manifest.json
chunks.jsonl
dataset_manifest.json
dataset/train.jsonl
dataset/validation.jsonl
dataset/test.jsonl
eval/heldout.jsonl
license/license_text.txt        (optional; exact license text the owner read)
checksums.json
```

Raw source files are **not** in the package (they stay on the phone; only their hash/size/metadata and the extracted chunk text
travel). Because the chunk text is included, the desktop rebuilds corpus, datasets and retrieval reference from the package alone.

### 1.1 `manifest.json`

```json
{
  "format": "llmtrainer-training-job",
  "version": 1,
  "job_id": "job-20261005-ab12cd",
  "created_at": "2026-10-05T12:00:00Z",
  "created_by": {"app": "llmtrainer-studio", "app_version": "0.1.0"},
  "project": {
    "id": "proj-motorcycle-mechanic", "name": "Motorcycle Mechanic", "domain": "motorcycle service",
    "purpose": "Diagnose and repair common faults", "created_at": "2026-10-01T09:00:00Z",
    "intended_use": {"commercial": false, "redistribute_model": false}
  },
  "device": {"name": "Pixel 9", "total_ram_mb": 12288},
  "base_model": {
    "registry_id": "Qwen3@Qwen3-1.7B (2504 generation, hybrid thinking)",
    "family": "Qwen3", "exact_version": "Qwen3-1.7B (2504 generation, hybrid thinking)",
    "variant": "hf-weights", "source_url": "https://huggingface.co/Qwen/Qwen3-1.7B"
  },
  "license_evidence": {
    "model_id": "Qwen3@Qwen3-1.7B (2504 generation, hybrid thinking)",
    "license_url": "https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/LICENSE",
    "fetched_at": "2026-10-05T11:00:00Z",
    "text_sha256": "sha256:<hex>",
    "evidence_level": "primary_license_text_read",
    "attested_by": "owner", "attested_on": "2026-10-05",
    "permissions": {"commercial_use": "yes", "fine_tuning_permitted": "yes", "derivative_adapter_permitted": "yes",
                    "redistribution_permitted": "yes", "attribution_required": "yes"},
    "scope_note": "Owner read the license text fetched from the URL above; permissions as attested by owner."
  },
  "method": {"requested": "qlora", "rationale": "Domain vocabulary and diagnostic reasoning; exact specs stay in retrieval."},
  "sources": [ { "...": "see 1.4" } ],
  "extensions": {}
}
```

Rules:

* `format`/`version` exact. `job_id`: identifier pattern; chosen by the phone, unique per export.
* `project.id` identifier pattern; `project.intended_use` optional (defaults: both `false`); `project.purpose` may be empty.
* `device`: optional free-form summary (never used for decisions on desktop).
* `base_model.registry_id` MUST equal `<family>@<exact_version>` (the factory registry `entry_id`). `variant` and `source_url` optional (null).
* `license_evidence` is **optional**. Absent, partial or inconsistent ⇒ the model stays **UNVERIFIED** (fail closed; see 1.6).
  All of the following are required for a complete attestation: `model_id` (== `base_model.registry_id`), `license_url` (https), `fetched_at`,
  `text_sha256`, `evidence_level == "primary_license_text_read"`, `attested_by == "owner"`, `attested_on`, all five `permissions`
  (each `yes|no|conditional`) and a non-empty `scope_note`. The app does not interpret legal text and says so; the owner attests.
  If `license/license_text.txt` is present its SHA-256 MUST equal `text_sha256`.
* `method.requested`: `rag_only` | `prompt_only` (no parameter training; the desktop produces no adapter) | `lora` | `qlora`
  (adapter training). Other strings are accepted by the schema but reported as "unsupported on desktop, planned only".
  `rationale` free text.

### 1.4 `sources[]`

```json
{
  "id": "src-1a2b3c4d5e6f", "file_name": "service_manual.pdf", "title": "Service manual", "sha256": "sha256:<hex of original file>",
  "size": 1048576, "mime": "application/pdf", "ingested_at": "2026-10-05T10:00:00Z",
  "extractor": "pdfbox-text", "extractor_version": "1.0", "origin": "owner library", "source_version": null,
  "rights": {"status": "owned", "license_id": null, "permitted_training": "yes", "permitted_commercial": "yes",
             "permitted_redistribution": "no", "evidence": "purchased", "notes": null},
  "issues": [{"code": "needs_ocr", "detail": "page 4 has no text layer", "severity": "warning", "page": 4}]
}
```

`rights.status`: `owned|licensed|public_domain|open_license|synthetic|unverified|restricted`; `permitted_*`:
`yes|no|conditional|unverified`. Possession of a file never implies rights; the desktop rights gate only lets sources with
`permitted_training == "yes"` feed train/validation examples. `title`, `origin`, `source_version`, `issues[]` optional.
`severity`: `info|warning|error` (default `warning`).

### 1.2 `chunks.jsonl`

One chunk per line:

```json
{"id":"c0001","source_id":"src-1a2b3c4d5e6f","page":3,"section_path":["Brakes","Bleeding"],"char_start":1204,"char_end":1790,
 "role":"train","exclude_reason":null,"text":"...","sha256":"sha256:<hex of UTF-8 text>","origin":"extracted","group_id":null}
```

* `id` unique within its source. `source_id` must exist in `sources[]`. `page` int or null. `section_path` list of strings (may be empty).
  `char_start <= char_end` offsets into the extraction stream.
* `role`: `train` (prose usable for parameter-adaptation examples) | `reference` (tables/exact facts: retrieval only, never training prose)
  | `excluded` (kept for provenance only; `exclude_reason` required, e.g. `duplicate_chunk`, `low_extraction_quality`).
* `sha256` = hash of `text` encoded as UTF-8 (must match, else reject).
* `origin`: `extracted` | `ocr` | `table` | `edited` | `synthetic`. A chunk with `origin: "synthetic"` may not be in the test split.
* `group_id` optional; default `<source_id>#<slug(section)>` where `slug` = lower-case, runs of non `[a-z0-9]` → `-`, trimmed, `untitled` if empty,
  `section` = `" > ".join(section_path)` or `Document` (same as `ingest.slugify` / `chunk_blocks`).

### 1.3 `dataset_manifest.json` and `dataset/*.jsonl`

```json
{
  "dataset_id": "ds-phone-0001", "dataset_version": 1, "seed": 1234,
  "split_config": {"ratios": {"train": 0.7, "validation": 0.15, "test": 0.15}, "group_by": "document",
                   "near_duplicate_threshold": 0.8, "shingle_size": 5},
  "builder": {"name": "studio-template-builder", "version": "1", "type": "template"},
  "split_assignment": {"group_by": "document", "groups": {"src-1": "train", "src-2": "validation", "src-3": "test"}},
  "counts": {"train": 120, "validation": 20, "test": 20}
}
```

* `split_config` matches `SplitConfig` (ratios positive, sum 1; `group_by` `document|section`). `builder.type`: `template|llm|human`.
* `split_assignment.groups` maps every group id used by an example to `train|validation|test`. A group is a whole source
  (`group_by: document`, group id = `source_id`) or a section group (`group_by: section`, group id = chunk `group_id`).
  The same group may never appear in two splits.
* `counts` optional; if present it must equal the row counts.

Example rows (`dataset/<split>.jsonl`), the same shape the Python split files use:

```json
{"example_id":"ex-0001","group_id":"src-1","prompt":"...","response":"...","origin":"source_derived",
 "derived_from":[{"source_id":"src-1","chunk_id":"c0001"}],"task":"cloze_continuation","quality":0.8}
```

`origin`: `source_derived` | `synthetic`. `derived_from` non-empty; every referenced chunk must exist and have `role: "train"`.
`task` (default `imported`) and `quality` (0..1, default 0.5) optional. **Synthetic examples are forbidden in `dataset/test.jsonl`**; test
examples may derive only from chunks whose group is assigned `test`, and those chunks may not back any train/validation example.

### 1.5 `eval/heldout.jsonl`

Held-out evaluation items, derived ONLY from test-split chunks:

```json
{"item_id":"it-0001","kind":"fact","question":"In section \"Torque\": \"Tighten the bolt to ____.\" What is the missing value (number and unit)?",
 "gold_refs":["src-3/c0007"],"expected":{"value":12.0,"unit":"N·m"},"required_terms":[],"origin":"source_derived"}
```

`kind`: `fact` (needs `expected`) | `concept` (needs ≥1 `required_terms`). `gold_refs`: non-empty `<source_id>/<chunk_id>` list, every ref a test-split chunk and
never referenced by train/validation examples. `origin`: `source_derived` only (synthetic evaluation items are rejected).

### 1.6 Desktop import semantics (normative)

1. Verify zip hygiene, `checksums.json` (exact set, every hash), then schema of every file. Any failure ⇒ reject, **nothing is written**
   (the workspace is built in a temporary directory and renamed on success; an existing non-empty target is refused).
2. Rebuild the workspace (`project.json`, `sources/source_manifest.json`, `corpus/<id>.jsonl`, `datasets/<dataset_id>/`, `evals/heldout.jsonl`, `job/…`).
3. Re-run, from the written files: group-overlap check, MinHash near-duplicate check across splits, containment of train/validation text in test chunk text,
   synthetic-in-test, role checks, rights gate, held-out-items-vs-training check. Any finding ⇒ reject the job (the phone's claim of "no leakage" is ignored).
4. License: a complete `license_evidence` creates a workspace registry entry **VERIFIED, owner-attested** (see 4); otherwise the model's catalog entry
   (or an all-`unverified` placeholder) is copied into the workspace unchanged ⇒ UNVERIFIED. A catalog entry that is already `DISALLOWED` is never overridden.

## 2. Results package (desktop → phone)

File name convention: `<project_id>-<job_id>.llmtrainer-results.zip`.

```
manifest.json
evaluation_report.json
python/evaluation_run.json        (optional; the sealed Python EvaluationRun)
python/training_run.json          (optional; the sealed Python TrainingRun)
python/dataset_manifest.json      (optional; the sealed Python DatasetManifest)
artifacts/<name>                  (optional; only with --include-artifacts and only when export is allowed)
checksums.json                    (same format as 0)
```

### 2.1 `manifest.json`

```json
{
  "format": "llmtrainer-results", "version": 1,
  "project_id": "proj-motorcycle-mechanic", "job_id": "job-20261005-ab12cd", "job_content_hash": "sha256:<hex>",
  "created_at": "2026-10-05T12:00:00Z",
  "status": "completed",
  "base_model": {"registry_id": "...", "family": "...", "exact_version": "...", "variant": null, "license_entry_hash": "sha256:<hex>"},
  "license_state": {
    "state": "VERIFIED", "owner_attested": true, "attested_by": "owner", "attested_on": "2026-10-05",
    "evidence_level": "primary_license_text_read", "license_url": "https://...", "license_text_sha256": "sha256:<hex>",
    "scope_note": "...", "export_allowed": true, "reasons": []
  },
  "specialist": {
    "kind": "adapter", "artifact_refs": ["adapter/adapter_model.safetensors", "adapter/adapter_config.json"],
    "sha256s": {"adapter/adapter_model.safetensors": "sha256:<hex>"},
    "included_artifacts": [], "quantization": null,
    "training_run_id": "run-0123456789ab", "training_run_hash": "sha256:<hex>", "config_hash": "sha256:<hex>",
    "method": "qlora", "trainer": "hf_lora/1", "local_experiment": false, "is_pipeline_validation_stub": false
  },
  "dataset": {"dataset_id": "ds-…", "dataset_hash": "sha256:<hex>", "source_manifest_hash": "sha256:<hex>",
              "n_train": 120, "n_validation": 20, "n_test": 20},
  "evaluation": {"eval_id": "eval-…", "evaluation_hash": "sha256:<hex>", "is_stub": false, "improvement_claim_allowed": false},
  "stages": [{"stage": "import", "state": "executed", "detail": "..."}, {"stage": "train", "state": "planned", "detail": "..."}],
  "packaging": {"exportable": false, "blockers": ["..."]},
  "generated_by": {"tool": "llmtrainer", "tool_version": "0.1.0"},
  "notes": []
}
```

* `status`: `completed` (real training and a non-stub evaluation executed) | `stub` (only pipeline-validation stubs ran; nothing here is a quality claim) |
  `planned_only` (dry-run: nothing trained/evaluated) | `partial` (some stage failed/skipped) | `failed`.
* `specialist.kind`: `adapter` | `merged` | `none`. `none` = no parameter training happened (RAG-only, prompt-only, dry-run, or training not executed);
  `artifact_refs` is then `[]`. Refs are paths relative to the desktop run's output directory; artifact bytes are in the zip (`artifacts/<ref>`) only if
  listed in `included_artifacts`. `local_experiment: true` (run under an unverified license) means the artifacts can never be exported/packaged and are never included.
* `license_state.state`: `VERIFIED|UNVERIFIED|CONDITIONAL|RESTRICTED` (as `schemas.LicenseState`, from the redistribution-grade gate, see 3);
  `owner_attested` true when the VERIFIED state rests on the owner attestation from the job. `export_allowed` mirrors `packaging.exportable` license part.
* `stages[].stage`: `import|verify_dataset|train|evaluate|package_results`; `state`: `executed|planned|skipped|failed`.

### 2.2 `evaluation_report.json` (phone-displayable)

```json
{
  "format": "llmtrainer-evaluation-report", "version": 1,
  "project_id": "…", "job_id": "…", "eval_id": "eval-…", "evaluated_on": "2026-10-05",
  "held_out_only": true, "split": "test",
  "generated_by": {"stub": true, "backend": "stub", "evaluator": {"name": "evalsuite_heuristic", "version": "1", "is_stub": true},
                   "tool": "llmtrainer", "tool_version": "0.1.0", "subjects": {"base": "…", "specialist": "…"}},
  "rows": [
    {"metric": "terminology_coverage", "category": "domain", "higher_is_better": true, "base": 0.31, "specialist": 0.44,
     "delta": 0.13, "ci_low": 0.02, "ci_high": 0.24, "n": 12, "notes": "…", "quantized": null}
  ],
  "regressions": [],
  "caveats": ["…"],
  "sample_sizes": {"heldout_items": 20, "fact_items": 8, "concept_items": 12, "train_examples": 120, "validation_examples": 20,
                   "test_examples": 20, "per_metric_n": {"terminology_coverage": 12}},
  "improvement_claim_allowed": false,
  "improvement_claim_reason": "STUB evaluation: …",
  "performance": {"base": {"latency_ms_mean": 1.0}, "specialist": {}}
}
```

* `held_out_only` is always `true` and `split` always `test`: the report never contains numbers from items that were seen in training.
* `delta = specialist - base`; `ci_low`/`ci_high` are the 95% paired-bootstrap interval of that delta (null when unavailable). There is **no aggregate score**.
* `improvement_claim_allowed` can be `true` only when ALL hold: non-stub evaluator and non-stub specialist, non-synthetic held-out data, ≥ 50 held-out items,
  every domain-metric delta positive with a CI excluding 0, no regression, a real (non-local-experiment) trained specialist. `improvement_claim_reason` always states why or why not.
  The phone MUST show `caveats` and the stub/claim fields next to any numbers.

### 2.3 Integrity

`checksums.json` lists every member except itself. A reader verifies it before parsing. `manifest.job_id` must equal the job it was
imported for; the app rejects results for a different project/job.

## 3. Field mapping to the existing Python schemas

| Package field | Python artifact / field | Notes |
| --- | --- | --- |
| `project.id/name/domain` | `SpecialistProject.project_id/name/domain` | `id` is kept verbatim (not re-slugged). |
| `project.purpose` | `SpecialistProject.description` | |
| `project.created_at` | `SpecialistProject.created_on` | date part only. |
| `project.intended_use` | `SpecialistProject.intended_use` (`commercial`, `redistribute_model`) | defaults false. |
| `dataset_manifest.seed/split_config` | `SpecialistProject.seed/split_config`, `DatasetManifest.seed/split_config` | |
| `sources[]` | `SourceManifest.sources[]` = `SourceRecord` | `id→source_id`, `file_name→original_filename`, `size→size_bytes`, `mime→media_type`, `ingested_at→ingested_on` (date), `extractor/_version→transformations[0]`; `rights` is `RightsInfo` 1:1; `issues` → `corpus/<id>.provenance.json`. Raw bytes are not present on desktop (`sources/raw/` stays empty). |
| `chunks.jsonl` | `SourceRecord.chunks[]` = `ChunkRecord` + `corpus/<id>.jsonl` rows | `id→chunk_id`, `page→page_start/page_end`, `section_path`→`section`, `sha256→text_sha256`; `role`, `exclude_reason→excluded_reason`, `origin` kept in the corpus row. |
| `dataset/*.jsonl` | split files + `DatasetManifest.examples[]` = `ExampleRecord` | `text_sha256` is recomputed by Python. |
| `dataset_manifest.dataset_id` | not trusted | Python re-derives `DatasetManifest.dataset_id` from content (`ds-<hash>`); the phone's id is recorded in `leakage_report.notes`. |
| leakage claims | `DatasetManifest.leakage_report` | always computed by Python (`group_overlap_pairs`, `near_duplicate_pairs_found`, `residual_cross_split_near_duplicates`); an import that would need non-zero values is rejected instead. |
| `eval/heldout.jsonl` | `evalsuite.items.EvalItem` | `gold_refs`, `expected→Quantity`, `required_terms`; stored as `evals/heldout.jsonl`. |
| `base_model` | `BaseModelLicenseEntry` identity (`entry_id`), `ModelManifest.base_model` (`BaseModelRef`) | `variant`, `source_url` map to `ModelVariant`. |
| `license_evidence` | workspace `registry/<id>.json` = `BaseModelLicenseEntry` with `verification` (`LicenseVerification`) | `text_sha256→license_text_sha256`, `license_url→source_urls[0]/license_text_url`, `attested_on→verified_on`, `scope_note→verified_scope`, `permissions→commercial_use…attribution_required`. There is no `attested_by` field in the v2 entry schema (unchanged): owner attribution is carried in `verified_scope` (prefix `OWNER-ATTESTED:`), `uncertainties`, and the sidecar `job/license_attestation.json`; the entry hash (and so every `license_entry_hash` in later manifests/packages) covers it. |
| `method.requested` | `SpecializeConfig.method` (`lora|qlora`) → `TrainingRun.method` | `rag_only`/`prompt_only` have no `TrainingRun`. The `Literal` of `TrainingRun.method` also has `stub`/`full`, which jobs never request. |
| results `specialist.*` | `TrainingRun` (`run_id`, `content_hash`, `config_hash`, `output_artifacts`, `local_experiment_unverified_license`, `trainer`, `is_pipeline_validation_stub`), `ModelManifest` (`quantization`, `artifacts`) | `kind: merged` is reserved: no desktop path produces a merged model yet. |
| results `license_state` | `TrainingRun.license_state`, `ModelManifest.base_model.license_state`, `GateResult.verification_state` | state is computed with `specialize.gating.redistribution_gate`. |
| results `dataset.*` | `DatasetManifest` | |
| `evaluation_report.rows` | `EvaluationRun.comparisons` (+ `metrics[].n`) | `ci_low/ci_high` have no field in `EvaluationRun`; they come from the sidecar `evals/<eval_id>.report.json` written by `llmtrainer evaluate` (otherwise null). `quantized` ← `MetricComparison.quantized`. |
| `improvement_claim_allowed` | `EvaluationRun.improvement_claim_allowed`, AND-ed with the extra results-time conditions in 2.2 | the package may be stricter than the sealed `EvaluationRun`, never looser. |
| `generated_by.stub` | `EvaluationRun.evaluator.is_stub` (OR the specialist being a stub subject) | |
| `checksums.json` | `PackageFile`-style sha256 list | but a flat map, not a sealed Artifact. |

Differences worth knowing: (a) the formats are plain JSON objects without `schema_version`/`kind`/`content_hash` (their integrity is `checksums.json`;
`format`+`version` are the discriminator); sealed Python artifacts travel in `python/` where needed. (b) Dates are partly ISO timestamps here,
dates in the Python schemas. (c) Hash strings are identical (`sha256:<hex>`). (d) Canonical JSON for derived hashes is exactly `hashing.canonical_json`.

## 4. Owner-attested license override (normative)

An attested entry is **VERIFIED but owner-attested**: the evidence is the owner's reading of the license text (hash recorded), not LLM Trainer's
interpretation. It is created only from a complete `license_evidence`, on top of the catalog entry for `base_model.registry_id` (identity, restrictions,
formats, architecture are kept), with:

* `verification.state = VERIFIED`, `evidence_level = primary_license_text_read`, `license_text_sha256`, `license_text_url`, `source_urls = [license_url]`,
  `verified_on = attested_on`, `verified_scope = "OWNER-ATTESTED: owner-reviewed license text from <url>, hash <sha>; permissions as attested by owner. <scope_note>"`,
  `uncertainties += ["Permissions are owner-attested, not independently interpreted by LLM Trainer."]`;
* permission fields = the attested values (a `no` still blocks the gate).

Never overridden: a catalog entry with `state == DISALLOWED`; an entry the catalog already VERIFIED (kept as is, attestation recorded only as a note);
mismatching `model_id`; any missing field. Secondary-source claims never become VERIFIED.
Anything produced under an UNVERIFIED model can run only as a local experiment (`--allow-unverified-license-for-local-experiment`) and is never exportable.

## 5. Desktop commands

```
llmtrainer import-job JOB.zip --workspace DIR
llmtrainer evaluate DIR --backend stub|hf [--run RUN_ID] [--execute] [--allow-download] [--base-model-path P] [--adapter-path P]
llmtrainer export-results DIR --out RESULTS.zip [--include-artifacts] [--job-id ID]
llmtrainer run-job JOB.zip --workspace DIR [--execute] [--eval-backend stub|hf] [--allow-download] [--allow-unverified-license-for-local-experiment]
```

Dry-run is the default everywhere: nothing is trained, evaluated with a real model, or downloaded without an explicit flag, and the commands print
exactly what ran and what was only planned.
