# Device profiler v2 — per-artifact capabilities, choices, measurements

Spec: `factory/llmtrainer/device.py` (section "Device profiler v2"). Port: `qualify/.../Capability.kt`.
Shared golden vectors: `factory/tests/golden/device_capability.v1.json` (regenerate with
`python factory/tools/gen_golden_capability.py`; both test suites must pass). The v1 qualifier
(`device_qualification.v1.json`) is unchanged and still drives the three deployment profiles.

Reusable by other Hot Attic Games apps: everything here is generic (no LLM-Trainer names): `ArtifactSpec`,
`DeviceProfile`, `DeviceState`, `MeasurementRecord`, `Capability.classify/choose/mergeRecords`.

## Capabilities (one answer per downloadable file, never conflated)
| Capability | Meaning |
|---|---|
| `can_download` | file is a verifiable artifact (refreshed catalog entry) AND fits the free storage after the storage reserve |
| `can_load` | weights + minimal runtime fit the RAM budget (OS, background, host app, safety reserve already deducted) |
| `can_infer` | some context length is RAM-safe and the estimated sustained decode speed is usable; reports `infer_context`, `est_tokens_per_s` |
| `can_evaluate` | `can_infer` with context >= 1024 and >= 5 tok/s (evaluation runs batches of prompts) |
| `can_specialize_full` / `_partial` | F32 training fits (see below). `trainable_last_layers` = the largest trailing-layer count that fits |
| `external_compute_required` | no local specialization possible -> desktop/GPU job package (`llmtrainer import-job`) |

Inference feasibility is **never** training feasibility: a quantized file can be `can_infer` and still
`external_compute_required` (it points at its full-precision sibling via `specialize_via_artifact_id`).
Reasons are stable codes (`ram_insufficient_to_load`, `training_class_external_only`, ...); the app maps them to text.

## Training memory model (F32 trainer)
```
RAM = (F32 weights + 12 B x trainable params) x weights_inflation(1.05)
      + (activations + runtime overhead 300 MB) x estimate_inflation(1.25)
F32 weights = 4 B x params;  12 B = 4 B gradient + 8 B Adam moments
activations = (trainable_layers x 4 x T x (16 x embd + 2 x ffn) + 2 x 4 x T x vocab) / MiB,  T = 256 tokens
partial tuning of k trailing layers: trainable = params/layers x k   (conservative: includes the embedding share)
fits  <=>  RAM <= 0.85 x RAM budget  AND  2 x checkpoint + 256 MB dataset <= free storage - reserve  AND no recorded OOM/failure at >= k layers
```
Static class of the FILE (`registry/artifacts`, `training.class`): quantized -> `inference_only`; full precision
<= 0.45B -> `local_full`, <= 1.0B -> `local_partial`, larger -> `external_only`. The device then decides what actually fits.
(`docs/v2/TRAINING_FEASIBILITY.md` from `feat/engine` was not available when this was written; the assumption is an
F32-only trainer. Revisit the constants in `CapabilityPolicy`/`full_precision_class_for` when it lands.)

Training also has **conditions** (not capabilities): thermal status <= LIGHT, charging, battery >= 30 % (or charging), battery saver
off, storage for checkpoints. Each is `met | unmet | unknown`; unknown never counts as met (the current host snapshot has no
`charging` field yet: add `BatteryManager.isCharging` as `charging`). `ready_to_train_now` needs all met.

## Owner-facing choices
`fastest` (highest estimated tok/s, then smallest file), `balanced` (best quality with <= 92 % RAM utilisation, >= 4 tok/s,
>= 2048 context), `best_quality` (best quality inside the safe envelope), and separately `best_specialize` (best
full-precision file this phone can specialize locally; prefers full over partial tuning). Tier: `recommended` only when the
chosen configuration has a COMPLETE on-device measurement; `provisional` = estimates; `preview` = the file is not
downloadable yet (unrefreshed catalog). DISALLOWED licenses are never offered. Missing measurements keep confidence `low`.

Pixel 10 Pro XL regression (15.2 GB total, 2.1 GB "available"): `availMem` alone is transient and must not cap a capable
phone; in the illustrative catalog a 4B Q4_K_M model is comfortable (< 60 % of the budget) and the best specialize target
is a sub-1B full-precision file with partial tuning. Covered by `test_capability.py`, the golden vectors and `ModelsServiceTest`.

## Measurement records (`device_measurement.v1`)
One JSON object per run, created by the app's benchmark runner (needs the native runtime; not part of the v1 host):
```json
{"schema_id":"device_measurement.v1","record_id":"<uuid>","artifact_id":"qwen3-4b-q4_k_m","artifact_sha256":"sha256:...",
 "device_id":"Google-Pixel 10 Pro XL","runtime_id":"hag-engine 1; llama.cpp 0c1e570","app_version":"2.1",
 "recorded_at":"2026-10-06T12:00:00Z","kind":"inference",
 "context_tokens":2048,"ttft_ms":1800,"tokens_per_s":12.0,"peak_ram_mb":3200,"sustained_ram_mb":3100,"thermal_throttle_ratio":0.9,
 "ui_jank_pct":1.0,"crashes":0,"anrs":0,"background_kills":0,"sustained_minutes":15}
```
* `kind:"load"`: `load_time_ms`, `load_peak_ram_mb`.
* `kind:"inference"`: the fields above for ONE context length (the measured configuration).
* `kind:"training"`: `trainable_last_layers` (omit = full tuning), `train_peak_ram_mb`, `train_tokens_per_s`,
  `checkpoint_size_mb`, `train_minutes`, `train_completed`, `train_oom`, `thermal_max_status`, `battery_drop_pct`.

Recording: the benchmark screen calls `ModelsApi.recordMeasurement(json)`. The app validates the schema, requires
`device_id` to be THIS device, requires `artifact_id` to be a catalog file or an installed import, rejects a record whose
`artifact_sha256` differs from the catalog/installed hash, then merges it into `workspace/measurements.json`.

Merge: key = (artifact, device, kind, context, trainable layers); the latest `recorded_at` wins, an older record never
replaces a newer one, full and partial runs are different keys. Effects:
* inference records become v1 `Measurement`s for that context: measured RAM replaces the estimate, measured tok/s and
  stability replace the decode estimate; a crash/ANR/kill/low tok/s/throttle turns an estimated "yes" into "no";
  confidence is `medium` with any record, `high` only with a COMPLETE record of the very configuration that was chosen.
* load records replace the load RAM estimate.
* training records replace the training RAM/throughput/checkpoint estimates for that layer count; an OOM or incomplete run
  at k layers forbids >= k layers; a completed run raises training confidence to `high` and can extend the layer count
  beyond the estimate.
Records for another device or another file (hash) are ignored with a reason (`ignored_measurements`).
