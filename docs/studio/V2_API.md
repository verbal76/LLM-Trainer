# Studio v2 API (phone-first) — contract

Additive to `FACADE.md`. Source of truth: `studio-api/.../Backends.kt` (engine seams), `TypesV2.kt` (value types), `Studio.kt` (methods),
`FakeStudioV2.kt` (scripted implementation for UI work), `studio-core/.../LocalStudio.kt` (real implementation).

## Honesty rules (binding on every screen)
* Only a run that **changes model parameters** is "training". Retrieval (source excerpts in the prompt), prompt specialization and dataset
  building are never labelled training. `TrainingOption.isTraining/changesParameters` and `ChatMessageRecord.contextLabel` carry this.
* No fake inference/training. If the native engine is missing, a model is not installed, or a configuration is too large, calls return
  `Err` (`Blocked` with `reasons`) — never a pretend result. The core never silently reroutes to desktop; `EXTERNAL_COMPUTE` is offered.
* Local-first: nothing in v2 touches the network.

## Seams (implemented by a bundle adapter over the host EngineApi; scripted fakes in tests)
* `InferenceBackend`: `status`, `loadModel(path, patchPath?)`, `modelInfo`, `newChat(model, ctx, threads)`, `resetChat`, `chatFormat`,
  `generate(chat, prompt, sampling, cancel, sink)`, `score(chat, text, cancel)`, `closeChat`, `closeModel`. Throws `BackendException(code)`.
* `TrainingBackend`: `status`, `estimate(base, params)`, `train(base, texts, params, workDir, outPatch, cancel, progress)`, `patchInfo`.
  Semantics mirror `hag_train` (checkpoints in `workDir`, same inputs resume, patch holds only changed tensors, base never modified).
* Constructed via `StudioFactory.createLocal(rootDir, snapshot, host, inference, trainer)` (or `StudioCore(..., inference, trainer)`); with null backends everything v2 reports
  "engine unavailable" and v1 behaviour is unchanged. `StudioCore.releaseModels()` closes the resident model (call it when the app is backgrounded).
* One engine, limited RAM: at most one model is resident (chat keeps it between messages; evaluation and A/B load/close). While a training run is active inference calls fail with `Conflict`; a generation in flight makes a queued run wait (PAUSED, "busy").

## Studio methods
Engine/models: `engineStatus`, `installedModels`, `projectModelState` (capabilities with reasons for chat base/specialist, train, evaluate).
Chat: `createChat(project, BASE|SPECIALIST, specialistId?, ChatOptions)`, `listChats`, `chatHistory`, `deleteChat`, `sendMessage(chatId, text, cancel, onToken)`.
Training: `localTrainingPlan`, `startLocalTraining(project, TrainingSettings, confirmed)`, `trainingRuns/trainingRun`, `pauseTraining`, `cancelTraining`, `resumeTraining`.
Specialists: `specialists`, `verifySpecialist`, `selectSpecialist`, `deleteSpecialist`, `exportSpecialistPatch`.
Evaluation: `startLocalEvaluation`, `localEvaluations/localEvaluation`, `cancelLocalEvaluation`. A/B: `compareAB`, `abComparisons`, `saveABNote`, `deleteAB`.
`MethodOption` gained `changesParameters`; `MethodIds.PARTIAL_ON_DEVICE` was added next to `ADAPTER_ON_DEVICE` (all layers). Desktop job export/import is unchanged.

## Training plan
`localTrainingPlan` returns one `TrainingOption` per `TrainingMethodKind`: `LOCAL_FULL`, `LOCAL_PARTIAL(last N layers)`, `LOCAL_LORA` (rank-r adapters, base frozen; the patch is a standard llama.cpp LoRA adapter GGUF; preferred for quantized bases, learning rate default 2e-3; `TrainingSettings.loraRank/loraAlpha`, `TrainParams.loraRank/loraAlpha`), `EXTERNAL_COMPUTE` (desktop, optional fallback),
`RAG_ONLY` and `PROMPT_ONLY` (not training), each with reasons, blockers, requirements ("plug in the charger") and a `ResourceEstimate`
(estimates only; the `basis` field says from what). Recommended = the largest configuration inside the safe memory envelope.

## Safety gates (checked at plan time, at start, at resume, and continuously during a run)
Refuse unless: engine + base model installed; dataset APPROVED and license gate passes; free RAM within `availableRam * (1 - safetyReserve)` vs the engine's own estimate;
free storage for checkpoints + reserve; charging (or the owner confirmed for hosts that do not report it); battery >= 20%; thermal below MODERATE.
During a run, overheating, unplugging or low battery PAUSES the run at its checkpoint (state PAUSED, message explains). Chat/eval/A-B are refused while training runs.

## Training data
Only the user-approved dataset's TRAIN split examples are used. The validation and TEST splits never enter `texts`; leakage verification is re-run on the final rows
and refuses on any finding. Sequences are packed deterministically (seeded), capped by `maxSequences`, persisted with provenance (`sequences.jsonl`: example ids -> source/chunk).

## Run persistence
`projects/<id>/training/<run>/run.json` (+`.bak`), `sequences.jsonl`, `work/` (engine checkpoints). After process death a RUNNING run reloads as PAUSED/resumable.
`checkpoint = PRESENT` means checkpoint files exist; the engine validates them on resume. A checkpoint the engine rejects (CORRUPT) is set aside (`work.corrupt-*`, kept for inspection) and the run restarts from scratch with `checkpoint = RECOVERED_FROM_SCRATCH`.
Resuming re-checks the gates, the saved inputs hash, the base model, and that no source the run used was removed/changed (otherwise the run is cancelled and its checkpoints discarded).
Specialists: `projects/<id>/specialists/<id>/{specialist.json, patch.hagpatch}`; verified by re-hashing patch + base hash + (engine available) reload.
A specialist is `stale` when a source it was trained on was removed/changed.

## Local evaluation
Held-out items come from the TEST split only. Base and specialist answer the same prompts with the same greedy sampling, one model resident at a time.
Metrics (reported one by one, each with n and a paired-bootstrap 95% CI of the delta): `terminology_coverage`, `exact_fact_accuracy` (closed book; normalized numeric+unit),
`grounded_exact_fact_accuracy` and `grounded_unsupported_claim_rate` (source excerpts provided in the prompt = retrieval), `heldout_nll` (via `score`),
`general_probe_pass_rate` (12 fixed probes, retention), plus latency / tokens-per-second / load time / peak RSS in `performance`.
`improvementClaimAllowed` is false when: items < 50, any synthetic item, any domain metric lacking a significant gain, any regression beyond tolerance, or
general-probe pass rate dropped by more than 0.10. Semantics mirror `factory/llmtrainer/evalsuite` and `evaluation.py`; metrics are lexical/numeric heuristics, not judgements.

## A/B
`compareAB` answers the same prompt with the base then the specialist (streamed per side, per-model stats). Results are kept in the project; `saveABNote` attaches an owner note.

## Fake
`FakeStudio.sampleV2()` = base installed, one trained+selected specialist, engine available. Knobs on `fake.v2`: `engineAvailable`, `charging`, `batteryPercent`, `thermal`, `freeStorageMb`.
`tick()` advances training (3 steps/tick of 12) and completes evaluations; `simulateProcessDeath()` pauses runs and interrupts evaluations.

## Held-out integrity (evaluation)
The held-out chunks must be material the specialist did not train on. If the dataset hash equals the one the specialist was trained on this holds by construction.
If the dataset was rebuilt since, the chunk ids recorded in the training run (`sequences.jsonl`) are compared with the current TEST chunks; any overlap, a changed source,
or a missing training record refuses the evaluation (`HELDOUT_OVERLAP` / `DATASET_CHANGED`). The dataset must be APPROVED.

## Device snapshot keys read
`availRamBytes`, `procMemAvailableBytes`, `lowMemory`, `lowMemoryThresholdBytes`, `freeStorageBytes`, `batteryPct`, `thermalStatus` (0..6 or name), `powerSaveMode`, and
**`charging`** (the host `DeviceProbe` reports it: `BatteryManager.isCharging` or a plugged-in sticky battery intent; omitted when the platform does not answer; `isCharging`/`batteryCharging`/`plugged` are accepted too).
When a host omits it, training needs `TrainingSettings.ownerConfirmsPluggedIn = true` (the UI then shows an explicit "I have plugged in the charger" checkbox). `ready_to_train_now` (model manager) needs `charging == true`; an unknown charger never counts as met.

## Error codes worth handling in the UI
`ENGINE_UNAVAILABLE`, `CHAT_UNAVAILABLE`, `TRAINING_BLOCKED` (reasons list: `NOT_CHARGING`, `CHARGER_UNKNOWN`, `BATTERY_LOW`, `THERMAL`, `POWER_SAVE`, `RAM_UNKNOWN_OR_LOW`, `TOO_LARGE`, `STORAGE`,
`NOT_TRAINABLE`, `BASE_NOT_INSTALLED`, `DATASET_NOT_APPROVED`, `NO_SPLITS`, `TOO_LITTLE_DATA`, `LEAKAGE_DETECTED`, `LICENSE_*`), `CONFIRMATION_REQUIRED`, `SOURCE_CHANGED`,
`SPECIALIST_UNVERIFIED`, `NO_SOURCES_INDEXED`, `OUT_OF_MEMORY`, `UNSUPPORTED`, `HELDOUT_OVERLAP`, `DATASET_CHANGED`, `LICENSE_GATE` (patch export), `Conflict` (engine busy).
