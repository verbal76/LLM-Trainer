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
* Constructed via `StudioFactory.createLocal(rootDir, snapshot, host, inference, trainer)`; with null backends everything v2 reports
  "engine unavailable" and v1 behaviour is unchanged.

## Studio methods
Engine/models: `engineStatus`, `installedModels`, `projectModelState` (capabilities with reasons for chat base/specialist, train, evaluate).
Chat: `createChat(project, BASE|SPECIALIST, specialistId?, ChatOptions)`, `listChats`, `chatHistory`, `deleteChat`, `sendMessage(chatId, text, cancel, onToken)`.
Training: `localTrainingPlan`, `startLocalTraining(project, TrainingSettings, confirmed)`, `trainingRuns/trainingRun`, `pauseTraining`, `cancelTraining`, `resumeTraining`.
Specialists: `specialists`, `verifySpecialist`, `selectSpecialist`, `deleteSpecialist`, `exportSpecialistPatch`.
Evaluation: `startLocalEvaluation`, `localEvaluations/localEvaluation`, `cancelLocalEvaluation`. A/B: `compareAB`, `abComparisons`, `saveABNote`, `deleteAB`.
`MethodOption` gained `changesParameters`; `MethodIds.PARTIAL_ON_DEVICE` was added next to `ADAPTER_ON_DEVICE` (all layers). Desktop job export/import is unchanged.

## Training plan
`localTrainingPlan` returns one `TrainingOption` per `TrainingMethodKind`: `LOCAL_FULL`, `LOCAL_PARTIAL(last N layers)`, `EXTERNAL_COMPUTE` (desktop, optional fallback),
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
A checkpoint the engine rejects (CORRUPT) is set aside (`work.corrupt-*`) and the run restarts from scratch with `checkpoint = RECOVERED_FROM_SCRATCH`.
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
