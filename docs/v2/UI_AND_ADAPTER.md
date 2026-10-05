# Phone-first UI and the engine adapter (OTA bundle)

How the native v2 product is usable inside the OTA-capable bundle (pure Kotlin, dexed, no native code, framework `android.*` views only).

## Engine adapter (`engine-adapter/`)
* `HostInferenceBackend` / `HostTrainingBackend` implement the studio-api seams (`Backends.kt`) over the host `EngineApi` (host API level 2).
  They are thin: mapping, measurements and cancel only. Gates, persistence, resume and honesty rules stay in studio-core.
* `EngineWiring.create(engine, reason, capabilities)` is what `Ctl.buildStudio()` calls. Inference is wired only if the host advertises
  `inference.gguf.v1`, training only with `training.patch.v1` (capabilities read from `diagnosticsJson().capabilities`; if unreadable the presence
  of `host.engine` decides). Otherwise the backend reports `BackendStatus(false, "none", reason)` with the host's `engineUnavailableReason`,
  and studio-core answers `ENGINE_UNAVAILABLE`. Nothing is ever faked.
* Patch semantics (ENGINE.md): `modelApplyPatch` accepts both kinds, a changed-tensor patch (`kind=replace`, full/partial) and a standard
  llama.cpp LoRA adapter GGUF (`kind=lora`). It hashes the whole base file once per model handle, so `ModelInfo.loadMs` of a specialist includes that cost.
* LoRA: `TrainParams.loraRank/loraAlpha` -> config keys `lora_rank`/`lora_alpha` -> `TrainConfig` -> JNI -> `hag_train_params`.
  The adapter checks the engine's estimate echoes `lora=true, lora_rank=N`; if not (stale runtime) it refuses (`UNSUPPORTED`) instead of silently tuning base weights.
* Cancel: the engine clears its cancel flag when a call starts, so `CancelWatcher` re-sends `cancel(session)` every 20 ms until the call returns.
  A cancelled prompt phase (`HAG_ERR_CANCELLED`) is a normal stop (`StopReason.CANCELLED`), not a failure.
* The thread count of a training run is pinned once from `n_cores` (3/4, max 8) because the engine's run fingerprint includes it (bit-exact resume).
* Build: the module compiles `host-api/.../EngineApi.kt` (copied at build time, no android imports) for its own classpath only and excludes it from its jar;
  at runtime the bundle uses the host's `EngineApi`. Tests: `:engine-adapter:test` (fake `EngineApi`, plus a StudioCore smoke test).

## Screens (all plain `android.*` views, tags are the test contract)
| Route kind | Screen | Honesty rules shown |
|---|---|---|
| `MODELS` | Model manager: storage, four picks (fastest / balanced / best quality / best to specialize), per-file capability class badges, license state, download with explicit confirmation (size, free-after, fit, license), progress / cancel / resume, manual GGUF import, remove | never auto-downloads; unrefreshed catalog files are not downloadable; estimates are `PROVISIONAL`; import is `UNVERIFIED` |
| `CHAT` | Base vs specialist, streaming, Stop, per-answer load time / first token / tok/s / process RAM, retrieval toggle | retrieval is labelled "not training" next to every answer that used excerpts |
| `TRAIN_LOCAL` | Plan (LoRA, last layers, all layers, desktop job, reference package, prompt), device conditions, blockers, settings, start / pause / resume / cancel, progress, losses, checkpoint status | only parameter-changing runs are "training"; forgetting caveat (+4 nats measured by the engine); desktop is an optional, externally-labelled fallback, never chosen for the user |
| `EVAL` | Local evaluation (held-out TEST split) above the existing desktop import | per-metric deltas with n and CI, retention row, `improvementClaimAllowed` shown as supported / not supported with the reason |
| `AB` | Same question to base and specialist side by side (greedy), note, saved comparisons | "one answer is an anecdote" |
| `SPECIALISTS` | Select / deselect, verify, export patch zip, delete, stale and unverified badges | export fails closed on the license gate |
| `ABOUT` | The five identities (native version, app version, OTA sequence, runtime id/ABI, git sha), engine status, "what can this phone do" | all speeds / memory / durations are estimates until measured |

The project hub has an "On this phone" card (status line from `projectModelState`) linking to all of them; the dashboard links to the model manager and About.

## Test seams (`StudioTestHooks`, inert until a test sets them)
`useFakeStudio(sampleV2)` swaps in `FakeStudio` (+ `FakeModelsApi`) and redraws; `fakeKnob(name, value)` (engineAvailable, charging, batteryPercent, thermal,
freeStorageMb, availableRamMb, `tick`, `death`) runs on the studio worker thread; `supplyModelFile(name, text)` / `supplyModelPath(path)` replace the system picker;
`setAutoConfirm(true)` skips confirmation dialogs (not part of the view tree). Instrumented tests: `StudioV2FlowTest` (scripted flow), `RealEngineChatFlowTest`
(real engine + the runtime tests' tiny SmolLM2 Q8_0, skipped locally without the fixture, failing in CI when `hagRequireModels=true`), `StudioUiFlowTest` (extended).
