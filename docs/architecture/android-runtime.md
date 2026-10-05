# Android runtime (native v2)

LLM Trainer is phone-first: the APK runs the LLM locally and specializes it on-device. Nothing is mocked and there is
no cloud/desktop fallback. If the engine cannot run on a device the host still boots, advertises no inference/training
capability, and says why (diagnostics: `engineStatus`, `nativeVersion`).

## Modules
| Module | Role |
|---|---|
| `native/engine` | C++ engine behind the stable C API `hag_engine.h` (owned by the engine implementer) |
| `runtime/` (`com.hotatticgames.hag.runtime`) | Generic, reusable Android library: CMake build of `native/engine` + llama.cpp, JNI, Kotlin binding (`HagRuntime`, `HagEngine`). No LLM-Trainer code. |
| `host-api/` | Level 2 adds `EngineApi`, `HostServices.engine/nativeVersion/buildSha/engineUnavailableReason`, capabilities `inference.gguf.v1`, `training.patch.v1` |
| `app/` | `NativeEngine` (guarded bring-up), `EngineAdapter` (runtime -> `EngineApi`), capability gating in `HostRuntime` |

## Two native libraries, one gate
* `libhagrt.so` – JNI bridge + CPU-feature gate, compiled for the plain baseline ISA, no link dependency on the engine.
* `libhagengine.so` – engine + llama.cpp/ggml (CPU backend only, static), compiled for the higher ISA baseline.

`HagRuntime.load()`: load `hagrt` -> CPU gate -> only then load `hagengine` -> bind symbols (dlsym) -> `hag_engine_init`.
A device that lacks the required ISA features never loads the engine library (no SIGILL, not even from static
initialisers) and gets `EngineStatus.Unavailable(stage="cpuGate", reason)`. Every failure stage is reported, none crashes.

## ISA / ABI decisions
* **arm64-v8a** (ships): `-march=armv8.2-a+dotprod+fp16` (`GGML_CPU_ARM_ARCH`). Gate = `AT_HWCAP` has FPHP + ASIMDHP + ASIMDDP.
  i8mm/SVE/SME are intentionally NOT in the baseline. Upgrade path: `GGML_BACKEND_DL=ON` + `GGML_CPU_ALL_VARIANTS=ON`
  (ggml now has `android_armv8.*/armv9.*` variants) which needs the engine to call `ggml_backend_load_all_from_path()`,
  the variant `.so` files packaged and 16 KB-checked; do it only when it can be benchmarked on real hardware.
* **x86_64** (emulator qualification only, never in the release APK): SSE4.2 only (AVX2 is not guaranteed on emulator CPUs).
  Gate = SSE4.2 + POPCNT.
* Native code is always built Release (also for the Gradle debug variant); Debug ggml is unusably slow.

## 16 KB page size
Link flags `-Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384 -Wl,-z,text` (no text relocations), NDK r28c
(`28.2.13676358`, pinned in `gradle.properties`; preinstalled on ubuntu-24.04 runners), `useLegacyPackaging=false`
(stored + page-aligned). `tools/ci/check-16kb.sh` verifies every `.so` in the APK (PT_LOAD `p_align >= 0x4000`, no TEXTREL,
stored, `zipalign -c -P 16`) and the ABI set: release = arm64-v8a only, debug = arm64-v8a + x86_64.

## Compatibility rules
* `NATIVE_ABI = 2` (exact gate). Native v1 hosts are ABI 1; a bundle needing the new runtime is never offered to them
  and v1-era bundles are never installed on v2.
* Built-in bundle is OTA sequence **#3** (> every v1 slot), so `UpdateStore.reconcileBuiltin` drops stale v1 slots.
* Host API level 2 (major unchanged, append-only). Inference/training capabilities exist **only** if the engine
  initialised (`HostCapabilities.advertised`), so a bundle that needs them is refused on an unsupported CPU instead of crashing.
* `nativeRuntimeId` = the engine's own `hag_engine_version()`; `buildSha` = source commit.

## Native build inputs
* `native/fetch-llama.sh <dest-dir>` (pins the llama.cpp commit; idempotent) – called by the Gradle task `fetchLlamaCpp`;
  `LLAMA_CPP_SRC=<checkout>` overrides it. Cache: `runtime/build/llama.cpp`.
* Engine sources: `native/engine/src/**/*.{c,cc,cpp}` (test/tool dirs excluded), headers `native/engine/include`.

## Tests
* JVM, no SDK: `runtime/host-test/run.sh` builds the REAL `hag_jni.cpp` against a stub C engine and drives it from Kotlin
  with `-Xcheck:jni` (marshalling, UTF-8, callbacks, exceptions, cancel from another thread, deferred free).
* `runtime` androidTest: real engine + real SmolLM2-135M-Instruct GGUF (Q8_0 inference, F16 on-device training with
  patch/base-untouched/resume checks). `app` androidTest: capability gating, engine-unavailable boot, OTA ABI rules.
* CI: `native-engine` job on API 36 and on the 16 KB page-size image (`getconf PAGE_SIZE` must print 16384).
