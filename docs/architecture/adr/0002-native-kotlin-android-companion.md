# ADR 0002: Native Kotlin Android app as product UI/companion

- Status: accepted (fixed by lead; recorded, not re-litigated)
- Date: 2026-10-05

## Decision
The Android app is native Kotlin: a host APK plus an OTA-updatable Kotlin bundle (mechanics in the lead's `docs/OTA.md` when it exists). It is the product UI/companion: project and package management, device profiling/qualification/benchmarking, running exported specialist packages with an on-device runtime, and update diagnostics. It does not train 7B models on-phone.

## Why it fits the requirements
- The no-degradation device rule needs first-class access to Android system signals: `ActivityManager.MemoryInfo` (`availMem`, `totalMem`, `threshold`, `lowMemory`), `PowerManager` thermal status/headroom (thermal API added at API 29/30), and `ApplicationExitInfo` exit reasons such as `REASON_LOW_MEMORY` (all from developer.android.com references fetched 2026-10-05; see ADR 0011). These are Kotlin/Java APIs.
- Inference runtimes ship Kotlin/JNI bindings (LiteRT-LM has a stable Kotlin API per its README; llama.cpp ships an Android example, https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md).

## Alternatives considered and rejected (not relitigating the fixed choice; recorded for completeness)
- Cross-platform JS/React-Native/Expo: excluded by the EAS-free decision (ADR 0003) and adds a bridge between UI and native runtime/profiling code.
- Flutter/KMP-UI: no requirement justifies a second UI runtime; Kotlin is sufficient. Kotlin Multiplatform *logic* (qualifier) remains possible later; not adopted now.

## Consequences
The qualification algorithm must exist in Kotlin as the on-device authority and in Python as the executable spec (ADR 0011), kept honest by shared golden test vectors. The app consumes Export Packages only; it never imports factory code.
