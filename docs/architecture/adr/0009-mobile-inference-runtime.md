# ADR 0009: On-device runtime = llama.cpp (GGUF) via JNI as primary; LiteRT-LM as secondary

- Status: proposed (candidate evaluated; needs on-device benchmark before it is promoted to accepted)
- Date: 2026-10-05

## Decision
Primary runtime: **llama.cpp** (MIT) built with the Android NDK for `arm64-v8a`, wrapped in a thin Kotlin/JNI bridge owned by the app. Interface abstracted (`InferenceRuntime`) so a second runtime can be added without touching the qualifier or UI. Secondary: **LiteRT-LM** for Gemma-class `.litertlm` exports once a fine-tuned-model conversion path is verified.

## Why llama.cpp first
- Runs the format we already produce (ADR 0008) from the same toolchain on workstation and phone, so the thing evaluated is the thing shipped.
- Broadest model coverage among candidates for the base models we track (Qwen3/3.5, Llama, Gemma 4 at launch, SmolLM3, Olmo, Ministral families; see BASE_MODELS.md for per-model verification state).
- Official Android docs: Android Studio example (`examples/llama.android`) auto-detects hardware kernels (up to SME2 on Arm), Termux, and NDK cross-compile (`GGML_NATIVE=OFF`, OpenMP/llamafile off); KleidiAI acceleration for Arm. Very active: release tag b11400 seen on 2026-10-05 (the fetch summarizer printed a 2024 year, which is inconsistent with tag/PyPI dates; treat date as unverified).
- Supports memory-mapping, context-size and KV-cache controls (needed by the qualifier) - flags not re-fetched this round; the main-tool README URL 404'd. Confirm at implementation.

## Alternatives evaluated
| Runtime | Evidence (2026-10-05) | Verdict |
|---|---|---|
| LiteRT-LM (Google AI Edge) | Apache-2.0, v0.16.0, Kotlin API marked stable, `.litertlm` format, GPU/NPU, Gemma/Llama/Phi-4/Qwen listed (https://github.com/google-ai-edge/LiteRT-LM) | Strong secondary: best vendor GPU/NPU story. Not primary because model conversion for arbitrary fine-tunes + LoRA is not documented in the README excerpt; requires its own export artifact |
| MediaPipe LLM Inference API | Android/iOS implementations deprecated, maintenance-only, migrate to LiteRT-LM (ai.google.dev pages via search) | Rejected: deprecated |
| ExecuTorch | BSD; `.pte` per backend (XNNPACK/Vulkan/Qualcomm/MediaTek); Llama 2/3.x and Qwen2.5/3 supported LLMs (docs) | Candidate later for Qualcomm NPU grade; per-backend export cost, narrower model list |
| MLC LLM | Apache-2.0; OpenCL Adreno/Mali on Android; model compile step per model | Rejected as primary: compilation pipeline per model/device is heavy for a factory that emits many specialists; revisit for GPU-throughput grade |
| ONNX Runtime GenAI | MIT; v0.14.0 includes Qwen 3.5 builder work; repo README says Android "under development" | Rejected for now |

## Risks
- JNI bridge and native build maintenance on GH Actions (NDK, 16 KB page-size alignment on new Android versions - concern flagged by sqlite-vec PR title found in search; verify for our `.so` files).
- Throughput on phone CPUs vs vendor accelerators is unknown until benchmarked on target devices (no measurements exist yet).
- Hybrid-architecture models (Qwen3.5) may use recurrent-state kernels with different memory behavior; qualifier must measure, not assume.

## Consequences
The runtime manifest in the Export Package names runtime + minimum version/commit; the app refuses to load a package whose runtime requirement it cannot satisfy.
