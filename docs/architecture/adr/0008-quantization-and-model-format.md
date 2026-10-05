# ADR 0008: GGUF (llama.cpp quantization) as the primary deployment format; safetensors as master

- Status: proposed
- Date: 2026-10-05

## Decision
- Master/training artifacts: Hugging Face safetensors + PEFT adapter.
- Primary deployment format: **GGUF** produced by pinned llama.cpp tools: `convert_hf_to_gguf`, `convert_lora_to_gguf.py`, `llama-imatrix`, `llama-quantize`. Each deployment grade is a distinct GGUF file with its own hash and manifest identity (quant type, imatrix hash, llama.cpp commit).
- Secondary/optional formats: LiteRT-LM `.litertlm` (Gemma-class on-device, ADR 0009), ExecuTorch `.pte`, ONNX (ORT GenAI). These are emitted only when a runtime backend is enabled for a target class; they are never the only copy.

## Evidence
- GGUF spec (https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md, fetched 2026-10-05): version 3, typed key/value metadata (extensible without breaking readers), tensor info table, global alignment, little-endian default, metadata keys including `general.architecture`, `general.license` as SPDX expression. Single self-describing file.
- llama.cpp quantize README (fetched): types from IQ1_S to Q8_0/F16; for Llama-3.1-8B: Q4_K_M 4.58 GiB, Q5_K_M 5.33 GiB, Q6_K 6.14 GiB, F16 14.96 GiB; importance-matrix (`--imatrix`) to reduce accuracy loss; quality measured by perplexity/divergence.
- llama.cpp is MIT-licensed and builds for Android (NDK, `arm64-v8a`, KleidiAI) per its android.md.
- LoRA adapters convert to GGUF and can be applied at load time (adapter-vs-merged is a per-model decision; merged is simpler for runtimes lacking adapter support).

## Alternatives evaluated
| Option | Why not primary |
|---|---|
| GPTQ / AWQ (GPU weight-only quant) | Aimed at GPU serving stacks (vLLM etc.), not phone CPU/GPU runtimes; useful workstation/server grade later |
| bitsandbytes NF4/INT8 | A training/inference-in-PyTorch technique, not a portable file format |
| ExecuTorch `.pte` | Backend-specific binaries; export per backend/device class; strong Qualcomm/Vulkan paths but per-model export work (docs list Llama 2/3/3.x and Qwen 2.5/3 as supported LLMs) |
| ONNX / ORT GenAI | Mature builder, but Android listed as "under development" in the repo README (fetched) |
| MLC compiled libs | Per-model/per-device compilation artifacts; heavier release pipeline |
| Custom format | Violates portability |

## Risks / uncertainties
- New architectures need llama.cpp support before GGUF export works (Qwen3.5's Gated DeltaNet op landed in llama.cpp per search results; Gemma 4 at launch per secondary sources). The registry `supported_formats` records this and the export step must test-load the GGUF.
- Quantization degrades quality; each quantized export is re-evaluated (ADR 0007). No quality claim without measurement.
- GGUF metadata is not a license enforcement mechanism; the license bundle is separate.
