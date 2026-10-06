# Technology Survey (research of 2026-10-05)

Purpose: evidence base for ADRs 0004-0014. Method and limits: GitHub READMEs/raw files, PyPI JSON, apache.org and developer.android.com were fetched; `huggingface.co`, `ai.google.dev`, `blog.google`, `mistral.ai` were blocked by the egress proxy and seen only as web-search excerpts. Fetch results were produced through a summarizing model, so numbers should be re-checked at pin time. Versions are as reported by PyPI JSON on 2026-10-05. Nothing here was benchmarked; no performance or quality claim is made.

## 1. Training and fine-tuning

| Tool | Version / date seen | Licence | Status / note | Role |
|---|---|---|---|---|
| transformers | 5.18.0 | Apache-2.0 | model-definition framework; Python>=3.10, PyTorch>=2.5 | core |
| PEFT | 0.21.2 (2026-10-01) | Apache-2.0 | LoRA/QLoRA etc. | core |
| TRL | 1.14.1 | Apache-2.0 | SFT, DPO, GRPO, KTO, reward trainers | core |
| bitsandbytes | 0.50.2 | MIT | LLM.int8, 4-bit QLoRA | core (CUDA GPUs) |
| datasets | 5.0.1 | Apache-2.0 | Arrow-backed loading | utility |
| PyTorch | 2.14.1 | BSD-style/Apache mix | Python 3.10-3.14 | core |
| Unsloth | 2026.9.14 | Apache-2.0 core + AGPL-3.0 Studio UI (PyPI text) | speed/memory optimizations; model coverage must be checked per model | optional plugin |
| Axolotl | 0.20.0 (2026-09-30) | Apache-2.0 | YAML-driven fine-tuning | possible plugin |
| LLaMA-Factory | 0.9.5 (2026-05-30) | Apache-2.0 | broad model coverage, own UI; Python>=3.11 | possible plugin |
| torchtune | wound down 2025 (GitHub issue #2883, via search) | BSD | no longer actively maintained | rejected |
| vLLM | 0.30.0 | Apache-2.0 | serving/generation for data generation + eval | optional |

Uncertainty: text-only LoRA on multimodal/hybrid checkpoints (Qwen3.5, Gemma 4, Ministral 3) not verified with these versions.

## 2. Dataset handling, corpus processing, retrieval

| Tool | Version | Licence | Note |
|---|---|---|---|
| Docling | 2.133.0 | MIT (models may carry separate terms) | PDF/DOCX/HTML parsing to unified representation |
| PyMuPDF | 1.28.2 | AGPL-3.0 or commercial (Artifex) | not default; licence decision needed |
| datasketch | 2.0.0 | MIT | MinHash/LSH dedupe |
| sentence-transformers | reported 6.1.0 (PyPI summary also showed 5.2.3 Feb 2026; inconsistent) | Apache-2.0 | embeddings/rerankers |
| sqlite-vec | 0.1.9 observed; project "pre-v1" | MIT / Apache-2.0 | vector search in SQLite; maintenance revived 2026; Android PRs open (search) |
| SQLite FTS5 | n/a | public domain | keyword/BM25 (availability in Android system SQLite unverified; ship own build) |

## 3. Evaluation

| Tool | Version | Licence | Note |
|---|---|---|---|
| lm-evaluation-harness | 0.4.13 (2026-08-31) | MIT | general capability regression |
| lighteval | 0.13.0 (2025-11-24) | MIT | optional |
| inspect-ai | 0.3.x | MIT | UK AISI; optional |

## 4. Quantization and formats

- GGUF v3: single-file, extensible typed KV metadata, aligned tensors, little-endian default, SPDX licence key in metadata (spec raw doc, ggml repo).
- llama.cpp quantization: 1.5- to 8-bit integer; example sizes for Llama-3.1-8B: IQ1_S 1.87 GiB, IQ2_XXS 2.23, IQ3_XXS 3.04, IQ4_XS 4.17, Q4_K_M 4.58, Q5_K_M 5.33, Q6_K 6.14, F16 14.96; `--imatrix` for importance-matrix quantization (tools/quantize README). `gguf` Python package 0.19.0 (MIT, 2026-05-06).
- `convert_lora_to_gguf.py` converts PEFT LoRA adapters to GGUF (script argparse).
- Alternatives (GPTQ/AWQ/NF4/ONNX/PTE) in ADR 0008.
- Uncertainty: quality loss per quant per specialist is unknown until measured; a candidate quant ladder is a default proposal (Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS), not a recommendation.

## 5. Mobile and desktop inference runtimes

| Runtime | Licence | Version/status (2026-10-05) | Android | Fit for us |
|---|---|---|---|---|
| llama.cpp | MIT | tag b11400 seen; very active (11k+ commits) | NDK/arm64-v8a, Android Studio example, Termux, KleidiAI/SME2 notes | primary candidate (ADR 0009) |
| LiteRT-LM | Apache-2.0 | v0.16.0; Kotlin API "stable" | yes, GPU/NPU; `.litertlm`; Gemma/Llama/Phi-4/Qwen listed | secondary; fine-tune conversion unverified |
| MediaPipe LLM Inference | Apache-2.0 (assumed; not fetched) | Android/iOS deprecated, maintenance-only | migrate to LiteRT-LM | rejected |
| ExecuTorch | BSD | 1.x (docs 1.0/1.2; release v1.3.1 seen) | XNNPACK, Vulkan, Qualcomm, MediaTek; Llama 2/3.x, Qwen2.5/3 export | later NPU grade |
| MLC LLM | Apache-2.0 | active | OpenCL Adreno/Mali | rejected as primary |
| ONNX Runtime GenAI | MIT | v0.14.0 | "under development" | rejected for now |
| Desktop: llama.cpp (CUDA/Metal/Vulkan), vLLM (Apache-2.0) | | | | workstation/server grades |

## 6. Device profiling (Android)
APIs verified from developer.android.com: `ActivityManager.MemoryInfo{availMem,totalMem,threshold,lowMemory}`; `PowerManager` thermal status (NONE..SHUTDOWN), `addThermalStatusListener`, `getThermalHeadroom` (API 29/30 per fetched page); `ApplicationExitInfo` + `getHistoricalProcessExitReasons` (`REASON_LOW_MEMORY`, `REASON_ANR`, `REASON_CRASH`, `REASON_NATIVE_CRASH`, ...). Not fetched: onTrimMemory, Debug.MemoryInfo/smaps_rollup, FrameMetrics, ADPF hints, Perfetto/Macrobenchmark - confirm during implementation. Detailed design: ADR 0011.

## 7. CLI/UI/application framework
- CLI: Typer 0.27.2 (MIT, PyPI) over pydantic 2.13.5 (MIT) models; jsonschema 4.26.0 (MIT).
- UI: native Kotlin (fixed, ADR 0002). Compose is the expected UI toolkit but was not researched in this round.
- Build: GitHub Actions only (ADR 0003).

## 8. Registries/manifests
No off-the-shelf model registry (MLflow, HF Hub) was adopted: they either require service infrastructure or lack licence/provenance gates. Decision in ADR 0012 and 0014: plain versioned JSON + content hashing. MLflow evaluation was not performed (gap).

## 9. Risks surfaced
1. New architectures need runtime support first (hybrid attention, per-layer embeddings, sliding windows).
2. Licence verification incomplete due to blocked hosts (see BASE_MODELS.md section 0).
3. PyPI summaries had inconsistencies (sentence-transformers versions, inspect-ai); pin from the lock resolver, not this table.
4. Fast churn: transformers 5.x / TRL 1.x / PEFT 0.21 releases within weeks; require pinned versions and an integration test on a tiny model.
5. The summarizer reported llama.cpp release dates as 2024 while tag/PyPI dates indicate 2026; treat release dates from that page as unverified.
