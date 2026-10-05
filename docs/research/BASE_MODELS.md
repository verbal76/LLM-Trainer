# Base-Model Candidates (research of 2026-10-05)

Machine-readable entries: `registry/base-models/*.json` (19 entries). This page explains them. **Nothing here is legal advice and no legal conclusion is asserted.** Models are listed by size class and licence verifiability, not popularity. Nothing in this document claims any model is "better" for a domain; that needs evaluation evidence.

## 0. Evidence quality and its limits (read first)

Access constraints in this research session: `huggingface.co`, `ai.google.dev`, `blog.google` and `mistral.ai` were blocked by the network egress proxy, so no model card, `config.json` or per-repo `LICENSE` file could be read directly. Reachable and used: GitHub (READMEs, raw files), PyPI JSON, apache.org, developer.android.com, and the web-search tool (whose answers are model-written summaries of results, sometimes echoing my query terms). Therefore:

- **Level A (read the primary document):** Apache-2.0 text (apache.org); Llama 3.1/3.2/4 licence texts and Llama 3.2 AUP (Meta GitHub raw files; read through a summarizer, not verbatim); Qwen3 README licence statement; SmolLM3 README (huggingface/smollm); olmo-core README (code).
- **Level B (official statement seen only as a search-result excerpt/title):** per-model HF cards for Qwen3.5, Olmo 3, Gemma 4 (DeepMind post title "releasing them under an Apache 2.0 license"; Gemma 4 overview excerpt).
- **Level C (third-party only):** Ministral 3 licence, Phi-4-mini licence/size.
- **Registry rule:** permissions are `yes` only for Apache-2.0/MIT-style grants supported by Level A/B evidence and with the licence text read; Llama permissions are `conditional`; Level C is `unverified`. Every entry lists `uncertainties`. Architecture numbers (layers/kv heads/head_dim/hidden) are mostly from search excerpts of `config.json` and must be re-read from the actual file at ingestion time (`factory` should parse it, not trust this table).
- The licence gate (ADR 0012) must re-read the LICENSE file from the downloaded repo; that step is not possible until Hub access exists.

Also note: Apache-2.0 on weights says nothing about rights in the model's pretraining data or about *your* source corpus.

## 1. Licence terms relevant to this product

### 1.1 Apache-2.0 (verified text: https://www.apache.org/licenses/LICENSE-2.0.txt, 2026-10-05)
- s.2: perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright licence to reproduce, prepare Derivative Works, publicly display/perform, sublicense and distribute.
- s.4 (as summarized): recipients of the Work or Derivative Works must receive a copy of the licence; modified files must carry prominent change notices; copyright/patent/trademark/attribution notices must be retained in the Source form of Derivative Works you distribute.
- s.3: patent licences terminate if you sue alleging the Work infringes a patent. s.6: no trademark licence.
- Consequence for us (engineering reading, not legal advice): fine-tuning, adapters, merged weights, quantized files and redistribution are within the grant; the export bundle must carry the licence text, notices and a statement that the model was modified. No naming rule, no use-case restrictions in the licence text itself (model repos may add separate usage policies - check per repo).

### 1.2 Llama 3.x / 4 Community Licenses (Meta GitHub raw texts, read via summarizer 2026-10-05)
- Grant: non-exclusive, worldwide, non-transferable, royalty-free *limited* licence to use, reproduce, distribute, copy, create derivative works of, and modify the Llama Materials.
- Redistribution: provide the agreement; display "Built with Llama"; retain the "Llama X is licensed under the Llama X Community License, Copyright (c) Meta Platforms, Inc." notice.
- Naming: any AI model created, trained, fine-tuned or improved using Llama Materials and distributed must have "Llama" at the beginning of its name. Llama 3.1 text additionally applies this to models improved with Llama *outputs* (relevant to distillation).
- 700M monthly-active-user threshold requires a separate Meta licence.
- Acceptable Use Policy incorporated by reference (prohibited-use categories incl. unlawful acts, harm, deception, undisclosed dangers). Llama 3.2 AUP: rights under s.1(a) not granted to individuals domiciled/companies principally in the EU **for multimodal models** (summary; the exact wording must be read before relying).
- Open questions: whether adapter-only distribution is itself an "AI model" for the naming rule; how "Built with Llama" displays in an on-device app. Owner/legal decision.

### 1.3 Gemma
- Gemma 3 and earlier: custom Gemma Terms of Use (secondary sources: Wikipedia, mindstudio, ibl.ai articles). **Gemma 4:** reported Apache-2.0 (Google DeepMind announcement as quoted in search result title; Gemma overview excerpts). The Gemma terms page (ai.google.dev/gemma/terms) could not be fetched. Treat Gemma 4 as "Apache-2.0 reported, per-repo LICENSE must be read".

### 1.4 Others
- Qwen: Qwen3 GitHub README: "All our open-weight models are licensed under Apache 2.0". Qwen3.5: README defers to the licence file shipped with weights; summaries and secondary sources say Apache-2.0.
- SmolLM3: huggingface/smollm README says Apache 2.0.
- Olmo 3: HF card excerpt "The code and model are released under Apache 2.0"; announcement coverage says all models Apache 2.0 and that data/code/checkpoints are released. olmo-core code is Apache-2.0 (README).
- Ministral 3: reported Apache-2.0 for base/instruct/reasoning (search summaries); unverified.
- Phi-4-mini-instruct: reported MIT (third-party pages); the microsoft/PhiCookBook repo is MIT but that is the cookbook, not weights. Unverified.
- Mistral custom licences exist for other models (MNPL, MRL) per mistral-inference README: never assume by vendor.

## 2. Candidates by size class

"Nominal" parameter counts come from model names. Context lengths are as reported in config excerpts; **usable context on a phone is bounded by KV memory, not by the advertised maximum** (ADR 0011).

### 2.1 ~1-2B class (smallest tier; test fixtures and lightweight device grade)
| Entry | Licence (evidence) | Context | Formats seen | Notes |
|---|---|---|---|---|
| Qwen3-1.7B (`qwen3-1.7b`) | Apache-2.0 (A, README) | unverified | HF, GGUF (README), ExecuTorch (1.7B documented in demo) | plain dense architecture; **chosen for the first experiment** (see FIRST_EXPERIMENT.md) |
| Qwen3.5-0.8B / 2B | Apache-2.0 (A-/B) | 262,144 reported | HF, GGUF (llama.cpp text+vision), MLX | hybrid Gated DeltaNet + full attention (every 4th layer); multimodal checkpoint; llama.cpp text support merged (search); 2B: 24 layers, 2 KV heads, head_dim 256, hidden 2048 (excerpt) |
| Gemma 4 E2B | Apache-2.0 reported (B) | ~128K reported | HF, GGUF, LiteRT-LM | "2.3B effective" per secondary; total params larger due to per-layer embeddings - affects storage |

### 2.2 ~3-4B class
| Entry | Licence | Context | Formats | Notes |
|---|---|---|---|---|
| Qwen3-4B | Apache-2.0 (A) | 40,960 (config excerpt) | HF, GGUF, ExecuTorch (documented) | 36 layers, 8 KV heads, head_dim 128, hidden 2560 (excerpt). Qwen3-4B-2507 refresh exists (not registered) |
| Qwen3.5-4B | Apache-2.0 (A-/B) | 262,144 | HF, GGUF, MLX | 32 layers, 4 KV heads, head_dim 256, hidden 2560; hybrid |
| SmolLM3-3B-Base | Apache-2.0 (A) | 65,536 (128k via YaRN) | HF transformers; GGUF/ONNX not verified | fully open training recipe/data per card excerpt; 6 languages |
| Llama-3.2-3B | Llama 3.2 Community (A, custom) | 131,072 (mirror config) | HF (gated), GGUF, ExecuTorch (QNN/Vulkan tutorials), MLC | 28 layers, 8 KV heads, head_dim 128, hidden 3072 (mirror excerpt); naming/AUP obligations |
| Ministral-3-3B-Base-2512 | Apache-2.0 reported, **unverified** (C) | 262,144 | HF; GGUF not confirmed for 3B | 3.4B LM + 0.4B vision; 26 layers, 8 KV, hidden 3072 |
| Gemma 4 E4B | Apache-2.0 reported (B) | 131,072 | HF, GGUF, LiteRT-LM (.litertlm) | 4.5B effective / ~8B with embeddings; 42 layers, hidden 2560; 5 sliding(512):1 full pattern |
| Phi-4-mini-instruct | MIT reported, **unverified** (C) | 131,072 | HF, ORT GenAI, LiteRT-LM | instruct-only checkpoint |

### 2.3 ~7-9B class (normal, important target)
| Entry | Licence | Context | Formats | Notes |
|---|---|---|---|---|
| Qwen3-8B | Apache-2.0 (A) | 40,960 | HF, GGUF | 36 layers, 8 KV, head_dim 128, hidden 4096 (excerpt) |
| Qwen3.5-9B | Apache-2.0 (A-/B) | 262,144 | HF, GGUF, MLX | 32 layers, 4 KV heads, hidden 4096; hybrid: KV only in 1 of 4 layers => much lower KV cost per token than a dense 8B (estimate to verify on device) |
| Llama-3.1-8B | Llama 3.1 Community (A, custom) | 131,072 | HF (gated), GGUF, ExecuTorch | 32 layers, 8 KV, 128, 4096; reference model in llama.cpp quantize README (Q4_K_M 4.58 GiB) |
| Olmo-3-1025-7B | Apache-2.0 (B) | 65,536 | HF, olmo-core, vLLM | 32 layers, **32 KV heads (no GQA)** => KV cache ~4x per token vs 8-KV-head models of the same depth: poor phone fit at long context; fully open data/code/checkpoints (good for provenance/audit) |
| Ministral-3-8B-Base-2512 | Apache-2.0 reported, unverified (C) | 262,144 | HF; official GGUF repos exist for Instruct (listing) | 8.4B LM + 0.4B vision; 34 layers, 8 KV, head_dim 128, hidden 4096 |

### 2.4 Larger (server/workstation grade; "master" model for distillation)
| Entry | Licence | Notes |
|---|---|---|
| Qwen3.5-27B | Apache-2.0 (A-/B) | released 2026-02-24 (README); Qwen3.6-27B (2026-04-22) and Qwen3.8-27B (2026-08-14) also appear in the Qwen3.5 README release list but are **not registered** (not researched) |
| Gemma 4 31B dense | Apache-2.0 reported (B) | ~30.7B; 256K context reported for "medium" models, unverified for 31B |
| Olmo-3-1125-32B | Apache-2.0 (B) | repo id from a search-result title; context/arch not obtained |
| Llama 3.3 70B / Llama 4 | custom Llama licences | listed in meta-llama/llama-models README (3.3 70B 12/04/2024, Llama 4 Scout/Maverick 4/5/2025); not registered; Llama 4 licence text read (same obligations as above) |

QLoRA of a 27-32B model needs a large-VRAM GPU; resource figures are not measured here.

## 3. Observations for architecture
1. **Do not assume the dense-transformer KV formula.** Qwen3.5 (hybrid linear/full attention), Gemma 4 (sliding-window layers), Olmo 3 7B (full MHA) all deviate. The registry stores layers/kv_heads/head_dim/hidden_size plus free-text `notes`; the qualifier needs a per-family KV model and must fall back conservatively when fields are null.
2. **Multimodal checkpoints** (Qwen3.5, Gemma 4, Ministral 3) include vision/audio components that inflate file size; text-only specialist export may need component stripping and runtime support - unverified for each runtime.
3. **Newest is not automatically the safest.** Qwen3.5 relies on newer llama.cpp ops (GATED_DELTA_NET, PR #19504 per search); Qwen3/Llama 3.x have the most mature GGUF/ExecuTorch paths. Support must be verified per llama.cpp commit by a load-and-generate test in the export step.
4. **Provenance advantage:** Olmo 3 and SmolLM3 publish training data/recipes, useful if the owner wants an auditable lineage including pretraining data (not assessed here).
5. **Default-enabled set (proposal):** Apache-2.0/MIT entries whose licence is verified `yes`; Llama entries opt-in because of naming/AUP/notice obligations; `unverified` entries visible but blocked.

## 4. Sources (all accessed 2026-10-05)
Qwen: https://github.com/QwenLM/Qwen3 , https://github.com/QwenLM/Qwen3.5 , https://raw.githubusercontent.com/QwenLM/Qwen3.5/main/README.md , https://www.therundown.ai/tools/qwen3-5-small (secondary). Llama: https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE , .../llama3_2/LICENSE , .../llama3_2/USE_POLICY.md , .../llama4/LICENSE , https://github.com/meta-llama/llama-models . Apache: https://www.apache.org/licenses/LICENSE-2.0.txt . Gemma: https://github.com/google-deepmind/gemma , https://x.com/GoogleDeepMind/status/2039735446628925907 (search result), https://developers.google.com/edge/litert-lm/models/gemma-4 (search result), https://en.wikipedia.org/wiki/Gemma_(language_model) (secondary). Ministral: https://mistral.ai/news/mistral-3/ , https://huggingface.co/mistralai/Ministral-3-8B-Base-2512 (search results only), https://github.com/mistralai/mistral-inference . SmolLM3: https://github.com/huggingface/smollm . Olmo 3: https://github.com/allenai/OLMo-core , https://www.hpcwire.com/aiwire/2025/11/20/ai2-announces-olmo-3-family-of-open-frontier-language-models/ , https://simonwillison.net/2025/Nov/22/olmo-3/ . Phi: https://github.com/microsoft/PhiCookBook . Config excerpts (search results, not fetched): https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json , .../Qwen3-8B/..., .../Qwen3.5-9B/..., https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512/blob/main/config.json , https://huggingface.co/docs/transformers/en/model_doc/smollm3 .

## 5. Next verification steps
Obtain Hub access (or owner-provided offline copies) and: read each repo's LICENSE/NOTICE/usage policy verbatim and hash it; parse `config.json`; test GGUF conversion + load with a pinned llama.cpp; read Gemma 4 repo terms; read Ministral 3 and Phi-4-mini licence files; have a lawyer review Llama obligations for adapter distribution.
