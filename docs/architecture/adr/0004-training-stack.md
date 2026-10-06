# ADR 0004: Training stack = Hugging Face transformers + PEFT + TRL (+ bitsandbytes)

- Status: proposed
- Date: 2026-10-05

## Context
We need a maintained, inspectable stack for LoRA/QLoRA SFT across Qwen, Llama, Gemma, Mistral, SmolLM, OLMo-class models, with deterministic seeding and recorded configuration.

## Decision
Core: `transformers` (5.18.0 on PyPI, Apache-2.0), `peft` (0.21.2, uploaded 2026-10-01, Apache-2.0), `trl` (1.14.1, Apache-2.0, provides SFTTrainer/DPOTrainer/GRPOTrainer), `bitsandbytes` (0.50.2, MIT, 4-bit QLoRA), `datasets` (5.0.1, Apache-2.0), PyTorch (2.14.1). All versions from PyPI JSON read 2026-10-05; re-pin at implementation time. The Training Orchestrator wraps these behind our own training-config schema; other frameworks may be added as plugins that consume the same config.

Why: first-party model definitions track new architectures first (e.g. Qwen3.5's hybrid Gated DeltaNet, Gemma 4), the exact stack is what base-model cards and TRL docs assume (https://huggingface.co/docs/trl/peft_integration, via search), and all components are permissively licensed.

## Alternatives evaluated
| Option | Evidence | Verdict |
|---|---|---|
| torchtune | Reported no longer actively maintained; development wound down in 2025 (https://github.com/meta-pytorch/torchtune issue "[IMPORTANT] The future of torchtune" #2883, via search) | Rejected: deprecated |
| Axolotl 0.20.0 (Apache-2.0, PyPI 2026-09-30) | Config-driven, active | Not the core: would put a third-party YAML schema between us and our manifests. Candidate *plugin backend*; our config can emit Axolotl YAML |
| LLaMA-Factory 0.9.5 (Apache-2.0, PyPI 2026-05-30, Python >=3.11) | Broad model coverage, has its own UI | Not the core: duplicates our UI/CLI concerns; candidate plugin |
| Unsloth 2026.9.14 | PyPI lists dual license: core Apache-2.0, Studio UI AGPL-3.0 | Optional accelerator plugin only (speed/memory); keep AGPL components out of the shipped product; verify per-model support before relying on it |
| Custom PyTorch loop | Maximum control | Rejected: re-implements well-tested trainers for no product gain |
| Cloud fine-tuning APIs | Convenient | Rejected as default: would transmit private corpora externally (violates `CLAUDE.md` security rule); weights often not exportable |

## Consequences
We own: training-config schema, run recorder (seed, versions, CUDA/driver, GPU, hashes), resume logic, and adapter/artifact layout. Upstream churn risk (transformers 5.x, TRL 1.x APIs) is mitigated by pinning and by integration tests on a tiny model.

## Uncertainty
Multimodal base models (Qwen3.5, Gemma 4, Ministral 3 are multimodal checkpoints) need verified text-only fine-tune paths in TRL/PEFT; not tested in this documentation round.
