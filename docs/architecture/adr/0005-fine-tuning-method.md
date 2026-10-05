# ADR 0005: Fine-tuning method = LoRA/QLoRA SFT first; exact facts go to retrieval

- Status: proposed
- Date: 2026-10-05

## Decision
Default method is supervised fine-tuning with LoRA (bf16/fp16 base) on GPUs with sufficient VRAM and QLoRA (4-bit NF4 base via bitsandbytes) when VRAM-constrained. Adapters are the primary artifact; merging is a separate, license-gated step. Preference tuning (DPO/GRPO via TRL) and full fine-tuning are later options behind the same orchestrator. Continued pretraining on raw text is not a default (`CLAUDE.md`: never blindly convert text to training data).

## Rationale
- Fits the portable-output goal: adapter + base reference is small, versioned and keeps the base license obligations explicit. llama.cpp can convert PEFT LoRA adapters to GGUF (`convert_lora_to_gguf.py`: "Convert a Hugging Face PEFT LoRA adapter to a GGUF file", args `--base`, `--outtype`), so an adapter can ship without a merged copy where the runtime applies adapters.
- Matches `CLAUDE.md` principle 4: fine-tuning learns domain language and behavior; exact facts (torques, clearances) come from the reference package.

## Alternatives evaluated
- Full fine-tune: higher VRAM, produces a derivative full-weight model whose redistribution is governed by the base license; reserve for a later "maximum quality" mode with licence gate.
- Continued pretraining: risks memorizing/regurgitating protected text, hard to prove value; rejected as default.
- Prompt-only/RAG-only: does not meet the "specialist model" product definition; retained as the baseline in evaluation (base + retrieval vs specialist + retrieval).
- Distillation: supported later as an optimization (`CLAUDE.md` principle 9); Llama licenses state outputs used to improve another distributed model trigger naming rules (per fetched Llama 3.1 summary) - license gate must model this.

## Hybrid-attention caveat
Qwen3.5 uses Gated DeltaNet + full attention hybrid layers (config excerpt). LoRA target-module selection must be validated per architecture; no assumption of a standard q/k/v/o set.

## Consequences
Record: rank, alpha, target modules, dropout, learning rate, seed, quantization, packing, max length, and data order hash in the Training Run contract.
