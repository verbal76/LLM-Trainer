# First Reproducible End-to-End Experiment (design only; nothing has been run)

Goal: prove the pipeline boundaries, not domain value. Chain to demonstrate:

`source -> processed corpus -> dataset -> LoRA/QLoRA fine-tune -> unseen evaluation -> versioned artifact -> export validation`

No real training was executed in this documentation round. All resource numbers below are order-of-magnitude estimates from parameter counts, not measurements. Do not claim improvement until the evaluation (section 5) is actually run.

## 1. Choices

| Item | Choice | Reason |
|---|---|---|
| Corpus | **Fully synthetic fixture** "Aurelia-7 Maintenance Guide" (fictional appliance), generated deterministically by `factory/fixtures/make_synthetic_corpus.py` from a seed (CC0 by our own authorship; no third-party content, no rights ambiguity) | Rights are clean by construction; held-out facts are guaranteed unseen because they exist nowhere else; the model cannot know them from pretraining, so exact-fact recall measures retrieval/training honestly |
| Size | ~40 synthetic "chapters", ~400 sections, ~1,500 atomic facts (torque-like values, intervals, fault codes), ~120 procedures, ~60 diagnostic decision trees; fictional entities only | tiny; runs in seconds on CPU for ingest/corpus/dataset stages |
| Base model | `Qwen/Qwen3-1.7B` (registry `qwen3-1.7b`, Apache-2.0; Qwen3 README states all open-weight models are Apache-2.0, 2026-10-05). Secondary run: `Qwen/Qwen3.5-0.8B` (hybrid architecture) to exercise the architecture-aware KV logic | plain dense architecture with the most mature GGUF/ExecuTorch paths; small enough for one consumer GPU. Caveat: Qwen3-1.7B layers/heads/context were not verified - the first pipeline step must parse the downloaded config.json and fill the registry |
| Method | LoRA (bf16) as default; QLoRA (NF4) as a variant test | ADR 0005 |
| Retrieval | FTS5-only reference package built from the same synthetic corpus (no vectors in v1 experiment) | ADR 0010 |
| Eval | custom domain eval + small general regression slice | ADR 0007 |
| Export | GGUF F16 + Q8_0 + Q4_K_M via pinned llama.cpp; adapter as GGUF LoRA | ADR 0008 |

## 2. Split design (leakage-safe by construction)
- Synthetic generator emits each chapter with a `split_group`. Assignment is by chapter and by *fictional entity*: 70% train, 10% val, 20% test chapters. Test chapters mention components/entities that never appear in train/val.
- Example generation happens after the split; examples inherit the split.
- A deliberately injected **leak canary** (a duplicate passage copied into test and train in a separate "negative-control" fixture) must be caught by the leakage gate; the gate test fails the pipeline if not detected.
- Test set hash frozen in the Dataset Manifest.

## 3. Stage list with commands (shape; flags provisional until the CLI exists)

```bash
# 0. environment (CPU is enough through step 4)
python -m venv .venv && . .venv/bin/activate
pip install -e "factory[dev]"            # pinned via lock file
llmt --version && llmt env snapshot --out runs/env.json

# 1. workspace + project
llmt workspace init ./ws
llmt project create ./ws --name aurelia-demo --config factory/experiments/exp001/project.yaml

# 2. source ingestion (synthetic fixture generated deterministically)
python factory/fixtures/make_synthetic_corpus.py --seed 1337 --out ws/inbox/
llmt source add ./ws ws/inbox/ --rights-file factory/experiments/exp001/rights.yaml   # status: owned-synthetic, train+redistribute allowed
llmt source ingest ./ws --parser plaintext        # no Docling needed for text fixture

# 3. corpus processing
llmt corpus build ./ws --config factory/experiments/exp001/corpus.yaml
llmt corpus report ./ws --latest                 # dedupe/noise/quality report

# 4. dataset generation (template-based generator for determinism; no model call, no network)
llmt dataset build ./ws --config factory/experiments/exp001/dataset.yaml --seed 1337
llmt dataset check ./ws --latest --leakage       # must pass; canary fixture must fail
llmt reference build ./ws --latest               # FTS5 reference package

# 5. license gate (reads registry/base-models/qwen3-1.7b.json)
llmt model check Qwen/Qwen3-1.7B --for train,export --acknowledge-conditions

# 6. training  (GPU needed from here; manual / owner-authorized)
llmt train run ./ws --dataset latest --base qwen3-1.7b \
     --config factory/experiments/exp001/train_lora.yaml --seed 1337
# train_lora.yaml (recorded in the Training Run): r=16, alpha=32, dropout=0.05,
# target_modules=all-linear (validate for architecture), lr=2e-4 cosine, warmup 3%,
# epochs=3, max_seq_len=1024, bf16, per-device batch 8 x grad-accum 4, packing off.

# 7. evaluation on unseen test split
llmt eval run ./ws --model base:qwen3-1.7b           --suite exp001-domain --split test
llmt eval run ./ws --model base:qwen3-1.7b+ref       --suite exp001-domain --split test
llmt eval run ./ws --model run:latest                --suite exp001-domain --split test
llmt eval run ./ws --model run:latest+ref            --suite exp001-domain --split test
llmt eval regress ./ws --model run:latest --vs base:qwen3-1.7b --suite lm-eval-slice
llmt eval compare ./ws --runs latest~4..latest  # paired differences + CIs, no single score

# 8. optimize / quantize (pinned llama.cpp)
llmt optimize gguf ./ws --run latest --merge --quants F16,Q8_0,Q4_K_M --imatrix auto
llmt eval run ./ws --model gguf:Q4_K_M --suite exp001-domain --split test   # quantized specialist is evaluated too

# 9. export + validation
llmt export package ./ws --run latest --grade desktop --out exports/
llmt export validate exports/<export-id>        # schema, hashes, license bundle, runtime manifest, GGUF load+generate smoke test
llmt export consume-check exports/<export-id> --reference-consumer   # import by a minimal consumer without factory code
```

## 4. Resource estimates (unmeasured; to be replaced by measured values after the first run)
- Steps 1-4, 9 (validation without GPU): CPU, <1 GB RAM, seconds to a few minutes; runs in GitHub Actions on hosted runners (this is the CI integration test, using a ~1% corpus and a tiny random-init model to exercise the trainer for 5 steps).
- Step 6 LoRA bf16, Qwen3-1.7B: weights ~3.4 GB (1.7B x 2 bytes); with LoRA/optimizer state, gradient checkpointing and seq 1024 expect roughly 10-16 GB VRAM -> a single 16-24 GB GPU. QLoRA NF4 expect ~5-8 GB. Training tokens: ~1,000 examples x ~400 tokens x 3 epochs ~ 1.2M tokens -> order of minutes to tens of minutes on one modern GPU. Disk: <20 GB incl. HF cache, adapters, GGUFs (F16 ~3.4 GB, Q8_0 ~1.8 GB, Q4_K_M ~1.1 GB, all approximate).
- Step 8: llama.cpp conversion/quantization CPU-capable, RAM > model size.
- Cost: owner-authorized GPU only; one cloud-hour-scale job. No spend is made by this document.
- Larger follow-up (7-9B QLoRA) is a separate experiment: expect ~16-24 GB VRAM for QLoRA with short sequences (rough), not part of experiment 001.

## 5. Evaluation definition (all on the frozen test split)
1. **Held-out fact questions without retrieval** (expected: base ~ chance on fictional facts; specialist may improve only if facts leak through training - test split facts are *not* in training, so this measures the *absence* of memorization/hallucination: the specialist must say "not in my knowledge" rather than invent). Metric: correct / abstain / unsupported-claim rate.
2. **Held-out fact questions with the reference package** (FTS5 top-k in the prompt): metric exact-answer accuracy and citation correctness (cited chunk contains the answer).
3. **Procedure/diagnostic format tasks** on test chapters: structured output validity and step coverage (rule-checked against the synthetic ground truth).
4. **Train-seen control** (reported separately, not graded as success).
5. **General regression slice** via lm-eval (small subset, `--limit`), base vs specialist vs quantized.
6. **Operational:** tokens/s, peak RSS, file size per quant (workstation only; phone benchmarks are a separate Android task).
Statistics: fixed seeds, paired bootstrap CIs; report per-category, never one aggregate. Success of the *pipeline* = all stages produce valid, hash-consistent artifacts and the leakage canary is caught; success of the *model* is not claimed.

## 6. Reproducibility checklist (recorded in manifests)
Seeds (python/numpy/torch/data order), library versions (lock), CUDA/driver/GPU, base-model repo revision + weights hash + license hash, dataset hash, config hash, llama.cpp commit, quant types, imatrix hash, judge identity (none here), wall-clock and nondeterminism flag. Re-running dataset/corpus/reference steps must yield byte-identical manifests.

## 7. Export validation criteria
Package manifest validates against schema v1; every file hash matches; license/attribution bundle contains the Apache-2.0 text, upstream copyright notice for Qwen3, and a "modified" statement; provenance manifest lists all sources (synthetic-owned); runtime manifest names llama.cpp commit; GGUF loads and generates deterministic-seed output; a minimal reference consumer (no factory imports) opens the package and answers a reference query with a citation.

## 8. Risks / uncertainties
- Qwen3-1.7B architecture and context length not verified in this round; first step must record them.
- Target-module choice for LoRA (`all-linear`) must be validated; hybrid models (Qwen3.5-0.8B) may need a custom list.
- Synthetic data is too clean to test OCR/table/noise handling; a later experiment needs authorized real documents.
- Tiny-model GPU nondeterminism can break byte-identical weights; manifests record this.
- `llmt` flags above are a design sketch; the CLI is not implemented.
