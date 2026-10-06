# Specialization runbook (desktop / GPU)

Status: written alongside the code, **not yet validated on real hardware**. No real model has been trained
by this repository's automation. Library calls in `factory/llmtrainer/specialize/hf_lora.py` are untested
against installed `torch/transformers/peft/trl`; expect to fix API drift on first contact.

## What runs where

| Stage | Where | Notes |
| --- | --- | --- |
| Ingest, dataset build, leakage checks, license gates, dry-run planning, evaluation harness (with fake generators), packaging, reference consumer | Anywhere (CPU, CI, laptop) | Pure Python, pydantic only |
| LoRA / QLoRA fine-tuning (`llmtrainer train --execute`) | **Desktop/workstation with an NVIDIA GPU** (QLoRA needs `bitsandbytes`/CUDA). ~1-2B model: roughly 8-16 GB VRAM; 7-8B QLoRA: ~16-24 GB. Use `llmtrainer train --dry-run` for the factory's order-of-magnitude estimate (it is an estimate, not a measurement) | Not a phone workload |
| Base-vs-specialist evaluation with real models | Desktop/GPU (or CPU for tiny models, slowly) | Needs a `generate(prompt)` callable per subject |
| Merge, quantize (GGUF etc.), on-device benchmark | Desktop for conversion; **the target phone for benchmarking** | Not implemented in this round |
| Inference of the finished specialist | Phone only if device qualification passes (no degraded device operation); otherwise desktop/server | See `qualify-device` |

## Steps

1. Create the workspace, add sources with explicit rights, `llmtrainer build-dataset <ws>`.
2. Put a verified license entry for the base model in `<ws>/registry/` (or pass `registry_path`). Only
   `yes` permissions with `verification.source_urls` + `verified_on` count as `VERIFIED`.
3. Download the base weights yourself into a local directory (the trainer is offline by default).
4. Write a config (see `factory/llmtrainer/configs/example_lora.json`) and plan:

       llmtrainer train --config my.json --dry-run     # validates, estimates, prints the exact command; launches nothing

5. Train (explicit opt-in; may take hours and a lot of VRAM):

       pip install -e 'factory[train]'                  # plus a CUDA-matched torch
       llmtrainer train --config my.json --execute

   Output: `<ws>/runs/<run_id>/output/{adapter,tokenizer}/` and `training_run.json` (config hash, dataset
   hash, base-model identity, library versions, seed, hardware, timings). Existing runs are never overwritten.
6. Evaluate on the **test split only** with `llmtrainer.evalsuite` (base vs specialist, optionally with
   retrieval), then package with `llmtrainer package-specialist`.

## Guard rails (enforced in code)

* Non-`VERIFIED` base-model license: refused. `--allow-unverified-license-for-local-experiment` permits
  `UNVERIFIED`/`CONDITIONAL` only (never an explicit `no`), is recorded in the TrainingRun, and **permanently
  blocks packaging/export** of that run.
* The trainer receives train/validation rows only; the test split loader raises.
* No weights are downloaded unless `--allow-download` is passed.
* A completed run directory is never silently overwritten.

## Not executed in the development sandbox

Real training, real model loading, QLoRA/bitsandbytes, GPU resource measurement, any quantization,
any on-device benchmark, and any claim that a specialist beats its base model.
