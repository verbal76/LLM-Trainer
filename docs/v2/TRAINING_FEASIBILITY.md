# On-device training feasibility (measured)

**Scope and honesty.** Everything below was measured on one Linux **x86_64 VM (4 vCPU Xeon @2.8 GHz with AVX-512, 15 GB RAM, noisy
neighbours)**, CPU backend only, llama.cpp `0c1e570` + our patch. **No ARM device, emulator or NDK run happened in this sandbox.**
Phone numbers are *extrapolations* (formulas at the end) and must be replaced by measurements from the Android qualifier.
Wall times on this VM fluctuate by up to 2x (and rarely 10x) between runs; tables use medians/steady-state steps.
Reproduce with `native/engine/tools/bench_engine.py` and `native/engine/tests/test_engine.py`
(`engine.yml` runs both on the CI runner and also real pre-trained GGUFs from huggingface.co, sha256-verified against the HF API).

## What is proven (all passing locally, 24 tests, no mocks of the ML path)
Pipeline PHONE-style: *model file -> data -> training -> learned artifact -> reload -> inference -> evaluation*:

| Claim | Evidence (test) |
|---|---|
| Load, deterministic greedy/seeded generation, UTF-8-safe streaming, async cancel, chat template, tokenize, score | `test_generation_*`, `test_streaming_utf8_*`, `test_cancel_*`, `hag_unit` |
| Parameter-changing training works and the learned artifact is a small patch; base file byte-identical | `test_specialization_*` (patch is a fraction of the base size when only the last layers are tuned) |
| Behaviour changes and matches the trained facts | greedy exact-match of trained facts **0/8 (base) -> 7/8 (specialist)** |
| Generalisation of the *format* to unseen entities | held-out fact sentences NLL **3.78 -> 2.80** nats/token |
| **Cost, not hidden** | unrelated prose NLL **3.00 -> 7.06** (+4.05 nats): 150 epochs on 12 sentences with a constant LR forgets general text almost completely (this base is a 2.5M-param model pre-trained by the engine on docs; a real model will degrade less but the mechanism is the same). LoRA r=8 (40 epochs): held-out facts 3.78 -> 2.14, prose +2.41 nats. Evaluate with a general set before shipping any specialist |
| SIGKILL mid-run then rerun -> **bit-identical patch** to an uninterrupted run (full and LoRA) | `test_sigkill_*`, `test_lora_sigkill_*` |
| Truncated / bit-flipped / foreign checkpoint is rejected (previous one used, else restart, never partial load) and the result is still identical | `test_corrupt_checkpoint_*` |
| Cancel keeps a valid checkpoint; completed run is idempotent; same inputs -> same patch hash; different seed -> different | `test_cancel_keeps_*`, `test_completed_*`, `test_training_is_deterministic_*` |
| Wrong base / corrupt / truncated patch rejected, model stays unpatched | `test_patch_validation_*` |
| Independent consumer (Python) merges base+patch; scores bit-identically to the engine | `test_reference_consumer_*` |
| Memory pre-flight refuses before any work | `test_memory_preflight_*` |
| Quantized bases: Q8_0 and Q4_0 train (last layers via F32 working copy; LoRA with a frozen Q4_0 base); patch stored in base type; cost of re-quantizing measured by the engine: held-out NLL F32 weights vs deployed type: **Q8_0 1.0374 -> 1.0365 (none), Q4_0 0.8973 -> 0.9348 (+0.04)** | `test_quantized_*`, `test_q4_*`, `test_lora_*` |
| Q8_0/Q4_0 base inference quality: prose NLL 3.0043 (F32) vs 3.0042 / 3.0046 | `test_quantized_bases_run_and_train` |

Not yet proven here: real pre-trained models (CI job `real-model`, needs internet), ARM/Android execution, thermal/sustained behaviour.

## Answers to the design questions
**(a) Full vs partial tuning.** Both work via `param_filter` on tensor names. Memory = F32 weights + 12 B x trainable params (4 grad
accumulator + 8 Adam) + activations. Backward only runs through layers at/after the first trainable one, so `trainable_last_layers=N`
saves optimizer memory *and* compute (135M: all layers 0.75 s/step, last 2 layers 0.18 s/step at ctx 64). Token embeddings are **not
trainable in llama.cpp** (upstream FIXME); tied-embedding models (SmolLM2) therefore never train the head.

**(b) Quantized base.** Backward through frozen Q8_0/Q4_0 matmuls works (activation gradients use `out_prod` with a quantized
operand). Trainable tensors need F32 weights, so a *working copy* GGUF is built in `work_dir` (trainable layers dequantized, rest
verbatim): disk = quantized size + 4 B x trainable params; the frozen early layers stay quantized and forward-only.
Norm-only tuning of a fully frozen quantized model also works but is too weak to matter. **Repacked CPU weight layouts must be
disabled for training** (they have no OUT_PROD) - done (`use_extra_bufts=false`).

**(c) Patch.** Done: GGUF containing only the changed tensors, converted back to the base tensor type (Q8_0/Q4_0/F16/BF16 non-imatrix
types), plus base sha256/size, tensor identity, hyper-parameters, dataset/token hashes, step counts, losses. `hag_model_apply_patch`
swaps private buffers into the loaded (mmap'd) model; the base file is never touched. Format: `docs/v2/ENGINE.md`.

**(d) Deterministic order.** Seeded SplitMix64 (not `std::shuffle`): document order per epoch and window order per epoch. Documents are
re-packed every epoch; **a fixed packing let the model memorise neighbours instead of facts** (loss 0.00 but 3/8 recall) - found by
testing, fixed (algorithm version 2).

**(e) Checkpoints.** Atomic (tmp + fsync + rename + dir fsync), sha256 trailer + manifest, 2 generations, exact state
(weights, Adam m/v, Adam iteration, position, loss stats). Bit-exact resume holds for equal `n_threads` (float reduction order).
Cost: `12 B x trainable` per checkpoint plus hashing (~0.3 GB/s here): 135M full-weight tuning = 1.3 GB per checkpoint, so use
`checkpoint_every_steps` thoughtfully (LoRA/last-layers checkpoints are MBs).

**(f) Cancel.** Callback != 0 stops at the next sequence boundary; a checkpoint at the last completed step is written first.

**(g) Pre-flight.** `hag_train_estimate` / `max_memory_bytes`. Measured peak RSS vs the estimate (the estimate must not be below the
measurement; ratio = estimate/measured):

| config (135M-shape, 32k vocab) | peak RSS MB | estimate MB | ratio |
|---|---|---|---|
| all layers, ctx 64 | 1835 | 2000 | 1.09 |
| all layers, ctx 128 | 1953 | 2193 | 1.12 |
| last 2, ctx 64 | 629 | 745 | 1.18 |
| last 4, ctx 256 | 847 | 1038 | 1.23 |
| LoRA r8, F32 base, ctx 128 | 762 | 944 | 1.24 |
| **LoRA r8, Q4_0 base, ctx 128** | **354** | 536 | 1.51 |
| LoRA r8, Q4_0 base, ctx 256 | 592 | 971 | 1.64 |
| last 4 on Q4_0 base, ctx 128 | 398 | 538 | 1.35 |
| 360M-shape, last 2, ctx 64 | 1629 | 1789 | 1.10 |

Small models are dominated by a constant 64 MB allowance (ratio 2-3x for 0.5-5M params). Never below the measurement in any run.

**(h) Held-out loss** is tracked separately (whole documents, never in the training stream), recorded first/last in the patch.

**(i) LoRA.** llama.cpp has no LoRA trainer, but it applies adapters in the graph builder, so a small patch (adapter tensors
marked trainable) was enough: it **works and is tested** (F32 and Q4_0 bases, resume bit-identical, standard llama.cpp adapter output).
This is the phone-relevant mode: base frozen and mmap'd/quantized, optimizer state ~16 B x (2.4M adapter params) = 39 MB for a
135M model, **354 MB total RSS at ctx 128** vs 1953 MB for full-weight tuning. Limits: activations still scale with *all* layers
below the lowest adapted one; rank/alpha/lr need tuning (lr 5e-3 diverged in one probe, 2-3e-3 worked); with 12 sentences LoRA r8
recalled fewer facts than last-2-layer tuning at the same epochs (still reduced loss and improved held-out NLL).

## Throughput (x86, 4 threads; NOT phone numbers)
Inference, F32 random-weight models of the stated shape (prompt ~110 tokens, 48 generated, median of 3):

| model | params M | file MB | gen tok/s | prompt tok/s | RSS MB |
|---|---|---|---|---|---|
| 1m | 0.5 | 2 | 894 | 43040 | 15 |
| 5m | 2.5 | 10 | 628 | 16840 | 25 |
| 40m | 26 | 100 | 162 | 3666 | 124 |
| 135m F32 | 125 | 476 | 48 | 840 | 507 |
| 135m Q8_0 | 125 | 127 | 101 | 787 | 158 |
| 135m Q4_0 | 125 | 68 | 197 | 927 | 164 |

(15m row in the raw file shows 0.0 tok/s: a stop-at-EOS artefact of random weights, ignore.) Tiny models are faster with fewer threads.

Training (ctx 64, steady-state step time, AdamW):

| model | trainable | params M | step s | tokens/s | peak RSS MB | patch MB |
|---|---|---|---|---|---|---|
| 5m | all | 2.4 | 0.023 | 2767 | 58 | 9 |
| 40m | all | 25.8 | 0.188 | 341 | 441 | 98 |
| 40m | last 2 | 4.3 | 0.059 | 1093 | 176 | 16 |
| 135m | all | 106 | 0.754 | 85 | 1835 | 405 |
| 135m | last 2 | 7.1 | 0.175 | 365 | 629 | 27 |
| 360m | last 2 | 19.7 | 0.337 | 190 | 1629 | 75 |
| 135m | LoRA r8 / Q4_0 base (ctx 128) | 2.4 | 2.7 | 47 | 354 | 10 |

(The last row: 7 steps in 18.8 s wall including model load; ctx 128; steady-state is better than this.)
Time per epoch = steps/epoch x step time, steps/epoch = corpus tokens / n_ctx / sequences-per-step. Pre-training the 2.5M-param
fixture for 2 epochs of 286 K characters (4500 steps at ctx 128, 3 threads) took 6 minutes.

## Planning for an ARM phone (extrapolation, unmeasured)
* **FLOPs per step** ~ `2 N T` (forward, all layers) + `4 N_b T` (backward over the N_b parameters at or after the first trainable layer),
  T = tokens/step. Here sustained throughput came out at ~60-100 GFLOP/s (4 AVX-512 threads, F32 training). A phone big-core cluster is
  plausibly 20-50 GFLOP/s sustained for F32 NEON before thermal throttling - i.e. 2-5x slower than this VM; measure it.
* **Memory**: full/partial tuning `4 B x N (F32 weights incl. working copy) + 12 B x N_trainable + activations`; LoRA on a quantized base
  `bytes(base) + 16 B x N_adapter + activations`, activations ~ `4 B x T x layers_with_grad x (14 d + 5 d_ff + 4 d_kv) + 8 B x heads x T^2 x layers + 20 B x T x vocab`.
* **Believed limits on a 15 GB phone** (budget ~6-8 GB for the job after OS/app reserve, leaving the device responsive):
  * full-weight F32 tuning: <= ~0.3-0.5 B params all layers; larger models only with `trainable_last_layers` (still F32 working copy of those layers) - e.g. 1.5B with the last 4 of 28 layers ~ 6 GB + activations;
  * **LoRA on a Q4/Q8 base: memory allows 3B (~2 GB weights) comfortably and 7B (~4 GB + ~2.6 GB activations at ctx 128) barely**;
  * **time is the binding limit, not memory**. Measured LoRA-on-Q4_0 at ctx 128: ~2.7 s/step for the 135M shape (backward must traverse every layer through quantized weights), i.e. roughly 20 ms per million parameters per step: ~20 s/step at 1B and ~2.5 min/step at 7B on this VM, x2-5 on a phone. Last-layers tuning is much cheaper per step (135M: 0.18 s at ctx 64). Useful specialization (hundreds of steps) is realistic for <= ~0.5B models in minutes-to-an-hour, 1-1.5B takes hours (overnight on charge), and is **not practical for >= 3B** even though the memory fits.
* Sustained thermal throttling, battery drain and background kills are not measured at all; run the qualifier before claiming a profile fits.
