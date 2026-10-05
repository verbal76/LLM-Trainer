# HAG native engine (`native/engine`)

C++17 wrapper over a **pinned, patched llama.cpp** (CPU backend only) behind the stable C API in
`native/engine/include/hag_engine.h`. Generic: nothing in it is LLM-Trainer specific.

| Piece | Where |
|---|---|
| C API contract | `native/engine/include/hag_engine.h` (names stable; later changes additive, `struct_size`-guarded) |
| Sources (inference, patch apply / training) | `native/engine/src/{engine,train,hag_util}.cpp` |
| llama.cpp pin | `native/llama.cpp.pin` (commit `0c1e57098bba43ac29e6e3b677cdceebdd22334f`) |
| Fetch + patch | `native/fetch-llama.sh <dest>` (idempotent; `LLAMA_CPP_SRC=<tree>` for offline/sandbox) |
| Patch to llama.cpp | `native/patches/0001-hag-training-hooks.patch` |
| CLI / tests / tools | `hag_cli`, `native/engine/tests/`, `native/engine/tools/` (never part of the shipped library) |
| CI | `.github/workflows/engine.yml` |

Build: `native/fetch-llama.sh native/.deps/llama.cpp && cmake -S native/engine -B build/engine && cmake --build build/engine -j`.
Targets: `hag_engine` (shared, exports only `hag_*`: 19 symbols, enforced in CI), `hag_engine_static`, `hag_cli`, `hag_unit`.
The runtime/Android build compiles `native/engine/src/**` itself against `llama` + `ggml`; non-API symbols are
hidden with `#pragma GCC visibility push(hidden)`, so it does not depend on the embedding build's flags.

## Why a patch to llama.cpp
llama.cpp ships a training API (`llama_opt_init`/`llama_opt_epoch`) but, at the pinned commit, it cannot be used for a
product as is. The patch (all additive, `native/patches/0001-...`) fixes or exposes:

1. **Gradient accumulators are never zeroed** between steps in the non-static-graph path llama uses (upstream
   `llama-finetune` therefore trains on a running sum of gradients). `ggml_opt_zero_grad_accs()` + our loop zero them per window.
2. **Forward-only (validation) evals advanced the accumulation phase**; fixed.
3. **Thread count ignored in training** (ggml default of 4 threads was used whatever `-t` said); fixed in `llama_opt_sequence`.
4. **K/V projections received no gradient unless `n_ctx` was a multiple of 256** (KV padding). The cache now reports an exact
   `n_kv` while training, so any sequence length works.
5. `llama_opt_sequence()` (one sequence = one forward/backward, no dataset/epoch machinery, usable with our own deterministic
   data order, cancel points and checkpoints), `ggml_opt_{get,set}_iter`, `ggml_opt_n_state/state_at`, `ggml_opt_set_state_init_cb`
   (exact optimizer-state save/restore), `llama_opt_params.opt_period` (gradient accumulation), tensor accessors
   (`llama_model_tensor_*`), LoRA adapter tensors are made trainable (`llama_adapter_lora_tensor`).

## API usage
```c
hag_engine_init();
hag_model *m; hag_model_load("model.gguf", /*mmap*/1, progress_cb, user, &m);
hag_model_apply_patch(m, "specialist.patch");              // optional; before sessions
hag_session *s; hag_session_params sp = {sizeof sp, 2048, 0, 0}; hag_session_new(m, &sp, &s);
hag_sample_params p = {sizeof p, 0.8f, 40, 0.95f, 0.f, 1.1f, /*seed*/42, /*max_new*/256};
hag_generate(s, prompt, &p, on_piece, user, &stats);        // on_piece gets complete UTF-8 only
hag_score(s, "held-out text", &nll, &n);                    // mean NLL (nats/token)
hag_train(base, texts, n, &tp, work_dir, "out.patch", on_event, user);   // blocks: run on a worker thread
```
`hag_cli` exposes all of it (`info|generate|score|tokenize|train|estimate|patch-info`), JSON on stdout.

Semantics worth knowing (also in the header): sampling is greedy at `temperature<=0`; a fixed seed reproduces the stream;
`hag_cancel` is async-safe and aborts inside the running decode (cancelled prompt phases leave the session unchanged); pieces
passed to the sink are always valid UTF-8 (partial multi-byte characters are held back, invalid bytes become U+FFFD);
`hag_score` windows long text into 256-token windows (1-token overlap) and clobbers the session KV; chat formatting uses the
model's own template **if it is one of llama.cpp's built-in templates** (no Jinja engine linked) and otherwise returns
`HAG_ERR_UNSUPPORTED` rather than guessing.

## Threading and memory model
* A model may be shared by sessions. One session = one thread at a time, except `hag_cancel`. `hag_train` blocks its caller.
* `n_threads=0` picks `min(8, 3/4 of cores)` (leave headroom for the UI). **Tiny models run faster with fewer threads**
  (measured: 0.5M-param model 1500 tok/s at 1 thread vs 360 at 4); pin the count for training: the float summation order, hence
  bit-exact resume, depends on it.
* Inference: weights are mmap'd (evictable page cache) when `use_mmap=1`; plus KV cache (F16) and compute buffers per session.
  `hag_apply_patch` of a *replace* patch allocates private memory for the patched tensors only.
* Training memory (weights tuning): `F32 weights of the working model + 12 B x trainable params (4 grad accumulator + 8 Adam) +
  activations + logits`. With **LoRA** the base stays frozen (and mmap'd, quantized): `16 B x adapter params + activations`.
  `hag_train_estimate` returns the breakdown; `max_memory_bytes` makes `hag_train` refuse (HAG_ERR_OOM) before writing anything.

## Training design (what runs)
* **Data packing** (deterministic): each text -> tokens with BOS (if the model adds one), special tokens parsed, `EOS` appended;
  documents concatenated into one stream. Validation documents are chosen by a seeded permutation (`val_fraction`, whole
  documents, no leakage between windows). Windows of `n_ctx` inputs + 1 shifted label, stride `n_ctx`, last window anchored at the
  end (may overlap). `n_ctx` shrinks to fit tiny corpora (and to the validation stream); recorded as `hag.train.n_ctx`.
  Per epoch the windows are shuffled with SplitMix64 (platform independent), `n_batch/n_ctx` windows are accumulated per
  AdamW step (lr constant, betas 0.9/0.999, eps 1e-8, no weight decay), the remainder of the epoch is dropped (differently each epoch).
* **What is trained**: all blocks, or the last `trainable_last_layers` blocks, plus final norm; `train_embeddings` adds an *untied*
  `output.weight` (token embeddings are not trainable in llama.cpp; tied-embedding models like SmolLM2 ignore the flag - reported
  as `train_embeddings_effective`). Only the `llama` architecture is validated; others get `trainable=false`.
* **Quantized bases**: trainable tensors are dequantized into an F32 *working copy* GGUF in `work_dir` (the other tensors are
  copied verbatim and run forward-only in their quantized type), trained, then re-quantized to the base type for the patch.
  The engine measures that cost (`requant_eval.json`: held-out NLL with F32 vs deployed-type weights). Types that need an imatrix
  (IQ*) are refused. Repacked weight layouts are disabled for training (backward needs the plain layout).
* **LoRA** (`lora_rank>0`): rank-r adapters on q/k/v/o/gate/up/down of the trainable layers (A ~ U(+-1/sqrt(n_in)), B = 0, scale
  alpha/r). Backward flows through the frozen (even Q4_0) weights. Patch = standard llama.cpp LoRA GGUF.
* **Checkpoints**: `work_dir/ckpt-<step>.bin` = header (fingerprint, step, next epoch/position, Adam iteration, loss stats) +
  per trainable tensor `weights | m | v` (F32) + sha256 trailer; written to `.tmp`, fsync, rename, directory fsync; two kept;
  `ckpt.manifest` records `step size sha256 file`. Resume verifies the trailer, the manifest and the run fingerprint
  (base sha256, text hash, token-stream hash, all hyper-parameters incl. seed, n_threads, LoRA config, algorithm version); a bad
  file is renamed `.corrupt` and the previous one tried, else the run restarts. Cancel (callback != 0) writes a checkpoint first.
  A finished run is idempotent (same fingerprint in the existing patch => no-op). `train_run.log` in work_dir records every decision.
* A non-finite loss aborts with `HAG_ERR_INTERNAL` (latest valid checkpoint kept).

## Patch format v1 (GGUF)
Common keys: `hag.patch.format_version=1` (u32), `hag.patch.kind` (`replace`|`lora`), `hag.patch.base.{arch,sha256,size_bytes,n_tensors}`,
`hag.patch.payload.sha256` (sha256 over all tensor payload bytes in file order), `hag.patch.run_fingerprint`, `hag.patch.engine`,
`hag.train.*` (n_ctx, tokens_per_step, epochs, steps, learning_rate, seed, threads, trainable_last_layers, dataset_sha256,
token_stream_sha256, n_*_docs/tokens/sequences, train_loss_first/last, val_loss_first/last, lora_rank/alpha ...).
No timestamps or run-time data: identical inputs give a byte-identical patch (this is what the resume tests assert).
* **kind=replace**: `general.architecture = "hag-patch"`; tensors have the base names, base types and shapes (re-quantized to
  the base type) and contain ONLY the changed tensors.
* **kind=lora**: `general.architecture = <base arch>`, `general.type = adapter`, `adapter.type = lora`, `adapter.lora.alpha`,
  tensors `<base tensor>.lora_a/.lora_b` (F32): loadable by stock llama.cpp (`--lora`) as well.

Apply (`hag_model_apply_patch`): reject unless arch, base size and **full-file sha256 of the base** match, every tensor matches the
base's type/shape, and the payload hash verifies; nothing is modified on failure (`HAG_ERR_CORRUPT`). The base file is never
written. Cost: hashing the base once per model handle (plain C sha256, about 0.3 GB/s on this x86; slower on phones - ARM SHA2
intrinsics are a possible later optimisation). `native/engine/tools/apply_patch.py` is an independent reference consumer
(verification + merge into a standalone GGUF); the tests prove it scores bit-identically to the engine.

## Android notes
* No exceptions cross the API; no `std::filesystem`; POSIX `fsync/rename/opendir` only; no AVX assumptions (ISA is the embedding
  build's choice: `runtime/src/main/cpp/CMakeLists.txt` sets armv8.2-a+dotprod+fp16 / SSE4.2).
* Link targets `llama` and `ggml` only; `-Wl,--exclude-libs,ALL` plus hidden visibility leave only `hag_*`.
* `hag_train` needs a writable `work_dir` with room for `disk_total_bytes` from `hag_train_estimate`, and the app must treat it as
  a long-running foreground job (the engine checkpoints but cannot survive process death mid-step beyond the last checkpoint).
* Not verified here: ARM/NDK execution (no device or emulator in this sandbox); CI has an informational NDK compile job.
