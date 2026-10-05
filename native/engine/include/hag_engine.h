/*
 * HAG engine C API  (Hot Attic Games shared on-device LLM engine; first consumer: LLM Trainer native v2)
 *
 * CONTRACT between: the C++ engine (native/engine), the JNI/Kotlin binding (runtime/), and the tests.
 * Owner of this file: the engine implementer. Rules: names/semantics below are stable; later changes are ADDITIVE only
 * (new functions / new trailing struct fields guarded by struct_size). Everything is UTF-8. No exceptions cross this API.
 * Thread-safety: a model may be shared by many sessions; one session is used by one thread at a time except
 * hag_cancel() which is async-safe. hag_train() blocks the calling thread (callers run it on a worker).
 * Generic by design: nothing here is LLM-Trainer specific, so other HAG apps can reuse the engine unchanged.
 */
#ifndef HAG_ENGINE_H
#define HAG_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HAG_ENGINE_API_VERSION 1

#if defined(_WIN32)
#  define HAG_API __declspec(dllexport)
#else
#  define HAG_API __attribute__((visibility("default")))
#endif

typedef struct hag_model   hag_model;    /* loaded weights (optionally + specialist patch) */
typedef struct hag_session hag_session;  /* inference context: KV cache, sampler, cancel flag */

typedef struct hag_status {
    int  code;            /* 0 = OK; negative = error (HAG_ERR_*) */
    char message[240];    /* human-readable, empty on success */
} hag_status;

enum {
    HAG_OK = 0,
    HAG_ERR_INVALID_ARG = -1,
    HAG_ERR_IO = -2,
    HAG_ERR_BAD_MODEL = -3,       /* unreadable / unsupported GGUF */
    HAG_ERR_OOM = -4,
    HAG_ERR_CANCELLED = -5,
    HAG_ERR_UNSUPPORTED = -6,     /* e.g. quantized weights cannot be trained, missing CPU feature */
    HAG_ERR_CORRUPT = -7,         /* checkpoint / patch failed hash or structure validation */
    HAG_ERR_INTERNAL = -8
};

/* ---- lifecycle / diagnostics ------------------------------------------------------------------------------ */
HAG_API hag_status  hag_engine_init(void);                 /* idempotent */
HAG_API void        hag_engine_shutdown(void);
HAG_API const char *hag_engine_version(void);              /* "hag-engine <api>; llama.cpp <commit>; ggml <ver>" (static) */
HAG_API const char *hag_system_info_json(void);            /* {"abi":"arm64-v8a","cpu_features":[...],"n_cores":8,...} (static) */
HAG_API void        hag_free(void *p);                     /* free memory returned by this API */

/* ---- models ----------------------------------------------------------------------------------------------- */
/* Called with 0..1 during load; return nonzero to cancel. May be NULL. */
typedef int (*hag_progress_fn)(float fraction, void *user);

HAG_API hag_status  hag_model_load(const char *gguf_path, int use_mmap, hag_progress_fn progress, void *user, hag_model **out);
/* Apply a specialist patch (written by hag_train) onto a loaded base. Verifies the patch's base-model sha256/shape match. */
HAG_API hag_status  hag_model_apply_patch(hag_model *model, const char *patch_path);
HAG_API void        hag_model_free(hag_model *model);
/* JSON: n_params, n_layer, n_embd, n_vocab, n_ctx_train, file_type, size_bytes, arch, chat_template present, patched, ... (owned by model) */
HAG_API const char *hag_model_info_json(const hag_model *model);

/* ---- inference -------------------------------------------------------------------------------------------- */
typedef struct hag_session_params {
    uint32_t struct_size;   /* sizeof(hag_session_params) */
    int32_t  n_ctx;         /* context tokens (clamped to the model's training context) */
    int32_t  n_threads;     /* 0 = auto */
    int32_t  n_batch;       /* 0 = default */
} hag_session_params;

typedef struct hag_sample_params {
    uint32_t struct_size;
    float    temperature;       /* 0 = greedy */
    int32_t  top_k;             /* 0 = off */
    float    top_p;             /* 1 = off */
    float    min_p;             /* 0 = off */
    float    repeat_penalty;    /* 1 = off */
    uint32_t seed;              /* fixed seed => reproducible */
    int32_t  max_new_tokens;
} hag_sample_params;

typedef struct hag_gen_stats {
    int32_t n_prompt_tokens;
    int32_t n_generated;
    double  prompt_ms;
    double  gen_ms;
    int32_t stop_reason;        /* 0 = end-of-generation token, 1 = max_new_tokens, 2 = cancelled, 3 = context full */
    size_t  peak_rss_bytes;     /* best effort, process-wide */
} hag_gen_stats;

/* Streaming sink: receives complete UTF-8 pieces (never a split multi-byte character). Return nonzero to stop. */
typedef int (*hag_token_fn)(const char *piece_utf8, void *user);

HAG_API hag_status  hag_session_new(hag_model *model, const hag_session_params *params, hag_session **out);
HAG_API void        hag_session_free(hag_session *s);
HAG_API hag_status  hag_session_reset(hag_session *s);       /* clears conversation/KV */
/* Renders the model's own chat template. roles: "system"/"user"/"assistant". *out_text must be released with hag_free. */
HAG_API hag_status  hag_chat_format(const hag_model *model, const char *const *roles, const char *const *contents,
                                    int n_messages, int add_generation_prompt, char **out_text);
/* Continues from the session's current KV state with `prompt_utf8` (call hag_session_reset first for a fresh chat). */
HAG_API hag_status  hag_generate(hag_session *s, const char *prompt_utf8, const hag_sample_params *sp,
                                 hag_token_fn sink, void *user, hag_gen_stats *stats);
HAG_API void        hag_cancel(hag_session *s);               /* async-safe; also cancels hag_score */
HAG_API hag_status  hag_tokenize(const hag_model *model, const char *text, int add_special, int32_t *out_tokens, int cap, int *n_out);
/* Mean negative log-likelihood per token of `text` under the model (lower = better fit). Used for held-out evaluation. */
HAG_API hag_status  hag_score(hag_session *s, const char *text, double *mean_nll, int *n_tokens);

/* ---- parameter-changing specialization (on-device) -------------------------------------------------------- */
typedef struct hag_train_params {
    uint32_t struct_size;
    int32_t  n_ctx;                 /* training sequence length */
    int32_t  n_batch;               /* tokens per optimizer step (micro-batching is the implementation's choice) */
    int32_t  epochs;
    float    learning_rate;
    float    val_fraction;          /* held out from `texts` for loss tracking (NOT the user's evaluation set) */
    uint32_t seed;                  /* deterministic data order / init */
    int32_t  n_threads;             /* 0 = auto */
    int32_t  trainable_last_layers; /* 0 = all layers; N = only the last N transformer blocks (+ final norm) */
    int32_t  train_embeddings;      /* nonzero: also train token embeddings / output head */
    int32_t  checkpoint_every_steps;/* 0 = once per epoch */
    size_t   max_memory_bytes;      /* refuse (HAG_ERR_OOM) instead of starting if the estimate exceeds this; 0 = no limit */
} hag_train_params;

enum { HAG_PHASE_PREPARE = 0, HAG_PHASE_TRAIN = 1, HAG_PHASE_EVAL = 2, HAG_PHASE_SAVE = 3, HAG_PHASE_DONE = 4 };

typedef struct hag_train_event {
    int32_t phase;
    int32_t epoch, epochs;
    int32_t step, steps;            /* steps within the run so far / total (best estimate) */
    int64_t examples_done;
    double  train_loss;             /* mean over the last reporting window; NaN if n/a */
    double  val_loss;               /* latest held-out loss; NaN if n/a */
    double  elapsed_s;
    size_t  rss_bytes;
    int32_t resumed_from_step;      /* >0 if this run resumed from a checkpoint */
} hag_train_event;

/* Return nonzero from the callback to cancel; the latest valid checkpoint is kept. */
typedef int (*hag_train_fn)(const hag_train_event *ev, void *user);

/*
 * Specialize `base_gguf_path` on `texts` (UTF-8 training sequences, already reviewed by the user).
 *  - Checkpoints (atomic, hash-protected) are written to work_dir; a later call with the same inputs RESUMES.
 *  - On success writes a specialist PATCH to out_patch_path containing ONLY the changed tensors plus metadata
 *    (base sha256, tensor names, hyper-parameters, dataset hash, step counts). hag_model_apply_patch() reloads it.
 *  - Never modifies the base file. Unsupported bases (e.g. quantized weights that cannot be trained) return
 *    HAG_ERR_UNSUPPORTED with an explanation; the engine must not silently fall back to anything fake.
 */
HAG_API hag_status  hag_train(const char *base_gguf_path, const char *const *texts, int n_texts,
                              const hag_train_params *params, const char *work_dir, const char *out_patch_path,
                              hag_train_fn progress, void *user);
/* Cheap pre-flight: estimated peak bytes for training with these params (JSON also reports whether the base is trainable). */
HAG_API hag_status  hag_train_estimate(const char *base_gguf_path, const hag_train_params *params, char **out_json);
HAG_API hag_status  hag_patch_info_json(const char *patch_path, char **out_json);   /* hag_free the result */

#ifdef __cplusplus
}
#endif
#endif /* HAG_ENGINE_H */
