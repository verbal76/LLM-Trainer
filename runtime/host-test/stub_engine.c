// Host-side STUB of the HAG C API, used ONLY to exercise the JNI marshalling on a JVM (never shipped).
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdatomic.h>
#include "hag_engine.h"

struct hag_model { int patched; };
struct hag_session { atomic_int cancel; struct hag_model *m; };

static hag_status ok(void) { hag_status s; s.code = 0; s.message[0] = 0; return s; }
static hag_status err(int c, const char *m) { hag_status s; s.code = c; snprintf(s.message, sizeof s.message, "%s", m); return s; }

HAG_API hag_status hag_engine_init(void) { return ok(); }
HAG_API void hag_engine_shutdown(void) {}
HAG_API const char *hag_engine_version(void) { return "hag-engine 1; stub"; }
HAG_API const char *hag_system_info_json(void) { return "{\"abi\":\"x86_64\",\"cpu_features\":[],\"n_cores\":4}"; }
HAG_API void hag_free(void *p) { free(p); }

HAG_API hag_status hag_model_load(const char *path, int mmap, hag_progress_fn pg, void *u, hag_model **out) {
    (void)mmap;
    if (strstr(path, "nonexistent")) return err(HAG_ERR_IO, "cannot open file");
    if (pg) { float f[3] = {0.f, 0.5f, 1.f}; for (int i = 0; i < 3; i++) if (pg(f[i], u)) return err(HAG_ERR_CANCELLED, "load cancelled"); }
    *out = calloc(1, sizeof(hag_model));
    return ok();
}
HAG_API hag_status hag_model_apply_patch(hag_model *m, const char *p) { if (strstr(p, "bad")) return err(HAG_ERR_CORRUPT, "bad patch"); m->patched = 1; return ok(); }
HAG_API void hag_model_free(hag_model *m) { free(m); }
HAG_API const char *hag_model_info_json(const hag_model *m) { return m->patched ? "{\"patched\":true,\"n_params\":135}" : "{\"patched\":false,\"n_params\":135}"; }

HAG_API hag_status hag_session_new(hag_model *m, const hag_session_params *p, hag_session **out) {
    if (p->struct_size != sizeof *p) return err(HAG_ERR_INVALID_ARG, "bad struct_size");
    if (p->n_ctx == 777 && p->n_threads == 3 && p->n_batch == 5) { /* marshalling probe */ }
    else if (p->n_ctx == 777) return err(HAG_ERR_INVALID_ARG, "marshalling mismatch in session params");
    hag_session *s = calloc(1, sizeof *s); s->m = m; *out = s; return ok();
}
HAG_API void hag_session_free(hag_session *s) { free(s); }
HAG_API hag_status hag_session_reset(hag_session *s) { atomic_store(&s->cancel, 0); return ok(); }

HAG_API hag_status hag_chat_format(const hag_model *m, const char *const *roles, const char *const *contents, int n, int gen, char **out) {
    (void)m; char *b = malloc(4096); b[0] = 0;
    for (int i = 0; i < n; i++) { strcat(b, "<|"); strcat(b, roles[i]); strcat(b, "|>"); strcat(b, contents[i]); strcat(b, "\n"); }
    if (gen) strcat(b, "<|assistant|>");
    *out = b; return ok();
}

HAG_API hag_status hag_generate(hag_session *s, const char *prompt, const hag_sample_params *sp, hag_token_fn sink, void *u, hag_gen_stats *st) {
    if (sp->struct_size != sizeof *sp) return err(HAG_ERR_INVALID_ARG, "bad sample struct_size");
    memset(st, 0, sizeof *st);
    st->n_prompt_tokens = (int)strlen(prompt);
    int slow = strstr(prompt, "SLOW") != NULL;
    // piece 0 echoes the prompt (UTF-8 integrity), piece 1 is a 4-byte emoji, then filler. seed/temperature echoed once.
    char hdr[256]; snprintf(hdr, sizeof hdr, "[t=%.2f k=%d p=%.2f m=%.2f r=%.2f seed=%u max=%d]", sp->temperature, sp->top_k, sp->top_p, sp->min_p, sp->repeat_penalty, sp->seed, sp->max_new_tokens);
    const char *pieces[4] = { prompt, hdr, "\xF0\x9F\x98\x80", "w\xC3\xB6rld " };
    int n = 0;
    for (int i = 0; i < sp->max_new_tokens; i++) {
        if (atomic_load(&s->cancel)) { st->stop_reason = 2; st->n_generated = n; return ok(); }
        if (slow) usleep(10000);
        const char *p = i < 4 ? pieces[i] : "w\xC3\xB6rld ";
        n++;
        if (sink(p, u)) { st->stop_reason = 2; st->n_generated = n; return ok(); }
    }
    st->n_generated = n; st->stop_reason = 1; st->prompt_ms = 1.5; st->gen_ms = 2.5; st->peak_rss_bytes = 12345;
    return ok();
}
HAG_API void hag_cancel(hag_session *s) { atomic_store(&s->cancel, 1); }

HAG_API hag_status hag_tokenize(const hag_model *m, const char *text, int special, int32_t *out, int cap, int *n_out) {
    (void)m; int n = (int)strlen(text) + (special ? 1 : 0);
    *n_out = n; if (cap < n) return ok(); /* reports the needed size, writes nothing */
    int k = 0; if (special) out[k++] = -1;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) out[k++] = *p;
    return ok();
}
HAG_API hag_status hag_score(hag_session *s, const char *text, double *nll, int *n) { (void)s; *nll = 1.5; *n = (int)strlen(text); return ok(); }

HAG_API hag_status hag_train(const char *base, const char *const *texts, int n, const hag_train_params *p, const char *work, const char *out_patch, hag_train_fn pg, void *u) {
    if (p->struct_size != sizeof *p) return err(HAG_ERR_INVALID_ARG, "bad train struct_size");
    (void)base; (void)work;
    for (int i = 0; i < 3; i++) {
        hag_train_event ev; memset(&ev, 0, sizeof ev);
        ev.phase = i == 2 ? HAG_PHASE_DONE : HAG_PHASE_TRAIN; ev.epoch = i + 1; ev.epochs = 3; ev.step = i; ev.steps = 3; ev.examples_done = 100 + i;
        ev.train_loss = 2.0 - i; ev.val_loss = 0.0 / 0.0; ev.elapsed_s = 0.25 * i; ev.rss_bytes = 1u << 20; ev.resumed_from_step = 7;
        if (pg && pg(&ev, u)) return err(HAG_ERR_CANCELLED, "train cancelled");
    }
    FILE *f = fopen(out_patch, "w");
    if (!f) return err(HAG_ERR_IO, "cannot write patch");
    fprintf(f, "n=%d first=%s last=%s ctx=%d batch=%d ep=%d lr=%.6f val=%.3f seed=%u thr=%d last=%d emb=%d ck=%d mem=%zu\n", n, texts[0], texts[n - 1],
            p->n_ctx, p->n_batch, p->epochs, p->learning_rate, p->val_fraction, p->seed, p->n_threads, p->trainable_last_layers, p->train_embeddings,
            p->checkpoint_every_steps, p->max_memory_bytes);
    fclose(f);
    return ok();
}
HAG_API hag_status hag_train_estimate(const char *base, const hag_train_params *p, char **out) {
    (void)base; char *b = malloc(256); snprintf(b, 256, "{\"trainable\":true,\"bytes\":%d}", p->n_ctx * 1000); *out = b; return ok();
}
HAG_API hag_status hag_patch_info_json(const char *path, char **out) { char *b = malloc(256); snprintf(b, 256, "{\"path\":\"%s\"}", path); *out = b; return ok(); }
