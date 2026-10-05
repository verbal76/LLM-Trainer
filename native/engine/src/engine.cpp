// HAG engine: lifecycle, model loading, specialist-patch application, sessions, generation, scoring.
// Training lives in train.cpp. No exception may cross the C API: every entry point is wrapped.
#include "hag_internal.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>


#ifndef HAG_LLAMA_COMMIT
#define HAG_LLAMA_COMMIT "0c1e570"   /* = first 7 of native/llama.cpp.pin; CMake overrides; tests/test_engine.py checks they agree */
#endif

// everything except the hag_* C API (explicit default visibility in hag_engine.h) stays hidden even when the
// embedding build does not pass -fvisibility=hidden
#pragma GCC visibility push(hidden)

using namespace hag;

#define HAG_TRY try {
#define HAG_CATCH_STATUS \
    } catch (const std::bad_alloc &) { return make_status(HAG_ERR_OOM, "out of memory"); } \
      catch (const std::exception & e) { return make_status(HAG_ERR_INTERNAL, "internal error: %s", e.what()); } \
      catch (...) { return make_status(HAG_ERR_INTERNAL, "internal error"); }

// ---- session ------------------------------------------------------------------------------------------------------
struct hag_session {
    hag_model *          model = nullptr;
    llama_context *      ctx = nullptr;
    int                  n_ctx = 0;
    int                  n_batch = 0;
    int                  n_threads = 0;
    int                  n_past = 0;
    std::atomic<bool>    cancel{false};
};

static std::mutex g_mu;
static bool       g_inited = false;

namespace hag {
std::string engine_version_string() {
    std::string s = "hag-engine " + std::to_string(HAG_ENGINE_API_VERSION) + "; llama.cpp " HAG_LLAMA_COMMIT "; ggml ";
    s += ggml_version();
    return s;
}
}  // namespace hag

static int auto_threads() {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) return 4;
    if (hw <= 4) return (int)hw;
    // leave headroom for the UI/OS: three quarters of the cores, at most 8 (spin-waiting threads hurt when oversubscribed)
    return (int)std::min<unsigned>(8, hw * 3 / 4);
}

// ---- lifecycle ----------------------------------------------------------------------------------------------------
hag_status hag_engine_init(void) {
    HAG_TRY
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_inited) return ok_status();
    install_log_capture();
    llama_backend_init();
    g_inited = true;
    return ok_status();
    HAG_CATCH_STATUS
}

void hag_engine_shutdown(void) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_inited) return;
    llama_backend_free();
    g_inited = false;
}

const char * hag_engine_version(void) {
    static const std::string v = engine_version_string();
    return v.c_str();
}

const char * hag_system_info_json(void) {
    static const std::string js = [] {
        Json j;
        j.begin_obj();
#if defined(__aarch64__)
        j.kv("abi", "arm64-v8a");
#elif defined(__x86_64__) || defined(_M_X64)
        j.kv("abi", "x86_64");
#elif defined(__arm__)
        j.kv("abi", "armeabi-v7a");
#elif defined(__i386__)
        j.kv("abi", "x86");
#else
        j.kv("abi", "unknown");
#endif
        j.begin_arr("cpu_features");
        struct F { const char * n; int (*fn)(void); };
        static const F feats[] = {
            {"neon", ggml_cpu_has_neon}, {"arm_fma", ggml_cpu_has_arm_fma}, {"fp16_va", ggml_cpu_has_fp16_va},
            {"dotprod", ggml_cpu_has_dotprod}, {"i8mm", ggml_cpu_has_matmul_int8}, {"sve", ggml_cpu_has_sve},
            {"sme", ggml_cpu_has_sme}, {"sse3", ggml_cpu_has_sse3}, {"ssse3", ggml_cpu_has_ssse3}, {"avx", ggml_cpu_has_avx},
            {"avx_vnni", ggml_cpu_has_avx_vnni}, {"avx2", ggml_cpu_has_avx2}, {"avx512", ggml_cpu_has_avx512},
            {"fma", ggml_cpu_has_fma}, {"f16c", ggml_cpu_has_f16c}, {"bmi2", ggml_cpu_has_bmi2},
        };
        for (const F & f : feats) if (f.fn()) j.val(f.n);
        j.end_arr();
        j.kv("n_cores", (int)std::thread::hardware_concurrency());
        j.kv("auto_threads", auto_threads());
        j.kv("pointer_bits", (int)(sizeof(void *) * 8));
        // memory (Linux/Android); 0 when unavailable
        uint64_t total = 0, avail = 0;
        if (FILE * f = fopen("/proc/meminfo", "r")) {
            char line[200];
            while (fgets(line, sizeof(line), f)) {
                unsigned long long kb;
                if (sscanf(line, "MemTotal: %llu kB", &kb) == 1) total = kb * 1024;
                if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) avail = kb * 1024;
            }
            fclose(f);
        }
        j.kv("mem_total_bytes", total);
        j.kv("mem_available_bytes", avail);
        j.kv("engine", engine_version_string());
        j.end_obj();
        return j.str();
    }();
    return js.c_str();
}

void hag_free(void * p) { free(p); }

// ---- models -------------------------------------------------------------------------------------------------------
struct LoadCtx { hag_progress_fn fn; void * user; bool cancelled; };

static bool load_progress(float p, void * ud) {
    LoadCtx * c = (LoadCtx *)ud;
    if (c->fn && c->fn(p, c->user) != 0) { c->cancelled = true; return false; }
    return true;
}

static std::string meta_str(const llama_model * m, const char * key) {
    char buf[512];
    int n = llama_model_meta_val_str(m, key, buf, sizeof(buf));
    return n >= 0 ? std::string(buf) : std::string();
}

static void build_model_info(hag_model * hm) {
    const llama_model * m = hm->model;
    const llama_vocab * v = llama_model_get_vocab(m);
    char desc[256];
    llama_model_desc(m, desc, sizeof(desc));
    Json j;
    j.begin_obj();
    j.kv("arch", hm->arch);
    j.kv("name", meta_str(m, "general.name"));
    j.kv("desc", desc);
    j.kv("n_params", (uint64_t)llama_model_n_params(m));
    j.kv("n_layer", (int)llama_model_n_layer(m));
    j.kv("n_embd", (int)llama_model_n_embd(m));
    j.kv("n_head", (int)llama_model_n_head(m));
    j.kv("n_head_kv", (int)llama_model_n_head_kv(m));
    j.kv("n_vocab", (int)llama_vocab_n_tokens(v));
    j.kv("n_ctx_train", (int)llama_model_n_ctx_train(m));
    j.kv("file_type", meta_str(m, "general.file_type"));
    j.kv("size_bytes", (int64_t)hm->file_size);
    j.kv("model_bytes", (uint64_t)llama_model_size(m));
    j.kv("chat_template", llama_model_chat_template(m, nullptr) != nullptr);
    j.kv("mmap", hm->mmap);
    j.kv("patched", hm->patched);
    if (hm->patched) {
        j.kv("patch_path", hm->patch_path);
        j.kv("patch_kind", hm->patch_kind);
        j.kv("patch_sha256", hm->patch_file_sha256);
    }
    if (!hm->base_sha256.empty()) j.kv("base_sha256", hm->base_sha256);
    j.end_obj();
    hm->info_json = j.str();
}

hag_status hag_model_load(const char * gguf_path, int use_mmap, hag_progress_fn progress, void * user, hag_model ** out) {
    HAG_TRY
    if (!gguf_path || !out) return make_status(HAG_ERR_INVALID_ARG, "null argument");
    *out = nullptr;
    hag_status st = hag_engine_init();
    if (st.code) return st;
    int64_t fsz = file_size(gguf_path);
    if (fsz < 0) return make_status(HAG_ERR_IO, "cannot open model file: %s", gguf_path);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    LoadCtx lc{progress, user, false};
    mp.progress_callback = load_progress;
    mp.progress_callback_user_data = &lc;
    clear_native_error();
    llama_model * m = llama_model_load_from_file(gguf_path, mp);
    if (!m) {
        if (lc.cancelled) return make_status(HAG_ERR_CANCELLED, "model load cancelled");
        std::string e = last_native_error();
        return make_status(HAG_ERR_BAD_MODEL, "cannot load model: %s", e.empty() ? "unsupported or corrupt GGUF" : e.c_str());
    }
    hag_model * hm = new hag_model();
    hm->model = m;
    hm->path = gguf_path;
    hm->mmap = use_mmap != 0;
    hm->file_size = fsz;
    hm->arch = meta_str(m, "general.architecture");
    build_model_info(hm);
    *out = hm;
    return ok_status();
    HAG_CATCH_STATUS
}

void hag_model_free(hag_model * hm) {
    if (!hm) return;
    if (hm->adapter) llama_adapter_lora_free(hm->adapter);
    if (hm->model) llama_model_free(hm->model);
    for (ggml_backend_buffer_t b : hm->patch_bufs) ggml_backend_buffer_free(b);
    delete hm;
}

const char * hag_model_info_json(const hag_model * hm) { return hm ? hm->info_json.c_str() : "{}"; }

// ---- specialist patches --------------------------------------------------------------------------------------------
namespace hag {

static std::string kv_string(const gguf_context * g, const char * key, bool required, std::string & err) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) { if (required) err = std::string("missing key ") + key; return ""; }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) { err = std::string("key has wrong type: ") + key; return ""; }
    return gguf_get_val_str(g, id);
}

std::string patch_read_header(const char * path, PatchHeader & h) {
    h.path = path;
    h.file_size = file_size(path);
    if (h.file_size < 0) return "cannot open patch file";
    gguf_init_params ip = {/*no_alloc=*/true, /*ctx=*/nullptr};
    h.gguf = gguf_init_from_file(path, ip);
    if (!h.gguf) return "not a valid GGUF file (truncated or corrupt)";
    gguf_context * g = h.gguf;
    std::string err;
    std::string arch = kv_string(g, "general.architecture", true, err);
    if (!err.empty()) return err;
    int64_t id;
    if (arch == HAG_PATCH_ARCH) {
        h.kind = "replace";
    } else {   // a standard llama.cpp LoRA adapter that carries hag.* provenance keys
        std::string e2, gtype = kv_string(g, "general.type", false, e2), atype = kv_string(g, "adapter.type", false, e2);
        std::string kind = kv_string(g, HAG_KEY_KIND, false, e2);
        if (gtype != "adapter" || atype != "lora" || kind != "lora") return "not a HAG specialist patch (architecture '" + arch + "')";
        h.kind = "lora";
        id = gguf_find_key(g, "adapter.lora.alpha");
        if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT32) h.lora_alpha = gguf_get_val_f32(g, id);
    }
    id = gguf_find_key(g, HAG_KEY_VERSION);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) return "missing patch format version";
    h.version = gguf_get_val_u32(g, id);
    if (h.version != HAG_PATCH_FORMAT_VERSION) return "unsupported patch format version " + std::to_string(h.version);
    h.base_arch = kv_string(g, HAG_KEY_BASE_ARCH, true, err);
    h.base_sha256 = kv_string(g, HAG_KEY_BASE_SHA256, true, err);
    h.payload_sha256 = kv_string(g, HAG_KEY_PAYLOAD_SHA256, true, err);
    h.fingerprint = kv_string(g, HAG_KEY_FINGERPRINT, false, err);
    h.engine = kv_string(g, HAG_KEY_ENGINE, false, err);
    if (!err.empty()) return err;
    if (h.base_sha256.size() != 64 || h.payload_sha256.size() != 64) return "malformed hash fields";
    id = gguf_find_key(g, HAG_KEY_BASE_SIZE);
    if (id < 0) return "missing base size";
    h.base_size = gguf_get_val_u64(g, id);
    id = gguf_find_key(g, HAG_KEY_BASE_NTENSORS);
    if (id < 0) return "missing base tensor count";
    h.base_n_tensors = gguf_get_val_u32(g, id);
    h.data_offset = gguf_get_data_offset(g);
    // structure: every tensor must lie inside the file
    int64_t nt = gguf_get_n_tensors(g);
    if (nt <= 0) return "patch contains no tensors";
    for (int64_t i = 0; i < nt; i++) {
        uint64_t end = (uint64_t)h.data_offset + gguf_get_tensor_offset(g, i) + gguf_get_tensor_size(g, i);
        if (end > (uint64_t)h.file_size) return "patch file is truncated (tensor beyond end of file)";
    }
    return "";
}

std::string patch_info_json(const PatchHeader & h, const std::string & file_sha256) {
    const gguf_context * g = h.gguf;
    Json j;
    j.begin_obj();
    j.kv("format", "hag-patch");
    j.kv("kind", h.kind);
    j.kv("format_version", (uint64_t)h.version);
    j.kv("file_bytes", (int64_t)h.file_size);
    j.kv("patch_sha256", file_sha256);
    j.kv("payload_sha256", h.payload_sha256);
    j.begin_obj("base");
    j.kv("arch", h.base_arch);
    j.kv("sha256", h.base_sha256);
    j.kv("size_bytes", h.base_size);
    j.kv("n_tensors", (uint64_t)h.base_n_tensors);
    j.end_obj();
    j.kv("run_fingerprint", h.fingerprint);
    j.kv("engine", h.engine);
    j.begin_obj("train");
    int64_t nkv = gguf_get_n_kv(g);
    for (int64_t i = 0; i < nkv; i++) {
        const char * k = gguf_get_key(g, i);
        if (strncmp(k, "hag.train.", 10) != 0) continue;
        switch (gguf_get_kv_type(g, i)) {
            case GGUF_TYPE_UINT32: j.kv(k + 10, (uint64_t)gguf_get_val_u32(g, i)); break;
            case GGUF_TYPE_INT32:  j.kv(k + 10, (int64_t)gguf_get_val_i32(g, i)); break;
            case GGUF_TYPE_UINT64: j.kv(k + 10, (uint64_t)gguf_get_val_u64(g, i)); break;
            case GGUF_TYPE_INT64:  j.kv(k + 10, (int64_t)gguf_get_val_i64(g, i)); break;
            case GGUF_TYPE_FLOAT32: j.kv(k + 10, (double)gguf_get_val_f32(g, i)); break;
            case GGUF_TYPE_FLOAT64: j.kv(k + 10, gguf_get_val_f64(g, i)); break;
            case GGUF_TYPE_BOOL:   j.kv(k + 10, gguf_get_val_bool(g, i)); break;
            case GGUF_TYPE_STRING: j.kv(k + 10, gguf_get_val_str(g, i)); break;
            default: break;
        }
    }
    j.end_obj();
    j.begin_arr("tensors");
    int64_t nt = gguf_get_n_tensors(g);
    uint64_t total = 0;
    for (int64_t i = 0; i < nt; i++) {
        const int64_t * ne = gguf_get_tensor_ne(g, i);
        j.begin_obj();
        j.kv("name", gguf_get_tensor_name(g, i));
        j.kv("type", ggml_type_name(gguf_get_tensor_type(g, i)));
        j.begin_arr("ne");
        for (int d = 0; d < 4; d++) j.val(std::to_string((long long)ne[d]));  // strings keep the tiny writer simple
        j.end_arr();
        j.kv("bytes", (uint64_t)gguf_get_tensor_size(g, i));
        j.end_obj();
        total += gguf_get_tensor_size(g, i);
    }
    j.end_arr();
    j.kv("n_tensors", (int64_t)nt);
    j.kv("payload_bytes", total);
    j.end_obj();
    return j.str();
}

}  // namespace hag

hag_status hag_patch_info_json(const char * patch_path, char ** out_json) {
    HAG_TRY
    if (!patch_path || !out_json) return make_status(HAG_ERR_INVALID_ARG, "null argument");
    *out_json = nullptr;
    PatchHeader h;
    std::string err = patch_read_header(patch_path, h);
    if (!err.empty()) return make_status(HAG_ERR_CORRUPT, "patch rejected: %s", err.c_str());
    std::string fh;
    if (!sha256_file(patch_path, fh)) return make_status(HAG_ERR_IO, "cannot read patch file");
    std::string js = patch_info_json(h, fh);
    char * p = (char *)malloc(js.size() + 1);
    if (!p) return make_status(HAG_ERR_OOM, "out of memory");
    memcpy(p, js.c_str(), js.size() + 1);
    *out_json = p;
    return ok_status();
    HAG_CATCH_STATUS
}

static bool never_cancel(void *) { return false; }

hag_status hag_model_apply_patch(hag_model * hm, const char * patch_path) {
    HAG_TRY
    if (!hm || !patch_path) return make_status(HAG_ERR_INVALID_ARG, "null argument");
    if (hm->patched) return make_status(HAG_ERR_INVALID_ARG, "model already has a patch applied (load the base again to switch)");
    if (hm->n_sessions.load() != 0) return make_status(HAG_ERR_INVALID_ARG, "apply the patch before creating sessions");

    PatchHeader h;
    std::string err = patch_read_header(patch_path, h);
    if (!err.empty()) return make_status(HAG_ERR_CORRUPT, "patch rejected: %s", err.c_str());
    if (h.base_arch != hm->arch) return make_status(HAG_ERR_CORRUPT, "patch is for architecture '%s', model is '%s'", h.base_arch.c_str(), hm->arch.c_str());
    if (h.base_size != (uint64_t)hm->file_size) return make_status(HAG_ERR_CORRUPT, "patch was made for a different base file (size mismatch)");

    if (hm->base_sha256.empty()) {
        std::string hx;
        if (!sha256_file(hm->path.c_str(), hx, never_cancel, nullptr)) return make_status(HAG_ERR_IO, "cannot hash base model file");
        hm->base_sha256 = hx;
    }
    if (hm->base_sha256 != h.base_sha256) {
        return make_status(HAG_ERR_CORRUPT, "patch was made for a different base model (sha256 %.12s... != %.12s...)", h.base_sha256.c_str(), hm->base_sha256.c_str());
    }

    if (h.kind == "lora") {
        if (h.base_arch != hm->arch) return make_status(HAG_ERR_CORRUPT, "patch rejected: architecture mismatch");
        // verify the payload hash by streaming the tensor bytes, then let llama.cpp attach the adapter (it validates every shape)
        FILE * f = fopen(patch_path, "rb");
        if (!f) return make_status(HAG_ERR_IO, "cannot open patch");
        Sha256 payload;
        std::vector<uint8_t> buf(1 << 20);
        const gguf_context * gg = h.gguf;
        for (int64_t i = 0; i < gguf_get_n_tensors(gg); i++) {
            uint64_t off = (uint64_t)h.data_offset + gguf_get_tensor_offset(gg, i), left = gguf_get_tensor_size(gg, i);
            if (fseek64(f, (int64_t)off, SEEK_SET) != 0) { fclose(f); return make_status(HAG_ERR_CORRUPT, "patch rejected: seek failed"); }
            while (left) {
                size_t n = (size_t)std::min<uint64_t>(left, buf.size());
                if (fread(buf.data(), 1, n, f) != n) { fclose(f); return make_status(HAG_ERR_CORRUPT, "patch rejected: short read"); }
                payload.update(buf.data(), n);
                left -= n;
            }
        }
        fclose(f);
        if (payload.final_hex() != h.payload_sha256) return make_status(HAG_ERR_CORRUPT, "patch rejected: payload hash mismatch (patch file corrupted)");
        clear_native_error();
        llama_adapter_lora * ad = llama_adapter_lora_init(hm->model, patch_path);
        if (!ad) {
            std::string e = last_native_error();
            return make_status(HAG_ERR_CORRUPT, "patch rejected: adapter does not fit the base: %s", e.c_str());
        }
        hm->adapter = ad;
        hm->patched = true;
        hm->patch_kind = "lora";
        hm->patch_path = patch_path;
        std::string fh;
        sha256_file(patch_path, fh);
        hm->patch_file_sha256 = fh;
        build_model_info(hm);
        return ok_status();
    }

    // validate every tensor against the loaded base, then stream payloads into private buffers (hash verified)
    gguf_context * g = h.gguf;
    int64_t nt = gguf_get_n_tensors(g);
    struct Item { ggml_tensor * t; ggml_backend_buffer_t buf; };
    std::vector<Item> items;
    items.reserve(nt);
    auto cleanup = [&] { for (auto & it : items) if (it.buf) ggml_backend_buffer_free(it.buf); };

    FILE * f = fopen(patch_path, "rb");
    if (!f) return make_status(HAG_ERR_IO, "cannot open patch");
    Sha256 payload;
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    std::string fail;
    for (int64_t i = 0; i < nt && fail.empty(); i++) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = llama_model_tensor_by_name(hm->model, name);
        if (!t) { fail = std::string("patch tensor not in base model: ") + name; break; }
        int dup = 0;
        for (int32_t k = 0; k < llama_model_n_tensors(hm->model); k++) {
            ggml_tensor * o = llama_model_tensor_at(hm->model, k);
            if (o && strcmp(ggml_get_name(o), name) == 0) dup++;
        }
        if (dup != 1) { fail = std::string("tensor is shared/duplicated in this architecture and cannot be patched: ") + name; break; }
        const int64_t * ne = gguf_get_tensor_ne(g, i);
        if (gguf_get_tensor_type(g, i) != t->type || memcmp(ne, t->ne, sizeof(int64_t) * 4) != 0) {
            fail = std::string("patch tensor shape/type does not match base: ") + name;
            break;
        }
        size_t nb = ggml_nbytes(t);
        if (nb != gguf_get_tensor_size(g, i)) { fail = std::string("size mismatch for ") + name; break; }
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(cpu_buft, nb);
        if (!buf) { fclose(f); cleanup(); return make_status(HAG_ERR_OOM, "cannot allocate %zu bytes for patched tensor %s", nb, name); }
        items.push_back({t, buf});
        void * dst = ggml_backend_buffer_get_base(buf);
        uint64_t off = (uint64_t)h.data_offset + gguf_get_tensor_offset(g, i);
        if (fseek64(f, (int64_t)off, SEEK_SET) != 0 || fread(dst, 1, nb, f) != nb) { fail = std::string("short read for ") + name; break; }
        payload.update(dst, nb);
    }
    fclose(f);
    if (fail.empty() && payload.final_hex() != h.payload_sha256) fail = "payload hash mismatch (patch file corrupted)";
    if (!fail.empty()) { cleanup(); return make_status(HAG_ERR_CORRUPT, "patch rejected: %s", fail.c_str()); }

    // all verified: rebind the tensors to the private buffers (the mmap'd base file is never written)
    for (auto & it : items) {
        it.t->data = ggml_backend_buffer_get_base(it.buf);
        it.t->buffer = it.buf;
        it.t->extra = nullptr;  // drop any repacked-layout handle: the patched data is in the plain GGUF layout
        hm->patch_bufs.push_back(it.buf);
    }
    hm->patched = true;
    hm->patch_kind = "replace";
    hm->patch_path = patch_path;
    {
        std::string fh;
        sha256_file(patch_path, fh);
        hm->patch_file_sha256 = fh;
    }
    build_model_info(hm);
    return ok_status();
    HAG_CATCH_STATUS
}

// ---- sessions -----------------------------------------------------------------------------------------------------
static bool abort_cb(void * ud) { return ((hag_session *)ud)->cancel.load(std::memory_order_relaxed); }

hag_status hag_session_new(hag_model * hm, const hag_session_params * sp, hag_session ** out) {
    HAG_TRY
    if (!hm || !out) return make_status(HAG_ERR_INVALID_ARG, "null argument");
    *out = nullptr;
    int32_t n_ctx_train = llama_model_n_ctx_train(hm->model);
    int n_ctx = 2048, n_threads = 0, n_batch = 0;
    if (sp) {
        if (sp->struct_size < offsetof(hag_session_params, n_batch) + sizeof(int32_t) && sp->struct_size != 0)
            return make_status(HAG_ERR_INVALID_ARG, "session params struct_size too small");
        n_ctx = sp->n_ctx; n_threads = sp->n_threads; n_batch = sp->n_batch;
    }
    if (n_ctx <= 0) n_ctx = 2048;
    n_ctx = std::min(n_ctx, (int)n_ctx_train);
    if (n_ctx < 8) n_ctx = std::min(8, (int)n_ctx_train);
    if (n_threads <= 0) n_threads = auto_threads();
    if (n_batch <= 0) n_batch = 512;
    n_batch = std::min(n_batch, n_ctx);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_batch;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.no_perf = true;
    clear_native_error();
    llama_context * c = llama_init_from_model(hm->model, cp);
    if (!c) {
        std::string e = last_native_error();
        return make_status(HAG_ERR_OOM, "cannot create context (n_ctx=%d): %s", n_ctx, e.empty() ? "allocation failed" : e.c_str());
    }
    if (hm->adapter) {
        float one = 1.0f;
        if (llama_set_adapters_lora(c, &hm->adapter, 1, &one) != 0) {
            llama_free(c);
            return make_status(HAG_ERR_INTERNAL, "cannot activate the LoRA patch");
        }
    }
    hag_session * s = new hag_session();
    s->model = hm; s->ctx = c; s->n_ctx = n_ctx; s->n_batch = n_batch; s->n_threads = n_threads;
    llama_set_abort_callback(c, abort_cb, s);
    hm->n_sessions++;
    *out = s;
    return ok_status();
    HAG_CATCH_STATUS
}

void hag_session_free(hag_session * s) {
    if (!s) return;
    if (s->ctx) llama_free(s->ctx);
    s->model->n_sessions--;
    delete s;
}

hag_status hag_session_reset(hag_session * s) {
    HAG_TRY
    if (!s) return make_status(HAG_ERR_INVALID_ARG, "null session");
    llama_memory_clear(llama_get_memory(s->ctx), true);
    s->n_past = 0;
    return ok_status();
    HAG_CATCH_STATUS
}

void hag_cancel(hag_session * s) { if (s) s->cancel.store(true); }

hag_status hag_chat_format(const hag_model * hm, const char * const * roles, const char * const * contents, int n,
                           int add_gen, char ** out_text) {
    HAG_TRY
    if (!hm || !roles || !contents || n <= 0 || !out_text) return make_status(HAG_ERR_INVALID_ARG, "bad arguments");
    *out_text = nullptr;
    const char * tmpl = llama_model_chat_template(hm->model, nullptr);
    if (!tmpl) return make_status(HAG_ERR_UNSUPPORTED, "model has no chat template");
    std::vector<llama_chat_message> msgs(n);
    for (int i = 0; i < n; i++) {
        if (!roles[i] || !contents[i]) return make_status(HAG_ERR_INVALID_ARG, "null message");
        msgs[i].role = roles[i];
        msgs[i].content = contents[i];
    }
    int32_t need = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_gen != 0, nullptr, 0);
    if (need < 0) return make_status(HAG_ERR_UNSUPPORTED, "the model's chat template is not one of llama.cpp's built-in templates");
    std::vector<char> buf((size_t)need + 1);
    int32_t got = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_gen != 0, buf.data(), (int32_t)buf.size());
    if (got < 0 || got > need) return make_status(HAG_ERR_INTERNAL, "chat template formatting failed");
    char * p = (char *)malloc((size_t)got + 1);
    if (!p) return make_status(HAG_ERR_OOM, "out of memory");
    memcpy(p, buf.data(), got);
    p[got] = 0;
    *out_text = p;
    return ok_status();
    HAG_CATCH_STATUS
}

hag_status hag_tokenize(const hag_model * hm, const char * text, int add_special, int32_t * out_tokens, int cap, int * n_out) {
    HAG_TRY
    if (!hm || !text || !n_out) return make_status(HAG_ERR_INVALID_ARG, "null argument");
    const llama_vocab * v = llama_model_get_vocab(hm->model);
    int32_t len = (int32_t)strlen(text);
    std::vector<llama_token> tmp((size_t)len + 8);
    int32_t n = llama_tokenize(v, text, len, tmp.data(), (int32_t)tmp.size(), add_special != 0, true);
    if (n < 0) {
        tmp.resize((size_t)(-n));
        n = llama_tokenize(v, text, len, tmp.data(), (int32_t)tmp.size(), add_special != 0, true);
    }
    if (n < 0) return make_status(HAG_ERR_INTERNAL, "tokenization failed");
    *n_out = n;
    if (out_tokens && cap > 0) memcpy(out_tokens, tmp.data(), sizeof(int32_t) * (size_t)std::min(n, cap));
    if (n > cap && out_tokens) return make_status(HAG_ERR_INVALID_ARG, "token buffer too small: need %d", n);
    return ok_status();
    HAG_CATCH_STATUS
}

// ---- generation ---------------------------------------------------------------------------------------------------
static llama_sampler * build_sampler(const llama_vocab * v, const hag_sample_params & p) {
    llama_sampler_chain_params cp = llama_sampler_chain_default_params();
    cp.no_perf = true;
    llama_sampler * ch = llama_sampler_chain_init(cp);
    if (p.repeat_penalty > 0.0f && p.repeat_penalty != 1.0f) {
        llama_sampler_chain_add(ch, llama_sampler_init_penalties(llama_vocab_n_tokens(v), 64, p.repeat_penalty, 0.0f, 0.0f));
    }
    if (p.temperature <= 0.0f) {
        llama_sampler_chain_add(ch, llama_sampler_init_greedy());
    } else {
        if (p.top_k > 0) llama_sampler_chain_add(ch, llama_sampler_init_top_k(p.top_k));
        if (p.top_p > 0.0f && p.top_p < 1.0f) llama_sampler_chain_add(ch, llama_sampler_init_top_p(p.top_p, 1));
        if (p.min_p > 0.0f) llama_sampler_chain_add(ch, llama_sampler_init_min_p(p.min_p, 1));
        llama_sampler_chain_add(ch, llama_sampler_init_temp(p.temperature));
        llama_sampler_chain_add(ch, llama_sampler_init_dist(p.seed));
    }
    return ch;
}

hag_status hag_generate(hag_session * s, const char * prompt, const hag_sample_params * spp, hag_token_fn sink, void * user,
                        hag_gen_stats * stats) {
    HAG_TRY
    if (!s || !prompt) return make_status(HAG_ERR_INVALID_ARG, "null argument");
    hag_sample_params sp;
    memset(&sp, 0, sizeof(sp));
    sp.temperature = 0.0f; sp.top_p = 1.0f; sp.repeat_penalty = 1.0f; sp.max_new_tokens = 128;
    if (spp) {
        // honour only the fields the caller's struct actually contains
        size_t n = std::min<size_t>(spp->struct_size ? spp->struct_size : sizeof(sp), sizeof(sp));
        memcpy(&sp, spp, n);
    }
    if (sp.max_new_tokens <= 0) sp.max_new_tokens = 128;
    hag_gen_stats st;
    memset(&st, 0, sizeof(st));
    s->cancel.store(false);

    const llama_vocab * v = llama_model_get_vocab(s->model->model);
    llama_memory_t mem = llama_get_memory(s->ctx);
    const int n_past0 = s->n_past;

    // prompt -> tokens (BOS only at the very start of a conversation)
    int32_t plen = (int32_t)strlen(prompt);
    std::vector<llama_token> ptoks((size_t)plen + 8);
    int32_t np = llama_tokenize(v, prompt, plen, ptoks.data(), (int32_t)ptoks.size(), s->n_past == 0, true);
    if (np < 0) {
        ptoks.resize((size_t)(-np));
        np = llama_tokenize(v, prompt, plen, ptoks.data(), (int32_t)ptoks.size(), s->n_past == 0, true);
    }
    if (np < 0) return make_status(HAG_ERR_INTERNAL, "tokenization failed");
    ptoks.resize((size_t)np);
    if (np == 0) return make_status(HAG_ERR_INVALID_ARG, "empty prompt");
    if (s->n_past + np >= s->n_ctx) {
        return make_status(HAG_ERR_INVALID_ARG, "prompt (%d tokens) does not fit the remaining context (%d of %d used)", np, s->n_past, s->n_ctx);
    }
    st.n_prompt_tokens = np;

    llama_sampler * chain = build_sampler(v, sp);
    struct ChainGuard { llama_sampler * c; ~ChainGuard() { llama_sampler_free(c); } } guard{chain};
    for (int i = std::max(0, np - 64); i < np; i++) llama_sampler_accept(chain, ptoks[i]);

    // ---- prompt phase (cancel-aware, chunked)
    double t0 = now_ms();
    int done = 0;
    while (done < np) {
        int n = std::min(np - done, s->n_batch);
        llama_batch b = llama_batch_get_one(ptoks.data() + done, n);
        int32_t rc = llama_decode(s->ctx, b);
        if (rc == 2 || s->cancel.load()) {   // aborted
            llama_memory_seq_rm(mem, 0, n_past0, -1);
            s->n_past = n_past0;
            st.stop_reason = 2;
            st.prompt_ms = now_ms() - t0;
            st.peak_rss_bytes = peak_rss_bytes();
            if (stats) *stats = st;
            return make_status(HAG_ERR_CANCELLED, "cancelled");
        }
        if (rc != 0) {
            llama_memory_seq_rm(mem, 0, n_past0, -1);
            s->n_past = n_past0;
            return make_status(rc == 1 ? HAG_ERR_OOM : HAG_ERR_INTERNAL, "decode failed (%d)", rc);
        }
        done += n;
        s->n_past += n;
    }
    st.prompt_ms = now_ms() - t0;

    // ---- generation phase
    double t1 = now_ms();
    std::string pending, out;
    char piece[256];
    int stop = 1;
    llama_token tok = 0;
    for (int i = 0; i < sp.max_new_tokens; i++) {
        if (s->cancel.load()) { stop = 2; break; }
        tok = llama_sampler_sample(chain, s->ctx, -1);
        llama_sampler_accept(chain, tok);
        if (llama_vocab_is_eog(v, tok)) { stop = 0; break; }
        int32_t pl = llama_token_to_piece(v, tok, piece, sizeof(piece), 0, false);
        if (pl < 0) {
            std::vector<char> big((size_t)(-pl));
            pl = llama_token_to_piece(v, tok, big.data(), (int32_t)big.size(), 0, false);
            out.clear();
            if (pl > 0) utf8_push(pending, big.data(), (size_t)pl, out);
        } else {
            out.clear();
            if (pl > 0) utf8_push(pending, piece, (size_t)pl, out);
        }
        st.n_generated++;
        if (!out.empty() && sink && sink(out.c_str(), user) != 0) { stop = 2; break; }
        if (s->n_past + 1 >= s->n_ctx) { stop = 3; break; }
        llama_batch b = llama_batch_get_one(&tok, 1);
        int32_t rc = llama_decode(s->ctx, b);
        if (rc == 2) { stop = 2; break; }
        if (rc != 0) { st.gen_ms = now_ms() - t1; if (stats) *stats = st; return make_status(HAG_ERR_INTERNAL, "decode failed (%d)", rc); }
        s->n_past++;
    }
    out.clear();
    utf8_flush(pending, out);
    if (!out.empty() && sink) sink(out.c_str(), user);
    st.gen_ms = now_ms() - t1;
    st.stop_reason = stop;
    st.peak_rss_bytes = peak_rss_bytes();
    if (stats) *stats = st;
    return ok_status();
    HAG_CATCH_STATUS
}

// ---- scoring ------------------------------------------------------------------------------------------------------
hag_status hag_score(hag_session * s, const char * text, double * mean_nll, int * n_tokens) {
    HAG_TRY
    if (!s || !text || !mean_nll) return make_status(HAG_ERR_INVALID_ARG, "null argument");
    s->cancel.store(false);
    const llama_vocab * v = llama_model_get_vocab(s->model->model);
    const int n_vocab = llama_vocab_n_tokens(v);
    int32_t len = (int32_t)strlen(text);
    std::vector<llama_token> toks((size_t)len + 8);
    int32_t n = llama_tokenize(v, text, len, toks.data(), (int32_t)toks.size(), true, true);
    if (n < 0) { toks.resize((size_t)(-n)); n = llama_tokenize(v, text, len, toks.data(), (int32_t)toks.size(), true, true); }
    if (n < 2) return make_status(HAG_ERR_INVALID_ARG, "text is too short to score (%d tokens)", n);
    toks.resize((size_t)n);

    // The session KV is used as scratch: scoring clobbers any conversation state.
    llama_memory_t mem = llama_get_memory(s->ctx);
    llama_memory_clear(mem, true);
    s->n_past = 0;

    const int W = std::min(std::min(s->n_ctx, s->n_batch), 256);  // window; consecutive windows overlap by one token
    double sum = 0.0;
    int64_t counted = 0;
    std::vector<llama_token> win;
    for (int start = 0; start < n - 1;) {
        int wlen = std::min(W, n - start);
        if (wlen < 2) break;
        llama_memory_clear(mem, true);
        llama_batch b = llama_batch_init(wlen, 0, 1);
        b.n_tokens = wlen;
        for (int i = 0; i < wlen; i++) {
            b.token[i] = toks[start + i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        int32_t rc = llama_decode(s->ctx, b);
        llama_batch_free(b);
        if (rc == 2 || s->cancel.load()) { llama_memory_clear(mem, true); return make_status(HAG_ERR_CANCELLED, "cancelled"); }
        if (rc != 0) { llama_memory_clear(mem, true); return make_status(HAG_ERR_INTERNAL, "decode failed (%d)", rc); }
        for (int i = 0; i < wlen - 1; i++) {
            const float * lg = llama_get_logits_ith(s->ctx, i);
            if (!lg) { llama_memory_clear(mem, true); return make_status(HAG_ERR_INTERNAL, "no logits"); }
            float mx = lg[0];
            for (int k = 1; k < n_vocab; k++) mx = std::max(mx, lg[k]);
            double z = 0.0;
            for (int k = 0; k < n_vocab; k++) z += std::exp((double)lg[k] - mx);
            sum += -((double)lg[toks[start + i + 1]] - mx - std::log(z));
            counted++;
        }
        start += wlen - 1;
    }
    llama_memory_clear(mem, true);
    s->n_past = 0;
    *mean_nll = sum / (double)counted;
    if (n_tokens) *n_tokens = (int)counted;
    return ok_status();
    HAG_CATCH_STATUS
}
