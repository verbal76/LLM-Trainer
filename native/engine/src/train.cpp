// HAG engine: on-device specialization (parameter-changing training), checkpoints and patch files.
// See docs/v2/ENGINE.md (patch format v1, checkpoint format) and docs/v2/TRAINING_FEASIBILITY.md (measured limits).
#include "hag_internal.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#if !defined(_WIN32)
#  include <dirent.h>
#  include <sys/stat.h>
#endif

using namespace hag;

#define HAG_TRY try {
#define HAG_CATCH_STATUS \
    } catch (const std::bad_alloc &) { return make_status(HAG_ERR_OOM, "out of memory"); } \
      catch (const std::exception & e) { return make_status(HAG_ERR_INTERNAL, "internal error: %s", e.what()); } \
      catch (...) { return make_status(HAG_ERR_INTERNAL, "internal error"); }

namespace hag {

bool type_convertible_from_f32(ggml_type t) {
    if (t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16) return true;
    return ggml_is_quantized(t) && !ggml_quantize_requires_imatrix(t);
}

size_t convert_rows_from_f32(ggml_type t, const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    switch (t) {
        case GGML_TYPE_F32: memcpy(dst, src, sizeof(float) * (size_t)(nrows * n_per_row)); return sizeof(float) * (size_t)(nrows * n_per_row);
        case GGML_TYPE_F16: ggml_fp32_to_fp16_row(src, (ggml_fp16_t *)dst, nrows * n_per_row); return 2 * (size_t)(nrows * n_per_row);
        case GGML_TYPE_BF16: ggml_fp32_to_bf16_row(src, (ggml_bf16_t *)dst, nrows * n_per_row); return 2 * (size_t)(nrows * n_per_row);
        default:
            if (!type_convertible_from_f32(t)) return 0;
            ggml_quantize_init(t);
            return ggml_quantize_chunk(t, src, dst, 0, nrows, n_per_row, nullptr);
    }
}

}  // namespace hag

namespace {

constexpr uint32_t kAlgoVersion = 1;  // bump when training maths / data packing / checkpoint semantics change

// ---- parameters ----------------------------------------------------------------------------------------------------
struct Cfg {
    int      n_ctx_req = 128, tokens_per_step = 128, accum = 1, epochs = 1, threads = 4, last_layers = 0, ckpt_every = 0;
    float    lr = 1e-4f, val_fraction = 0.f;
    uint32_t seed = 1;
    bool     train_emb = false;
    size_t   max_mem = 0;
};

static int auto_threads_train() {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) return 4;
    if (hw <= 4) return (int)hw;
    return (int)std::min<unsigned>(8, hw * 3 / 4);
}

static std::string normalize(const hag_train_params * in, Cfg & c) {
    hag_train_params p;
    memset(&p, 0, sizeof(p));
    p.struct_size = sizeof(p);
    p.n_ctx = 128; p.epochs = 1; p.learning_rate = 1e-4f;
    if (in) {
        if (in->struct_size < offsetof(hag_train_params, max_memory_bytes) && in->struct_size != 0) {
            // allow older shorter structs, but not ones that miss the fundamental fields
            if (in->struct_size < offsetof(hag_train_params, n_threads)) return "train params struct_size too small";
        }
        size_t n = std::min<size_t>(in->struct_size ? in->struct_size : sizeof(p), sizeof(p));
        memcpy(&p, in, n);
    }
    c.n_ctx_req = p.n_ctx > 0 ? p.n_ctx : 128;
    if (c.n_ctx_req < 8) return "n_ctx must be >= 8";
    c.tokens_per_step = p.n_batch > 0 ? p.n_batch : c.n_ctx_req;
    c.accum = std::max(1, c.tokens_per_step / c.n_ctx_req);
    c.epochs = p.epochs > 0 ? p.epochs : 1;
    c.lr = p.learning_rate > 0 ? p.learning_rate : 1e-4f;
    if (!(c.lr > 0) || !std::isfinite(c.lr)) return "invalid learning rate";
    c.val_fraction = std::min(0.5f, std::max(0.f, p.val_fraction));
    c.seed = p.seed;
    c.threads = p.n_threads > 0 ? p.n_threads : auto_threads_train();
    c.last_layers = std::max(0, p.trainable_last_layers);
    c.train_emb = p.train_embeddings != 0;
    c.ckpt_every = std::max(0, p.checkpoint_every_steps);
    c.max_mem = p.max_memory_bytes;
    return "";
}

// ---- base model description ---------------------------------------------------------------------------------------
struct TInfo {
    std::string name;
    ggml_type   type;
    int64_t     ne[4];
    size_t      nbytes;      // as stored
    uint64_t    offset;      // within the data section of the base file
    int64_t     nelem;
    int         layer = -1;  // blk.N
    bool        trainable = false;
};

struct BaseInfo {
    std::string arch;
    int         n_layer = 0;
    std::vector<TInfo> t;
    size_t      data_offset = 0;
    size_t      alignment = 32;
    int64_t     file_size = 0;
    bool        has_output = false;   // separate output.weight (untied)
    uint64_t    n_params = 0;
    std::string file_type;
};

static int parse_layer(const std::string & n) {
    if (n.compare(0, 4, "blk.") != 0) return -1;
    return atoi(n.c_str() + 4);
}

static std::string read_base(const char * path, BaseInfo & b) {
    b.file_size = file_size(path);
    if (b.file_size < 0) return "cannot open base model file";
    gguf_init_params ip = {true, nullptr};
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) return "base file is not a valid GGUF";
    struct G { gguf_context * g; ~G() { gguf_free(g); } } guard{g};
    int64_t id = gguf_find_key(g, "general.architecture");
    if (id < 0) return "base GGUF has no general.architecture";
    b.arch = gguf_get_val_str(g, id);
    id = gguf_find_key(g, (b.arch + ".block_count").c_str());
    if (id < 0) return "base GGUF has no block_count";
    b.n_layer = (int)gguf_get_val_u32(g, id);
    id = gguf_find_key(g, "general.file_type");
    if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_UINT32) b.file_type = std::to_string(gguf_get_val_u32(g, id));
    b.data_offset = gguf_get_data_offset(g);
    b.alignment = gguf_get_alignment(g);
    int64_t nt = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < nt; i++) {
        TInfo t;
        t.name = gguf_get_tensor_name(g, i);
        t.type = gguf_get_tensor_type(g, i);
        const int64_t * ne = gguf_get_tensor_ne(g, i);
        for (int d = 0; d < 4; d++) t.ne[d] = ne[d];
        t.nbytes = gguf_get_tensor_size(g, i);
        t.offset = gguf_get_tensor_offset(g, i);
        t.nelem = ne[0] * ne[1] * ne[2] * ne[3];
        t.layer = parse_layer(t.name);
        if (t.name == "output.weight") b.has_output = true;
        b.n_params += (uint64_t)t.nelem;
        b.t.push_back(std::move(t));
    }
    return "";
}

// ---- training plan --------------------------------------------------------------------------------------------------
struct Plan {
    bool        ok = false;
    std::string reason;                 // if !ok
    int         first_layer = 0;
    uint64_t    p_train = 0;
    int         n_train_tensors = 0;
    bool        needs_working_copy = false;
    uint64_t    weights_bytes = 0;      // what the (non-mmap) working model occupies
    uint64_t    base_bytes = 0;         // sum of stored tensor bytes
    bool        tied = false;
    bool        emb_effective = false;  // train_embeddings actually trains something
    bool        base_quantized = false;
};

static bool type_to_f32_ok(ggml_type t) {
    if (t == GGML_TYPE_F32) return true;
    const ggml_type_traits * tr = ggml_get_type_traits(t);
    return tr && tr->to_float != nullptr;
}

static Plan make_plan(BaseInfo & b, const Cfg & c) {
    Plan p;
    if (b.arch != "llama") {
        p.reason = "architecture '" + b.arch + "' is not validated for on-device training (validated: llama)";
        return p;
    }
    p.first_layer = (c.last_layers <= 0 || c.last_layers >= b.n_layer) ? 0 : b.n_layer - c.last_layers;
    p.tied = !b.has_output;
    for (TInfo & t : b.t) {
        t.trainable = false;
        if (t.layer >= p.first_layer) t.trainable = true;
        if (t.name == "output_norm.weight" || t.name == "output_norm.bias") t.trainable = true;
        if (c.train_emb && (t.name == "output.weight" || t.name == "output.bias")) { t.trainable = true; p.emb_effective = true; }
        if (t.type != GGML_TYPE_F32) p.base_quantized = true;
        p.base_bytes += t.nbytes;
        if (t.trainable) {
            if (!type_to_f32_ok(t.type)) { p.reason = "tensor " + t.name + " has a type that cannot be converted to F32 for training"; return p; }
            if (!type_convertible_from_f32(t.type)) { p.reason = "tensor " + t.name + ": cannot convert trained weights back to this type (needs an importance matrix)"; return p; }
            p.p_train += (uint64_t)t.nelem;
            p.n_train_tensors++;
            if (t.type != GGML_TYPE_F32) p.needs_working_copy = true;
            p.weights_bytes += (uint64_t)t.nelem * 4;
        } else {
            p.weights_bytes += t.nbytes;
        }
    }
    if (p.n_train_tensors == 0) { p.reason = "no trainable tensors selected"; return p; }
    if (p.tied) p.weights_bytes += 0;  // duplicated head accounted for in the activation model (measured)
    p.ok = true;
    return p;
}

// Peak-memory model (bytes). Calibrated against measured RSS, see docs/v2/TRAINING_FEASIBILITY.md.
struct Est {
    uint64_t weights = 0, optimizer = 0, activations = 0, kv = 0, logits = 0, overhead = 0, total = 0;
    uint64_t disk_working = 0, disk_ckpt = 0, patch_bytes = 0;
};

static Est estimate(const BaseInfo & b, const Plan & p, int n_ctx, int n_embd, int n_ff, int n_head, int n_vocab, int n_embd_kv) {
    Est e;
    const uint64_t n = (uint64_t)n_ctx;
    const int n_tr_layers = b.n_layer - p.first_layer;
    e.weights = p.weights_bytes;
    // duplicated output head for tied embeddings (loaded as its own tensor in the non-mmap model)
    if (p.tied) e.weights += (uint64_t)n_vocab * n_embd * 4;  // upper bound (F32)
    e.optimizer = 12ull * p.p_train;  // 4 B gradient accumulator + 8 B Adam moments per trainable parameter
    // per trained layer: activations kept for the backward pass, ~ (c_e*d + c_f*ff + c_kv*dkv) floats per token + attention scores
    const double per_tok = 4.0 * (14.0 * n_embd + 5.0 * n_ff + 4.0 * n_embd_kv);
    const double scores  = 4.0 * 2.0 * (double)n_head * (double)n * (double)n;
    e.activations = (uint64_t)((per_tok * (double)n + scores) * n_tr_layers);
    // frozen (forward-only) layers only need transient buffers
    e.activations += (uint64_t)(per_tok * (double)n * 1.0);
    e.logits = n * (uint64_t)n_vocab * 4ull * 5ull;  // logits, one-hot labels, softmax, gradients
    e.kv = n * (uint64_t)b.n_layer * 2ull * (uint64_t)n_embd_kv * 4ull;  // F32 K/V
    e.overhead = 64ull << 20;  // runtime, vocab, graph metadata
    e.total = e.weights + e.optimizer + e.activations + e.logits + e.kv + e.overhead;
    e.disk_working = p.needs_working_copy ? p.weights_bytes : 0;
    e.disk_ckpt = 2ull * 12ull * p.p_train + (1ull << 20);
    for (const TInfo & t : b.t) if (t.trainable) e.patch_bytes += t.nbytes;
    return e;
}

// vocabulary/hparams only (cheap): needed for the estimate and for tokenization
struct HParams { int n_embd = 0, n_ff = 0, n_head = 0, n_vocab = 0, n_embd_kv = 0, n_ctx_train = 0; };

static std::string read_hparams(const char * path, const BaseInfo & b, HParams & h) {
    gguf_init_params ip = {true, nullptr};
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) return "cannot re-read base";
    auto u32 = [&](const char * suffix) -> int {
        int64_t id = gguf_find_key(g, (b.arch + "." + suffix).c_str());
        if (id < 0) return 0;
        switch (gguf_get_kv_type(g, id)) {
            case GGUF_TYPE_UINT32: return (int)gguf_get_val_u32(g, id);
            case GGUF_TYPE_INT32: return (int)gguf_get_val_i32(g, id);
            case GGUF_TYPE_UINT64: return (int)gguf_get_val_u64(g, id);
            default: return 0;
        }
    };
    h.n_embd = u32("embedding_length");
    h.n_ff = u32("feed_forward_length");
    h.n_head = u32("attention.head_count");
    int nkv = u32("attention.head_count_kv");
    if (nkv <= 0) nkv = h.n_head;
    h.n_ctx_train = u32("context_length");
    h.n_embd_kv = h.n_head > 0 ? h.n_embd / h.n_head * nkv : h.n_embd;
    for (const TInfo & t : b.t) if (t.name == "token_embd.weight") h.n_vocab = (int)t.ne[1];
    gguf_free(g);
    if (!h.n_embd || !h.n_head || !h.n_vocab) return "base model hyper-parameters missing";
    return "";
}

// ---- dataset ---------------------------------------------------------------------------------------------------------
struct Dataset {
    std::vector<int32_t> train, val;     // token streams
    std::vector<int64_t> train_win, val_win;  // window start offsets (each window = n_ctx inputs + 1 label)
    int       n_ctx = 0;
    int       n_train_docs = 0, n_val_docs = 0;
    std::string text_sha, tokens_sha;
};

static void make_windows(size_t len, int n, std::vector<int64_t> & out) {
    out.clear();
    if (len < (size_t)n + 1) return;
    size_t nwin = (len - 1 + n - 1) / n;  // ceil((len-1)/n)
    for (size_t k = 0; k + 1 < nwin; k++) out.push_back((int64_t)(k * n));
    out.push_back((int64_t)(len - 1 - n));  // last window anchored at the end (may overlap the previous one)
}

static std::string build_dataset(const llama_vocab * vocab, const char * const * texts, int n_texts, const Cfg & c, Dataset & d) {
    if (n_texts <= 0 || !texts) return "no training texts";
    Sha256 th;
    th.update("HAGDS1", 6);
    uint64_t nt = (uint64_t)n_texts;
    th.update(&nt, 8);
    std::vector<std::vector<int32_t>> docs((size_t)n_texts);
    const llama_token eos = llama_vocab_eos(vocab);
    for (int i = 0; i < n_texts; i++) {
        if (!texts[i]) return "null training text";
        size_t len = strlen(texts[i]);
        uint64_t l64 = len;
        th.update(&l64, 8);
        th.update(texts[i], len);
        std::vector<llama_token> tk(len + 8);
        int32_t n = llama_tokenize(vocab, texts[i], (int32_t)len, tk.data(), (int32_t)tk.size(), true, true);
        if (n < 0) { tk.resize((size_t)-n); n = llama_tokenize(vocab, texts[i], (int32_t)len, tk.data(), (int32_t)tk.size(), true, true); }
        if (n < 0) return "tokenization failed";
        tk.resize((size_t)n);
        if (eos != LLAMA_TOKEN_NULL && (tk.empty() || tk.back() != eos)) tk.push_back(eos);
        docs[i].assign(tk.begin(), tk.end());
    }
    d.text_sha = th.final_hex();

    // split documents: val chosen by a seeded permutation, the rest stay in their original order
    int n_val = 0;
    if (c.val_fraction > 0.f && n_texts >= 2) {
        n_val = std::max(1, (int)std::floor(c.val_fraction * n_texts));
        n_val = std::min(n_val, n_texts - 1);
    }
    std::vector<int> perm((size_t)n_texts);
    for (int i = 0; i < n_texts; i++) perm[i] = i;
    SplitMix64 rng(((uint64_t)c.seed << 1) ^ 0xD1CE5EEDull);
    for (int i = n_texts - 1; i > 0; i--) std::swap(perm[i], perm[(size_t)rng.below((uint64_t)i + 1)]);
    std::vector<char> is_val((size_t)n_texts, 0);
    for (int i = 0; i < n_val; i++) is_val[perm[i]] = 1;
    for (int i = 0; i < n_texts; i++) {
        auto & dst = is_val[i] ? d.val : d.train;
        dst.insert(dst.end(), docs[i].begin(), docs[i].end());
        (is_val[i] ? d.n_val_docs : d.n_train_docs)++;
    }
    if (d.train.size() < 9) return "training texts are too short (fewer than 9 tokens after tokenization)";
    int n = c.n_ctx_req;
    n = std::min<int64_t>(n, (int64_t)d.train.size() - 1);
    if (n_val > 0) n = std::min<int64_t>(n, (int64_t)d.val.size() - 1);
    if (n < 4) return "validation text is too short for a window; use val_fraction=0 or longer texts";
    d.n_ctx = n;
    make_windows(d.train.size(), n, d.train_win);
    if (n_val > 0) make_windows(d.val.size(), n, d.val_win);
    Sha256 ts;
    ts.update("HAGTOK1", 7);
    ts.update(d.train.data(), d.train.size() * 4);
    uint64_t sep = d.train.size();
    ts.update(&sep, 8);
    ts.update(d.val.data(), d.val.size() * 4);
    d.tokens_sha = ts.final_hex();
    return "";
}

// ---- checkpoints ------------------------------------------------------------------------------------------------------
struct CkptTensor { std::string name; uint64_t n; bool has_state; uint64_t off; };  // off = file offset of the weights blob

struct CkptMeta {
    std::string fingerprint;
    uint64_t step = 0, next_epoch = 0, next_k = 0;
    int64_t  opt_iter = 1;
    uint64_t bits_train_first = 0, bits_train_last = 0, bits_val_first = 0, bits_val_last = 0;
    int      has_train_first = 0, has_val_first = 0, has_val_last = 0;
    std::vector<CkptTensor> tensors;
    uint64_t header_len = 0;
};

static uint64_t dbits(double v) { uint64_t u; memcpy(&u, &v, 8); return u; }
static double   bitsd(uint64_t u) { double v; memcpy(&v, &u, 8); return v; }

static std::string ckpt_name(uint64_t step) { char b[48]; snprintf(b, sizeof(b), "ckpt-%010" PRIu64 ".bin", step); return b; }

static std::vector<std::string> list_dir(const std::string & dir) {
    std::vector<std::string> out;
#if !defined(_WIN32)
    DIR * d = opendir(dir.c_str());
    if (!d) return out;
    while (dirent * e = readdir(d)) out.push_back(e->d_name);
    closedir(d);
#else
    (void)dir;
#endif
    std::sort(out.begin(), out.end());
    return out;
}

struct Manifest { struct E { uint64_t step; uint64_t size; std::string sha; std::string file; }; std::vector<E> e; };

static Manifest read_manifest(const std::string & dir) {
    Manifest m;
    std::string s;
    if (!read_file(join_path(dir, "ckpt.manifest"), s, 1 << 20)) return m;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t nl = s.find('\n', pos);
        if (nl == std::string::npos) break;
        std::string line = s.substr(pos, nl - pos);
        pos = nl + 1;
        unsigned long long st, sz;
        char sha[80], fn[128];
        if (sscanf(line.c_str(), "%llu %llu %64s %127s", &st, &sz, sha, fn) == 4) m.e.push_back({st, sz, sha, fn});
    }
    return m;
}

static bool write_manifest(const std::string & dir, const Manifest & m) {
    std::string s;
    for (auto & e : m.e) {
        char b[400];
        snprintf(b, sizeof(b), "%llu %llu %s %s\n", (unsigned long long)e.step, (unsigned long long)e.size, e.sha.c_str(), e.file.c_str());
        s += b;
    }
    AtomicWriter w(join_path(dir, "ckpt.manifest"));
    return w.ok() && w.write(s.data(), s.size()) && w.commit();
}

// Verifies integrity (single pass) and parses the header. Returns "" on success.
static std::string ckpt_verify(const std::string & path, CkptMeta & m, std::string * sha_out) {
    int64_t sz = file_size(path);
    if (sz < 128) return "file too small";
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return "cannot open";
    struct F { FILE * f; ~F() { fclose(f); } } fg{f};
    const int64_t body = sz - 65;
    Sha256 h;
    std::vector<uint8_t> buf(1 << 20);
    int64_t left = body;
    while (left > 0) {
        size_t n = (size_t)std::min<int64_t>(left, (int64_t)buf.size());
        if (fread(buf.data(), 1, n, f) != n) return "short read";
        h.update(buf.data(), n);
        left -= (int64_t)n;
    }
    char trailer[66];
    if (fread(trailer, 1, 65, f) != 65) return "short read (trailer)";
    trailer[65] = 0;
    std::string have = h.final_hex();
    if (strncmp(trailer, have.c_str(), 64) != 0 || trailer[64] != '\n') return "sha256 mismatch";
    if (sha_out) *sha_out = have;
    // header
    if (fseek64(f, 0, SEEK_SET) != 0) return "seek";
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "HAGCKP1\n", 8) != 0) return "bad magic";
    uint64_t hl;
    if (fread(&hl, 8, 1, f) != 1 || hl == 0 || hl > (1u << 24) || (int64_t)(16 + hl) > body) return "bad header length";
    std::string hdr(hl, '\0');
    if (fread(&hdr[0], 1, hl, f) != hl) return "short header";
    m.header_len = 16 + hl;
    uint64_t off = m.header_len;
    size_t pos = 0;
    while (pos < hdr.size()) {
        size_t nl = hdr.find('\n', pos);
        if (nl == std::string::npos) break;
        std::string line = hdr.substr(pos, nl - pos);
        pos = nl + 1;
        char key[64], val[256];
        if (line.compare(0, 2, "t ") == 0) {
            unsigned long long n; int hs; char nm[200];
            if (sscanf(line.c_str(), "t %llu %d %199s", &n, &hs, nm) != 3) return "bad tensor line";
            m.tensors.push_back({nm, n, hs != 0, off});
            off += n * 4 * (hs ? 3 : 1);
        } else if (sscanf(line.c_str(), "%63[^=]=%255s", key, val) == 2) {
            std::string k = key;
            if (k == "fingerprint") m.fingerprint = val;
            else if (k == "step") m.step = strtoull(val, nullptr, 10);
            else if (k == "next_epoch") m.next_epoch = strtoull(val, nullptr, 10);
            else if (k == "next_k") m.next_k = strtoull(val, nullptr, 10);
            else if (k == "opt_iter") m.opt_iter = strtoll(val, nullptr, 10);
            else if (k == "train_first") { m.has_train_first = 1; m.bits_train_first = strtoull(val, nullptr, 16); }
            else if (k == "train_last") m.bits_train_last = strtoull(val, nullptr, 16);
            else if (k == "val_first") { m.has_val_first = 1; m.bits_val_first = strtoull(val, nullptr, 16); }
            else if (k == "val_last") { m.has_val_last = 1; m.bits_val_last = strtoull(val, nullptr, 16); }
        }
    }
    if ((int64_t)off != body) return "tensor sizes do not add up to file size";
    if (m.fingerprint.size() != 64) return "no fingerprint";
    return "";
}

struct RunLog {
    std::string path;
    void line(const char * fmt, ...) __attribute__((format(printf, 2, 3))) {
        if (path.empty()) return;
        char b[512];
        va_list ap; va_start(ap, fmt); vsnprintf(b, sizeof(b), fmt, ap); va_end(ap);
        FILE * f = fopen(path.c_str(), "ab");
        if (!f) return;
        fprintf(f, "%s\n", b);
        fclose(f);
    }
};

}  // namespace

// ---- the trainer -----------------------------------------------------------------------------------------------------
namespace {

struct Hooks {
    hag_train_fn cb = nullptr;
    void * user = nullptr;
    double t0 = 0;
    bool   cancelled = false;
    int32_t resumed_from = 0;
    int32_t epochs = 0, steps_total = 0;
    double last_event_ms = 0;
    // returns true if the caller asked to cancel
    bool emit(int phase, int epoch, int step, int64_t examples, double train_loss, double val_loss) {
        if (!cb) return false;
        hag_train_event ev;
        memset(&ev, 0, sizeof(ev));
        ev.phase = phase; ev.epoch = epoch; ev.epochs = epochs; ev.step = step; ev.steps = steps_total;
        ev.examples_done = examples; ev.train_loss = train_loss; ev.val_loss = val_loss;
        ev.elapsed_s = (now_ms() - t0) / 1000.0; ev.rss_bytes = rss_bytes(); ev.resumed_from_step = resumed_from;
        last_event_ms = now_ms();
        if (cb(&ev, user) != 0) cancelled = true;
        return cancelled;
    }
};

static const double kNaN = std::nan("");

struct OptCtlState {
    const std::vector<std::string> * names = nullptr;
    // resume source
    std::string    ckpt_path;
    CkptMeta       meta;
    std::unordered_map<std::string, const CkptTensor *> by_name;
    bool           restore = false;
    bool           restore_failed = false;
};

static void state_init_cb(ggml_opt_context_t opt, void * ud) {
    OptCtlState * st = (OptCtlState *)ud;
    if (!st->restore) return;
    FILE * f = fopen(st->ckpt_path.c_str(), "rb");
    if (!f) { st->restore_failed = true; return; }
    ggml_opt_set_iter(opt, st->meta.opt_iter);
    int n = ggml_opt_n_state(opt);
    std::vector<float> tmp;
    for (int i = 0; i < n; i++) {
        ggml_tensor *m, *v;
        if (!ggml_opt_state_at(opt, i, &m, &v)) continue;
        std::string nm = m->name;
        const std::string pre = "AdamW m for ";
        if (nm.compare(0, pre.size(), pre) != 0) { st->restore_failed = true; break; }
        nm = nm.substr(pre.size());
        auto it = st->by_name.find(nm);
        if (it == st->by_name.end() || !it->second->has_state || (int64_t)it->second->n != ggml_nelements(m)) { st->restore_failed = true; break; }
        const CkptTensor * ct = it->second;
        tmp.resize(ct->n);
        for (int k = 0; k < 2; k++) {
            if (fseek64(f, (int64_t)(ct->off + ct->n * 4 * (1 + k)), SEEK_SET) != 0 || fread(tmp.data(), 4, ct->n, f) != ct->n) { st->restore_failed = true; break; }
            ggml_backend_tensor_set(k == 0 ? m : v, tmp.data(), 0, ct->n * 4);
        }
        if (st->restore_failed) break;
    }
    fclose(f);
}

struct LrState { float lr; };
static ggml_opt_optimizer_params get_pars(void * ud) {
    ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
    p.adamw.alpha = ((LrState *)ud)->lr;
    p.adamw.beta1 = 0.9f; p.adamw.beta2 = 0.999f; p.adamw.eps = 1e-8f; p.adamw.wd = 0.0f;
    return p;
}
static bool param_filter(const ggml_tensor * t, void * ud) {
    return ((std::unordered_set<std::string> *)ud)->count(t->name) != 0;
}

static std::string base_sha_cached(const char * base_path, const std::string & work_dir, bool (*cancel)(void *), void * ud, bool & cancelled) {
    int64_t sz = file_size(base_path);
    int64_t mt = 0;
#if !defined(_WIN32)
    struct stat st;
    if (stat(base_path, &st) == 0) mt = (int64_t)st.st_mtime;
#endif
    Sha256 k;
    k.update(base_path, strlen(base_path));
    k.update(&sz, 8);
    k.update(&mt, 8);
    std::string cache = join_path(work_dir, "base_" + k.final_hex().substr(0, 16) + ".sha256");
    std::string s;
    if (read_file(cache, s, 256) && s.size() >= 64) {
        std::string hx = s.substr(0, 64);
        bool ok = true;
        for (char ch : hx) if (!isxdigit((unsigned char)ch)) ok = false;
        if (ok) return hx;
    }
    std::string hx;
    if (!sha256_file(base_path, hx, cancel, ud)) { cancelled = true; return ""; }
    AtomicWriter w(cache);
    if (w.ok()) { w.write(hx.data(), hx.size()); w.commit(); }
    return hx;
}

struct Ctx {
    const char * base_path;
    Cfg c;
    Hooks hk;
    RunLog log;
    std::string work_dir;
};

static bool cancel_poll_cb(void * ud) {
    Hooks * h = (Hooks *)ud;
    if (now_ms() - h->last_event_ms > 700) h->emit(HAG_PHASE_PREPARE, 0, 0, 0, kNaN, kNaN);
    return h->cancelled;
}

// Builds `out_path`: the base with every tensor flagged in `convert` expanded to F32; others copied verbatim.
static std::string build_working_copy(const char * base_path, const BaseInfo & b, const std::string & out_path, Hooks & hk) {
    gguf_init_params ip = {true, nullptr};
    gguf_context * src = gguf_init_from_file(base_path, ip);
    if (!src) return "cannot re-open base";
    struct G { gguf_context * g; ~G() { if (g) gguf_free(g); } } sg{src};
    ggml_init_params gp = {b.t.size() * ggml_tensor_overhead() + (1u << 20), nullptr, true};
    ggml_context * ctx = ggml_init(gp);
    struct C { ggml_context * c; ~C() { ggml_free(c); } } cg{ctx};
    gguf_context * out = gguf_init_empty();
    G og{out};
    gguf_set_kv(out, src);
    for (const TInfo & t : b.t) {
        ggml_type ty = t.trainable ? GGML_TYPE_F32 : t.type;
        int nd = 4;
        while (nd > 1 && t.ne[nd - 1] == 1) nd--;
        ggml_tensor * g = ggml_new_tensor(ctx, ty, nd, t.ne);
        ggml_set_name(g, t.name.c_str());
        gguf_add_tensor(out, g);
    }
    const size_t meta = gguf_get_meta_size(out);
    std::vector<uint8_t> mbuf(meta);
    gguf_get_meta_data(out, mbuf.data());

    AtomicWriter w(out_path);
    if (!w.ok()) return "cannot create working copy file";
    w.write(mbuf.data(), meta);
    FILE * f = fopen(base_path, "rb");
    if (!f) return "cannot open base";
    struct F { FILE * f; ~F() { fclose(f); } } fg{f};
    std::vector<uint8_t> raw;
    std::vector<float> fl;
    static const uint8_t zeros[64] = {0};
    for (size_t i = 0; i < b.t.size(); i++) {
        const TInfo & t = b.t[i];
        uint64_t want = gguf_get_tensor_offset(out, (int64_t)i);
        uint64_t have = w.written() - meta;
        while (have < want) { size_t n = (size_t)std::min<uint64_t>(sizeof(zeros), want - have); w.write(zeros, n); have += n; }
        const int64_t ne0 = t.ne[0];
        const int64_t nrows = t.ne[1] * t.ne[2] * t.ne[3];
        const size_t src_row = ggml_row_size(t.type, ne0);
        if (fseek64(f, (int64_t)(b.data_offset + t.offset), SEEK_SET) != 0) return "seek failed in base";
        const int64_t rows_per_chunk = std::max<int64_t>(1, (4 << 20) / std::max<int64_t>(1, ne0));
        for (int64_t r = 0; r < nrows; r += rows_per_chunk) {
            int64_t nr = std::min(rows_per_chunk, nrows - r);
            raw.resize(src_row * (size_t)nr);
            if (fread(raw.data(), 1, raw.size(), f) != raw.size()) return "short read from base";
            if (t.trainable && t.type != GGML_TYPE_F32) {
                fl.resize((size_t)(nr * ne0));
                const ggml_type_traits * tr = ggml_get_type_traits(t.type);
                for (int64_t k = 0; k < nr; k++) tr->to_float(raw.data() + src_row * (size_t)k, fl.data() + (size_t)(k * ne0), ne0);
                if (!w.write(fl.data(), fl.size() * 4)) return "write failed (disk full?)";
            } else {
                if (!w.write(raw.data(), raw.size())) return "write failed (disk full?)";
            }
        }
        if (hk.emit(HAG_PHASE_PREPARE, 0, 0, 0, kNaN, kNaN)) return "cancelled";
    }
    if (!w.commit()) return "cannot finalize working copy";
    return "";
}

}  // namespace

// ---- public: estimate --------------------------------------------------------------------------------------------------
static hag_status copy_json(const std::string & js, char ** out) {
    char * p = (char *)malloc(js.size() + 1);
    if (!p) return make_status(HAG_ERR_OOM, "out of memory");
    memcpy(p, js.c_str(), js.size() + 1);
    *out = p;
    return ok_status();
}

hag_status hag_train_estimate(const char * base_gguf_path, const hag_train_params * params, char ** out_json) {
    HAG_TRY
    if (!base_gguf_path || !out_json) return make_status(HAG_ERR_INVALID_ARG, "null argument");
    *out_json = nullptr;
    Cfg c;
    std::string err = normalize(params, c);
    if (!err.empty()) return make_status(HAG_ERR_INVALID_ARG, "%s", err.c_str());
    BaseInfo b;
    err = read_base(base_gguf_path, b);
    if (!err.empty()) return make_status(HAG_ERR_BAD_MODEL, "%s", err.c_str());
    Plan p = make_plan(b, c);
    HParams hp;
    if (p.ok) { err = read_hparams(base_gguf_path, b, hp); if (!err.empty()) { p.ok = false; p.reason = err; } }
    Json j;
    j.begin_obj();
    j.kv("trainable", p.ok);
    j.kv("arch", b.arch);
    j.kv("n_layer", b.n_layer);
    j.kv("n_params", b.n_params);
    j.kv("base_quantized", p.base_quantized);
    if (!p.ok) {
        j.kv("reason", p.reason);
        j.end_obj();
        return copy_json(j.str(), out_json);
    }
    int n_ctx = std::min(c.n_ctx_req, hp.n_ctx_train > 0 ? hp.n_ctx_train : c.n_ctx_req);
    Est e = estimate(b, p, n_ctx, hp.n_embd, hp.n_ff, hp.n_head, hp.n_vocab, hp.n_embd_kv);
    j.kv("n_ctx", n_ctx);
    j.kv("sequences_per_step", c.accum);
    j.kv("trainable_tensors", p.n_train_tensors);
    j.kv("trainable_params", p.p_train);
    j.kv("first_trainable_layer", p.first_layer);
    j.kv("tied_embeddings", p.tied);
    j.kv("train_embeddings_effective", p.emb_effective);
    j.kv("working_copy_needed", p.needs_working_copy);
    j.begin_obj("bytes");
    j.kv("weights", e.weights); j.kv("optimizer", e.optimizer); j.kv("activations", e.activations);
    j.kv("logits", e.logits); j.kv("kv", e.kv); j.kv("overhead", e.overhead);
    j.end_obj();
    j.kv("estimated_peak_bytes", e.total);
    j.kv("disk_working_copy_bytes", e.disk_working);
    j.kv("disk_checkpoint_bytes", e.disk_ckpt);
    j.kv("patch_bytes", e.patch_bytes);
    j.kv("disk_total_bytes", e.disk_working + e.disk_ckpt + e.patch_bytes);
    if (c.max_mem) j.kv("fits_max_memory", e.total <= c.max_mem);
    j.end_obj();
    return copy_json(j.str(), out_json);
    HAG_CATCH_STATUS
}

// ---- public: train ------------------------------------------------------------------------------------------------------
namespace {

struct Loaded {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    ~Loaded() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
};

static std::string write_ckpt(const std::string & dir, uint64_t step, const std::string & fingerprint, uint64_t next_epoch, uint64_t next_k,
                              ggml_opt_context_t opt, llama_model * model, const std::vector<TInfo> & trainable,
                              bool has_tf, double tf, double tl, bool has_vf, double vf, bool has_vl, double vl, Manifest & mf,
                              std::string & out_file) {
    // optimizer state slots by param name
    std::unordered_map<std::string, std::pair<ggml_tensor *, ggml_tensor *>> st;
    int n = ggml_opt_n_state(opt);
    for (int i = 0; i < n; i++) {
        ggml_tensor *m, *v;
        if (ggml_opt_state_at(opt, i, &m, &v)) st[std::string(m->name).substr(strlen("AdamW m for "))] = {m, v};
    }
    std::string hdr;
    {
        char b[512];
        snprintf(b, sizeof(b), "fingerprint=%s\nstep=%" PRIu64 "\nnext_epoch=%" PRIu64 "\nnext_k=%" PRIu64 "\nopt_iter=%" PRId64 "\n",
                 fingerprint.c_str(), step, next_epoch, next_k, ggml_opt_get_iter(opt));
        hdr += b;
        if (has_tf) { snprintf(b, sizeof(b), "train_first=%016" PRIx64 "\n", dbits(tf)); hdr += b; }
        snprintf(b, sizeof(b), "train_last=%016" PRIx64 "\n", dbits(tl)); hdr += b;
        if (has_vf) { snprintf(b, sizeof(b), "val_first=%016" PRIx64 "\n", dbits(vf)); hdr += b; }
        if (has_vl) { snprintf(b, sizeof(b), "val_last=%016" PRIx64 "\n", dbits(vl)); hdr += b; }
        for (const TInfo & t : trainable) {
            bool hs = st.count(t.name) != 0;
            snprintf(b, sizeof(b), "t %" PRId64 " %d %s\n", t.nelem, hs ? 1 : 0, t.name.c_str());
            hdr += b;
        }
    }
    out_file = ckpt_name(step);
    AtomicWriter w(join_path(dir, out_file));
    if (!w.ok()) return "cannot create checkpoint file";
    w.write("HAGCKP1\n", 8);
    uint64_t hl = hdr.size();
    w.write(&hl, 8);
    w.write(hdr.data(), hdr.size());
    std::vector<float> buf;
    for (const TInfo & t : trainable) {
        ggml_tensor * mt = llama_model_tensor_by_name(model, t.name.c_str());
        if (!mt) return "trainable tensor vanished";
        buf.resize((size_t)t.nelem);
        ggml_backend_tensor_get(mt, buf.data(), 0, (size_t)t.nelem * 4);
        if (!w.write(buf.data(), buf.size() * 4)) return "checkpoint write failed (disk full?)";
        auto it = st.find(t.name);
        if (it != st.end()) {
            ggml_backend_tensor_get(it->second.first, buf.data(), 0, (size_t)t.nelem * 4);
            w.write(buf.data(), buf.size() * 4);
            ggml_backend_tensor_get(it->second.second, buf.data(), 0, (size_t)t.nelem * 4);
            if (!w.write(buf.data(), buf.size() * 4)) return "checkpoint write failed (disk full?)";
        }
    }
    std::string body = w.hash_hex();
    std::string tr = body + "\n";
    w.write(tr.data(), tr.size());
    uint64_t size = w.written();
    if (!w.commit()) return "cannot commit checkpoint (disk full?)";
    // manifest: keep the previous valid entry and the new one, drop the rest
    std::vector<Manifest::E> keep;
    for (auto & e : mf.e) if (e.file != out_file) keep.push_back(e);
    if (keep.size() > 1) keep.erase(keep.begin(), keep.end() - 1);
    keep.push_back({step, size, body, out_file});
    Manifest nm;
    nm.e = keep;
    if (!write_manifest(dir, nm)) return "cannot write checkpoint manifest";
    for (auto & e : mf.e) {
        bool kept = false;
        for (auto & k : keep) if (k.file == e.file) kept = true;
        if (!kept) remove_file(join_path(dir, e.file));
    }
    mf = nm;
    return "";
}

}  // namespace

hag_status hag_train(const char * base_path, const char * const * texts, int n_texts, const hag_train_params * params,
                     const char * work_dir_c, const char * out_patch, hag_train_fn progress, void * user) {
    HAG_TRY
    if (!base_path || !texts || !work_dir_c || !out_patch) return make_status(HAG_ERR_INVALID_ARG, "null argument");
    Ctx X;
    X.base_path = base_path;
    X.work_dir = work_dir_c;
    X.hk.cb = progress; X.hk.user = user; X.hk.t0 = now_ms(); X.hk.last_event_ms = X.hk.t0;
    std::string err = normalize(params, X.c);
    if (!err.empty()) return make_status(HAG_ERR_INVALID_ARG, "%s", err.c_str());
    const Cfg & c = X.c;
    hag_status st0 = hag_engine_init();
    if (st0.code) return st0;
    if (!make_dir(X.work_dir)) return make_status(HAG_ERR_IO, "cannot create work_dir");
    X.log.path = join_path(X.work_dir, "train_run.log");
    X.log.line("--- hag_train start (%s)", engine_version_string().c_str());

    X.hk.emit(HAG_PHASE_PREPARE, 0, 0, 0, kNaN, kNaN);

    // ---- base + plan
    BaseInfo b;
    err = read_base(base_path, b);
    if (!err.empty()) return make_status(HAG_ERR_BAD_MODEL, "%s", err.c_str());
    Plan plan = make_plan(b, c);
    if (!plan.ok) return make_status(HAG_ERR_UNSUPPORTED, "cannot train this model: %s", plan.reason.c_str());
    HParams hp;
    err = read_hparams(base_path, b, hp);
    if (!err.empty()) return make_status(HAG_ERR_BAD_MODEL, "%s", err.c_str());

    // vocab-only load for tokenization
    Dataset ds;
    {
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        clear_native_error();
        llama_model * vm = llama_model_load_from_file(base_path, mp);
        if (!vm) return make_status(HAG_ERR_BAD_MODEL, "cannot read vocabulary: %s", last_native_error().c_str());
        struct VM { llama_model * m; ~VM() { llama_model_free(m); } } vg{vm};
        err = build_dataset(llama_model_get_vocab(vm), texts, n_texts, c, ds);
        if (!err.empty()) return make_status(HAG_ERR_INVALID_ARG, "%s", err.c_str());
    }
    const int n_ctx = std::min(ds.n_ctx, hp.n_ctx_train > 0 ? hp.n_ctx_train : ds.n_ctx);
    if (n_ctx != ds.n_ctx) return make_status(HAG_ERR_INVALID_ARG, "sequence length %d exceeds the model's training context %d", ds.n_ctx, hp.n_ctx_train);

    // ---- memory pre-flight (refuse BEFORE doing any work)
    Est est = estimate(b, plan, n_ctx, hp.n_embd, hp.n_ff, hp.n_head, hp.n_vocab, hp.n_embd_kv);
    X.log.line("plan: n_ctx=%d accum=%d trainable_params=%" PRIu64 " est_peak=%" PRIu64, n_ctx, c.accum, plan.p_train, est.total);
    if (c.max_mem && est.total > c.max_mem) {
        return make_status(HAG_ERR_OOM, "estimated training memory %.0f MB exceeds the limit %.0f MB (weights %.0f, optimizer %.0f, activations %.0f)",
                           est.total / 1048576.0, c.max_mem / 1048576.0, est.weights / 1048576.0, est.optimizer / 1048576.0,
                           (est.activations + est.logits + est.kv) / 1048576.0);
    }

    // ---- schedule
    const size_t S = ds.train_win.size();
    const int accum = (int)std::min<size_t>((size_t)c.accum, S);
    const size_t S_used = (S / accum) * accum;
    const size_t steps_per_epoch = S_used / accum;
    const uint64_t total_steps = (uint64_t)c.epochs * steps_per_epoch;
    X.hk.epochs = c.epochs; X.hk.steps_total = (int32_t)total_steps;

    // ---- identity of this run
    bool cancelled = false;
    std::string base_sha = base_sha_cached(base_path, X.work_dir, cancel_poll_cb, &X.hk, cancelled);
    if (cancelled) return make_status(HAG_ERR_CANCELLED, "cancelled while hashing the base model");
    std::string fingerprint;
    {
        Sha256 f;
        char b2[512];
        snprintf(b2, sizeof(b2), "HAGRUN|algo=%u|api=%d|base=%s|text=%s|tok=%s|n_ctx=%d|accum=%d|epochs=%d|lr=%.9g|val=%.9g|seed=%u|thr=%d|last=%d|emb=%d|S=%zu|",
                 kAlgoVersion, HAG_ENGINE_API_VERSION, base_sha.c_str(), ds.text_sha.c_str(), ds.tokens_sha.c_str(), n_ctx, accum, c.epochs,
                 (double)c.lr, (double)c.val_fraction, c.seed, c.threads, c.last_layers, c.train_emb ? 1 : 0, S_used);
        f.update(b2, strlen(b2));
        fingerprint = f.final_hex();
    }

    // ---- already finished? (idempotent: same inputs -> same patch)
    {
        PatchHeader ph;
        if (file_exists(out_patch) && patch_read_header(out_patch, ph).empty() && ph.fingerprint == fingerprint && ph.base_sha256 == base_sha) {
            X.log.line("output patch already complete for this run fingerprint: nothing to do");
            X.hk.emit(HAG_PHASE_DONE, c.epochs, (int)total_steps, (int64_t)total_steps * accum, kNaN, kNaN);
            return ok_status();
        }
    }

    // ---- working model
    std::string load_path = base_path;
    std::unordered_set<std::string> train_names;
    std::vector<TInfo> trainable;
    for (const TInfo & t : b.t) if (t.trainable) { train_names.insert(t.name); trainable.push_back(t); }
    if (plan.needs_working_copy) {
        Sha256 k;
        k.update(base_sha.data(), base_sha.size());
        for (auto & t : trainable) k.update(t.name.data(), t.name.size());
        load_path = join_path(X.work_dir, "train_base_" + k.final_hex().substr(0, 16) + ".gguf");
        if (!file_exists(load_path)) {
            X.log.line("building F32 working copy of %d trainable tensors -> %s", plan.n_train_tensors, load_path.c_str());
            err = build_working_copy(base_path, b, load_path, X.hk);
            if (!err.empty()) {
                if (X.hk.cancelled) return make_status(HAG_ERR_CANCELLED, "cancelled while preparing the working copy");
                return make_status(HAG_ERR_IO, "cannot build working copy: %s", err.c_str());
            }
        }
    }
    Loaded L;
    {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_NONE;   // private, writable weights (no mmap)
        clear_native_error();
        L.model = llama_model_load_from_file(load_path.c_str(), mp);
        if (!L.model) return make_status(HAG_ERR_BAD_MODEL, "cannot load model for training: %s", last_native_error().c_str());
    }
    for (const TInfo & t : trainable) {
        ggml_tensor * mt = llama_model_tensor_by_name(L.model, t.name.c_str());
        if (!mt || mt->type != GGML_TYPE_F32) return make_status(HAG_ERR_INTERNAL, "tensor %s is not F32 in the working model", t.name.c_str());
    }
    {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t)n_ctx; cp.n_batch = (uint32_t)n_ctx; cp.n_ubatch = (uint32_t)n_ctx;
        cp.n_threads = c.threads; cp.n_threads_batch = c.threads;
        cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.no_perf = true;
        clear_native_error();
        L.ctx = llama_init_from_model(L.model, cp);
        if (!L.ctx) return make_status(HAG_ERR_OOM, "cannot create training context: %s", last_native_error().c_str());
    }
    LrState lrs{c.lr};
    {
        llama_opt_params op;
        memset(&op, 0, sizeof(op));
        op.n_ctx_train = (uint32_t)n_ctx;
        op.param_filter = param_filter;
        op.param_filter_ud = &train_names;
        op.get_opt_pars = get_pars;
        op.get_opt_pars_ud = &lrs;
        op.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
        op.opt_period = accum;
        llama_opt_init(L.ctx, L.model, op);
    }
    ggml_opt_context_t opt = llama_opt_get_context(L.ctx);
    if (!opt) return make_status(HAG_ERR_INTERNAL, "optimizer not initialised");

    // ---- resume?
    OptCtlState rs;
    uint64_t step = 0, start_epoch = 0, start_k = 0;
    bool has_tf = false, has_vf = false, has_vl = false;
    double tf = 0, tl = kNaN, vf = kNaN, vl = kNaN;
    Manifest mf = read_manifest(X.work_dir);
    {
        std::vector<std::string> cands;
        for (auto & n : list_dir(X.work_dir)) if (n.compare(0, 5, "ckpt-") == 0 && n.size() > 4 && n.substr(n.size() - 4) == ".bin") cands.push_back(n);
        std::sort(cands.rbegin(), cands.rend());
        for (const std::string & n : cands) {
            CkptMeta m;
            std::string sha;
            std::string e = ckpt_verify(join_path(X.work_dir, n), m, &sha);
            if (e.empty()) {
                for (auto & me : mf.e) if (me.file == n && me.sha != sha) e = "manifest sha256 mismatch";
            }
            if (e.empty() && m.fingerprint != fingerprint) {
                X.log.line("checkpoint %s belongs to a different run (inputs changed): ignored", n.c_str());
                continue;
            }
            if (e.empty()) {
                // tensor set must match exactly
                if (m.tensors.size() != trainable.size()) e = "tensor set mismatch";
                else for (size_t i = 0; i < trainable.size(); i++) if (m.tensors[i].name != trainable[i].name || (int64_t)m.tensors[i].n != trainable[i].nelem) { e = "tensor set mismatch"; break; }
            }
            if (!e.empty()) {
                X.log.line("checkpoint %s REJECTED: %s", n.c_str(), e.c_str());
                rename_file(join_path(X.work_dir, n), join_path(X.work_dir, n + ".corrupt"));
                continue;
            }
            // accept: restore weights now, optimizer state at the first graph build
            FILE * f = fopen(join_path(X.work_dir, n).c_str(), "rb");
            bool okr = f != nullptr;
            std::vector<float> tmp;
            for (size_t i = 0; okr && i < m.tensors.size(); i++) {
                tmp.resize(m.tensors[i].n);
                ggml_tensor * mt = llama_model_tensor_by_name(L.model, m.tensors[i].name.c_str());
                okr = mt && fseek64(f, (int64_t)m.tensors[i].off, SEEK_SET) == 0 && fread(tmp.data(), 4, tmp.size(), f) == tmp.size();
                if (okr) ggml_backend_tensor_set(mt, tmp.data(), 0, tmp.size() * 4);
            }
            if (f) fclose(f);
            if (!okr) { X.log.line("checkpoint %s could not be loaded", n.c_str()); continue; }
            rs.restore = true;
            rs.ckpt_path = join_path(X.work_dir, n);
            rs.meta = m;
            for (auto & ct : rs.meta.tensors) rs.by_name[ct.name] = &ct;
            step = m.step; start_epoch = m.next_epoch; start_k = m.next_k;
            has_tf = m.has_train_first; tf = bitsd(m.bits_train_first); tl = bitsd(m.bits_train_last);
            has_vf = m.has_val_first; vf = bitsd(m.bits_val_first); has_vl = m.has_val_last; vl = bitsd(m.bits_val_last);
            X.hk.resumed_from = (int32_t)step;
            X.log.line("RESUMED from %s (step %" PRIu64 ", epoch %" PRIu64 ", k %" PRIu64 ")", n.c_str(), step, start_epoch, start_k);
            break;
        }
        if (!rs.restore) X.log.line("no valid checkpoint: starting from scratch");
    }
    rs.names = nullptr;
    ggml_opt_set_state_init_cb(opt, state_init_cb, &rs);

    // ---- loop helpers
    ggml_opt_result_t res_train = ggml_opt_result_init();
    ggml_opt_result_t res_val = ggml_opt_result_init();
    struct RG { ggml_opt_result_t a, b; ~RG() { ggml_opt_result_free(a); ggml_opt_result_free(b); } } rg{res_train, res_val};
    std::vector<llama_token> tok((size_t)n_ctx), lab((size_t)n_ctx);
    auto load_window = [&](const std::vector<int32_t> & stream, int64_t start) {
        for (int i = 0; i < n_ctx; i++) { tok[i] = stream[(size_t)start + i]; lab[i] = stream[(size_t)start + i + 1]; }
    };
    auto fail_step = [&](int rc) { return make_status(HAG_ERR_INTERNAL, "training step failed (%d)", rc); };

    auto run_val = [&](double & out_loss, bool & failed) -> bool {   // returns false if cancelled
        failed = false;
        ggml_opt_result_reset(res_val);
        for (size_t i = 0; i < ds.val_win.size(); i++) {
            load_window(ds.val, ds.val_win[i]);
            int rc = llama_opt_sequence(L.ctx, tok.data(), lab.data(), false, res_val);
            if (rc != 0) { failed = true; return true; }
            if (now_ms() - X.hk.last_event_ms > 700) { if (X.hk.emit(HAG_PHASE_EVAL, 0, (int)step, 0, kNaN, vl)) return false; }
        }
        double unc;
        ggml_opt_result_loss(res_val, &out_loss, &unc);
        return true;
    };

    auto save_ckpt = [&](uint64_t next_epoch, uint64_t next_k) -> std::string {
        X.hk.emit(HAG_PHASE_SAVE, (int)next_epoch, (int)step, (int64_t)step * accum, tl, vl);
        std::string fn;
        std::string e = write_ckpt(X.work_dir, step, fingerprint, next_epoch, next_k, opt, L.model, trainable, has_tf, tf, tl, has_vf, vf, has_vl, vl, mf, fn);
        if (e.empty()) X.log.line("checkpoint written: %s", fn.c_str());
        else X.log.line("checkpoint FAILED: %s", e.c_str());
        return e;
    };

    // baseline held-out loss (fresh runs only)
    if (!rs.restore && !ds.val_win.empty()) {
        bool failed;
        double v;
        if (!run_val(v, failed)) { return make_status(HAG_ERR_CANCELLED, "cancelled"); }
        if (failed) return fail_step(-1);
        vf = v; vl = v; has_vf = has_vl = true;
        X.hk.emit(HAG_PHASE_EVAL, 0, 0, 0, kNaN, vl);
    }
    if (rs.restore_failed) return make_status(HAG_ERR_CORRUPT, "checkpoint optimizer state mismatch");

    // ---- training loop
    uint64_t last_saved_step = rs.restore ? step : UINT64_MAX;
    bool done_all = (start_epoch >= (uint64_t)c.epochs);
    std::vector<uint32_t> perm(S);
    for (uint64_t epoch = start_epoch; epoch < (uint64_t)c.epochs && !done_all; epoch++) {
        for (size_t i = 0; i < S; i++) perm[i] = (uint32_t)i;
        {
            SplitMix64 rng(((uint64_t)c.seed * 0x9E3779B97F4A7C15ull) ^ (epoch + 1) * 0xD6E8FEB86659FD93ull);
            for (size_t i = S - 1; i > 0; i--) std::swap(perm[i], perm[(size_t)rng.below((uint64_t)i + 1)]);
        }
        for (uint64_t k = (epoch == start_epoch ? start_k : 0); k < steps_per_epoch; k++) {
            ggml_opt_zero_grad_accs(opt);
            ggml_opt_result_reset(res_train);
            for (int a = 0; a < accum; a++) {
                load_window(ds.train, ds.train_win[perm[k * accum + a]]);
                int rc = llama_opt_sequence(L.ctx, tok.data(), lab.data(), true, res_train);
                if (rc != 0) return fail_step(rc);
                if (a + 1 < accum && now_ms() - X.hk.last_event_ms > 700) {
                    if (X.hk.emit(HAG_PHASE_TRAIN, (int)epoch, (int)step, (int64_t)step * accum + a, tl, vl)) break;
                }
            }
            if (X.hk.cancelled) {
                // the partial step is discarded (weights/moments are untouched until the optimizer step)
                std::string e;
                if (last_saved_step != step && step > 0) e = save_ckpt(epoch, k);
                X.log.line("cancelled at step %" PRIu64 " (checkpoint %s)", step, e.empty() ? "kept/written" : e.c_str());
                return make_status(HAG_ERR_CANCELLED, "training cancelled; latest checkpoint kept");
            }
            double loss, unc;
            ggml_opt_result_loss(res_train, &loss, &unc);
            if (!std::isfinite(loss)) {
                X.log.line("loss became non-finite at step %" PRIu64 ": aborting (latest valid checkpoint kept)", step + 1);
                return make_status(HAG_ERR_INTERNAL, "training diverged (non-finite loss at step %" PRIu64 "); lower the learning rate", step + 1);
            }
            step++;
            if (!has_tf) { has_tf = true; tf = loss; }
            tl = loss;
            bool last_of_epoch = (k + 1 == steps_per_epoch);
            if (last_of_epoch && !ds.val_win.empty()) {
                bool failed;
                double v;
                if (!run_val(v, failed)) {
                    if (last_saved_step != step) save_ckpt(epoch, k + 1);
                    return make_status(HAG_ERR_CANCELLED, "training cancelled; latest checkpoint kept");
                }
                if (failed) return fail_step(-2);
                if (!std::isfinite(v)) return make_status(HAG_ERR_INTERNAL, "validation loss is not finite");
                vl = v; has_vl = true;
                X.hk.emit(HAG_PHASE_EVAL, (int)epoch + 1, (int)step, (int64_t)step * accum, loss, vl);
            }
            bool cancel_now = X.hk.emit(HAG_PHASE_TRAIN, (int)epoch + (last_of_epoch ? 1 : 0), (int)step, (int64_t)step * accum, loss, vl);
            uint64_t ne = last_of_epoch ? epoch + 1 : epoch, nk = last_of_epoch ? 0 : k + 1;
            bool due = c.ckpt_every > 0 ? (step % (uint64_t)c.ckpt_every == 0) : last_of_epoch;
            bool final_step = (step == total_steps);
            if ((due && !final_step) || cancel_now) {
                std::string e = save_ckpt(ne, nk);
                if (e.empty()) last_saved_step = step;
                else if (!cancel_now) return make_status(HAG_ERR_IO, "checkpoint failed: %s", e.c_str());
            }
            if (cancel_now) {
                X.log.line("cancelled after step %" PRIu64, step);
                return make_status(HAG_ERR_CANCELLED, "training cancelled; latest checkpoint kept");
            }
        }
    }

    // ---- write the patch (free optimizer state & activations first)
    X.hk.emit(HAG_PHASE_SAVE, c.epochs, (int)step, (int64_t)step * accum, tl, vl);
    llama_free(L.ctx);
    L.ctx = nullptr;

    gguf_context * pg = gguf_init_empty();
    struct PG { gguf_context * g; ~PG() { gguf_free(g); } } pgg{pg};
    gguf_set_val_str(pg, "general.architecture", HAG_PATCH_ARCH);
    gguf_set_val_str(pg, "general.name", "hag specialist patch");
    gguf_set_val_u32(pg, HAG_KEY_VERSION, HAG_PATCH_FORMAT_VERSION);
    gguf_set_val_str(pg, HAG_KEY_BASE_ARCH, b.arch.c_str());
    gguf_set_val_str(pg, HAG_KEY_BASE_SHA256, base_sha.c_str());
    gguf_set_val_u64(pg, HAG_KEY_BASE_SIZE, (uint64_t)b.file_size);
    gguf_set_val_u32(pg, HAG_KEY_BASE_NTENSORS, (uint32_t)b.t.size());
    gguf_set_val_str(pg, HAG_KEY_PAYLOAD_SHA256, std::string(64, '0').c_str());
    gguf_set_val_str(pg, HAG_KEY_FINGERPRINT, fingerprint.c_str());
    gguf_set_val_str(pg, HAG_KEY_ENGINE, engine_version_string().c_str());
    gguf_set_val_u32(pg, "hag.train.n_ctx", (uint32_t)n_ctx);
    gguf_set_val_u32(pg, "hag.train.n_ctx_requested", (uint32_t)c.n_ctx_req);
    gguf_set_val_u32(pg, "hag.train.sequences_per_step", (uint32_t)accum);
    gguf_set_val_u32(pg, "hag.train.tokens_per_step", (uint32_t)(accum * n_ctx));
    gguf_set_val_u32(pg, "hag.train.epochs", (uint32_t)c.epochs);
    gguf_set_val_u64(pg, "hag.train.steps", step);
    gguf_set_val_f32(pg, "hag.train.learning_rate", c.lr);
    gguf_set_val_f32(pg, "hag.train.val_fraction", c.val_fraction);
    gguf_set_val_u32(pg, "hag.train.seed", c.seed);
    gguf_set_val_u32(pg, "hag.train.threads", (uint32_t)c.threads);
    gguf_set_val_u32(pg, "hag.train.trainable_last_layers", (uint32_t)c.last_layers);
    gguf_set_val_bool(pg, "hag.train.train_embeddings", c.train_emb);
    gguf_set_val_bool(pg, "hag.train.train_embeddings_effective", plan.emb_effective);
    gguf_set_val_str(pg, "hag.train.optimizer", "adamw(beta1=0.9,beta2=0.999,eps=1e-8,wd=0,constant_lr)");
    gguf_set_val_u32(pg, "hag.train.algo_version", kAlgoVersion);
    gguf_set_val_u32(pg, "hag.train.n_trainable_tensors", (uint32_t)plan.n_train_tensors);
    gguf_set_val_u64(pg, "hag.train.n_trainable_params", plan.p_train);
    gguf_set_val_str(pg, "hag.train.dataset_sha256", ds.text_sha.c_str());
    gguf_set_val_str(pg, "hag.train.token_stream_sha256", ds.tokens_sha.c_str());
    gguf_set_val_u32(pg, "hag.train.n_texts", (uint32_t)n_texts);
    gguf_set_val_u32(pg, "hag.train.n_train_docs", (uint32_t)ds.n_train_docs);
    gguf_set_val_u32(pg, "hag.train.n_val_docs", (uint32_t)ds.n_val_docs);
    gguf_set_val_u64(pg, "hag.train.n_train_tokens", ds.train.size());
    gguf_set_val_u64(pg, "hag.train.n_val_tokens", ds.val.size());
    gguf_set_val_u32(pg, "hag.train.n_train_sequences", (uint32_t)S);
    gguf_set_val_u32(pg, "hag.train.n_val_sequences", (uint32_t)ds.val_win.size());
    gguf_set_val_f64(pg, "hag.train.train_loss_first", has_tf ? tf : kNaN);
    gguf_set_val_f64(pg, "hag.train.train_loss_last", tl);
    gguf_set_val_f64(pg, "hag.train.val_loss_first", has_vf ? vf : kNaN);
    gguf_set_val_f64(pg, "hag.train.val_loss_last", has_vl ? vl : kNaN);
    gguf_set_val_str(pg, "hag.train.base_file_type", b.file_type.c_str());

    ggml_init_params gp = {trainable.size() * ggml_tensor_overhead() + (1u << 20), nullptr, true};
    ggml_context * gctx = ggml_init(gp);
    struct GC { ggml_context * c; ~GC() { ggml_free(c); } } gcg{gctx};
    for (const TInfo & t : trainable) {
        int nd = 4;
        while (nd > 1 && t.ne[nd - 1] == 1) nd--;
        ggml_tensor * g = ggml_new_tensor(gctx, t.type, nd, t.ne);
        ggml_set_name(g, t.name.c_str());
        gguf_add_tensor(pg, g);
    }
    const size_t meta = gguf_get_meta_size(pg);
    // pass 1: payload hash
    std::vector<float> fbuf;
    std::vector<uint8_t> qbuf;
    auto tensor_bytes = [&](const TInfo & t, const uint8_t *& ptr, size_t & n) -> bool {
        ggml_tensor * mt = llama_model_tensor_by_name(L.model, t.name.c_str());
        if (!mt) return false;
        fbuf.resize((size_t)t.nelem);
        ggml_backend_tensor_get(mt, fbuf.data(), 0, (size_t)t.nelem * 4);
        if (t.type == GGML_TYPE_F32) { ptr = (const uint8_t *)fbuf.data(); n = fbuf.size() * 4; return true; }
        const int64_t nrows = t.ne[1] * t.ne[2] * t.ne[3];
        qbuf.resize(t.nbytes);
        size_t w = convert_rows_from_f32(t.type, fbuf.data(), qbuf.data(), nrows, t.ne[0]);
        if (w != t.nbytes) return false;
        ptr = qbuf.data(); n = qbuf.size();
        return true;
    };
    Sha256 payload;
    for (const TInfo & t : trainable) {
        const uint8_t * p; size_t n;
        if (!tensor_bytes(t, p, n)) return make_status(HAG_ERR_INTERNAL, "cannot convert tensor %s to its stored type", t.name.c_str());
        payload.update(p, n);
    }
    std::string payload_hex = payload.final_hex();
    gguf_set_val_str(pg, HAG_KEY_PAYLOAD_SHA256, payload_hex.c_str());
    if (gguf_get_meta_size(pg) != meta) return make_status(HAG_ERR_INTERNAL, "patch header size changed");
    std::vector<uint8_t> mbuf(meta);
    gguf_get_meta_data(pg, mbuf.data());
    // pass 2: write
    {
        AtomicWriter w(out_patch);
        if (!w.ok()) return make_status(HAG_ERR_IO, "cannot create output patch file");
        w.write(mbuf.data(), meta);
        static const uint8_t zeros[64] = {0};
        for (size_t i = 0; i < trainable.size(); i++) {
            uint64_t want = gguf_get_tensor_offset(pg, (int64_t)i);
            uint64_t have = w.written() - meta;
            while (have < want) { size_t nn = (size_t)std::min<uint64_t>(sizeof(zeros), want - have); w.write(zeros, nn); have += nn; }
            const uint8_t * p; size_t n;
            if (!tensor_bytes(trainable[i], p, n)) return make_status(HAG_ERR_INTERNAL, "tensor conversion failed");
            if (!w.write(p, n)) return make_status(HAG_ERR_IO, "write failed (disk full?)");
        }
        if (!w.commit()) return make_status(HAG_ERR_IO, "cannot finalize patch file");
    }
    X.log.line("patch written: %s (payload %s)", out_patch, payload_hex.c_str());

    // verify what we wrote, then clean up the work area
    {
        PatchHeader ph;
        std::string e = patch_read_header(out_patch, ph);
        if (!e.empty()) return make_status(HAG_ERR_INTERNAL, "written patch failed validation: %s", e.c_str());
    }
    // Best effort: what does deploying in the BASE tensor type cost? (quantized bases only; the patch is already committed)
    double vl_deploy = vl;
    if (plan.base_quantized && !ds.val_win.empty()) {
        try {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = (uint32_t)n_ctx; cp.n_batch = (uint32_t)n_ctx; cp.n_ubatch = (uint32_t)n_ctx;
            cp.n_threads = c.threads; cp.n_threads_batch = c.threads; cp.no_perf = true;
            llama_context * ic = llama_init_from_model(L.model, cp);
            if (ic) {
                const llama_vocab * vv = llama_model_get_vocab(L.model);
                const int nv = llama_vocab_n_tokens(vv);
                auto score_val = [&]() -> double {
                    double sum = 0; int64_t cnt = 0;
                    for (int64_t start : ds.val_win) {
                        llama_memory_clear(llama_get_memory(ic), true);
                        llama_batch bt = llama_batch_init(n_ctx, 0, 1);
                        bt.n_tokens = n_ctx;
                        for (int i = 0; i < n_ctx; i++) { bt.token[i] = ds.val[(size_t)start + i]; bt.pos[i] = i; bt.n_seq_id[i] = 1; bt.seq_id[i][0] = 0; bt.logits[i] = 1; }
                        int rc = llama_decode(ic, bt);
                        llama_batch_free(bt);
                        if (rc != 0) return kNaN;
                        for (int i = 0; i < n_ctx; i++) {
                            const float * lg = llama_get_logits_ith(ic, i);
                            float mx = lg[0];
                            for (int k = 1; k < nv; k++) mx = std::max(mx, lg[k]);
                            double z = 0;
                            for (int k = 0; k < nv; k++) z += std::exp((double)lg[k] - mx);
                            sum += -((double)lg[ds.val[(size_t)start + i + 1]] - mx - std::log(z));
                            cnt++;
                        }
                    }
                    return sum / (double)cnt;
                };
                double nll_f32 = score_val();
                std::vector<float> q32, dq;
                std::vector<uint8_t> qq;
                for (const TInfo & t : trainable) {
                    if (t.type == GGML_TYPE_F32) continue;
                    ggml_tensor * mt = llama_model_tensor_by_name(L.model, t.name.c_str());
                    q32.resize((size_t)t.nelem);
                    ggml_backend_tensor_get(mt, q32.data(), 0, (size_t)t.nelem * 4);
                    qq.resize(t.nbytes);
                    const int64_t nrows = t.ne[1] * t.ne[2] * t.ne[3];
                    if (convert_rows_from_f32(t.type, q32.data(), qq.data(), nrows, t.ne[0]) != t.nbytes) continue;
                    dq.resize((size_t)t.nelem);
                    const ggml_type_traits * tr = ggml_get_type_traits(t.type);
                    const size_t rs = ggml_row_size(t.type, t.ne[0]);
                    for (int64_t r = 0; r < nrows; r++) tr->to_float(qq.data() + rs * (size_t)r, dq.data() + (size_t)(r * t.ne[0]), t.ne[0]);
                    ggml_backend_tensor_set(mt, dq.data(), 0, (size_t)t.nelem * 4);
                }
                double nll_rq = score_val();
                llama_free(ic);
                vl_deploy = nll_rq;
                X.log.line("deploy-type check on held-out windows: val NLL f32 working weights %.6f -> %s patch weights %.6f", nll_f32, b.file_type.c_str(), nll_rq);
                Json rj;
                rj.begin_obj().kv("val_nll_f32_weights", nll_f32).kv("val_nll_deployed_type", nll_rq).kv("base_file_type", b.file_type).end_obj();
                AtomicWriter rw(join_path(X.work_dir, "requant_eval.json"));
                if (rw.ok()) { rw.write(rj.str().data(), rj.str().size()); rw.commit(); }
            }
        } catch (...) { X.log.line("deploy-type check failed (ignored)"); }
    }
    for (auto & n : list_dir(X.work_dir)) {
        if ((n.compare(0, 5, "ckpt-") == 0) || n == "ckpt.manifest" || n.compare(0, 11, "train_base_") == 0) remove_file(join_path(X.work_dir, n));
    }
    X.hk.emit(HAG_PHASE_DONE, c.epochs, (int)step, (int64_t)step * accum, tl, vl_deploy);
    return ok_status();
    HAG_CATCH_STATUS
}
