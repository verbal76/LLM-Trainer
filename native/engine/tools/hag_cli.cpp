// hag_cli: command-line front-end to the HAG engine C API (used by the test suite and for on-device qualification).
//   hag_cli info <model>
//   hag_cli generate <model> --prompt TEXT [--chat] [--n N] [--seed S] [--temp T] [--top-k K] [--top-p P] [--min-p P] [--repeat R]
//                            [--ctx N] [--threads N] [--patch FILE] [--no-mmap] [--cancel-after N] [--stats]
//   hag_cli score <model> (--text TEXT | --file FILE) [--patch FILE] [--ctx N] [--threads N]
//   hag_cli tokenize <model> --text TEXT
//   hag_cli train <model> --data FILE --work DIR --out PATCH [--epochs N] [--lr F] [--ctx N] [--batch TOKENS] [--val F] [--seed S]
//                         [--threads N] [--last-layers N] [--emb] [--ckpt-every N] [--max-mem BYTES] [--quiet]
//                         [--lora-rank R [--lora-alpha A]]
//                         [--cancel-after-steps N]
//   hag_cli estimate <model> [train options]
//   hag_cli patch-info <patch>
//   hag_cli sha256 <file>
// Output: one JSON object on stdout (generation text is JSON-escaped in "text"); progress/events as JSON lines on stderr.
// Exit status: 0 ok, otherwise the (negated) hag_status code.
#include "hag_engine.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <atomic>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static std::string jesc(const std::string & s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; } else o += (char)c;
        }
    }
    return o;
}

struct Args {
    std::string cmd, pos;
    std::map<std::string, std::string> kv;
    bool has(const char * k) const { return kv.count(k) != 0; }
    std::string get(const char * k, const std::string & d = "") const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
    double num(const char * k, double d) const { auto it = kv.find(k); return it == kv.end() ? d : atof(it->second.c_str()); }
};

static bool is_flag(const std::string & k) {
    return k == "--chat" || k == "--no-mmap" || k == "--stats" || k == "--emb" || k == "--quiet";
}

static Args parse(int argc, char ** argv) {
    Args a;
    if (argc < 2) return a;
    a.cmd = argv[1];
    int i = 2;
    if (i < argc && strncmp(argv[i], "--", 2) != 0) a.pos = argv[i++];
    for (; i < argc; i++) {
        std::string k = argv[i];
        if (k.compare(0, 2, "--") != 0) continue;
        if (is_flag(k)) a.kv[k.substr(2)] = "1";
        else if (i + 1 < argc) a.kv[k.substr(2)] = argv[++i];
    }
    return a;
}

static int fail(hag_status s) {
    printf("{\"ok\":false,\"code\":%d,\"error\":\"%s\"}\n", s.code, jesc(s.message).c_str());
    fprintf(stderr, "error %d: %s\n", s.code, s.message);
    return s.code < 0 ? -s.code : 1;
}

static bool valid_utf8(const std::string & s) {
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        size_t len = c < 0x80 ? 1 : (c >= 0xC2 && c <= 0xDF) ? 2 : (c >= 0xE0 && c <= 0xEF) ? 3 : (c >= 0xF0 && c <= 0xF4) ? 4 : 0;
        if (!len || i + len > n) return false;
        for (size_t k = 1; k < len; k++) if (((unsigned char)s[i + k] & 0xC0) != 0x80) return false;
        i += len;
    }
    return true;
}

static double peak_rss_mb() {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    double mb = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned long long kb;
        if (sscanf(line, "VmHWM: %llu kB", &kb) == 1) { mb = kb / 1024.0; break; }
    }
    fclose(f);
    return mb;
}

// cancels `s` from another thread after a delay (exercises the async-safe hag_cancel)
struct DelayedCancel {
    std::thread th;
    std::atomic<bool> stop{false};
    void start(hag_session * s, int ms) {
        th = std::thread([this, s, ms] {
            for (int t = 0; t < ms && !stop.load(); t++) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (!stop.load()) hag_cancel(s);
        });
    }
    ~DelayedCancel() { stop = true; if (th.joinable()) th.join(); }
};

struct GenCtx {
    std::string text;
    int pieces = 0, cancel_after = 0;
    bool utf8_ok = true;
    hag_session * s = nullptr;
};

static int on_piece(const char * p, void * u) {
    GenCtx * g = (GenCtx *)u;
    std::string ps = p;
    if (!valid_utf8(ps)) g->utf8_ok = false;
    g->text += ps;
    g->pieces++;
    if (g->cancel_after > 0 && g->pieces >= g->cancel_after) hag_cancel(g->s);
    return 0;
}

static hag_train_params train_params(const Args & a) {
    hag_train_params p;
    memset(&p, 0, sizeof(p));
    p.struct_size = sizeof(p);
    p.n_ctx = (int)a.num("ctx", 128);
    p.n_batch = (int)a.num("batch", 0);
    p.epochs = (int)a.num("epochs", 1);
    p.learning_rate = (float)a.num("lr", 1e-4);
    p.val_fraction = (float)a.num("val", 0.0);
    p.seed = (uint32_t)a.num("seed", 1);
    p.n_threads = (int)a.num("threads", 0);
    p.trainable_last_layers = (int)a.num("last-layers", 0);
    p.train_embeddings = a.has("emb") ? 1 : 0;
    p.checkpoint_every_steps = (int)a.num("ckpt-every", 0);
    p.max_memory_bytes = (size_t)a.num("max-mem", 0);
    p.lora_rank = (int)a.num("lora-rank", 0);
    p.lora_alpha = (float)a.num("lora-alpha", 0);
    return p;
}

struct TrainUI { bool quiet; int cancel_after; int steps_seen; };

static int on_event(const hag_train_event * e, void * u) {
    TrainUI * ui = (TrainUI *)u;
    if (e->phase == 1 && e->step > ui->steps_seen) ui->steps_seen = e->step;
    if (!ui->quiet) {
        fprintf(stderr, "{\"phase\":%d,\"epoch\":%d,\"epochs\":%d,\"step\":%d,\"steps\":%d,\"examples\":%lld,\"train_loss\":%s,\"val_loss\":%s,\"elapsed_s\":%.3f,\"rss_mb\":%.1f,\"resumed_from\":%d}\n",
                e->phase, e->epoch, e->epochs, e->step, e->steps, (long long)e->examples_done,
                std::isnan(e->train_loss) ? "null" : std::to_string(e->train_loss).c_str(),
                std::isnan(e->val_loss) ? "null" : std::to_string(e->val_loss).c_str(), e->elapsed_s, e->rss_bytes / 1048576.0, e->resumed_from_step);
    }
    if (ui->cancel_after > 0 && e->phase == 1 && e->step >= ui->cancel_after) return 1;
    return 0;
}

int main(int argc, char ** argv) {
    Args a = parse(argc, argv);
    if (a.cmd.empty()) { fprintf(stderr, "usage: hag_cli <info|generate|score|tokenize|train|estimate|patch-info|sha256> ...\n"); return 64; }
    hag_status st = hag_engine_init();
    if (st.code) return fail(st);

    if (a.cmd == "version") { printf("{\"version\":\"%s\",\"system\":%s}\n", jesc(hag_engine_version()).c_str(), hag_system_info_json()); return 0; }

    if (a.cmd == "patch-info") {
        char * js = nullptr;
        st = hag_patch_info_json(a.pos.c_str(), &js);
        if (st.code) return fail(st);
        printf("%s\n", js);
        hag_free(js);
        return 0;
    }
    if (a.cmd == "estimate") {
        hag_train_params p = train_params(a);
        char * js = nullptr;
        st = hag_train_estimate(a.pos.c_str(), &p, &js);
        if (st.code) return fail(st);
        printf("%s\n", js);
        hag_free(js);
        return 0;
    }
    if (a.cmd == "sha256") {
        // via the patch-info machinery would be wrong; tiny local implementation is not needed: use sha256sum in tests
        fprintf(stderr, "use sha256sum\n");
        return 64;
    }
    if (a.cmd == "train") {
        std::ifstream in(a.get("data"));
        if (!in) { printf("{\"ok\":false,\"error\":\"cannot read --data\"}\n"); return 66; }
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(in, line)) {
            while (!line.empty() && (line.back() == '\r')) line.pop_back();
            if (!line.empty()) lines.push_back(line);
        }
        std::vector<const char *> texts;
        for (auto & l : lines) texts.push_back(l.c_str());
        hag_train_params p = train_params(a);
        TrainUI ui{a.has("quiet"), (int)a.num("cancel-after-steps", 0), 0};
        auto t0 = std::chrono::steady_clock::now();
        st = hag_train(a.pos.c_str(), texts.data(), (int)texts.size(), &p, a.get("work").c_str(), a.get("out").c_str(), on_event, &ui);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (st.code) return fail(st);
        printf("{\"ok\":true,\"wall_s\":%.3f,\"steps\":%d,\"peak_rss_mb\":%.1f}\n", secs, ui.steps_seen, peak_rss_mb());
        return 0;
    }

    // everything below needs a model
    hag_model * m = nullptr;
    st = hag_model_load(a.pos.c_str(), a.has("no-mmap") ? 0 : 1, nullptr, nullptr, &m);
    if (st.code) return fail(st);
    if (a.has("patch")) {
        st = hag_model_apply_patch(m, a.get("patch").c_str());
        if (st.code) return fail(st);
    }
    if (a.cmd == "info") {
        printf("{\"ok\":true,\"version\":\"%s\",\"system\":%s,\"model\":%s}\n", jesc(hag_engine_version()).c_str(), hag_system_info_json(), hag_model_info_json(m));
        hag_model_free(m);
        return 0;
    }
    if (a.cmd == "tokenize") {
        std::string text = a.get("text");
        int n = 0;
        std::vector<int32_t> t(text.size() + 16);
        st = hag_tokenize(m, text.c_str(), 1, t.data(), (int)t.size(), &n);
        if (st.code) return fail(st);
        printf("{\"ok\":true,\"n\":%d,\"tokens\":[", n);
        for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", t[i]);
        printf("]}\n");
        hag_model_free(m);
        return 0;
    }
    hag_session_params sp;
    sp.struct_size = sizeof(sp);
    sp.n_ctx = (int)a.num("ctx", 512);
    sp.n_threads = (int)a.num("threads", 0);
    sp.n_batch = 0;
    hag_session * s = nullptr;
    st = hag_session_new(m, &sp, &s);
    if (st.code) return fail(st);
    int rc = 0;
    if (a.cmd == "score") {
        std::string text = a.get("text");
        if (a.has("file")) { std::ifstream f(a.get("file")); std::stringstream ss; ss << f.rdbuf(); text = ss.str(); }
        double nll = 0; int n = 0;
        DelayedCancel dc;
        if (a.has("cancel-after-ms")) dc.start(s, (int)a.num("cancel-after-ms", 0));
        auto t0 = std::chrono::steady_clock::now();
        st = hag_score(s, text.c_str(), &nll, &n);
        if (st.code) { fprintf(stderr, "score took %.3f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()); rc = fail(st); }
        else printf("{\"ok\":true,\"mean_nll\":%.9g,\"n_tokens\":%d,\"patched\":%s}\n", nll, n, a.has("patch") ? "true" : "false");
    } else if (a.cmd == "generate") {
        std::string prompt = a.get("prompt");
        if (a.has("chat")) {
            const char * roles[1] = {"user"};
            const char * contents[1] = {prompt.c_str()};
            char * f = nullptr;
            st = hag_chat_format(m, roles, contents, 1, 1, &f);
            if (st.code) { rc = fail(st); goto done; }
            prompt = f;
            hag_free(f);
        }
        {
            hag_sample_params sampler;
            memset(&sampler, 0, sizeof(sampler));
            sampler.struct_size = sizeof(sampler);
            sampler.temperature = (float)a.num("temp", 0);
            sampler.top_k = (int)a.num("top-k", 0);
            sampler.top_p = (float)a.num("top-p", 1);
            sampler.min_p = (float)a.num("min-p", 0);
            sampler.repeat_penalty = (float)a.num("repeat", 1);
            sampler.seed = (uint32_t)a.num("seed", 1);
            sampler.max_new_tokens = (int)a.num("n", 64);
            GenCtx g;
            g.s = s;
            g.cancel_after = (int)a.num("cancel-after", 0);
            hag_gen_stats stats;
            memset(&stats, 0, sizeof(stats));
            DelayedCancel dc;
            if (a.has("cancel-after-ms")) dc.start(s, (int)a.num("cancel-after-ms", 0));
            st = hag_generate(s, prompt.c_str(), &sampler, on_piece, &g, &stats);
            if (st.code && st.code != HAG_ERR_CANCELLED) { rc = fail(st); goto done; }
            printf("{\"ok\":true,\"text\":\"%s\",\"pieces\":%d,\"utf8_ok\":%s,\"n_prompt\":%d,\"n_generated\":%d,\"stop_reason\":%d,"
                   "\"prompt_ms\":%.2f,\"gen_ms\":%.2f,\"tok_per_s\":%.3f,\"peak_rss_mb\":%.1f,\"status\":%d}\n",
                   jesc(g.text).c_str(), g.pieces, g.utf8_ok ? "true" : "false", stats.n_prompt_tokens, stats.n_generated, stats.stop_reason,
                   stats.prompt_ms, stats.gen_ms, stats.gen_ms > 0 ? stats.n_generated * 1000.0 / stats.gen_ms : 0.0,
                   stats.peak_rss_bytes / 1048576.0, st.code);
        }
    } else {
        fprintf(stderr, "unknown command %s\n", a.cmd.c_str());
        rc = 64;
    }
done:
    hag_session_free(s);
    hag_model_free(m);
    return rc;
}
