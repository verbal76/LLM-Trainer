// HAG runtime JNI bridge (libhagrt.so).
//
// Design:
//  * libhagrt.so is tiny and compiled for the plain baseline ISA (armv8-a / x86-64). It runs the CPU-feature gate
//    BEFORE the real engine is touched. The engine (libhagengine.so: engine + llama.cpp/ggml, built for a higher ISA
//    baseline) is only loaded and bound (dlsym) after the gate passes, so a device lacking the required features
//    never executes an instruction it cannot decode (no SIGILL, not even from static initialisers).
//  * All strings cross JNI as UTF-8 byte[] (never NewStringUTF: that is *modified* UTF-8 and aborts under CheckJNI
//    for supplementary characters). Kotlin does the encode/decode.
//  * Handles are raw pointers as jlong; the Kotlin layer wraps them in refcounted, validated ids.
//  * Callbacks run synchronously on the calling thread; if the engine ever calls from another thread we attach it
//    for the duration of the callback and detach right after.
//  * Failures throw com.hotatticgames.hag.runtime.HagException(code, message). No C++ exception crosses JNI.
#include <jni.h>
#include <dlfcn.h>
#include <sys/auxv.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hag_engine.h"

#ifdef __ANDROID__
#  define HAG_ATTACH_ARG(p) (p)               /* Android: AttachCurrentThread(JNIEnv**, void*) */
#else
#  define HAG_ATTACH_ARG(p) (reinterpret_cast<void **>(p))  /* JDK headers (host syntax check) */
#endif

namespace {

// ------------------------------------------------------------------------------------------ engine function table
struct Api {
    hag_status  (*engine_init)(void) = nullptr;
    void        (*engine_shutdown)(void) = nullptr;
    const char *(*engine_version)(void) = nullptr;
    const char *(*system_info_json)(void) = nullptr;
    void        (*free_)(void *) = nullptr;
    hag_status  (*model_load)(const char *, int, hag_progress_fn, void *, hag_model **) = nullptr;
    hag_status  (*model_apply_patch)(hag_model *, const char *) = nullptr;
    void        (*model_free)(hag_model *) = nullptr;
    const char *(*model_info_json)(const hag_model *) = nullptr;
    hag_status  (*session_new)(hag_model *, const hag_session_params *, hag_session **) = nullptr;
    void        (*session_free)(hag_session *) = nullptr;
    hag_status  (*session_reset)(hag_session *) = nullptr;
    hag_status  (*chat_format)(const hag_model *, const char *const *, const char *const *, int, int, char **) = nullptr;
    hag_status  (*generate)(hag_session *, const char *, const hag_sample_params *, hag_token_fn, void *, hag_gen_stats *) = nullptr;
    void        (*cancel)(hag_session *) = nullptr;
    hag_status  (*tokenize)(const hag_model *, const char *, int, int32_t *, int, int *) = nullptr;
    hag_status  (*score)(hag_session *, const char *, double *, int *) = nullptr;
    hag_status  (*train)(const char *, const char *const *, int, const hag_train_params *, const char *, const char *,
                         hag_train_fn, void *) = nullptr;
    hag_status  (*train_estimate)(const char *, const hag_train_params *, char **) = nullptr;
    hag_status  (*patch_info_json)(const char *, char **) = nullptr;
};

Api g_api;
bool g_bound = false;
JavaVM *g_vm = nullptr;

const char *kHagException = "com/hotatticgames/hag/runtime/HagException";

// ------------------------------------------------------------------------------------------------- JNI helpers
std::string bytesToString(JNIEnv *env, jbyteArray a) {
    if (a == nullptr) return std::string();
    jsize n = env->GetArrayLength(a);
    std::string s(static_cast<size_t>(n), '\0');
    if (n > 0) env->GetByteArrayRegion(a, 0, n, reinterpret_cast<jbyte *>(&s[0]));
    return s;
}

jbyteArray stringToBytes(JNIEnv *env, const char *s, size_t n) {
    jbyteArray a = env->NewByteArray(static_cast<jsize>(n));
    if (a != nullptr && n > 0) env->SetByteArrayRegion(a, 0, static_cast<jsize>(n), reinterpret_cast<const jbyte *>(s));
    return a;
}

jbyteArray cstrToBytes(JNIEnv *env, const char *s) { return s ? stringToBytes(env, s, strlen(s)) : stringToBytes(env, "", 0); }

void throwHag(JNIEnv *env, int code, const char *msg) {
    if (env->ExceptionCheck()) return;  // keep the first pending exception
    jclass c = env->FindClass(kHagException);
    if (c == nullptr) return;           // FindClass already left a NoClassDefFoundError pending
    jmethodID ctor = env->GetMethodID(c, "<init>", "(I[B)V");
    if (ctor != nullptr) {
        jbyteArray m = cstrToBytes(env, msg);
        jobject ex = env->NewObject(c, ctor, static_cast<jint>(code), m);
        if (ex != nullptr) env->Throw(static_cast<jthrowable>(ex));
        env->DeleteLocalRef(m);
    }
    env->DeleteLocalRef(c);
}

bool checkStatus(JNIEnv *env, const hag_status &st) {
    if (st.code == HAG_OK) return true;
    char buf[sizeof(st.message) + 1];
    memcpy(buf, st.message, sizeof(st.message));
    buf[sizeof(st.message)] = '\0';
    throwHag(env, st.code, buf);
    return false;
}

bool requireBound(JNIEnv *env) {
    if (g_bound) return true;
    throwHag(env, -100, "engine library is not bound");
    return false;
}

// Gets a JNIEnv for the current thread, attaching if needed. Detaches in the destructor only if it attached.
class EnvScope {
public:
    EnvScope() {
        if (g_vm == nullptr) return;
        jint r = g_vm->GetEnv(reinterpret_cast<void **>(&env_), JNI_VERSION_1_6);
        if (r == JNI_EDETACHED) {
            if (g_vm->AttachCurrentThread(HAG_ATTACH_ARG(&env_), nullptr) == JNI_OK) attached_ = true; else env_ = nullptr;
        } else if (r != JNI_OK) {
            env_ = nullptr;
        }
    }
    ~EnvScope() { if (attached_) g_vm->DetachCurrentThread(); }
    EnvScope(const EnvScope &) = delete;
    EnvScope &operator=(const EnvScope &) = delete;
    JNIEnv *env() const { return env_; }
private:
    JNIEnv *env_ = nullptr;
    bool attached_ = false;
};

// Global ref to a Kotlin callback object, released on scope exit (also on exceptions/early returns).
struct CbRef {
    jobject obj = nullptr;
    jmethodID mid = nullptr;
    CbRef(JNIEnv *env, jobject cb, const char *name, const char *sig) {
        if (cb == nullptr) return;
        jclass c = env->GetObjectClass(cb);
        mid = env->GetMethodID(c, name, sig);
        env->DeleteLocalRef(c);
        if (mid == nullptr) { env->ExceptionClear(); return; }
        obj = env->NewGlobalRef(cb);
    }
    ~CbRef() {
        if (obj == nullptr) return;
        EnvScope s;
        if (s.env() != nullptr) s.env()->DeleteGlobalRef(obj);
    }
    bool valid() const { return obj != nullptr && mid != nullptr; }
    CbRef(const CbRef &) = delete;
    CbRef &operator=(const CbRef &) = delete;
};

// A Kotlin callback must never leave an exception pending inside native code: clear it and request a stop.
// (The Kotlin wrappers catch Throwable themselves and re-throw after the call, so this is only a safety net.)
int finishCallback(JNIEnv *env, jboolean r) {
    if (env->ExceptionCheck()) { env->ExceptionClear(); return 1; }
    return r ? 1 : 0;
}

int onProgress(float fraction, void *user) {
    auto *cb = static_cast<CbRef *>(user);
    EnvScope s;
    if (s.env() == nullptr || !cb->valid()) return 0;
    jboolean r = s.env()->CallBooleanMethod(cb->obj, cb->mid, static_cast<jfloat>(fraction));
    return finishCallback(s.env(), r);
}

int onToken(const char *piece, void *user) {
    auto *cb = static_cast<CbRef *>(user);
    EnvScope s;
    if (s.env() == nullptr || !cb->valid()) return 1;
    JNIEnv *env = s.env();
    jbyteArray a = cstrToBytes(env, piece);
    if (a == nullptr) { env->ExceptionClear(); return 1; }
    jboolean r = env->CallBooleanMethod(cb->obj, cb->mid, a);
    env->DeleteLocalRef(a);
    return finishCallback(env, r);
}

int onTrain(const hag_train_event *ev, void *user) {
    auto *cb = static_cast<CbRef *>(user);
    EnvScope s;
    if (s.env() == nullptr || !cb->valid()) return 0;
    jboolean r = s.env()->CallBooleanMethod(
        cb->obj, cb->mid, static_cast<jint>(ev->phase), static_cast<jint>(ev->epoch), static_cast<jint>(ev->epochs),
        static_cast<jint>(ev->step), static_cast<jint>(ev->steps), static_cast<jlong>(ev->examples_done),
        static_cast<jdouble>(ev->train_loss), static_cast<jdouble>(ev->val_loss), static_cast<jdouble>(ev->elapsed_s),
        static_cast<jlong>(ev->rss_bytes), static_cast<jint>(ev->resumed_from_step));
    return finishCallback(s.env(), r);
}

// Double-array layout shared with NativeBridge.kt (TrainConfig.toNative()).
enum { P_N_CTX, P_N_BATCH, P_EPOCHS, P_LR, P_VAL_FRACTION, P_SEED, P_N_THREADS, P_LAST_LAYERS, P_TRAIN_EMB, P_CKPT_EVERY,
       P_MAX_MEM, P_COUNT };

bool readTrainParams(JNIEnv *env, jdoubleArray p, hag_train_params *out) {
    if (p == nullptr || env->GetArrayLength(p) < P_COUNT) { throwHag(env, -100, "bad train params array"); return false; }
    double v[P_COUNT];
    env->GetDoubleArrayRegion(p, 0, P_COUNT, v);
    memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(*out);
    out->n_ctx = static_cast<int32_t>(v[P_N_CTX]);
    out->n_batch = static_cast<int32_t>(v[P_N_BATCH]);
    out->epochs = static_cast<int32_t>(v[P_EPOCHS]);
    out->learning_rate = static_cast<float>(v[P_LR]);
    out->val_fraction = static_cast<float>(v[P_VAL_FRACTION]);
    out->seed = static_cast<uint32_t>(static_cast<int64_t>(v[P_SEED]));
    out->n_threads = static_cast<int32_t>(v[P_N_THREADS]);
    out->trainable_last_layers = static_cast<int32_t>(v[P_LAST_LAYERS]);
    out->train_embeddings = static_cast<int32_t>(v[P_TRAIN_EMB]);
    out->checkpoint_every_steps = static_cast<int32_t>(v[P_CKPT_EVERY]);
    out->max_memory_bytes = v[P_MAX_MEM] > 0 ? static_cast<size_t>(v[P_MAX_MEM]) : 0;
    return true;
}

template <typename T> T *asPtr(jlong h) { return reinterpret_cast<T *>(static_cast<intptr_t>(h)); }

template <typename F> bool bindSym(void *lib, const char *name, F *slot, std::string *err) {
    void *sym = dlsym(lib, name);
    if (sym == nullptr) { *err = std::string("engine library lacks symbol ") + name; return false; }
    *slot = reinterpret_cast<F>(sym);
    return true;
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
    g_vm = vm;
    return JNI_VERSION_1_6;
}

#define JFN(ret, name) extern "C" JNIEXPORT ret JNICALL Java_com_hotatticgames_hag_runtime_NativeBridge_##name

// ---------------------------------------------------------------------------------------------- CPU feature gate
// Returns null when this CPU can run the engine build, else a human-readable reason (UTF-8).
// The required set must match the compile baseline in CMakeLists.txt.
JFN(jbyteArray, nativeCpuCheck)(JNIEnv *env, jclass) {
#if defined(__aarch64__)
    // Linux hwcap bits (arm64): FPHP = bit 9, ASIMDHP = bit 10, ASIMDDP = bit 20.
    const unsigned long kFphp = 1UL << 9, kAsimdhp = 1UL << 10, kAsimddp = 1UL << 20;
    unsigned long hw = getauxval(AT_HWCAP);
    std::string miss;
    if (!(hw & kFphp)) miss += " fp16";
    if (!(hw & kAsimdhp)) miss += " fp16-simd";
    if (!(hw & kAsimddp)) miss += " dotprod";
    if (miss.empty()) return nullptr;
    char buf[200];
    snprintf(buf, sizeof(buf), "CPU lacks required ARMv8.2 features:%s (AT_HWCAP=0x%lx). This device cannot run the on-device engine.",
             miss.c_str(), hw);
    return cstrToBytes(env, buf);
#elif defined(__x86_64__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("sse4.2") && __builtin_cpu_supports("popcnt")) return nullptr;
    return cstrToBytes(env, "CPU lacks SSE4.2/POPCNT required by the x86_64 engine build.");
#else
    return cstrToBytes(env, "unsupported CPU architecture (engine ships for arm64-v8a and x86_64 only)");
#endif
}

JFN(jbyteArray, nativeCpuAbi)(JNIEnv *env, jclass) {
#if defined(__aarch64__)
    return cstrToBytes(env, "arm64-v8a");
#elif defined(__x86_64__)
    return cstrToBytes(env, "x86_64");
#else
    return cstrToBytes(env, "unsupported");
#endif
}

// ----------------------------------------------------------------------------------------------- bind + lifecycle
// Resolves every hag_* symbol from the already-loaded libhagengine.so. Returns null on success, else the reason.
JFN(jbyteArray, nativeBind)(JNIEnv *env, jclass) {
    if (g_bound) return nullptr;
    // Normally the Java side already System.loadLibrary()'d it, so the bare soname resolves to the loaded image.
    void *lib = dlopen("libhagengine.so", RTLD_NOW | RTLD_LOCAL);
    std::string firstErr;
    if (lib == nullptr) { const char *fe = dlerror(); if (fe != nullptr) firstErr = fe; }
    if (lib == nullptr) {
        // Fallback: the sibling of this very library (works for extracted libs and for "base.apk!/lib/<abi>/" paths).
        Dl_info info;
        if (dladdr(reinterpret_cast<void *>(&JNI_OnLoad), &info) != 0 && info.dli_fname != nullptr) {
            std::string self(info.dli_fname);
            size_t slash = self.rfind('/');
            if (slash != std::string::npos) lib = dlopen((self.substr(0, slash + 1) + "libhagengine.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        }
    }
    if (lib == nullptr) {
        const char *e = dlerror();
        return cstrToBytes(env, ("dlopen(libhagengine.so) failed: " + (e ? std::string(e) : firstErr)).c_str());
    }
    Api a;
    std::string err;
    bool ok =
        bindSym(lib, "hag_engine_init", &a.engine_init, &err) && bindSym(lib, "hag_engine_shutdown", &a.engine_shutdown, &err) &&
        bindSym(lib, "hag_engine_version", &a.engine_version, &err) && bindSym(lib, "hag_system_info_json", &a.system_info_json, &err) &&
        bindSym(lib, "hag_free", &a.free_, &err) && bindSym(lib, "hag_model_load", &a.model_load, &err) &&
        bindSym(lib, "hag_model_apply_patch", &a.model_apply_patch, &err) && bindSym(lib, "hag_model_free", &a.model_free, &err) &&
        bindSym(lib, "hag_model_info_json", &a.model_info_json, &err) && bindSym(lib, "hag_session_new", &a.session_new, &err) &&
        bindSym(lib, "hag_session_free", &a.session_free, &err) && bindSym(lib, "hag_session_reset", &a.session_reset, &err) &&
        bindSym(lib, "hag_chat_format", &a.chat_format, &err) && bindSym(lib, "hag_generate", &a.generate, &err) &&
        bindSym(lib, "hag_cancel", &a.cancel, &err) && bindSym(lib, "hag_tokenize", &a.tokenize, &err) &&
        bindSym(lib, "hag_score", &a.score, &err) && bindSym(lib, "hag_train", &a.train, &err) &&
        bindSym(lib, "hag_train_estimate", &a.train_estimate, &err) && bindSym(lib, "hag_patch_info_json", &a.patch_info_json, &err);
    if (!ok) return cstrToBytes(env, err.c_str());
    g_api = a;
    g_bound = true;
    return nullptr;
}

JFN(jbyteArray, nativeEngineInit)(JNIEnv *env, jclass) {
    if (!g_bound) return cstrToBytes(env, "engine library is not bound");
    hag_status st = g_api.engine_init();
    if (st.code == HAG_OK) return nullptr;
    char buf[sizeof(st.message) + 1];
    memcpy(buf, st.message, sizeof(st.message));
    buf[sizeof(st.message)] = '\0';
    return cstrToBytes(env, buf);
}

JFN(void, nativeEngineShutdown)(JNIEnv *, jclass) { if (g_bound) g_api.engine_shutdown(); }

JFN(jbyteArray, nativeVersion)(JNIEnv *env, jclass) {
    if (!requireBound(env)) return nullptr;
    return cstrToBytes(env, g_api.engine_version());
}

JFN(jbyteArray, nativeSystemInfo)(JNIEnv *env, jclass) {
    if (!requireBound(env)) return nullptr;
    return cstrToBytes(env, g_api.system_info_json());
}

// ------------------------------------------------------------------------------------------------------ models
JFN(jlong, nativeModelLoad)(JNIEnv *env, jclass, jbyteArray path, jboolean useMmap, jobject cb) {
    if (!requireBound(env)) return 0;
    std::string p = bytesToString(env, path);
    CbRef ref(env, cb, "onProgress", "(F)Z");
    hag_model *m = nullptr;
    hag_status st = g_api.model_load(p.c_str(), useMmap ? 1 : 0, ref.valid() ? onProgress : nullptr, &ref, &m);
    if (!checkStatus(env, st)) return 0;
    if (m == nullptr) { throwHag(env, HAG_ERR_INTERNAL, "engine returned OK with a null model"); return 0; }
    return static_cast<jlong>(reinterpret_cast<intptr_t>(m));
}

JFN(void, nativeModelApplyPatch)(JNIEnv *env, jclass, jlong model, jbyteArray path) {
    if (!requireBound(env)) return;
    std::string p = bytesToString(env, path);
    checkStatus(env, g_api.model_apply_patch(asPtr<hag_model>(model), p.c_str()));
}

JFN(jbyteArray, nativeModelInfo)(JNIEnv *env, jclass, jlong model) {
    if (!requireBound(env)) return nullptr;
    return cstrToBytes(env, g_api.model_info_json(asPtr<hag_model>(model)));  // owned by the model: copied right away
}

JFN(void, nativeModelFree)(JNIEnv *, jclass, jlong model) { if (g_bound) g_api.model_free(asPtr<hag_model>(model)); }

// ----------------------------------------------------------------------------------------------------- sessions
JFN(jlong, nativeSessionNew)(JNIEnv *env, jclass, jlong model, jint nCtx, jint nThreads, jint nBatch) {
    if (!requireBound(env)) return 0;
    hag_session_params sp;
    memset(&sp, 0, sizeof(sp));
    sp.struct_size = sizeof(sp);
    sp.n_ctx = nCtx; sp.n_threads = nThreads; sp.n_batch = nBatch;
    hag_session *s = nullptr;
    if (!checkStatus(env, g_api.session_new(asPtr<hag_model>(model), &sp, &s))) return 0;
    if (s == nullptr) { throwHag(env, HAG_ERR_INTERNAL, "engine returned OK with a null session"); return 0; }
    return static_cast<jlong>(reinterpret_cast<intptr_t>(s));
}

JFN(void, nativeSessionReset)(JNIEnv *env, jclass, jlong s) {
    if (!requireBound(env)) return;
    checkStatus(env, g_api.session_reset(asPtr<hag_session>(s)));
}

JFN(void, nativeSessionFree)(JNIEnv *, jclass, jlong s) { if (g_bound) g_api.session_free(asPtr<hag_session>(s)); }

JFN(void, nativeCancel)(JNIEnv *, jclass, jlong s) { if (g_bound) g_api.cancel(asPtr<hag_session>(s)); }

JFN(jbyteArray, nativeChatFormat)(JNIEnv *env, jclass, jlong model, jobjectArray roles, jobjectArray contents, jboolean addGen) {
    if (!requireBound(env)) return nullptr;
    jsize n = roles ? env->GetArrayLength(roles) : 0;
    if (contents == nullptr || env->GetArrayLength(contents) != n) { throwHag(env, -100, "roles/contents length mismatch"); return nullptr; }
    std::vector<std::string> r(static_cast<size_t>(n)), c(static_cast<size_t>(n));
    std::vector<const char *> rp(static_cast<size_t>(n)), cp(static_cast<size_t>(n));
    for (jsize i = 0; i < n; i++) {
        jbyteArray a = static_cast<jbyteArray>(env->GetObjectArrayElement(roles, i));
        jbyteArray b = static_cast<jbyteArray>(env->GetObjectArrayElement(contents, i));
        r[i] = bytesToString(env, a); c[i] = bytesToString(env, b);
        env->DeleteLocalRef(a); env->DeleteLocalRef(b);
        rp[i] = r[i].c_str(); cp[i] = c[i].c_str();
    }
    char *out = nullptr;
    hag_status st = g_api.chat_format(asPtr<hag_model>(model), rp.data(), cp.data(), static_cast<int>(n), addGen ? 1 : 0, &out);
    if (!checkStatus(env, st)) return nullptr;
    jbyteArray res = cstrToBytes(env, out);
    if (out != nullptr) g_api.free_(out);
    return res;
}

JFN(jbyteArray, nativeGenerate)(JNIEnv *env, jclass, jlong session, jbyteArray prompt, jfloat temperature, jint topK, jfloat topP,
                                jfloat minP, jfloat repeatPenalty, jlong seed, jint maxNew, jobject sink) {
    if (!requireBound(env)) return nullptr;
    std::string p = bytesToString(env, prompt);
    CbRef ref(env, sink, "onPiece", "([B)Z");
    if (!ref.valid()) { throwHag(env, -100, "token sink is required"); return nullptr; }
    hag_sample_params sp;
    memset(&sp, 0, sizeof(sp));
    sp.struct_size = sizeof(sp);
    sp.temperature = temperature; sp.top_k = topK; sp.top_p = topP; sp.min_p = minP; sp.repeat_penalty = repeatPenalty;
    sp.seed = static_cast<uint32_t>(seed); sp.max_new_tokens = maxNew;
    hag_gen_stats stats;
    memset(&stats, 0, sizeof(stats));
    hag_status st = g_api.generate(asPtr<hag_session>(session), p.c_str(), &sp, onToken, &ref, &stats);
    if (!checkStatus(env, st)) return nullptr;
    char buf[320];
    snprintf(buf, sizeof(buf),
             "{\"n_prompt_tokens\":%d,\"n_generated\":%d,\"prompt_ms\":%.3f,\"gen_ms\":%.3f,\"stop_reason\":%d,\"peak_rss_bytes\":%llu}",
             stats.n_prompt_tokens, stats.n_generated, stats.prompt_ms, stats.gen_ms, stats.stop_reason,
             static_cast<unsigned long long>(stats.peak_rss_bytes));
    return cstrToBytes(env, buf);
}

JFN(jintArray, nativeTokenize)(JNIEnv *env, jclass, jlong model, jbyteArray text, jboolean addSpecial) {
    if (!requireBound(env)) return nullptr;
    std::string t = bytesToString(env, text);
    // Every token consumes >= 1 byte of input; specials add a handful.
    int cap = static_cast<int>(t.size()) + 16;
    for (int attempt = 0; attempt < 2; attempt++) {
        std::vector<int32_t> toks(static_cast<size_t>(cap));
        int n = 0;
        if (!checkStatus(env, g_api.tokenize(asPtr<hag_model>(model), t.c_str(), addSpecial ? 1 : 0, toks.data(), cap, &n))) return nullptr;
        if (n > cap) { cap = n; continue; }  // engine reported the size it needed
        jintArray out = env->NewIntArray(n);
        if (out != nullptr && n > 0) env->SetIntArrayRegion(out, 0, n, reinterpret_cast<const jint *>(toks.data()));
        return out;
    }
    throwHag(env, HAG_ERR_INTERNAL, "tokenize: inconsistent required size");
    return nullptr;
}

JFN(jbyteArray, nativeScore)(JNIEnv *env, jclass, jlong session, jbyteArray text) {
    if (!requireBound(env)) return nullptr;
    std::string t = bytesToString(env, text);
    double nll = 0; int n = 0;
    if (!checkStatus(env, g_api.score(asPtr<hag_session>(session), t.c_str(), &nll, &n))) return nullptr;
    char buf[96];
    snprintf(buf, sizeof(buf), "{\"mean_nll\":%.9g,\"n_tokens\":%d}", std::isfinite(nll) ? nll : 0.0, n);
    return cstrToBytes(env, buf);
}

// ------------------------------------------------------------------------------------------------------ training
JFN(jbyteArray, nativeTrainEstimate)(JNIEnv *env, jclass, jbyteArray base, jdoubleArray params) {
    if (!requireBound(env)) return nullptr;
    std::string b = bytesToString(env, base);
    hag_train_params tp;
    if (!readTrainParams(env, params, &tp)) return nullptr;
    char *out = nullptr;
    if (!checkStatus(env, g_api.train_estimate(b.c_str(), &tp, &out))) return nullptr;
    jbyteArray res = cstrToBytes(env, out);
    if (out != nullptr) g_api.free_(out);
    return res;
}

JFN(void, nativeTrain)(JNIEnv *env, jclass, jbyteArray base, jobjectArray texts, jdoubleArray params, jbyteArray workDir,
                       jbyteArray outPatch, jobject progress) {
    if (!requireBound(env)) return;
    hag_train_params tp;
    if (!readTrainParams(env, params, &tp)) return;
    std::string b = bytesToString(env, base), w = bytesToString(env, workDir), o = bytesToString(env, outPatch);
    jsize n = texts ? env->GetArrayLength(texts) : 0;
    std::vector<std::string> store(static_cast<size_t>(n));
    std::vector<const char *> ptrs(static_cast<size_t>(n));
    for (jsize i = 0; i < n; i++) {
        jbyteArray a = static_cast<jbyteArray>(env->GetObjectArrayElement(texts, i));
        store[i] = bytesToString(env, a);
        env->DeleteLocalRef(a);
        ptrs[i] = store[i].c_str();
    }
    CbRef ref(env, progress, "onEvent", "(IIIIIJDDDJI)Z");
    checkStatus(env, g_api.train(b.c_str(), ptrs.data(), static_cast<int>(n), &tp, w.c_str(), o.c_str(),
                                 ref.valid() ? onTrain : nullptr, &ref));
}

JFN(jbyteArray, nativePatchInfo)(JNIEnv *env, jclass, jbyteArray path) {
    if (!requireBound(env)) return nullptr;
    std::string p = bytesToString(env, path);
    char *out = nullptr;
    if (!checkStatus(env, g_api.patch_info_json(p.c_str(), &out))) return nullptr;
    jbyteArray res = cstrToBytes(env, out);
    if (out != nullptr) g_api.free_(out);
    return res;
}
