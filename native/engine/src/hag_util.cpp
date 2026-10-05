#include "hag_util.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(_WIN32)
#  include <direct.h>
#  include <io.h>
#else
#  include <fcntl.h>
#  include <sys/resource.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace hag {

int fseek64(FILE * f, int64_t off, int whence) {
#if defined(_WIN32)
    return _fseeki64(f, off, whence);
#else
    return fseeko(f, (off_t)off, whence);
#endif
}
int64_t ftell64(FILE * f) {
#if defined(_WIN32)
    return _ftelli64(f);
#else
    return (int64_t)ftello(f);
#endif
}

// ---- status -----------------------------------------------------------------------------------------------------
hag_status make_status(int code, const char * fmt, ...) {
    hag_status s;
    s.code = code;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(s.message, sizeof(s.message), fmt, ap);
    va_end(ap);
    return s;
}

// ---- log capture ------------------------------------------------------------------------------------------------
static std::mutex  g_log_mu;
static std::string g_last_error;
static bool        g_verbose = false;

static void log_cb(enum ggml_log_level level, const char * text, void *) {
    if (!text) return;
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::lock_guard<std::mutex> lk(g_log_mu);
        g_last_error = text;
        while (!g_last_error.empty() && (g_last_error.back() == '\n' || g_last_error.back() == '\r')) g_last_error.pop_back();
    }
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

void install_log_capture() {
    const char * v = getenv("HAG_LOG");
    g_verbose = v && v[0] && v[0] != '0';
    llama_log_set(log_cb, nullptr);
}
std::string last_native_error() { std::lock_guard<std::mutex> lk(g_log_mu); return g_last_error; }
void clear_native_error() { std::lock_guard<std::mutex> lk(g_log_mu); g_last_error.clear(); }

// ---- SHA-256 ----------------------------------------------------------------------------------------------------
static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void Sha256::reset() {
    static const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(h_, init, sizeof(h_));
    total_ = 0;
    fill_ = 0;
}

void Sha256::block(const uint8_t * p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | (uint32_t)p[4 * i + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K256[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
}

void Sha256::update(const void * data, size_t len) {
    const uint8_t * p = (const uint8_t *)data;
    total_ += len;
    if (fill_) {
        size_t take = std::min(len, 64 - fill_);
        memcpy(buf_ + fill_, p, take);
        fill_ += take; p += take; len -= take;
        if (fill_ == 64) { block(buf_); fill_ = 0; }
    }
    while (len >= 64) { block(p); p += 64; len -= 64; }
    if (len) { memcpy(buf_, p, len); fill_ = len; }
}

void Sha256::final(uint8_t out[32]) {
    uint64_t bits = total_ * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    uint8_t z = 0;
    while (fill_ != 56) update(&z, 1);
    uint8_t lenb[8];
    for (int i = 0; i < 8; i++) lenb[i] = (uint8_t)(bits >> (56 - 8 * i));
    update(lenb, 8);
    for (int i = 0; i < 8; i++) {
        out[4 * i] = (uint8_t)(h_[i] >> 24); out[4 * i + 1] = (uint8_t)(h_[i] >> 16);
        out[4 * i + 2] = (uint8_t)(h_[i] >> 8); out[4 * i + 3] = (uint8_t)h_[i];
    }
}

std::string hex_encode(const uint8_t * p, size_t n) {
    static const char * d = "0123456789abcdef";
    std::string s(2 * n, '0');
    for (size_t i = 0; i < n; i++) { s[2 * i] = d[p[i] >> 4]; s[2 * i + 1] = d[p[i] & 15]; }
    return s;
}
std::string Sha256::final_hex() { uint8_t o[32]; final(o); return hex_encode(o, 32); }
std::string sha256_hex(const void * data, size_t len) { Sha256 s; s.update(data, len); return s.final_hex(); }

bool sha256_file(const char * path, std::string & hex_out, bool (*cancel)(void *), void * ud) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    Sha256 s;
    std::vector<uint8_t> buf(1 << 20);
    bool ok = true;
    for (;;) {
        size_t n = fread(buf.data(), 1, buf.size(), f);
        if (n) s.update(buf.data(), n);
        if (n < buf.size()) { ok = !ferror(f); break; }
        if (cancel && cancel(ud)) { ok = false; break; }
    }
    fclose(f);
    if (ok) hex_out = s.final_hex();
    return ok;
}

// ---- files ------------------------------------------------------------------------------------------------------
bool file_exists(const std::string & p) { return file_size(p) >= 0; }

int64_t file_size(const std::string & p) {
#if defined(_WIN32)
    struct _stat64 st;
    if (_stat64(p.c_str(), &st) != 0) return -1;
#else
    struct stat st;
    if (stat(p.c_str(), &st) != 0) return -1;
    if (S_ISDIR(st.st_mode)) return -1;
#endif
    return (int64_t)st.st_size;
}

bool make_dir(const std::string & p) {
#if defined(_WIN32)
    return _mkdir(p.c_str()) == 0 || errno == EEXIST;
#else
    if (mkdir(p.c_str(), 0755) == 0) return true;
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

bool remove_file(const std::string & p) { return remove(p.c_str()) == 0; }
bool rename_file(const std::string & a, const std::string & b) { return rename(a.c_str(), b.c_str()) == 0; }

void fsync_dir_of(const std::string & path) {
#if !defined(_WIN32)
    size_t slash = path.find_last_of('/');
    std::string dir = slash == std::string::npos ? "." : (slash == 0 ? "/" : path.substr(0, slash));
    int fd = open(dir.c_str(), O_RDONLY);
    if (fd >= 0) { fsync(fd); close(fd); }
#else
    (void)path;
#endif
}

std::string join_path(const std::string & a, const std::string & b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

bool read_file(const std::string & p, std::string & out, size_t max_bytes) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return false;
    out.clear();
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
        if (out.size() > max_bytes) { fclose(f); return false; }
    }
    bool ok = !ferror(f);
    fclose(f);
    return ok;
}

AtomicWriter::AtomicWriter(const std::string & final_path) : final_(final_path), tmp_(final_path + ".tmp") {
    f_ = fopen(tmp_.c_str(), "wb");
}
AtomicWriter::~AtomicWriter() {
    if (f_) { fclose(f_); f_ = nullptr; remove(tmp_.c_str()); }
}
bool AtomicWriter::write(const void * p, size_t n) {
    if (!f_ || err_) return false;
    if (n && fwrite(p, 1, n, f_) != n) { err_ = true; return false; }
    sha_.update(p, n);
    pos_ += n;
    return true;
}
bool AtomicWriter::seek_write(uint64_t off, const void * p, size_t n) {
    if (!f_ || err_) return false;
    int64_t cur = ftell64(f_);
    if (fseek64(f_, (int64_t)off, SEEK_SET) != 0 || fwrite(p, 1, n, f_) != n || fseek64(f_, cur, SEEK_SET) != 0) { err_ = true; return false; }
    return true;
}
std::string AtomicWriter::hash_hex() { Sha256 c = sha_; return c.final_hex(); }
bool AtomicWriter::commit() {
    if (!f_ || err_) return false;
    bool ok = fflush(f_) == 0;
#if !defined(_WIN32)
    ok = ok && fsync(fileno(f_)) == 0;
#endif
    ok = (fclose(f_) == 0) && ok;
    f_ = nullptr;
    if (!ok) { remove(tmp_.c_str()); return false; }
    if (rename(tmp_.c_str(), final_.c_str()) != 0) { remove(tmp_.c_str()); return false; }
    fsync_dir_of(final_);
    return true;
}

// ---- process info -----------------------------------------------------------------------------------------------
static size_t proc_status_kb(const char * key) {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    size_t kl = strlen(key);
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, key, kl) == 0) { kb = (size_t)strtoull(line + kl, nullptr, 10); break; }
    }
    fclose(f);
    return kb;
}

size_t rss_bytes() { return proc_status_kb("VmRSS:") * 1024; }
size_t peak_rss_bytes() {
    size_t hwm = proc_status_kb("VmHWM:") * 1024;
    if (hwm) return hwm;
#if !defined(_WIN32)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) return (size_t)ru.ru_maxrss * 1024;
#endif
    return rss_bytes();
}
double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---- JSON -------------------------------------------------------------------------------------------------------
void Json::key_(const char * key) {
    sep();
    if (key) { str_(key); s_ += ':'; }
}
void Json::str_(const std::string & v) {
    s_ += '"';
    for (unsigned char c : v) {
        switch (c) {
            case '"': s_ += "\\\""; break;
            case '\\': s_ += "\\\\"; break;
            case '\n': s_ += "\\n"; break;
            case '\r': s_ += "\\r"; break;
            case '\t': s_ += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); s_ += b; }
                else s_ += (char)c;
        }
    }
    s_ += '"';
}
Json & Json::kv(const char * key, double v) {
    key_(key);
    if (std::isnan(v) || std::isinf(v)) { s_ += "null"; return *this; }
    char b[48];
    snprintf(b, sizeof(b), "%.10g", v);
    s_ += b;
    return *this;
}

// ---- UTF-8 ------------------------------------------------------------------------------------------------------
static const char FFFD[] = "\xEF\xBF\xBD";

void utf8_push(std::string & pending, const char * in, size_t n, std::string & out) {
    std::string buf;
    buf.swap(pending);
    buf.append(in, n);
    const size_t len_all = buf.size();
    size_t i = 0;
    while (i < len_all) {
        unsigned char c = (unsigned char)buf[i];
        if (c == 0) { i++; continue; }
        if (c < 0x80) { out += (char)c; i++; continue; }
        size_t len;
        if (c >= 0xC2 && c <= 0xDF) len = 2;
        else if (c >= 0xE0 && c <= 0xEF) len = 3;
        else if (c >= 0xF0 && c <= 0xF4) len = 4;
        else { out += FFFD; i++; continue; }
        size_t avail = len_all - i;
        size_t check = std::min(len, avail);
        bool bad = false;
        for (size_t k = 1; k < check && !bad; k++) {
            unsigned char b = (unsigned char)buf[i + k];
            if ((b & 0xC0) != 0x80) { bad = true; break; }
            if (k == 1) {
                if (c == 0xE0 && b < 0xA0) bad = true;
                else if (c == 0xED && b > 0x9F) bad = true;
                else if (c == 0xF0 && b < 0x90) bad = true;
                else if (c == 0xF4 && b > 0x8F) bad = true;
            }
        }
        if (bad) { out += FFFD; i++; continue; }
        if (avail < len) break;  // incomplete but valid so far: wait for more bytes
        out.append(buf, i, len);
        i += len;
    }
    pending.assign(buf, i, std::string::npos);
}

void utf8_flush(std::string & pending, std::string & out) {
    // every remaining byte belongs to an incomplete sequence: one replacement character for the sequence
    if (!pending.empty()) out += FFFD;
    pending.clear();
}

}  // namespace hag
