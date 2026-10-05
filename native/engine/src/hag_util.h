// Internal helpers shared by the engine translation units. Not part of the public C API.
#pragma once
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include "hag_engine.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace hag {

// ---- status ---------------------------------------------------------------------------------------------------
hag_status make_status(int code, const char * fmt, ...) __attribute__((format(printf, 2, 3)));
inline hag_status ok_status() { hag_status s; s.code = HAG_OK; s.message[0] = 0; return s; }

// ---- logging: llama.cpp/ggml logs are captured; errors are kept so failures can carry their reason -------------
void  install_log_capture();
std::string last_native_error();   // most recent ERROR-level line from llama.cpp/ggml (trimmed), may be empty
void  clear_native_error();

// ---- SHA-256 (portable) ----------------------------------------------------------------------------------------
class Sha256 {
public:
    Sha256() { reset(); }
    void reset();
    void update(const void * data, size_t len);
    void final(uint8_t out[32]);
    std::string final_hex();
private:
    void block(const uint8_t * p);
    uint32_t h_[8];
    uint8_t  buf_[64];
    uint64_t total_;
    size_t   fill_;
};
std::string hex_encode(const uint8_t * p, size_t n);
std::string sha256_hex(const void * data, size_t len);
// Streams a whole file. `cancel` (optional) is polled between 1 MiB blocks. Returns false on I/O error or cancel.
bool sha256_file(const char * path, std::string & hex_out, bool (*cancel)(void *) = nullptr, void * ud = nullptr);

int     fseek64(FILE * f, int64_t off, int whence);
int64_t ftell64(FILE * f);

// ---- files -----------------------------------------------------------------------------------------------------
bool        file_exists(const std::string & p);
int64_t     file_size(const std::string & p);          // -1 if missing
bool        make_dir(const std::string & p);           // single level, ok if it exists
bool        remove_file(const std::string & p);
bool        rename_file(const std::string & from, const std::string & to);
void        fsync_dir_of(const std::string & path);
std::string join_path(const std::string & a, const std::string & b);
bool        read_file(const std::string & p, std::string & out, size_t max_bytes = 1 << 26);

// Write-to-temp, fsync, rename. Hashes everything written. Use finish() to commit or the destructor to discard.
class AtomicWriter {
public:
    explicit AtomicWriter(const std::string & final_path);
    ~AtomicWriter();
    bool ok() const { return f_ != nullptr && !err_; }
    bool write(const void * p, size_t n);
    bool seek_write(uint64_t off, const void * p, size_t n);   // does not touch the running hash
    bool commit();                                             // fflush + fsync + rename + fsync(dir)
    std::string hash_hex();                                    // of bytes written via write() so far (sequential)
    uint64_t    written() const { return pos_; }
    const std::string & tmp_path() const { return tmp_; }
private:
    std::string final_, tmp_;
    FILE * f_ = nullptr;
    bool err_ = false;
    uint64_t pos_ = 0;
    Sha256 sha_;
};

// ---- process info ----------------------------------------------------------------------------------------------
size_t rss_bytes();       // current resident set
size_t peak_rss_bytes();  // process high-water mark
double now_ms();          // steady clock

// ---- JSON writing (tiny, append-only) ---------------------------------------------------------------------------
class Json {
public:
    Json() { s_.reserve(512); }
    Json & begin_obj() { sep(); s_ += '{'; first_ = true; return *this; }
    Json & end_obj()   { s_ += '}'; first_ = false; return *this; }
    Json & begin_arr(const char * key = nullptr) { key_(key); s_ += '['; first_ = true; return *this; }
    Json & end_arr()   { s_ += ']'; first_ = false; return *this; }
    Json & begin_obj(const char * key) { key_(key); s_ += '{'; first_ = true; return *this; }
    Json & kv(const char * key, const std::string & v) { key_(key); str_(v); return *this; }
    Json & kv(const char * key, const char * v) { key_(key); str_(v ? v : ""); return *this; }
    Json & kv(const char * key, double v);
    Json & kv(const char * key, int64_t v) { key_(key); s_ += std::to_string((long long)v); return *this; }
    Json & kv(const char * key, uint64_t v) { key_(key); s_ += std::to_string((unsigned long long)v); return *this; }
    Json & kv(const char * key, int v) { return kv(key, (int64_t)v); }
    Json & kv(const char * key, unsigned v) { return kv(key, (uint64_t)v); }
    Json & kv(const char * key, bool v) { key_(key); s_ += v ? "true" : "false"; return *this; }
    Json & raw(const char * key, const std::string & json) { key_(key); s_ += json; return *this; }
    Json & val(const std::string & v) { sep(); str_(v); return *this; }
    const std::string & str() const { return s_; }
private:
    void sep() { if (!first_) s_ += ','; first_ = false; }
    void key_(const char * key);
    void str_(const std::string & v);
    std::string s_;
    bool first_ = true;
};

// ---- deterministic, platform-independent PRNG (data order must not depend on the C++ library) -------------------
struct SplitMix64 {
    uint64_t x;
    explicit SplitMix64(uint64_t seed) : x(seed) {}
    uint64_t next() {
        uint64_t z = (x += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint64_t below(uint64_t n) { return next() % n; }
};

// ---- UTF-8 -----------------------------------------------------------------------------------------------------
// Appends `in` to `pending`, then moves the longest prefix of complete & valid UTF-8 sequences to `out`
// (invalid bytes become U+FFFD, NUL bytes are dropped). A trailing incomplete-but-possibly-valid sequence stays in pending.
void utf8_push(std::string & pending, const char * in, size_t n, std::string & out);
// End of stream: whatever is left in pending is incomplete -> emitted as U+FFFD(s).
void utf8_flush(std::string & pending, std::string & out);

}  // namespace hag
