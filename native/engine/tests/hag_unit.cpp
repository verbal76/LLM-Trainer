// Unit tests for the engine's self-contained helpers (no model needed): SHA-256 vectors, UTF-8 stream sanitising,
// JSON writer, deterministic PRNG, atomic file writing.  Exit status 0 = all passed.
#include "hag_util.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using namespace hag;

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

static std::string feed_chunks(const std::string & bytes, size_t chunk) {
    std::string pending, out;
    for (size_t i = 0; i < bytes.size(); i += chunk) utf8_push(pending, bytes.data() + i, std::min(chunk, bytes.size() - i), out);
    utf8_flush(pending, out);
    return out;
}

int main() {
    // SHA-256 known answers (FIPS 180-4)
    CHECK(sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    {
        std::string m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        CHECK(sha256_hex(m.data(), m.size()) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        Sha256 s;  // incremental == one-shot
        for (char c : m) s.update(&c, 1);
        CHECK(s.final_hex() == sha256_hex(m.data(), m.size()));
        std::string mil(1000000, 'a');
        CHECK(sha256_hex(mil.data(), mil.size()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }
    // UTF-8: a multi-byte character split at every possible boundary is delivered intact
    const std::string euro = "\xE2\x82\xAC", deg = "\xC2\xB0", rocket = "\xF0\x9F\x9A\x80";
    std::string text = "T=20" + deg + "C " + euro + "5 " + rocket + " ok";
    for (size_t chunk = 1; chunk <= text.size(); chunk++) CHECK(feed_chunks(text, chunk) == text);
    // incomplete sequence at the end -> one replacement char; invalid bytes -> replacement; NUL dropped
    CHECK(feed_chunks("ab\xE2\x82", 1) == "ab\xEF\xBF\xBD");
    CHECK(feed_chunks("a\xFF" "b", 1) == "a\xEF\xBF\xBD" "b");
    CHECK(feed_chunks(std::string("a\0b", 3), 1) == "ab");
    CHECK(feed_chunks("\xC0\x80", 1) == "\xEF\xBF\xBD\xEF\xBF\xBD");        // overlong
    CHECK(feed_chunks("\xED\xA0\x80", 1) == "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD");  // UTF-16 surrogate
    // a held-back lead byte must not be emitted early
    {
        std::string pending, out;
        utf8_push(pending, "\xE2", 1, out);
        CHECK(out.empty());
        utf8_push(pending, "\x82", 1, out);
        CHECK(out.empty());
        utf8_push(pending, "\xAC", 1, out);
        CHECK(out == euro);
    }
    // JSON
    {
        Json j;
        j.begin_obj().kv("a", "x\"y\n").kv("n", std::nan("")).kv("i", 5).begin_arr("l").val("p").val("q").end_arr().end_obj();
        CHECK(j.str() == "{\"a\":\"x\\\"y\\n\",\"n\":null,\"i\":5,\"l\":[\"p\",\"q\"]}");
    }
    // deterministic PRNG (data order must never depend on the platform's C++ library)
    {
        SplitMix64 r(1234567);
        CHECK(r.next() == 6457827717110365317ull);
        CHECK(r.next() == 3203168211198807973ull);
    }
    // atomic writer: nothing visible until commit; destructor discards
    {
        std::string p = "/tmp/hag_unit_atomic.bin";
        remove_file(p);
        {
            AtomicWriter w(p);
            CHECK(w.ok());
            w.write("hello", 5);
        }
        CHECK(!file_exists(p));
        AtomicWriter w(p);
        w.write("hello", 5);
        CHECK(w.hash_hex() == sha256_hex("hello", 5));
        CHECK(w.commit());
        CHECK(file_size(p) == 5);
        std::string hx;
        CHECK(sha256_file(p.c_str(), hx) && hx == sha256_hex("hello", 5));
        remove_file(p);
    }
    if (g_fail) { fprintf(stderr, "%d check(s) failed\n", g_fail); return 1; }
    printf("hag_unit: all checks passed\n");
    return 0;
}
