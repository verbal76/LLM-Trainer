// Internal structures shared between engine.cpp (inference/patch apply) and train.cpp (training/patch write).
#pragma once

#include "hag_engine.h"
#include "hag_util.h"

#include "gguf.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <atomic>
#include <string>
#include <vector>

// ---- specialist patch format v1 (see docs/v2/ENGINE.md) ----------------------------------------------------------
#define HAG_PATCH_ARCH           "hag-patch"
#define HAG_PATCH_FORMAT_VERSION 1u
#define HAG_KEY_VERSION          "hag.patch.format_version"
#define HAG_KEY_BASE_ARCH        "hag.patch.base.arch"
#define HAG_KEY_BASE_SHA256      "hag.patch.base.sha256"
#define HAG_KEY_BASE_SIZE        "hag.patch.base.size_bytes"
#define HAG_KEY_BASE_NTENSORS    "hag.patch.base.n_tensors"
#define HAG_KEY_PAYLOAD_SHA256   "hag.patch.payload.sha256"
#define HAG_KEY_FINGERPRINT      "hag.patch.run_fingerprint"
#define HAG_KEY_ENGINE           "hag.patch.engine"
#define HAG_KEY_KIND             "hag.patch.kind"   /* "replace" (tensor replacement) | "lora" (llama.cpp LoRA adapter) */

struct hag_model {
    llama_model *                  model = nullptr;
    std::string                    path;
    bool                           mmap = false;
    int64_t                        file_size = 0;
    std::string                    arch;
    std::string                    info_json;
    std::string                    base_sha256;       // lazily computed
    bool                           patched = false;
    std::string                    patch_path;
    std::string                    patch_file_sha256;
    std::string                    patch_json;
    std::vector<ggml_backend_buffer_t> patch_bufs;    // owned replacement tensor storage
    llama_adapter_lora *           adapter = nullptr; // LoRA patch (owned)
    std::string                    patch_kind;
    std::atomic<int>               n_sessions{0};
};

namespace hag {

// Parsed + structurally validated patch header (tensor payloads are NOT read).
struct PatchHeader {
    gguf_context *       gguf = nullptr;
    std::string          path;
    int64_t              file_size = 0;
    uint32_t             version = 0;
    std::string          kind = "replace";
    float                lora_alpha = 0.f;
    std::string          base_arch, base_sha256, payload_sha256, fingerprint, engine;
    uint64_t             base_size = 0;
    uint32_t             base_n_tensors = 0;
    size_t               data_offset = 0;
    ~PatchHeader() { if (gguf) gguf_free(gguf); }
};

// Returns "" on success or an error message. Never throws.
std::string patch_read_header(const char * path, PatchHeader & out);
std::string patch_info_json(const PatchHeader & h, const std::string & file_sha256);

std::string engine_version_string();

// Quantize/convert F32 rows to `type`. Returns bytes written or 0 if unsupported (needs imatrix, etc.).
size_t convert_rows_from_f32(ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row);
bool   type_convertible_from_f32(ggml_type type);

}  // namespace hag
