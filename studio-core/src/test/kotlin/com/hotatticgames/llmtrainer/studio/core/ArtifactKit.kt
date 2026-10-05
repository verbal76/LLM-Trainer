package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.qualify.SafetyPolicy
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Test fixtures for the artifact registry: synthetic CI-refreshed artifact JSON, a StudioCore wired with it, and synthetic GGUF files. */
object ArtifactKit {
    const val REV = "0123456789abcdef0123456789abcdef01234567"
    val APACHE_SHA: String get() = ArtifactRegistry.parseCanonical(EmbeddedArtifacts.canonicalLicenses).getValue("Apache-2.0").sha256

    /** A REFRESHED artifact as the CI job would write it (synthetic values; the registry's real ones await the first CI refresh). */
    fun refreshed(
        id: String, baseRepo: String, repo: String, file: String, size: Long, sha: String, quant: String = "Q8_0", precision: String = "quantized",
        licenseVerified: Boolean = true, canonicalSha: String = APACHE_SHA, textSha: String = APACHE_SHA, match: String = "exact", cardAgrees: Boolean = true,
        spdx: String = "Apache-2.0", nominalB: Double = 1.7, training: String = "inference_only", trainingSource: String? = null, optional: Boolean = false,
        layers: Int? = 28, params: Long? = 1_720_000_000L, revision: String = REV, extra: Map<String, Any?> = emptyMap(),
    ): String {
        val m = linkedMapOf<String, Any?>(
            "schema_version" to 1, "artifact_id" to id, "refresh_state" to "refreshed", "refreshed_at" to "2026-10-06T00:00:00Z", "optional" to optional, "published" to true,
            "family" to "Test", "base_repo" to baseRepo,
            "source" to linkedMapOf("repo" to repo, "repo_url" to "https://huggingface.co/$repo", "file" to file, "file_glob" to "*", "revision" to revision,
                "download_url" to "https://huggingface.co/$repo/resolve/$revision/$file"),
            "format" to "gguf", "quantization" to quant, "precision" to precision, "size_bytes" to size, "sha256" to "sha256:$sha",
            "parameter_count_nominal_b" to nominalB, "parameter_count" to params,
            "architecture" to linkedMapOf("name" to "qwen3", "layers" to layers, "kv_heads" to 8, "heads" to 16, "head_dim" to 128, "embd" to 2048, "ffn" to 6144, "ctx_train" to 40960, "vocab" to 151936),
            "chat_template_present" to true, "chat_template_sha256" to "ab".repeat(32),
            "runtime_compat" to linkedMapOf("llama.cpp" to linkedMapOf("architecture" to "qwen3", "supported" to "yes", "min_build" to "b5401"),
                "hag-engine" to linkedMapOf("architecture" to "qwen3", "supported" to "unverified", "min_build" to null)),
            "training" to linkedMapOf("class" to training, "reasons" to listOf("test"), "training_source_artifact_id" to trainingSource, "model_class_if_full_precision" to training),
            "license" to linkedMapOf("state" to if (licenseVerified) "VERIFIED" else "UNVERIFIED", "expected_spdx" to "Apache-2.0", "spdx_id" to if (licenseVerified) spdx else null,
                "reasons" to emptyList<String>(), "evidence" to linkedMapOf(
                    "expected_spdx" to "Apache-2.0", "license_text_url" to "https://huggingface.co/$repo/resolve/$revision/LICENSE", "license_text_sha256" to "sha256:$textSha",
                    "license_text_bytes" to 11358, "canonical_spdx" to spdx, "canonical_sha256" to "sha256:$canonicalSha", "canonical_source_url" to "https://www.apache.org/licenses/LICENSE-2.0.txt",
                    "match" to match, "model_card_licenses" to linkedMapOf(repo to "apache-2.0"), "model_card_agrees" to cardAgrees, "fetched_at" to "2026-10-06T00:00:00Z")),
            "notes" to emptyList<String>(),
        )
        m.putAll(extra)
        return J.dump(m)
    }

    fun open(rig: TK.Rig, artifacts: List<Pair<String, String>>): StudioCore =
        StudioCore(rig.dir, { rig.snap() }, null, rig.http, rig.clock, rig.storage, rig.runner, rig.ids, "test", SafetyPolicy(), EmbeddedRegistry.files, artifacts, EmbeddedArtifacts.canonicalLicenses)

    /** A tiny but structurally valid GGUF v3 file (header + metadata + tensor infos + padding) for the header reader and the import path. */
    fun gguf(arch: String = "llama", layers: Int = 4, heads: Int = 8, kvHeads: Int = 2, embd: Int = 64, vocab: Int = 50, template: Boolean = true, pad: Int = 2048): ByteArray {
        val out = ByteArrayOutputStream()
        fun le(n: Int, v: Long) { for (i in 0 until n) out.write(((v shr (8 * i)) and 0xff).toInt()) }
        fun str(s: String) { val b = s.toByteArray(); le(8, b.size.toLong()); out.write(b) }
        fun kvU32(k: String, v: Int) { str(k); le(4, 4); le(4, v.toLong()) }
        fun kvStr(k: String, v: String) { str(k); le(4, 8); str(v) }
        out.write("GGUF".toByteArray()); le(4, 3)
        val kvCount = 8 + (if (template) 1 else 0) + 1
        le(8, 2); le(8, kvCount.toLong())
        kvStr("general.architecture", arch); kvStr("general.size_label", "tiny")
        kvU32("$arch.block_count", layers); kvU32("$arch.attention.head_count", heads); kvU32("$arch.attention.head_count_kv", kvHeads)
        kvU32("$arch.embedding_length", embd); kvU32("$arch.context_length", 4096); kvU32("$arch.feed_forward_length", 128)
        str("tokenizer.ggml.tokens"); le(4, 9); le(4, 8); le(8, vocab.toLong()); for (i in 0 until vocab) str("t$i")
        if (template) kvStr("tokenizer.chat_template", "{{ messages }}")
        // two tensors: [embd, vocab] and [embd, embd]
        for ((name, dims) in listOf("tok_embd" to listOf(embd.toLong(), vocab.toLong()), "w0" to listOf(embd.toLong(), embd.toLong()))) {
            str(name); le(4, dims.size.toLong()); dims.forEach { le(8, it) }; le(4, 0); le(8, 0)
        }
        out.write(ByteArray(pad))
        return out.toByteArray()
    }

    /** Streams [size] bytes of deterministic pattern into [part] without ever holding them in memory (sparse file; the pattern is zeros). */
    class SparseHttp(val url: String, val size: Long) : Http {
        var downloads = 0
        var maxHeapSeen = 0L
        override fun getBytes(url: String, maxBytes: Long, timeoutMs: Int): HttpBytes = throw HttpException(HttpException.Kind.STATUS, "404", 404)
        override fun downloadToFile(url: String, part: File, maxBytes: Long, cancelled: () -> Boolean, onProgress: (Long, Long) -> Unit, timeoutMs: Int): DownloadOutcome {
            downloads++
            java.io.RandomAccessFile(part, "rw").use { it.setLength(size) }
            onProgress(size, size)
            return DownloadOutcome(size, size)
        }
    }
}
