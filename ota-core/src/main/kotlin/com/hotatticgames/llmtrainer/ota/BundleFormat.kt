package com.hotatticgames.llmtrainer.ota

import java.io.ByteArrayOutputStream
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream

/**
 * A .hagb bundle is a zip containing exactly:
 *   manifest.json   - [BundleManifest]
 *   manifest.sig    - base64 DER ECDSA signature over the raw bytes of manifest.json
 *   classes.dex, classes2.dex, ...   - the Kotlin/Java code
 *   assets/...      - optional non-code payload
 * Every file other than manifest.json/.sig must be listed in manifest.files with sha256+size.
 * Native code (.so, lib/...) is never allowed.
 */
object BundleFormat {
    const val MANIFEST = "manifest.json"
    const val SIGNATURE = "manifest.sig"
    const val MAX_ENTRIES = 512
    const val MAX_TOTAL_BYTES = 128L * 1024 * 1024
    private val DEX = Regex("""classes(\d*)\.dex""")

    fun isAllowedPath(p: String): Boolean = DEX.matches(p) || (p.startsWith("assets/") && p.length > 7)

    /** Returns null when [p] is safe, else the reason. */
    fun pathProblem(p: String): Reject? {
        if (p.isEmpty() || p.startsWith("/") || p.contains('\\') || p.contains("\u0000") ||
            p.split('/').any { it == ".." || it == "." || it.isEmpty() }
        ) return Reject(RejectCode.UNSAFE_PATH, p)
        val lower = p.lowercase()
        if (lower.endsWith(".so") || lower.startsWith("lib/") || lower.endsWith(".jar") || lower.endsWith(".apk")) {
            return Reject(RejectCode.FORBIDDEN_FILE, "native/executable payload not allowed in bundle: $p")
        }
        if (!isAllowedPath(p)) return Reject(RejectCode.FORBIDDEN_FILE, p)
        return null
    }

    /** Build a signed bundle. Used by the CI tool and by tests. */
    fun pack(
        files: Map<String, ByteArray>,
        template: BundleManifest,
        privateKeyDerBase64: String,
    ): ByteArray {
        val entries = files.toSortedMap().map { (p, b) -> FileEntry(p, Hashing.sha256Hex(b), b.size.toLong()) }
        val manifest = template.copy(files = entries)
        val manifestBytes = OtaJson.encodeToString(BundleManifest.serializer(), manifest).toByteArray(Charsets.UTF_8)
        val sig = Signing.sign(manifestBytes, privateKeyDerBase64)
        return zip(
            buildMap {
                put(MANIFEST, manifestBytes)
                put(SIGNATURE, sig.toByteArray(Charsets.UTF_8))
                putAll(files)
            },
        )
    }

    /** Test/tool helper: zip arbitrary entries (also used to build deliberately hostile archives). */
    fun zip(entries: Map<String, ByteArray>): ByteArray {
        val bos = ByteArrayOutputStream()
        ZipOutputStream(bos).use { z ->
            for ((name, data) in entries) {
                val e = ZipEntry(name)
                e.time = 0L // reproducible
                z.putNextEntry(e)
                z.write(data)
                z.closeEntry()
            }
        }
        return bos.toByteArray()
    }

    class Verified(val manifest: BundleManifest, val entries: Map<String, ByteArray>)

    sealed interface Result {
        data class Ok(val bundle: Verified) : Result
        data class Rejected(val reasons: List<Reject>) : Result
    }

    /**
     * Fully verify a bundle against trusted keys and host compatibility.
     * Nothing is trusted (or written anywhere) until this returns Ok.
     */
    fun verify(zipFile: File, trusted: List<TrustedKey>, host: HostInfo): Result {
        val rejects = mutableListOf<Reject>()
        fun fail(r: Reject): Result.Rejected = Result.Rejected(rejects + r)

        val entries = LinkedHashMap<String, ByteArray>()
        try {
            ZipFile(zipFile).use { z ->
                if (z.size() > MAX_ENTRIES) return fail(Reject(RejectCode.TOO_LARGE, "too many entries"))
                var total = 0L
                val en = z.entries()
                while (en.hasMoreElements()) {
                    val e = en.nextElement()
                    if (e.isDirectory) continue
                    if (entries.containsKey(e.name)) return fail(Reject(RejectCode.BAD_ARCHIVE, "duplicate entry ${e.name}"))
                    if (e.size < 0 || e.size > MAX_TOTAL_BYTES) return fail(Reject(RejectCode.TOO_LARGE, e.name))
                    total += e.size
                    if (total > MAX_TOTAL_BYTES) return fail(Reject(RejectCode.TOO_LARGE, "total"))
                    // Bounded read: do not trust the declared size alone (zip-bomb guard).
                    val buf = ByteArrayOutputStream()
                    z.getInputStream(e).use { ins ->
                        val tmp = ByteArray(32 * 1024)
                        var read = 0L
                        while (true) {
                            val n = ins.read(tmp)
                            if (n < 0) break
                            read += n
                            if (read > e.size || read > MAX_TOTAL_BYTES) return fail(Reject(RejectCode.TOO_LARGE, e.name))
                            buf.write(tmp, 0, n)
                        }
                    }
                    entries[e.name] = buf.toByteArray()
                }
            }
        } catch (ex: Exception) {
            return fail(Reject(RejectCode.BAD_ARCHIVE, ex.message ?: ex.javaClass.simpleName))
        }

        val manifestBytes = entries[MANIFEST] ?: return fail(Reject(RejectCode.BAD_MANIFEST, "missing $MANIFEST"))
        val sigBytes = entries[SIGNATURE] ?: return fail(Reject(RejectCode.BAD_SIGNATURE, "missing $SIGNATURE"))

        val manifest = try {
            OtaJson.decodeFromString(BundleManifest.serializer(), manifestBytes.toString(Charsets.UTF_8))
        } catch (ex: Exception) {
            return fail(Reject(RejectCode.BAD_MANIFEST, ex.message ?: "unparseable"))
        }

        // Signature first: nothing in the manifest is believed until it verifies.
        val key = trusted.firstOrNull { it.keyId == manifest.keyId }
            ?: return fail(Reject(RejectCode.UNKNOWN_KEY, manifest.keyId))
        if (!Signing.verify(manifestBytes, sigBytes.toString(Charsets.UTF_8), key)) {
            return fail(Reject(RejectCode.BAD_SIGNATURE))
        }

        if (manifest.schema != BUNDLE_SCHEMA) return fail(Reject(RejectCode.BAD_SCHEMA, "schema ${manifest.schema}"))
        if (manifest.bundleId != host.bundleId) {
            return fail(Reject(RejectCode.WRONG_BUNDLE_ID, "${manifest.bundleId} != ${host.bundleId}"))
        }
        if (manifest.channel != host.channel) {
            return fail(Reject(RejectCode.WRONG_CHANNEL, "${manifest.channel} != ${host.channel}"))
        }

        // Payload files: exact set, hashes, sizes, safe names, no native code.
        val listed = manifest.files.associateBy { it.path }
        if (listed.size != manifest.files.size) rejects += Reject(RejectCode.BAD_MANIFEST, "duplicate file entries")
        for (f in manifest.files) pathProblem(f.path)?.let { rejects += it }
        for (name in entries.keys) {
            if (name == MANIFEST || name == SIGNATURE) continue
            pathProblem(name)?.let { rejects += it }
            if (name !in listed) rejects += Reject(RejectCode.UNLISTED_FILE, name)
        }
        for (f in manifest.files) {
            val data = entries[f.path]
            if (data == null) {
                rejects += Reject(RejectCode.MISSING_FILE, f.path)
            } else if (data.size.toLong() != f.size || Hashing.sha256Hex(data) != f.sha256) {
                rejects += Reject(RejectCode.HASH_MISMATCH, f.path)
            }
        }
        if (manifest.files.none { DEX.matches(it.path) }) rejects += Reject(RejectCode.NO_DEX)
        if (rejects.isNotEmpty()) return Result.Rejected(rejects)

        val compat = Compatibility.evaluate(manifest.requires, host)
        if (compat.isNotEmpty()) return Result.Rejected(compat)

        return Result.Ok(Verified(manifest, entries.filterKeys { it != MANIFEST && it != SIGNATURE }))
    }
}
