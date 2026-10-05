package com.hotatticgames.llmtrainer.studio.core

import java.io.BufferedInputStream
import java.io.EOFException
import java.io.File
import java.io.IOException
import java.io.InputStream
import org.json.JSONObject

/** Provenance label of an installed model file. Only [VERIFIED_DOWNLOAD] means "bytes match a catalog hash fetched by CI". */
object InstallProvenance {
    const val VERIFIED_DOWNLOAD = "verified_download"
    const val DOWNLOAD_UNVERIFIED_CHECKSUM = "download_unverified_checksum"
    const val VERIFIED_IMPORT = "verified_import"
    const val UNVERIFIED = "unverified_provenance"
}

/** One installed model file as recorded in its install manifest (`acquired.json`, schema 2). */
data class InstallRec(
    val variantId: String, val dir: File, val fileName: String, val size: Long, val sha256: String, val source: String, val url: String?,
    val revision: String?, val artifactId: String?, val repo: String?, val originalName: String?, val installedAt: Long,
    val checksumVerifiedAgainstRegistry: Boolean, val provenance: String, val licenseState: String, val licenseId: String?,
    val licenseEvidenceLevel: String?, val licenseTextSha256: String?, val architecture: String?, val layers: Int?, val parameterCount: Long?,
    val chatTemplatePresent: Boolean?,
) {
    val file: File get() = File(dir, fileName)
}

/**
 * Where downloaded/imported model files live (`workspace/models/<safe variant id>-<hash>/`) and the per-artifact install
 * manifest. A model is "installed" only if its manifest exists AND the final file is present with the recorded size; a
 * `.part` file is NEVER installed and never returned by [installedFile] (partials cannot be loaded).
 */
class ModelFiles(val dir: File) {
    fun dirOf(variantId: String) = File(dir, Fs.safeName(variantId.replace('#', '_').replace('@', '_'), 100) + "-" + Hashing.short(Hashing.sha256(variantId), 8))
    fun acquiredRecord(variantId: String): File = File(dirOf(variantId), "acquired.json")

    private fun parse(d: File, o: JSONObject): InstallRec? {
        val file = o.str("file") ?: return null
        val sha = o.str("sha256") ?: return null
        val checksum = o.bool("checksum_verified_against_registry") ?: false
        val source = o.str("source") ?: "download"
        return InstallRec(
            o.str("variant_id") ?: d.name, d, file, o.lng("size") ?: -1L, Hashing.bare(sha), source, o.str("url"), o.str("revision"), o.str("artifact_id"), o.str("repo"),
            o.str("original_name"), o.lng("at") ?: 0L, checksum,
            o.str("provenance") ?: if (source == "download" && checksum) InstallProvenance.VERIFIED_DOWNLOAD else if (source == "download") InstallProvenance.DOWNLOAD_UNVERIFIED_CHECKSUM
                else if (checksum) InstallProvenance.VERIFIED_IMPORT else InstallProvenance.UNVERIFIED,
            o.str("license_state") ?: "UNVERIFIED", o.str("license_id"), o.str("license_evidence_level"), o.str("license_text_sha256"),
            o.str("architecture"), o.int("layers"), o.lng("parameter_count"), o.bool("chat_template_present"),
        )
    }

    private fun readRec(d: File): InstallRec? {
        val o = Fs.readJson(File(d, "acquired.json")) ?: return null
        return parse(d, o)
    }

    /** The install manifest of [variantId] iff the final file exists with exactly the recorded size (a truncated/replaced file is not installed). */
    fun record(variantId: String): InstallRec? {
        val r = readRec(dirOf(variantId)) ?: return null
        val f = r.file
        if (r.fileName.endsWith(".part") || !f.isFile || (r.size >= 0 && f.length() != r.size)) return null
        return r
    }

    /** The loadable model file, or null. Never a partial download. */
    fun installedFile(variantId: String): File? = record(variantId)?.file
    fun isAcquired(variantId: String): Boolean = record(variantId) != null

    /** Every valid install (scans the model directories, so custom imports are included). */
    fun allInstalled(): List<InstallRec> {
        val dirs = dir.listFiles { f -> f.isDirectory } ?: return emptyList()
        return dirs.sortedBy { it.name }.mapNotNull { d ->
            val r = readRec(d) ?: return@mapNotNull null
            if (r.fileName.endsWith(".part") || !r.file.isFile || (r.size >= 0 && r.file.length() != r.size)) null else r
        }
    }

    fun partialFiles(): List<File> = (dir.listFiles { f -> f.isDirectory } ?: emptyArray()).flatMap { d ->
        (d.listFiles { f -> f.isFile && f.name.endsWith(".part") } ?: emptyArray()).toList()
    }

    fun installedBytes(): Long = allInstalled().sumOf { it.file.length() }
    fun partialBytes(): Long = partialFiles().sumOf { it.length() }

    /** Removes the model file, any partial file and the manifest. Returns the bytes freed. */
    fun uninstall(variantId: String): Long {
        val d = dirOf(variantId)
        var freed = 0L
        val modelName = readRec(d)?.fileName          // read BEFORE the manifest is deleted
        for (f in d.listFiles().orEmpty()) {
            if (f.name == "acquired.json" || f.name == "acquired.json.bak" || f.name.endsWith(".part") || f.name == modelName) {
                if (f.isFile) { freed += f.length(); f.delete() }
            }
        }
        d.delete()   // only succeeds when empty
        return freed
    }

    /** Deletes empty/unrecorded debris directories that have no files left (housekeeping after crashes). */
    fun sweepEmptyDirs() { for (d in dir.listFiles { f -> f.isDirectory }.orEmpty()) if (d.listFiles().isNullOrEmpty()) d.delete() }
}

/**
 * Minimal streaming GGUF header reader (mirrors `read_gguf_header` in scripts/catalog/refresh_catalog.py). Used to
 * describe a user-imported file; never throws (returns null for anything that is not a readable GGUF header).
 */
object GgufHeader {
    class Info(
        val version: Int, val architecture: String, val layers: Int?, val heads: Int?, val kvHeads: Int?, val headDim: Int?, val embd: Int?, val ffn: Int?,
        val ctxTrain: Int?, val vocab: Long?, val tensorCount: Long, val parameterCount: Long, val chatTemplatePresent: Boolean, val sizeLabel: String?,
        val fileType: Int?,
    )

    private const val CAP = 192L * 1024 * 1024

    fun hasMagic(f: File): Boolean = try {
        f.inputStream().use { val b = ByteArray(4); it.read(b) == 4 && b[0] == 'G'.code.toByte() && b[1] == 'G'.code.toByte() && b[2] == 'U'.code.toByte() && b[3] == 'F'.code.toByte() }
    } catch (e: IOException) { false }

    private class Rd(val ins: InputStream) {
        var pos = 0L
        fun fill(n: Int): ByteArray {
            val b = ByteArray(n)
            var o = 0
            while (o < n) { val r = ins.read(b, o, n - o); if (r < 0) throw EOFException(); o += r }
            pos += n
            if (pos > CAP) throw IOException("header too large")
            return b
        }
        fun u32(): Long { val b = fill(4); return (b[0].toLong() and 255) or ((b[1].toLong() and 255) shl 8) or ((b[2].toLong() and 255) shl 16) or ((b[3].toLong() and 255) shl 24) }
        fun u64(): Long { val lo = u32(); val hi = u32(); return lo or (hi shl 32) }
        fun skip(n: Long) {
            var left = n
            while (left > 0) { val s = ins.skip(left); if (s <= 0) { if (ins.read() < 0) throw EOFException(); pos++; left-- } else { pos += s; left -= s } }
            if (pos > CAP) throw IOException("header too large")
        }
        fun string(maxKeep: Int): String? {
            val n = u64()
            if (n < 0 || n > 64L * 1024 * 1024) throw IOException("bad string")
            if (n <= maxKeep) return String(fill(n.toInt()), Charsets.UTF_8)
            skip(n); return null
        }
    }

    private val SIZES = mapOf(0L to 1, 1L to 1, 2L to 2, 3L to 2, 4L to 4, 5L to 4, 6L to 4, 7L to 1, 10L to 8, 11L to 8, 12L to 8)

    fun read(f: File): Info? = try {
        BufferedInputStream(f.inputStream(), 1 shl 16).use { parse(Rd(it)) }
    } catch (e: IOException) { null } catch (e: RuntimeException) { null }

    private fun parse(rd: Rd): Info? {
        val magic = rd.fill(4)
        if (String(magic, Charsets.US_ASCII) != "GGUF") return null
        val version = rd.u32().toInt()
        if (version != 2 && version != 3) return null
        val tensors = rd.u64()
        val kvs = rd.u64()
        val nums = HashMap<String, Long>()
        val strs = HashMap<String, String>()
        var hasTemplate = false
        var vocab: Long? = null
        for (i in 0 until kvs) {
            val key = rd.string(256) ?: return null
            val t = rd.u32()
            when {
                SIZES.containsKey(t) -> {
                    val b = rd.fill(SIZES.getValue(t))
                    if (t in setOf(0L, 2L, 4L, 10L, 1L, 3L, 5L, 11L)) {
                        var v = 0L
                        for (k in b.indices.reversed()) v = (v shl 8) or (b[k].toLong() and 255)
                        nums[key] = when (t) { 1L -> b[0].toLong(); 3L -> v.toShort().toLong(); 5L -> v.toInt().toLong(); else -> v }
                    }
                }
                t == 8L -> {
                    if (key == "tokenizer.chat_template") { hasTemplate = true; rd.string(0) }
                    else rd.string(1024)?.let { strs[key] = it }
                }
                t == 9L -> {
                    val et = rd.u32()
                    val n = rd.u64()
                    if (SIZES.containsKey(et)) rd.skip(n * SIZES.getValue(et))
                    else if (et == 8L) for (j in 0 until n) rd.skip(rd.u64())
                    else return null
                    if (key == "tokenizer.ggml.tokens") vocab = n
                }
                else -> return null
            }
        }
        var params = 0L
        for (i in 0 until tensors) {
            rd.skip(rd.u64())
            val nd = rd.u32()
            var n = 1L
            for (d in 0 until nd) n *= rd.u64()
            rd.fill(4 + 8)
            params += n
        }
        val arch = strs["general.architecture"] ?: return null
        fun num(s: String) = nums["$arch.$s"]?.toInt()
        val heads = num("attention.head_count")
        val embd = num("embedding_length")
        val headDim = num("attention.key_length") ?: if (heads != null && heads > 0 && embd != null) embd / heads else null
        return Info(version, arch, num("block_count"), heads, num("attention.head_count_kv") ?: heads, headDim, embd, num("feed_forward_length"),
            num("context_length"), vocab, tensors, params, hasTemplate, strs["general.size_label"], nums["general.file_type"]?.toInt())
    }
}
