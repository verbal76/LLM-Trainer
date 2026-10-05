package com.hotatticgames.llmtrainer.studio.core

import java.io.ByteArrayOutputStream
import java.io.FilterInputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.security.MessageDigest
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

class PackageException(val code: String, message: String) : Exception(message)

/** Zip hygiene limits from PACKAGE_FORMATS section 0 (the desktop applies the same ones). */
object ZipLimits {
    const val MAX_MEMBERS = 10_000
    const val MAX_MEMBER_BYTES = 128L * 1024 * 1024
    const val MAX_TOTAL_BYTES = 256L * 1024 * 1024
    const val MAX_RATIO = 200
}

object Zips {
    private val DRIVE = Regex("^[A-Za-z]:")

    fun validName(n: String): String? = when {
        n.isEmpty() -> "empty member name"
        n.startsWith("/") -> "absolute path: $n"
        n.contains('\\') -> "backslash in member name: $n"
        n.contains('\u0000') -> "NUL in member name"
        DRIVE.containsMatchIn(n) -> "drive letter in member name: $n"
        n.split('/').any { it == ".." || it == "." || it.isEmpty() } -> "unsafe path segment in: $n"
        else -> null
    }

    private class Counting(i: InputStream) : FilterInputStream(i) {
        var count = 0L
        override fun read(): Int = super.read().also { if (it >= 0) count++ }
        override fun read(b: ByteArray, off: Int, len: Int): Int = super.read(b, off, len).also { if (it > 0) count += it }
    }

    /** Reads every member into memory, enforcing hygiene. Throws [PackageException] (code ZIP_*). */
    fun readAll(input: InputStream): LinkedHashMap<String, ByteArray> {
        val out = LinkedHashMap<String, ByteArray>()
        val counting = Counting(input)
        var total = 0L
        try {
            ZipInputStream(counting).use { zin ->
                while (true) {
                    val e = zin.nextEntry ?: break
                    if (e.isDirectory) continue
                    validName(e.name)?.let { throw PackageException("ZIP_UNSAFE_NAME", it) }
                    if (out.containsKey(e.name)) throw PackageException("ZIP_DUPLICATE", "duplicate member: ${e.name}")
                    if (out.size >= ZipLimits.MAX_MEMBERS) throw PackageException("ZIP_TOO_MANY", "more than ${ZipLimits.MAX_MEMBERS} members")
                    val startCount = counting.count
                    val buf = ByteArrayOutputStream()
                    val tmp = ByteArray(1 shl 16)
                    var n = 0L
                    while (true) {
                        val r = zin.read(tmp)
                        if (r < 0) break
                        n += r
                        total += r
                        if (n > ZipLimits.MAX_MEMBER_BYTES) throw PackageException("ZIP_MEMBER_TOO_LARGE", "member ${e.name} exceeds ${ZipLimits.MAX_MEMBER_BYTES} bytes")
                        if (total > ZipLimits.MAX_TOTAL_BYTES) throw PackageException("ZIP_TOO_LARGE", "package exceeds ${ZipLimits.MAX_TOTAL_BYTES} bytes uncompressed")
                        if (n > (1 shl 20) && (counting.count - startCount) > 0 && n / (counting.count - startCount) > ZipLimits.MAX_RATIO)
                            throw PackageException("ZIP_BOMB", "member ${e.name} has an implausible compression ratio")
                        buf.write(tmp, 0, r)
                    }
                    out[e.name] = buf.toByteArray()
                }
            }
        } catch (e: PackageException) { throw e
        } catch (e: IOException) { throw PackageException("ZIP_CORRUPT", "not a readable zip file: ${e.message}")
        } catch (e: IllegalArgumentException) { throw PackageException("ZIP_CORRUPT", "not a readable zip file: ${e.message}") }
        if (out.isEmpty()) throw PackageException("ZIP_EMPTY", "not a zip file, or the zip file has no members")
        return out
    }

    /** `checksums.json` must list every member except itself, and nothing else; every hash must match. Returns problems. */
    fun verifyChecksums(files: Map<String, ByteArray>): List<String> {
        val raw = files["checksums.json"] ?: return listOf("checksums.json is missing")
        val o = J.parseOrNull(String(raw, Charsets.UTF_8)) ?: return listOf("checksums.json is not valid JSON")
        if (o.str("algorithm") != "sha256") return listOf("checksums.json: algorithm must be sha256")
        val listed = o.obj("files") ?: return listOf("checksums.json has no files map")
        val problems = ArrayList<String>()
        val actual = files.keys.filter { it != "checksums.json" }.toSet()
        val claimed = listed.keySet()
        for (m in actual - claimed) problems.add("member not covered by checksums.json: $m")
        for (m in claimed - actual) problems.add("checksums.json lists a missing member: $m")
        for (m in actual intersect claimed) {
            val want = listed.optString(m, "")
            val got = Hashing.prefixed(Hashing.sha256(files.getValue(m)))
            if (want != got) problems.add("hash mismatch for $m")
        }
        return problems
    }
}

/** Collects package members, adds `checksums.json`, and streams a zip to the caller's OutputStream. */
class PackageWriter {
    private val files = LinkedHashMap<String, ByteArray>()
    fun add(name: String, bytes: ByteArray) {
        require(Zips.validName(name) == null) { "bad member name $name" }
        require(name != "checksums.json")
        files[name] = bytes
    }
    fun addJson(name: String, v: Any?, sortKeys: Boolean = false) = add(name, J.bytes(v, sortKeys) + "\n".toByteArray())
    fun addJsonl(name: String, rows: List<Any?>) = add(name, ByteArray(0).let { _ ->
        val b = ByteArrayOutputStream()
        for (r in rows) { b.write(J.bytes(r)); b.write('\n'.code) }
        b.toByteArray()
    })
    fun has(name: String) = files.containsKey(name)
    fun memberNames(): List<String> = (files.keys + "checksums.json").sorted()

    class Written(val sizeBytes: Long, val sha256: String, val files: List<String>, val checksums: Map<String, String>)

    fun write(out: OutputStream): Written {
        val sums = LinkedHashMap<String, String>()
        for (n in files.keys.sorted()) sums[n] = Hashing.prefixed(Hashing.sha256(files.getValue(n)))
        val checks = J.bytes(linkedMapOf("algorithm" to "sha256", "files" to sums), true) + "\n".toByteArray()
        val md = MessageDigest.getInstance("SHA-256")
        var size = 0L
        val counting = object : OutputStream() {
            override fun write(b: Int) { out.write(b); md.update(b.toByte()); size++ }
            override fun write(b: ByteArray, off: Int, len: Int) { out.write(b, off, len); md.update(b, off, len); size += len }
            override fun flush() = out.flush()
        }
        val zos = ZipOutputStream(counting)
        zos.setLevel(6)
        val all = files.toSortedMap().toMutableMap().also { it["checksums.json"] = checks }
        for ((n, bytes) in all.toSortedMap()) {
            val e = ZipEntry(n)
            e.time = 315532800000L    // fixed (1980-01-01): the member contents, not file times, define identity
            zos.putNextEntry(e)
            zos.write(bytes)
            zos.closeEntry()
        }
        zos.finish()
        zos.flush()
        return Written(size, Hashing.hex(md.digest()), all.keys.sorted(), sums)
    }
}
