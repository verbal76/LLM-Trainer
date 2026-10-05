package com.hotatticgames.llmtrainer.studio.core

import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

/** Hashing helpers. API objects use bare lowercase hex; package files use the `sha256:` prefixed rendering. */
object Hashing {
    const val PREFIX = "sha256:"
    private val HEX = "0123456789abcdef".toCharArray()

    fun hex(b: ByteArray): String {
        val c = CharArray(b.size * 2)
        for (i in b.indices) {
            val v = b[i].toInt() and 0xFF
            c[i * 2] = HEX[v ushr 4]
            c[i * 2 + 1] = HEX[v and 15]
        }
        return String(c)
    }

    fun sha256(b: ByteArray): String = hex(MessageDigest.getInstance("SHA-256").digest(b))
    fun sha256(s: String): String = sha256(s.toByteArray(Charsets.UTF_8))
    fun prefixed(hex: String) = if (hex.startsWith(PREFIX)) hex else PREFIX + hex
    fun bare(h: String) = h.removePrefix(PREFIX).lowercase()
    fun short(h: String, n: Int = 12) = bare(h).take(n)

    fun sha256File(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { ins ->
            val buf = ByteArray(1 shl 16)
            while (true) {
                val n = ins.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return hex(md.digest())
    }
}

/** ISO-8601 UTC timestamps/dates at whole-second precision (java.time exists from API 26). */
object Iso {
    fun ts(ms: Long): String = java.time.Instant.ofEpochSecond(Math.floorDiv(ms, 1000L)).toString()
    fun date(ms: Long): String = ts(ms).substring(0, 10)
    fun parseMs(s: String?): Long? {
        if (s.isNullOrBlank()) return null
        return try {
            if (s.length == 10) java.time.LocalDate.parse(s).atStartOfDay(java.time.ZoneOffset.UTC).toInstant().toEpochMilli()
            else java.time.OffsetDateTime.parse(s).toInstant().toEpochMilli()
        } catch (e: Exception) { null }
    }
}

/**
 * Deterministic JSON writer for plain Kotlin values (Map / List / String / Number / Boolean / null). Insertion order is
 * kept unless sortKeys. We do not write with org.json because its key order is unspecified across JVM/Android.
 */
object J {
    fun dump(v: Any?, sortKeys: Boolean = false): String = StringBuilder().also { write(it, v, sortKeys) }.toString()
    fun bytes(v: Any?, sortKeys: Boolean = false): ByteArray = dump(v, sortKeys).toByteArray(Charsets.UTF_8)
    fun canonical(v: Any?): String = dump(v, true)

    @Suppress("UNCHECKED_CAST")
    private fun write(sb: StringBuilder, v: Any?, sort: Boolean) {
        when (v) {
            null, JSONObject.NULL -> sb.append("null")
            is String -> quote(sb, v)
            is Boolean -> sb.append(v)
            is Int, is Long, is Short, is Byte -> sb.append(v.toString())
            is Double -> { require(v.isFinite()) { "non-finite number" }; sb.append(num(v)) }
            is Float -> { require(v.isFinite()) { "non-finite number" }; sb.append(num(v.toDouble())) }
            is Enum<*> -> quote(sb, v.name)
            is Map<*, *> -> {
                sb.append('{')
                val keys = v.keys.map { it.toString() }.let { if (sort) it.sorted() else it }
                var first = true
                for (k in keys) {
                    if (!first) sb.append(',')
                    first = false
                    quote(sb, k)
                    sb.append(':')
                    write(sb, (v as Map<String, Any?>)[k], sort)
                }
                sb.append('}')
            }
            is Iterable<*> -> { sb.append('['); var f = true; for (e in v) { if (!f) sb.append(','); f = false; write(sb, e, sort) }; sb.append(']') }
            is Array<*> -> write(sb, v.toList(), sort)
            is JSONObject -> write(sb, v.keySet().associateWith { v.get(it) }, sort)
            is JSONArray -> write(sb, (0 until v.length()).map { v.get(it) }, sort)
            else -> quote(sb, v.toString())
        }
    }

    private fun num(d: Double): String = if (d == Math.rint(d) && Math.abs(d) < 1e15) d.toLong().toString() + ".0" else d.toString()

    private fun quote(sb: StringBuilder, s: String) {
        sb.append('"')
        for (c in s) when {
            c == '"' -> sb.append("\\\"")
            c == '\\' -> sb.append("\\\\")
            c == '\n' -> sb.append("\\n")
            c == '\r' -> sb.append("\\r")
            c == '\t' -> sb.append("\\t")
            c == '\b' -> sb.append("\\b")
            c == '\u000c' -> sb.append("\\f")
            c.code < 0x20 || c == ' ' || c == ' ' || c.code == 0x7f -> sb.append("\\u").append("%04x".format(c.code))
            else -> sb.append(c)
        }
        sb.append('"')
    }

    fun parseOrNull(s: String): JSONObject? = try { JSONObject(s) } catch (e: Exception) { null }
}

fun JSONObject.str(k: String): String? = if (!has(k) || isNull(k)) null else opt(k) as? String
fun JSONObject.lng(k: String): Long? = if (!has(k) || isNull(k)) null else (opt(k) as? Number)?.toLong()
fun JSONObject.int(k: String): Int? = lng(k)?.toInt()
fun JSONObject.dbl(k: String): Double? = if (!has(k) || isNull(k)) null else (opt(k) as? Number)?.toDouble()
fun JSONObject.bool(k: String): Boolean? = if (!has(k) || isNull(k)) null else opt(k) as? Boolean
fun JSONObject.obj(k: String): JSONObject? = if (!has(k) || isNull(k)) null else opt(k) as? JSONObject
fun JSONObject.arr(k: String): JSONArray? = if (!has(k) || isNull(k)) null else opt(k) as? JSONArray
fun JSONArray.objs(): List<JSONObject> = (0 until length()).mapNotNull { optJSONObject(it) }
fun JSONArray.strs(): List<String> = (0 until length()).mapNotNull { if (isNull(it)) null else opt(it) as? String }
fun JSONObject.strList(k: String): List<String> = arr(k)?.strs().orEmpty()
fun JSONObject.objList(k: String): List<JSONObject> = arr(k)?.objs().orEmpty()

/** File helpers: atomic writes (tmp + fsync + rename), a one-generation backup for JSON, tolerant reads. */
object Fs {
    fun writeAtomic(target: File, keepBackup: Boolean = false, writer: (OutputStream) -> Unit) {
        val dir = target.absoluteFile.parentFile ?: throw java.io.IOException("no parent for $target")
        if (!dir.isDirectory && !dir.mkdirs() && !dir.isDirectory) throw java.io.IOException("cannot create $dir")
        val tmp = File(dir, ".${target.name}.tmp-${System.nanoTime()}")
        try {
            FileOutputStream(tmp).use { fos ->
                val bos = java.io.BufferedOutputStream(fos, 1 shl 16)
                writer(bos)
                bos.flush()
                try { fos.fd.sync() } catch (_: Exception) { }
            }
            if (keepBackup && target.isFile) {
                try { Files.copy(target.toPath(), File(dir, target.name + ".bak").toPath(), StandardCopyOption.REPLACE_EXISTING) } catch (_: Exception) { }
            }
            try {
                Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
            } catch (e: java.nio.file.AtomicMoveNotSupportedException) {
                Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
        } finally {
            if (tmp.exists()) tmp.delete()
        }
    }

    fun writeBytes(target: File, bytes: ByteArray, keepBackup: Boolean = false) = writeAtomic(target, keepBackup) { it.write(bytes) }
    fun writeJson(target: File, value: Any?) = writeAtomic(target, keepBackup = true) { it.write(J.bytes(value)) }

    /** Reads JSON; falls back to the `.bak` generation; a corrupt main file is quarantined (kept as `.corrupt-<n>`). */
    fun readJson(target: File, onCorrupt: (String) -> Unit = {}): JSONObject? {
        fun tryRead(f: File): JSONObject? = if (!f.isFile) null else try { JSONObject(f.readText(Charsets.UTF_8)) } catch (e: Exception) { null }
        val main = tryRead(target)
        if (main != null) return main
        if (target.isFile) {
            val q = File(target.parentFile, target.name + ".corrupt-" + System.currentTimeMillis())
            try { target.renameTo(q) } catch (_: Exception) { }
            onCorrupt("${target.name} was unreadable and was set aside as ${q.name}")
        }
        val bak = tryRead(File(target.parentFile, target.name + ".bak"))
        if (bak != null) onCorrupt("${target.name}: restored from the previous generation")
        return bak
    }

    fun readLines(f: File): List<String> = if (f.isFile) f.readLines(Charsets.UTF_8).filter { it.isNotEmpty() } else emptyList()

    fun readLimited(ins: InputStream, limit: Long): ByteArray? {
        val out = ByteArrayOutputStream()
        val buf = ByteArray(1 shl 16)
        var total = 0L
        while (true) {
            val n = ins.read(buf)
            if (n < 0) break
            total += n
            if (total > limit) return null
            out.write(buf, 0, n)
        }
        return out.toByteArray()
    }

    fun safeName(s: String, max: Int = 80): String {
        val base = s.substringAfterLast('/').substringAfterLast('\\')
        val clean = base.replace(Regex("[^A-Za-z0-9._-]+"), "_").trim('_', '.').ifEmpty { "file" }
        return if (clean.length <= max) clean else clean.take(max)
    }
}

object Text {
    fun slugify(s: String): String = s.lowercase().replace(Regex("[^a-z0-9]+"), "-").trim('-').ifEmpty { "untitled" }
}
