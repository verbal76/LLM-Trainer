package com.hotatticgames.llmtrainer.studio.core

import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.security.SecureRandom
import java.util.concurrent.Executors

// ===== Platform seams: tiny interfaces so the core stays JVM-testable and nothing hits the OS directly ============

interface Clock { fun nowMs(): Long }
object SystemClock : Clock { override fun nowMs() = System.currentTimeMillis() }

/** Free space of the volume holding [dir]. */
interface StorageProbe { fun freeBytes(dir: File): Long }
object FileStorageProbe : StorageProbe {
    override fun freeBytes(dir: File): Long {
        var d: File? = dir.absoluteFile
        while (d != null && !d.exists()) d = d.parentFile
        return d?.usableSpace ?: 0L
    }
}

/** Source of unique-enough ids. Tests inject a counter for deterministic ids. */
interface IdSource { fun hex(nChars: Int): String }
object RandomIds : IdSource {
    private val rnd = SecureRandom()
    override fun hex(nChars: Int): String {
        val b = ByteArray((nChars + 1) / 2)
        rnd.nextBytes(b)
        return Hashing.hex(b).take(nChars)
    }
}

/** Runs background work (downloads/imports). The default is one daemon thread; tests run tasks inline or manually. */
interface TaskRunner { fun submit(task: Runnable) }
class ThreadTaskRunner : TaskRunner {
    private val ex = Executors.newSingleThreadExecutor { r -> Thread(r, "studio-core-worker").also { it.isDaemon = true } }
    override fun submit(task: Runnable) { ex.execute(task) }
}
object InlineTaskRunner : TaskRunner { override fun submit(task: Runnable) = task.run() }

class HttpException(val kind: Kind, message: String, val status: Int? = null) : IOException(message) {
    enum class Kind { NOT_HTTPS, NETWORK, TIMEOUT, STATUS, TOO_LARGE, CANCELLED, PROTOCOL }
}

class HttpBytes(val finalUrl: String, val status: Int, val bytes: ByteArray, val contentType: String?)

class DownloadOutcome(val totalBytes: Long, val bytesOnDisk: Long)

/**
 * HTTPS-only network seam. Implementations MUST refuse non-HTTPS URLs (including redirects), enforce timeouts and size
 * caps, and never follow a redirect to a non-HTTPS location.
 */
interface Http {
    /** GET a small resource fully into memory. Throws [HttpException]. */
    fun getBytes(url: String, maxBytes: Long, timeoutMs: Int = 20_000): HttpBytes

    /**
     * GET [url] into [part], resuming from [part].length() with a `Range` request when the file already has bytes.
     * If the server ignores the range (200) the file is restarted from zero. [onProgress] gets (bytesOnDisk, total or -1).
     * Throws [HttpException] (kind CANCELLED when [cancelled] turned true; the partial file is left in place).
     */
    fun downloadToFile(
        url: String, part: File, maxBytes: Long, cancelled: () -> Boolean,
        onProgress: (done: Long, total: Long) -> Unit, timeoutMs: Int = 30_000,
    ): DownloadOutcome
}

class JavaHttp(private val userAgent: String = "LLMTrainerStudio/1") : Http {
    private val maxRedirects = 5

    private fun open(url0: String, timeoutMs: Int, range: Long?): Pair<HttpURLConnection, String> {
        var url = url0
        for (hop in 0..maxRedirects) {
            if (!url.startsWith("https://", ignoreCase = true)) throw HttpException(HttpException.Kind.NOT_HTTPS, "Only HTTPS is allowed: $url")
            val c = try { URL(url).openConnection() as HttpURLConnection } catch (e: Exception) { throw HttpException(HttpException.Kind.PROTOCOL, "Bad URL: $url") }
            c.instanceFollowRedirects = false
            c.connectTimeout = timeoutMs
            c.readTimeout = timeoutMs
            c.setRequestProperty("User-Agent", userAgent)
            c.setRequestProperty("Accept-Encoding", "identity")
            if (range != null && range > 0) c.setRequestProperty("Range", "bytes=$range-")
            val code = try { c.responseCode } catch (e: SocketTimeoutException) {
                throw HttpException(HttpException.Kind.TIMEOUT, "Timed out contacting ${hostOf(url)}")
            } catch (e: IOException) { throw HttpException(HttpException.Kind.NETWORK, "Network error contacting ${hostOf(url)}: ${e.message}") }
            if (code in 300..399) {
                val loc = c.getHeaderField("Location")
                c.disconnect()
                if (loc.isNullOrBlank()) throw HttpException(HttpException.Kind.PROTOCOL, "Redirect without Location")
                url = try { URL(URL(url), loc).toString() } catch (e: Exception) { throw HttpException(HttpException.Kind.PROTOCOL, "Bad redirect") }
                continue
            }
            return Pair(c, url)
        }
        throw HttpException(HttpException.Kind.PROTOCOL, "Too many redirects")
    }

    private fun hostOf(u: String) = try { URL(u).host } catch (e: Exception) { "server" }

    override fun getBytes(url: String, maxBytes: Long, timeoutMs: Int): HttpBytes {
        val (c, finalUrl) = open(url, timeoutMs, null)
        try {
            if (c.responseCode !in 200..299) throw HttpException(HttpException.Kind.STATUS, "Server answered HTTP ${c.responseCode}", c.responseCode)
            val declared = c.contentLengthLong
            if (declared > maxBytes) throw HttpException(HttpException.Kind.TOO_LARGE, "Response is larger than the allowed $maxBytes bytes")
            val data = try { c.inputStream.use { Fs.readLimited(it, maxBytes) } } catch (e: SocketTimeoutException) {
                throw HttpException(HttpException.Kind.TIMEOUT, "Timed out while reading")
            } catch (e: IOException) { throw HttpException(HttpException.Kind.NETWORK, "Network error: ${e.message}") }
                ?: throw HttpException(HttpException.Kind.TOO_LARGE, "Response is larger than the allowed $maxBytes bytes")
            return HttpBytes(finalUrl, c.responseCode, data, c.contentType)
        } finally { c.disconnect() }
    }

    override fun downloadToFile(
        url: String, part: File, maxBytes: Long, cancelled: () -> Boolean, onProgress: (Long, Long) -> Unit, timeoutMs: Int,
    ): DownloadOutcome {
        part.parentFile?.mkdirs()
        val have = if (part.isFile) part.length() else 0L
        val (c, _) = open(url, timeoutMs, have)
        try {
            val code = c.responseCode
            var start = have
            var total = -1L
            when (code) {
                206 -> {
                    val cr = c.getHeaderField("Content-Range") ?: ""
                    val m = Regex("""bytes (\d+)-(\d+)/(\d+|\*)""").find(cr)
                    val s = m?.groupValues?.get(1)?.toLongOrNull()
                    if (s == null || s != have) throw HttpException(HttpException.Kind.PROTOCOL, "Server returned an unexpected byte range")
                    total = m.groupValues[3].toLongOrNull() ?: -1L
                }
                200 -> { start = 0; total = c.contentLengthLong }
                416 -> {
                    val cr = c.getHeaderField("Content-Range") ?: ""
                    val t = Regex("""bytes \*/(\d+)""").find(cr)?.groupValues?.get(1)?.toLongOrNull()
                    if (t != null && t == have) return DownloadOutcome(t, have)
                    part.delete()
                    throw HttpException(HttpException.Kind.PROTOCOL, "Partial file did not match the server; it was discarded. Retry.", 416)
                }
                else -> throw HttpException(HttpException.Kind.STATUS, "Server answered HTTP $code", code)
            }
            if (total > maxBytes) throw HttpException(HttpException.Kind.TOO_LARGE, "Download is larger than allowed")
            RandomAccessFile(part, "rw").use { raf ->
                raf.setLength(start)
                raf.seek(start)
                var done = start
                val buf = ByteArray(1 shl 16)
                try {
                    c.inputStream.use { ins ->
                        while (true) {
                            if (cancelled()) throw HttpException(HttpException.Kind.CANCELLED, "Cancelled")
                            val n = ins.read(buf)
                            if (n < 0) break
                            raf.write(buf, 0, n)
                            done += n
                            if (done > maxBytes) throw HttpException(HttpException.Kind.TOO_LARGE, "Download is larger than allowed")
                            onProgress(done, total)
                        }
                    }
                } catch (e: SocketTimeoutException) {
                    throw HttpException(HttpException.Kind.TIMEOUT, "Timed out while downloading")
                } catch (e: HttpException) { throw e
                } catch (e: IOException) { throw HttpException(HttpException.Kind.NETWORK, "Connection lost: ${e.message}") }
                if (total >= 0 && done != total) throw HttpException(HttpException.Kind.NETWORK, "Connection closed early ($done of $total bytes)")
                return DownloadOutcome(if (total >= 0) total else done, done)
            }
        } finally { c.disconnect() }
    }
}
