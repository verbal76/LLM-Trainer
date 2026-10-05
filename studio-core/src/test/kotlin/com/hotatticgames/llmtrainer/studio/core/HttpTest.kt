package com.hotatticgames.llmtrainer.studio.core

import com.sun.net.httpserver.HttpExchange
import com.sun.net.httpserver.HttpServer
import java.io.File
import java.net.HttpURLConnection
import java.net.InetSocketAddress
import java.net.URL
import java.util.Random
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The real [JavaHttp] against a loopback server. The opener seam rewrites https://models.test/... to the loopback port, so the
 * HTTPS-only policy (checked on the URL string) and every protocol branch are exercised without a TLS certificate.
 */
class HttpTest {
    private val data = ByteArray(300_000).also { Random(9).nextBytes(it) }
    private val hits = AtomicInteger()
    private val rangeHeaders = java.util.concurrent.CopyOnWriteArrayList<String?>()
    @Volatile private var mode = "normal"
    private val server: HttpServer = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).also { srv ->
        srv.createContext("/") { ex -> hits.incrementAndGet(); try { handle(ex) } finally { ex.close() } }
        srv.start()
    }
    private val port = server.address.port
    private val http = JavaHttp("test-agent") { u -> URL("http://127.0.0.1:$port${u.file}").openConnection() as HttpURLConnection }
    private val dir = TK.tmp("http")

    @AfterTest fun stop() = server.stop(0)

    private fun handle(ex: HttpExchange) {
        val path = ex.requestURI.path
        val range = ex.requestHeaders.getFirst("Range")
        rangeHeaders.add(range)
        when {
            path == "/small" -> ex.sendResponseHeaders(200, 5).also { ex.responseBody.write("hello".toByteArray()) }
            path == "/missing" -> ex.sendResponseHeaders(404, -1)
            path == "/redir" -> { ex.responseHeaders.add("Location", "/small"); ex.sendResponseHeaders(302, -1) }
            path == "/redir-http" -> { ex.responseHeaders.add("Location", "http://evil.test/x"); ex.sendResponseHeaders(302, -1) }
            path == "/loop" -> { ex.responseHeaders.add("Location", "/loop"); ex.sendResponseHeaders(302, -1) }
            path == "/chunked-big" -> { ex.sendResponseHeaders(200, 0); ex.responseBody.write(ByteArray(2_000_000)) }
            path == "/slow" -> { Thread.sleep(2000); ex.sendResponseHeaders(200, 1) }
            path == "/cut" -> { ex.sendResponseHeaders(200, 100_000); ex.responseBody.write(data, 0, 40_000); ex.responseBody.flush() }
            path == "/file" || path == "/file-ignore-range" -> {
                val start = if (range != null && path == "/file") range.removePrefix("bytes=").removeSuffix("-").toInt() else 0
                when {
                    mode == "416-match" && range != null -> { ex.responseHeaders.add("Content-Range", "bytes */${data.size}"); ex.sendResponseHeaders(416, -1) }
                    mode == "416-mismatch" && range != null -> { ex.responseHeaders.add("Content-Range", "bytes */999"); ex.sendResponseHeaders(416, -1) }
                    mode == "bad-range" && range != null -> { ex.responseHeaders.add("Content-Range", "bytes 5-${data.size - 1}/${data.size}"); ex.sendResponseHeaders(206, (data.size - 5).toLong()); ex.responseBody.write(data, 5, data.size - 5) }
                    start > 0 -> { ex.responseHeaders.add("Content-Range", "bytes $start-${data.size - 1}/${data.size}"); ex.sendResponseHeaders(206, (data.size - start).toLong()); ex.responseBody.write(data, start, data.size - start) }
                    else -> { ex.sendResponseHeaders(200, data.size.toLong()); ex.responseBody.write(data) }
                }
            }
            else -> ex.sendResponseHeaders(500, -1)
        }
    }

    private fun err(f: () -> Unit): HttpException = assertFailsWith<HttpException> { f() }

    @Test fun getBytesHappyPathAndStatusAndRedirect() {
        assertEquals("hello", String(http.getBytes("https://models.test/small", 100).bytes))
        assertEquals("hello", String(http.getBytes("https://models.test/redir", 100).bytes))
        val e = err { http.getBytes("https://models.test/missing", 100) }
        assertEquals(HttpException.Kind.STATUS, e.kind); assertEquals(404, e.status)
        assertEquals(HttpException.Kind.PROTOCOL, err { http.getBytes("https://models.test/loop", 100) }.kind)
    }

    @Test fun plainHttpIsRefusedBeforeAnyConnectionAndRedirectsToHttpToo() {
        val before = hits.get()
        assertEquals(HttpException.Kind.NOT_HTTPS, err { http.getBytes("http://models.test/small", 100) }.kind)
        assertEquals(HttpException.Kind.NOT_HTTPS, err { http.downloadToFile("http://models.test/file", File(dir, "a.part"), 1 shl 30, { false }, { _, _ -> }) }.kind)
        assertEquals(before, hits.get(), "no request may be made for a non-HTTPS URL")
        assertEquals(HttpException.Kind.NOT_HTTPS, err { http.getBytes("https://models.test/redir-http", 100) }.kind)     // downgrade via redirect
    }

    @Test fun sizeCapsApplyToDeclaredAndUndeclaredLengths() {
        assertEquals(HttpException.Kind.TOO_LARGE, err { http.getBytes("https://models.test/file", 1000) }.kind)               // Content-Length known
        assertEquals(HttpException.Kind.TOO_LARGE, err { http.getBytes("https://models.test/chunked-big", 1000) }.kind)       // chunked, no length
    }

    @Test fun timeoutsAreEnforced() {
        val e = err { http.getBytes("https://models.test/slow", 100, timeoutMs = 300) }
        assertEquals(HttpException.Kind.TIMEOUT, e.kind)
    }

    @Test fun downloadFromScratchAndResumeWithRange() {
        val part = File(dir, "m.part")
        val out = http.downloadToFile("https://models.test/file", part, 10_000_000, { false }, { _, _ -> })
        assertEquals(data.size.toLong(), out.bytesOnDisk); assertTrue(part.readBytes().contentEquals(data))
        // pretend a previous run stopped after 123456 bytes
        part.writeBytes(data.copyOf(123_456))
        val progress = ArrayList<Pair<Long, Long>>()
        val out2 = http.downloadToFile("https://models.test/file", part, 10_000_000, { false }, { d, t -> progress.add(d to t) })
        assertTrue(part.readBytes().contentEquals(data))
        assertEquals("bytes=123456-", rangeHeaders.last())
        assertTrue(progress.first().first > 123_456 && progress.last() == data.size.toLong() to data.size.toLong())
        assertEquals(data.size.toLong(), out2.totalBytes)
    }

    @Test fun serverIgnoringRangeRestartsTheFile() {
        val part = File(dir, "n.part"); part.writeBytes(ByteArray(5000) { 7 })
        http.downloadToFile("https://models.test/file-ignore-range", part, 10_000_000, { false }, { _, _ -> })
        assertTrue(part.readBytes().contentEquals(data))
    }

    @Test fun unexpectedContentRangeIsRejected() {
        mode = "bad-range"
        val part = File(dir, "b.part"); part.writeBytes(data.copyOf(1000))
        val e = err { http.downloadToFile("https://models.test/file", part, 10_000_000, { false }, { _, _ -> }) }
        assertEquals(HttpException.Kind.PROTOCOL, e.kind)
        assertEquals(1000L, part.length(), "a mismatching range must not corrupt the partial file")
    }

    @Test fun rangeNotSatisfiableMeansDoneOnlyWhenTheSizesAgree() {
        mode = "416-match"
        val full = File(dir, "f.part"); full.writeBytes(data)
        assertEquals(data.size.toLong(), http.downloadToFile("https://models.test/file", full, 10_000_000, { false }, { _, _ -> }).bytesOnDisk)
        mode = "416-mismatch"
        val wrong = File(dir, "w.part"); wrong.writeBytes(data.copyOf(500))
        val e = err { http.downloadToFile("https://models.test/file", wrong, 10_000_000, { false }, { _, _ -> }) }
        assertEquals(416, e.status); assertTrue(!wrong.exists(), "inconsistent partial file is discarded")
    }

    @Test fun truncatedTransferIsANetworkErrorAndKeepsTheBytesForResume() {
        val part = File(dir, "c.part")
        val e = err { http.downloadToFile("https://models.test/cut", part, 10_000_000, { false }, { _, _ -> }) }
        assertEquals(HttpException.Kind.NETWORK, e.kind)
        assertEquals(40_000L, part.length())
    }

    @Test fun cancelAndHardCapStopTheTransfer() {
        val part = File(dir, "x.part")
        var n = 0
        val e = err { http.downloadToFile("https://models.test/file", part, 10_000_000, { n++ > 2 }, { _, _ -> }) }
        assertEquals(HttpException.Kind.CANCELLED, e.kind)
        assertTrue(part.exists() && part.length() < data.size)
        val capped = File(dir, "y.part")
        assertEquals(HttpException.Kind.TOO_LARGE, err { http.downloadToFile("https://models.test/file", capped, 1000, { false }, { _, _ -> }) }.kind)
    }

    @Test fun userAgentIsSentAndNoCredentialsLeak() {
        val seen = ArrayList<String>()
        server.removeContext("/")
        server.createContext("/") { ex -> seen.add(ex.requestHeaders.getFirst("User-Agent") + "|" + ex.requestHeaders.getFirst("Authorization") + "|" + ex.requestHeaders.getFirst("Cookie")); ex.sendResponseHeaders(200, 2); ex.responseBody.write("ok".toByteArray()); ex.close() }
        http.getBytes("https://models.test/x", 10)
        assertEquals(listOf("test-agent|null|null"), seen)
    }
}
