package com.hotatticgames.llmtrainer.host

import com.hotatticgames.llmtrainer.ota.Fetcher
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/** HTTPS-only fetcher with size caps and timeouts. GitHub release asset URLs redirect https->https. */
class HttpFetcher : Fetcher {
    private fun open(url: String): HttpURLConnection {
        require(url.startsWith("https://")) { "https required: $url" }
        val c = URL(url).openConnection() as HttpURLConnection
        c.connectTimeout = 10_000
        c.readTimeout = 30_000
        c.instanceFollowRedirects = true
        c.setRequestProperty("User-Agent", "LLM-Trainer-OTA/1")
        c.setRequestProperty("Cache-Control", "no-cache")
        if (c.responseCode !in 200..299) throw IOException("HTTP ${c.responseCode} for $url")
        return c
    }

    override fun getBytes(url: String, maxBytes: Long): ByteArray {
        val c = open(url)
        try {
            c.inputStream.use { ins ->
                val out = java.io.ByteArrayOutputStream()
                val buf = ByteArray(16 * 1024)
                var total = 0L
                while (true) {
                    val n = ins.read(buf)
                    if (n < 0) break
                    total += n
                    if (total > maxBytes) throw IOException("response exceeds $maxBytes bytes")
                    out.write(buf, 0, n)
                }
                return out.toByteArray()
            }
        } finally {
            c.disconnect()
        }
    }

    override fun download(url: String, dest: File, maxBytes: Long) {
        val c = open(url)
        try {
            c.inputStream.use { ins ->
                dest.outputStream().use { out ->
                    val buf = ByteArray(64 * 1024)
                    var total = 0L
                    while (true) {
                        val n = ins.read(buf)
                        if (n < 0) break
                        total += n
                        if (total > maxBytes) throw IOException("download exceeds $maxBytes bytes")
                        out.write(buf, 0, n)
                    }
                }
            }
        } catch (e: IOException) {
            dest.delete(); throw e
        } finally {
            c.disconnect()
        }
    }
}
