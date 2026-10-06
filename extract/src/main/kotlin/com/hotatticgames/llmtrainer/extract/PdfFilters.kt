package com.hotatticgames.llmtrainer.extract

import java.io.ByteArrayOutputStream
import java.util.zip.DataFormatException
import java.util.zip.Inflater

/** Stream filters needed for text extraction. Image filters (DCT, JPX, CCITT, JBIG2) are intentionally unsupported. */
internal object PdfFilters {
    const val MAX_OUT = 96 * 1024 * 1024

    fun flate(data: ByteArray, maxOut: Int = MAX_OUT): ByteArray {
        // Normal zlib first; if that yields nothing, retry as raw deflate after the 2-byte header (some writers are sloppy).
        val a = inflate(data, 0, false, maxOut)
        if (a.isNotEmpty() || data.size < 3) return a
        return inflate(data, 2, true, maxOut)
    }

    private fun inflate(data: ByteArray, off: Int, nowrap: Boolean, maxOut: Int): ByteArray {
        val inf = Inflater(nowrap)
        val out = ByteArrayOutputStream(maxOf(1024, minOf(data.size * 3, 1 shl 22)))
        try {
            inf.setInput(data, off, data.size - off)
            val buf = ByteArray(32768)
            while (!inf.finished()) {
                val n = try { inf.inflate(buf) } catch (e: DataFormatException) { break } // keep the partial output
                if (n == 0) break // needs input/dictionary or stalled: never loop
                out.write(buf, 0, n)
                if (out.size() > maxOut) throw PdfException("decoded stream exceeds ${maxOut / 1024 / 1024} MB limit")
            }
        } finally { inf.end() }
        return out.toByteArray()
    }

    fun asciiHex(data: ByteArray): ByteArray {
        val out = ByteArrayOutputStream()
        var hi = -1
        for (x in data) {
            val c = x.toInt() and 0xFF
            if (c == '>'.code) break
            val v = PdfLexer.hexVal(c)
            if (v < 0) continue
            if (hi < 0) hi = v else { out.write(hi * 16 + v); hi = -1 }
        }
        if (hi >= 0) out.write(hi * 16)
        return out.toByteArray()
    }

    fun ascii85(data: ByteArray): ByteArray {
        val out = ByteArrayOutputStream()
        val grp = LongArray(5)
        var n = 0
        var i = 0
        if (data.size >= 2 && data[0] == '<'.code.toByte() && data[1] == '~'.code.toByte()) i = 2
        while (i < data.size) {
            val c = data[i++].toInt() and 0xFF
            if (isPdfWs(c)) continue
            if (c == '~'.code) break
            if (c == 'z'.code && n == 0) { out.write(0); out.write(0); out.write(0); out.write(0); continue }
            if (c < '!'.code || c > 'u'.code) continue
            grp[n++] = (c - '!'.code).toLong()
            if (n == 5) {
                var v = 0L
                for (k in 0 until 5) v = v * 85 + grp[k]
                out.write((v shr 24).toInt() and 0xFF); out.write((v shr 16).toInt() and 0xFF)
                out.write((v shr 8).toInt() and 0xFF); out.write(v.toInt() and 0xFF)
                n = 0
            }
        }
        if (n > 1) {
            for (k in n until 5) grp[k] = 84
            var v = 0L
            for (k in 0 until 5) v = v * 85 + grp[k]
            for (k in 0 until n - 1) out.write((v shr (24 - 8 * k)).toInt() and 0xFF)
        }
        return out.toByteArray()
    }

    fun runLength(data: ByteArray): ByteArray {
        val out = ByteArrayOutputStream()
        var i = 0
        while (i < data.size) {
            val l = data[i++].toInt() and 0xFF
            if (l == 128) break
            if (l < 128) { val n = minOf(l + 1, data.size - i); out.write(data, i, n); i += n }
            else { if (i >= data.size) break; val v = data[i++].toInt(); repeat(257 - l) { out.write(v) } }
            if (out.size() > MAX_OUT) throw PdfException("RunLength output too large")
        }
        return out.toByteArray()
    }

    fun lzw(data: ByteArray, earlyChange: Int): ByteArray {
        val out = ByteArrayOutputStream()
        val table = arrayOfNulls<ByteArray>(4096)
        for (i in 0 until 256) table[i] = byteArrayOf(i.toByte())
        var next = 258
        var bits = 9
        var acc = 0
        var nb = 0
        var prev: ByteArray? = null
        var i = 0
        while (true) {
            while (nb < bits && i < data.size) { acc = (acc shl 8) or (data[i++].toInt() and 0xFF); nb += 8 }
            if (nb < bits) break
            val code = (acc shr (nb - bits)) and ((1 shl bits) - 1)
            nb -= bits
            acc = acc and ((1 shl nb) - 1)
            if (code == 256) { next = 258; bits = 9; prev = null; continue }
            if (code == 257) break
            val entry: ByteArray = when {
                code < next && table[code] != null -> table[code]!!
                code == next && prev != null -> prev + prev[0]
                else -> break // corrupt
            }
            out.write(entry)
            if (out.size() > MAX_OUT) throw PdfException("LZW output too large")
            if (prev != null && next < 4096) table[next++] = prev + entry[0]
            prev = entry
            bits = when { next + earlyChange >= 2048 -> 12; next + earlyChange >= 1024 -> 11; next + earlyChange >= 512 -> 10; else -> 9 }
        }
        return out.toByteArray()
    }

    /** PNG (10..15) and TIFF (2, 8-bit only) predictors, as used by xref and object streams. */
    fun predictor(data: ByteArray, predictor: Int, colors: Int, bpc: Int, columns: Int): ByteArray {
        if (predictor < 2) return data
        val rowBytes = (colors.toLong() * bpc * columns + 7) / 8
        if (rowBytes <= 0 || rowBytes > 1 shl 24) throw PdfException("bad predictor parameters")
        val rb = rowBytes.toInt()
        val bpp = maxOf(1, colors * bpc / 8)
        if (predictor == 2) {
            if (bpc != 8) throw PdfException("TIFF predictor with $bpc bits per component not supported")
            val out = data.copyOf()
            var r = 0
            while (r + rb <= out.size) {
                for (x in bpp until rb) out[r + x] = (out[r + x] + out[r + x - bpp]).toByte()
                r += rb
            }
            return out
        }
        val out = ByteArrayOutputStream(data.size)
        var prev = ByteArray(rb)
        var i = 0
        while (i < data.size) {
            val ft = data[i++].toInt() and 0xFF
            val cur = ByteArray(rb)
            val n = minOf(rb, data.size - i)
            System.arraycopy(data, i, cur, 0, n)
            i += n
            for (x in 0 until rb) {
                val a = if (x >= bpp) cur[x - bpp].toInt() and 0xFF else 0
                val b = prev[x].toInt() and 0xFF
                val c = if (x >= bpp) prev[x - bpp].toInt() and 0xFF else 0
                val add = when (ft) {
                    0 -> 0
                    1 -> a
                    2 -> b
                    3 -> (a + b) / 2
                    4 -> { val p = a + b - c; val pa = Math.abs(p - a); val pb = Math.abs(p - b); val pc = Math.abs(p - c); if (pa <= pb && pa <= pc) a else if (pb <= pc) b else c }
                    else -> 0
                }
                cur[x] = (cur[x] + add).toByte()
            }
            out.write(cur)
            prev = cur
        }
        return out.toByteArray()
    }
}
