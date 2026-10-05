package com.hotatticgames.llmtrainer.extract

/** Parse failure inside the PDF reader. Always caught at a boundary and turned into an IngestIssue. */
internal class PdfException(msg: String) : Exception(msg)

internal sealed class PObj
internal object PNull : PObj()
internal class PBool(val v: Boolean) : PObj()
internal class PNum(val v: Double, val isInt: Boolean) : PObj() {
    fun int(): Int = if (v >= Int.MAX_VALUE) Int.MAX_VALUE else if (v <= Int.MIN_VALUE) Int.MIN_VALUE else v.toInt()
}
internal class PName(val v: String) : PObj()
internal class PStr(val bytes: ByteArray) : PObj()
internal class PArr(val items: List<PObj>) : PObj()
internal class PDict(val map: Map<String, PObj>) : PObj()
internal class PRef(val num: Int, val gen: Int) : PObj()
/** [raw] is the still-encoded stream data. */
internal class PStream(val dict: PDict, val raw: ByteArray) : PObj()
/** Operators (content streams) and structural keywords (obj, endobj, stream, ...). */
internal class PKeyword(val v: String) : PObj()

internal fun isPdfWs(c: Int) = c == 0 || c == 9 || c == 10 || c == 12 || c == 13 || c == 32
internal fun isPdfDelim(c: Int) = c == '('.code || c == ')'.code || c == '<'.code || c == '>'.code || c == '['.code ||
    c == ']'.code || c == '{'.code || c == '}'.code || c == '/'.code || c == '%'.code

internal fun indexOfBytes(b: ByteArray, pat: String, from: Int, to: Int = b.size): Int {
    val p = pat.toByteArray(Charsets.ISO_8859_1)
    var i = maxOf(from, 0)
    val last = minOf(to, b.size) - p.size
    while (i <= last) {
        if (b[i] == p[0]) {
            var k = 1
            while (k < p.size && b[i + k] == p[k]) k++
            if (k == p.size) return i
        }
        i++
    }
    return -1
}

internal fun lastIndexOfBytes(b: ByteArray, pat: String): Int {
    val p = pat.toByteArray(Charsets.ISO_8859_1)
    var i = b.size - p.size
    while (i >= 0) {
        var k = 0
        while (k < p.size && b[i + k] == p[k]) k++
        if (k == p.size) return i
        i--
    }
    return -1
}

/** Tokenizer/parser for PDF syntax (file objects, content streams, CMaps). Never reads past [end]. */
internal class PdfLexer(val b: ByteArray, var pos: Int = 0, val end: Int = b.size, private val refs: Boolean = true) {
    private fun at(i: Int) = if (i in 0 until end) b[i].toInt() and 0xFF else -1

    fun skipWs() {
        while (pos < end) {
            val c = at(pos)
            if (isPdfWs(c)) pos++
            else if (c == '%'.code) { while (pos < end && at(pos) != 10 && at(pos) != 13) pos++ }
            else break
        }
    }

    fun startsWith(s: String): Boolean {
        if (pos + s.length > end) return false
        for (k in s.indices) if (at(pos + k) != s[k].code) return false
        return true
    }

    /** Next object, or null at end of input. Stray closers come back as [PKeyword]. */
    fun readObject(depth: Int = 0): PObj? {
        if (depth > 64) throw PdfException("object nesting too deep")
        skipWs()
        if (pos >= end) return null
        val c = at(pos)
        when {
            c == '/'.code -> return name()
            c == '('.code -> return literal()
            c == '<'.code -> return if (at(pos + 1) == '<'.code) { pos += 2; dict(depth) } else hex()
            c == '['.code -> { pos++; return array(depth) }
            c == ']'.code -> { pos++; return PKeyword("]") }
            c == '>'.code -> { if (at(pos + 1) == '>'.code) { pos += 2; return PKeyword(">>") }; pos++; return PKeyword(">") }
            c == ')'.code || c == '{'.code || c == '}'.code -> { pos++; return PKeyword(c.toChar().toString()) }
            c == '+'.code || c == '-'.code || c == '.'.code || (c in '0'.code..'9'.code) -> return numberOrKeyword()
        }
        val st = pos
        while (pos < end && !isPdfWs(at(pos)) && !isPdfDelim(at(pos))) pos++
        if (pos == st) pos++
        return when (val w = String(b, st, pos - st, Charsets.ISO_8859_1)) {
            "true" -> PBool(true)
            "false" -> PBool(false)
            "null" -> PNull
            else -> PKeyword(w)
        }
    }

    private fun name(): PName {
        pos++
        val sb = StringBuilder()
        while (pos < end) {
            val c = at(pos)
            if (isPdfWs(c) || isPdfDelim(c)) break
            if (c == '#'.code && pos + 2 < end + 0 && hexVal(at(pos + 1)) >= 0 && hexVal(at(pos + 2)) >= 0) {
                sb.append((hexVal(at(pos + 1)) * 16 + hexVal(at(pos + 2))).toChar()); pos += 3
            } else { sb.append(c.toChar()); pos++ }
        }
        return PName(sb.toString())
    }

    private fun literal(): PStr {
        pos++
        val out = java.io.ByteArrayOutputStream()
        var depth = 1
        while (pos < end) {
            val c = at(pos++)
            when (c) {
                '('.code -> { depth++; out.write(c) }
                ')'.code -> { if (--depth == 0) return PStr(out.toByteArray()); out.write(c) }
                '\\'.code -> {
                    if (pos >= end) break
                    val e = at(pos++)
                    when (e) {
                        'n'.code -> out.write(10); 'r'.code -> out.write(13); 't'.code -> out.write(9)
                        'b'.code -> out.write(8); 'f'.code -> out.write(12)
                        '('.code, ')'.code, '\\'.code -> out.write(e)
                        13 -> if (at(pos) == 10) pos++
                        10 -> {}
                        in '0'.code..'7'.code -> {
                            var v = e - '0'.code
                            var n = 1
                            while (n < 3 && at(pos) in '0'.code..'7'.code) { v = v * 8 + (at(pos) - '0'.code); pos++; n++ }
                            out.write(v and 0xFF)
                        }
                        else -> out.write(e)
                    }
                }
                13 -> { out.write(10); if (at(pos) == 10) pos++ }
                else -> out.write(c)
            }
        }
        return PStr(out.toByteArray()) // unterminated: keep what we have
    }

    private fun hex(): PStr {
        pos++
        val out = java.io.ByteArrayOutputStream()
        var hi = -1
        while (pos < end) {
            val c = at(pos++)
            if (c == '>'.code) break
            val v = hexVal(c)
            if (v < 0) continue
            if (hi < 0) hi = v else { out.write(hi * 16 + v); hi = -1 }
        }
        if (hi >= 0) out.write(hi * 16)
        return PStr(out.toByteArray())
    }

    private fun array(depth: Int): PArr {
        val items = ArrayList<PObj>()
        while (true) {
            val o = readObject(depth + 1) ?: break
            if (o is PKeyword) {
                if (o.v == "]") break
                if (o.v == "endobj" || o.v == "endstream") break
                continue
            }
            items.add(o)
            if (items.size > 5_000_000) throw PdfException("array too large")
        }
        return PArr(items)
    }

    private fun dict(depth: Int): PDict {
        val map = LinkedHashMap<String, PObj>()
        while (true) {
            val save = pos
            val k = readObject(depth + 1) ?: break
            if (k is PKeyword) {
                if (k.v == ">>") break
                if (k.v == "endobj" || k.v == "stream" || k.v == "endstream") { pos = save; break }
                continue
            }
            if (k !is PName) continue
            val save2 = pos
            val v = readObject(depth + 1) ?: break
            if (v is PKeyword) {
                if (v.v == ">>") break
                if (v.v == "endobj" || v.v == "stream") { pos = save2; break }
                continue
            }
            map[k.v] = v
            if (map.size > 200_000) throw PdfException("dictionary too large")
        }
        return PDict(map)
    }

    private fun numberOrKeyword(): PObj {
        val st = pos
        while (pos < end && !isPdfWs(at(pos)) && !isPdfDelim(at(pos))) pos++
        var t = String(b, st, pos - st, Charsets.ISO_8859_1)
        val neg = t.startsWith("-")
        t = t.trimStart('+', '-')
        if (t.isNotEmpty() && t.all { it in '0'..'9' }) {
            val v = (t.take(18).toLongOrNull() ?: 0L).toDouble().let { if (t.length > 18) Double.MAX_VALUE else it }
            val num = PNum(if (neg) -v else v, true)
            if (refs && !neg && v < Int.MAX_VALUE) {
                val save = pos
                skipWs()
                val s2 = pos
                while (pos < end && at(pos) in '0'.code..'9'.code) pos++
                if (pos > s2 && pos - s2 <= 10) {
                    val gen = String(b, s2, pos - s2, Charsets.ISO_8859_1).toLong()
                    skipWs()
                    if (at(pos) == 'R'.code && (pos + 1 >= end || isPdfWs(at(pos + 1)) || isPdfDelim(at(pos + 1)))) {
                        pos++
                        return PRef(v.toInt(), gen.coerceAtMost(65535).toInt())
                    }
                }
                pos = save
            }
            return num
        }
        val d = if (t.count { it == '.' } <= 1 && t.isNotEmpty() && t.all { it in '0'..'9' || it == '.' } && t != ".") t.toDoubleOrNull() else null
        if (d != null) return PNum(if (neg) -d else d, false)
        return PKeyword(String(b, st, pos - st, Charsets.ISO_8859_1))
    }

    companion object {
        fun hexVal(c: Int): Int = when (c) {
            in '0'.code..'9'.code -> c - '0'.code
            in 'a'.code..'f'.code -> c - 'a'.code + 10
            in 'A'.code..'F'.code -> c - 'A'.code + 10
            else -> -1
        }
    }
}
