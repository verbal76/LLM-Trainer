package com.hotatticgames.llmtrainer.qualify

/**
 * Minimal strict JSON reader (RFC 8259). Exists so `qualify` has no dependency beyond kotlin-stdlib:
 * objects -> Map<String, Any?> (insertion ordered), arrays -> List<Any?>, strings, Boolean, null,
 * integers without fraction/exponent -> Long, every other number -> Double.
 */
class JsonException(message: String) : RuntimeException(message)

object MiniJson {
    fun parse(text: String): Any? {
        val p = Parser(text)
        p.skipWs()
        val v = p.value(0)
        p.skipWs()
        if (p.pos != text.length) p.fail("trailing characters")
        return v
    }

    private class Parser(val s: String) {
        var pos = 0

        fun fail(msg: String): Nothing = throw JsonException("$msg at offset $pos")

        fun skipWs() {
            while (pos < s.length) {
                val c = s[pos]
                if (c == ' ' || c == '\n' || c == '\r' || c == '\t') pos++ else break
            }
        }

        fun value(depth: Int): Any? {
            if (depth > 64) fail("nesting too deep")
            if (pos >= s.length) fail("unexpected end")
            return when (val c = s[pos]) {
                '{' -> obj(depth)
                '[' -> arr(depth)
                '"' -> str()
                't' -> lit("true", true)
                'f' -> lit("false", false)
                'n' -> lit("null", null)
                else -> if (c == '-' || (c in '0'..'9')) num() else fail("unexpected '$c'")
            }
        }

        fun lit(word: String, v: Any?): Any? {
            if (!s.startsWith(word, pos)) fail("bad literal")
            pos += word.length
            return v
        }

        fun obj(depth: Int): Map<String, Any?> {
            pos++ // {
            val m = LinkedHashMap<String, Any?>()
            skipWs()
            if (pos < s.length && s[pos] == '}') { pos++; return m }
            while (true) {
                skipWs()
                if (pos >= s.length || s[pos] != '"') fail("expected string key")
                val k = str()
                skipWs()
                if (pos >= s.length || s[pos] != ':') fail("expected ':'")
                pos++
                skipWs()
                m[k] = value(depth + 1)
                skipWs()
                if (pos >= s.length) fail("unterminated object")
                when (s[pos++]) {
                    ',' -> continue
                    '}' -> return m
                    else -> fail("expected ',' or '}'")
                }
            }
        }

        fun arr(depth: Int): List<Any?> {
            pos++ // [
            val l = ArrayList<Any?>()
            skipWs()
            if (pos < s.length && s[pos] == ']') { pos++; return l }
            while (true) {
                skipWs()
                l.add(value(depth + 1))
                skipWs()
                if (pos >= s.length) fail("unterminated array")
                when (s[pos++]) {
                    ',' -> continue
                    ']' -> return l
                    else -> fail("expected ',' or ']'")
                }
            }
        }

        fun str(): String {
            pos++ // opening quote
            val sb = StringBuilder()
            while (true) {
                if (pos >= s.length) fail("unterminated string")
                val c = s[pos++]
                when {
                    c == '"' -> return sb.toString()
                    c == '\\' -> {
                        if (pos >= s.length) fail("bad escape")
                        when (val e = s[pos++]) {
                            '"' -> sb.append('"')
                            '\\' -> sb.append('\\')
                            '/' -> sb.append('/')
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000C')
                            'n' -> sb.append('\n')
                            'r' -> sb.append('\r')
                            't' -> sb.append('\t')
                            'u' -> {
                                if (pos + 4 > s.length) fail("bad \\u escape")
                                var code = 0
                                for (i in 0 until 4) {
                                    val d = hexVal(s[pos + i])
                                    if (d < 0) fail("bad \\u escape")
                                    code = code * 16 + d
                                }
                                pos += 4
                                sb.append(code.toChar())
                            }
                            else -> fail("bad escape '\\$e'")
                        }
                    }
                    c < ' ' -> fail("control character in string")
                    else -> sb.append(c)
                }
            }
        }

        fun hexVal(c: Char): Int = when (c) {
            in '0'..'9' -> c - '0'
            in 'a'..'f' -> c - 'a' + 10
            in 'A'..'F' -> c - 'A' + 10
            else -> -1
        }

        fun num(): Any {
            val start = pos
            if (s[pos] == '-') pos++
            if (pos >= s.length) fail("bad number")
            if (s[pos] == '0') pos++
            else if (s[pos] in '1'..'9') { while (pos < s.length && s[pos] in '0'..'9') pos++ }
            else fail("bad number")
            var integral = true
            if (pos < s.length && s[pos] == '.') {
                integral = false
                pos++
                val d = pos
                while (pos < s.length && s[pos] in '0'..'9') pos++
                if (pos == d) fail("bad fraction")
            }
            if (pos < s.length && (s[pos] == 'e' || s[pos] == 'E')) {
                integral = false
                pos++
                if (pos < s.length && (s[pos] == '+' || s[pos] == '-')) pos++
                val d = pos
                while (pos < s.length && s[pos] in '0'..'9') pos++
                if (pos == d) fail("bad exponent")
            }
            val tok = s.substring(start, pos)
            if (integral) {
                val l = tok.toLongOrNull()
                if (l != null) return l
            }
            return tok.toDouble()
        }
    }
}

// ---- typed accessors (lenient: wrong type or absent -> null) -------------------------------------------

@Suppress("UNCHECKED_CAST")
internal fun Any?.asObj(): Map<String, Any?>? = this as? Map<String, Any?>

@Suppress("UNCHECKED_CAST")
internal fun Any?.asList(): List<Any?>? = this as? List<Any?>

internal fun Map<String, Any?>.str(k: String): String? = this[k] as? String
internal fun Map<String, Any?>.bool(k: String): Boolean? = this[k] as? Boolean
internal fun Map<String, Any?>.dbl(k: String): Double? = when (val v = this[k]) {
    is Double -> v
    is Long -> v.toDouble()
    else -> null
}
internal fun Map<String, Any?>.lng(k: String): Long? = when (val v = this[k]) {
    is Long -> v
    is Double -> if (v.isNaN() || v.isInfinite()) null else v.toLong()
    else -> null
}
internal fun Map<String, Any?>.obj(k: String): Map<String, Any?>? = this[k].asObj()
internal fun Map<String, Any?>.list(k: String): List<Any?>? = this[k].asList()
