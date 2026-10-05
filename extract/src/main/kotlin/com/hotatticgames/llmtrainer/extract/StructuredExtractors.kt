package com.hotatticgames.llmtrainer.extract

private const val STRUCT_VERSION = "1"

/** One parsed delimited row with the exact char span of the whole (possibly multi-line) record. */
internal class Row(val cells: List<String>, val start: Int, val end: Int)

/**
 * RFC 4180-ish parser: quoted fields, doubled quotes, delimiters and newlines inside quotes, CRLF/LF/CR terminators.
 * A quote in the middle of an unquoted field is literal. An unterminated quote runs to EOF and sets [unterminated].
 */
internal fun parseDelimited(text: String, delim: Char): Pair<List<Row>, Boolean> {
    val rows = ArrayList<Row>()
    var i = 0
    val n = text.length
    var unterminated = false
    while (i < n) {
        val rowStart = i
        val cells = ArrayList<String>()
        val sb = StringBuilder()
        var rowEnd: Int
        while (true) {
            sb.setLength(0)
            if (i < n && text[i] == '"') {
                i++
                var closed = false
                while (i < n) {
                    val c = text[i]
                    if (c == '"') {
                        if (i + 1 < n && text[i + 1] == '"') { sb.append('"'); i += 2 } else { i++; closed = true; break }
                    } else { sb.append(c); i++ }
                }
                if (!closed) unterminated = true
                while (i < n && text[i] != delim && text[i] != '\n' && text[i] != '\r') { sb.append(text[i]); i++ } // junk after closing quote
            } else {
                while (i < n && text[i] != delim && text[i] != '\n' && text[i] != '\r') { sb.append(text[i]); i++ }
            }
            cells.add(sb.toString())
            if (i < n && text[i] == delim) { i++; continue }
            rowEnd = i
            if (i < n && text[i] == '\r') i++
            if (i < n && text[i] == '\n') i++
            break
        }
        if (cells.any { it.isNotBlank() }) rows.add(Row(cells.map { it.trim() }, rowStart, rowEnd))
    }
    return rows to unterminated
}

internal fun headerNames(raw: List<String>, width: Int): List<String> =
    (0 until maxOf(raw.size, width)).map { i -> raw.getOrNull(i)?.takeIf { it.isNotBlank() } ?: "col${i + 1}" }

internal fun recordText(header: List<String>, cells: List<String>): String =
    cells.indices.filter { cells[it].isNotBlank() }.joinToString("; ") { "${header.getOrElse(it) { "col${it + 1}" }}: ${cells[it]}" }

/**
 * CSV / TSV. The first non-blank row is the header; each following row becomes one `record` block
 * ("Header: value; Header: value", blank cells omitted) whose span covers the full source row, including multi-line quoted cells.
 */
class DelimitedExtractor : Extractor {
    override fun supports(fileName: String, mime: String?) =
        extOf(fileName) in setOf(".csv", ".tsv") ||
            (extOf(fileName) == "" && mime in setOf("text/csv", "text/tab-separated-values"))

    override fun extract(fileName: String, bytes: ByteArray): Extracted {
        val tsv = extOf(fileName) == ".tsv"
        val name = if (tsv) "tsv" else "csv"
        val d = decodeText(bytes, fileName) ?: return extractedOf(name, STRUCT_VERSION, listOf(IngestIssue(
            "undecodable_text", fileName, "not decodable text (binary data or invalid UTF-16)")))
        val issues = d.issues.toMutableList()
        val (rows, unterminated) = parseDelimited(d.text, if (tsv) '\t' else ',')
        if (unterminated) issues.add(IngestIssue("corrupt_structured", fileName, "unterminated quoted field; the rest of the file was read as one cell", "warning"))
        if (rows.size < 2) {
            issues.add(IngestIssue("empty_content", fileName, "no data rows (header only or empty)"))
            return Extracted(d.text, emptyList(), issues, emptyList(), name, STRUCT_VERSION, null)
        }
        val width = rows.maxOf { it.cells.size }
        val header = headerNames(rows[0].cells, rows[0].cells.size)
        val ragged = rows.drop(1).count { it.cells.size != header.size }
        if (ragged > 0) issues.add(IngestIssue("ragged_rows", fileName, "$ragged row(s) have a different column count than the header", "warning"))
        val blocks = rows.drop(1).map { RawBlock(recordText(headerNames(header, width), it.cells), BlockKind.RECORD, null, listOf(fileName), it.start, it.end) }
        return Extracted(d.text, blocks, issues, emptyList(), name, STRUCT_VERSION, null)
    }
}

// --- minimal order-preserving JSON parser (org.json on the JVM does not preserve key order, and has no offsets) ---

internal class JNum(val raw: String)

internal class JsonSyntax(msg: String, val pos: Int) : Exception("$msg at char $pos")

internal class JsonParser(private val s: String, private var i: Int = 0) {
    var depth = 0

    fun pos() = i
    fun skipWs() { while (i < s.length && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++ }
    fun atEnd(): Boolean { skipWs(); return i >= s.length }
    fun peek(): Char? { skipWs(); return s.getOrNull(i) }
    fun expect(c: Char) { skipWs(); if (s.getOrNull(i) != c) throw JsonSyntax("expected '$c'", i); i++ }

    fun value(): Any? {
        skipWs()
        if (i >= s.length) throw JsonSyntax("unexpected end", i)
        if (++depth > 200) throw JsonSyntax("nesting too deep", i)
        try {
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> lit("true", true)
                'f' -> lit("false", false)
                'n' -> lit("null", null)
                else -> if (c == '-' || c.isDigit()) num() else throw JsonSyntax("unexpected '$c'", i)
            }
        } finally { depth-- }
    }

    private fun lit(w: String, v: Any?): Any? {
        if (!s.startsWith(w, i)) throw JsonSyntax("bad literal", i)
        i += w.length; return v
    }

    private fun num(): JNum {
        val st = i
        if (s[i] == '-') i++
        while (i < s.length && (s[i].isDigit() || s[i] in ".eE+-")) i++
        val raw = s.substring(st, i)
        if (raw.toDoubleOrNull() == null) throw JsonSyntax("bad number", st)
        return JNum(raw)
    }

    private fun str(): String {
        i++
        val sb = StringBuilder()
        while (true) {
            if (i >= s.length) throw JsonSyntax("unterminated string", i)
            val c = s[i++]
            when (c) {
                '"' -> return sb.toString()
                '\\' -> {
                    if (i >= s.length) throw JsonSyntax("bad escape", i)
                    when (val e = s[i++]) {
                        'n' -> sb.append('\n'); 't' -> sb.append('\t'); 'r' -> sb.append('\r'); 'b' -> sb.append('\b')
                        'f' -> sb.append('\u000c'); '/' -> sb.append('/'); '\\' -> sb.append('\\'); '"' -> sb.append('"')
                        'u' -> {
                            val h = s.substring(i, minOf(i + 4, s.length)).toIntOrNull(16) ?: throw JsonSyntax("bad \\u escape", i)
                            if (i + 4 > s.length) throw JsonSyntax("bad \\u escape", i)
                            sb.append(h.toChar()); i += 4
                        }
                        else -> throw JsonSyntax("bad escape \\$e", i)
                    }
                }
                else -> sb.append(c)
            }
        }
    }

    private fun arr(): List<Any?> {
        i++
        val out = ArrayList<Any?>()
        if (peek() == ']') { i++; return out }
        while (true) {
            out.add(value())
            skipWs()
            when (s.getOrNull(i)) { ',' -> i++; ']' -> { i++; return out }; else -> throw JsonSyntax("expected ',' or ']'", i) }
        }
    }

    private fun obj(): Map<String, Any?> {
        i++
        val out = LinkedHashMap<String, Any?>()
        if (peek() == '}') { i++; return out }
        while (true) {
            skipWs()
            if (s.getOrNull(i) != '"') throw JsonSyntax("expected string key", i)
            val k = str()
            expect(':')
            out[k] = value()
            skipWs()
            when (s.getOrNull(i)) { ',' -> i++; '}' -> { i++; return out }; else -> throw JsonSyntax("expected ',' or '}'", i) }
        }
    }

    /** Parse an array and report each element's exact char span (used for record offsets). */
    fun arrayWithSpans(): List<Triple<Any?, Int, Int>> {
        expect('[')
        val out = ArrayList<Triple<Any?, Int, Int>>()
        if (peek() == ']') { i++; return out }
        while (true) {
            skipWs()
            val st = i
            val v = value()
            out.add(Triple(v, st, i))
            skipWs()
            when (s.getOrNull(i)) { ',' -> i++; ']' -> { i++; return out }; else -> throw JsonSyntax("expected ',' or ']'", i) }
        }
    }
}

internal fun jsonCompact(v: Any?): String = when (v) {
    null -> "null"
    is String -> "\"" + v.flatMap { c ->
        when (c) {
            '"' -> "\\\"".toList(); '\\' -> "\\\\".toList(); '\n' -> "\\n".toList(); '\r' -> "\\r".toList(); '\t' -> "\\t".toList()
            else -> if (c.code < 32) "\\u%04x".format(c.code).toList() else listOf(c)
        }
    }.joinToString("") + "\""
    is JNum -> v.raw
    is Map<*, *> -> v.entries.joinToString(",", "{", "}") { jsonCompact(it.key as String) + ":" + jsonCompact(it.value) }
    is List<*> -> v.joinToString(",", "[", "]") { jsonCompact(it) }
    else -> v.toString()
}

private fun flat(v: Any?): String = when (v) { is String -> v; is JNum -> v.raw; null -> ""; else -> jsonCompact(v) }

/**
 * JSON / JSONL. Each non-empty JSON object becomes one `record` block ("key: value; key: value", nested values as
 * compact JSON). Spans are exact: the object's slice in the file (JSON arrays and JSONL lines alike). A top-level object
 * with exactly one array-of-objects member is unwrapped; any other object is a single record. Malformed JSONL lines are
 * skipped with a warning; malformed JSON is `corrupt_structured`.
 */
class JsonExtractor : Extractor {
    override fun supports(fileName: String, mime: String?) =
        extOf(fileName) in setOf(".json", ".jsonl", ".ndjson") ||
            (extOf(fileName) == "" && mime in setOf("application/json", "application/x-ndjson"))

    override fun extract(fileName: String, bytes: ByteArray): Extracted {
        val lines = extOf(fileName).let { it == ".jsonl" || it == ".ndjson" }
        val name = if (lines) "jsonl" else "json"
        val d = decodeText(bytes, fileName) ?: return extractedOf(name, STRUCT_VERSION, listOf(IngestIssue(
            "undecodable_text", fileName, "not decodable text (binary data or invalid UTF-16)")))
        val issues = d.issues.toMutableList()
        val recs = ArrayList<Triple<Any?, Int, Int>>()
        try {
            if (lines) {
                for ((ix, ln) in scanLines(d.text).withIndex()) {
                    if (ln.body.isBlank()) continue
                    try {
                        val p = JsonParser(ln.body)
                        val v = p.value()
                        if (!p.atEnd()) throw JsonSyntax("trailing data", p.pos())
                        recs.add(Triple(v, ln.start, ln.end))
                    } catch (e: JsonSyntax) {
                        issues.add(IngestIssue("corrupt_structured", fileName, "line ${ix + 1}: ${e.message}", "warning"))
                    }
                }
            } else {
                val p = JsonParser(d.text)
                p.skipWs()
                when (p.peek()) {
                    '[' -> recs.addAll(p.arrayWithSpans())
                    '{' -> {
                        val st = p.pos()
                        val v = p.value()
                        val m = v as Map<*, *>
                        val lists = m.values.filter { it is List<*> && it.isNotEmpty() && it.all { x -> x is Map<*, *> } }
                        if (lists.size == 1) {
                            // re-parse the member list to get element spans: locate it by parsing again with spans
                            recs.addAll(memberSpans(d.text, st, m, lists[0] as List<*>))
                        } else recs.add(Triple(v, st, p.pos()))
                    }
                    else -> throw JsonSyntax("top level must be an object or an array of objects", p.pos())
                }
                if (!p.atEnd()) throw JsonSyntax("trailing data", p.pos())
            }
        } catch (e: JsonSyntax) {
            issues.add(IngestIssue("corrupt_structured", fileName, "JSON parse error: ${e.message}"))
            return Extracted(d.text, emptyList(), issues, emptyList(), name, STRUCT_VERSION, null)
        }
        val blocks = recs.mapNotNull { (v, s, e) ->
            val m = v as? Map<*, *> ?: return@mapNotNull null
            val txt = m.entries.filter { flat(it.value).isNotBlank() }.joinToString("; ") { "${it.key}: ${flat(it.value)}" }
            if (txt.isBlank()) null else RawBlock(txt, BlockKind.RECORD, null, listOf(fileName), s, e)
        }
        if (blocks.isEmpty()) issues.add(IngestIssue("empty_content", fileName, "no non-empty JSON object records"))
        return Extracted(d.text, blocks, issues, emptyList(), name, STRUCT_VERSION, null)
    }

    /** Walk the top-level object again to find the single array member and its element spans. */
    private fun memberSpans(text: String, objStart: Int, m: Map<*, *>, target: List<*>): List<Triple<Any?, Int, Int>> {
        val p = JsonParser(text, objStart)
        p.expect('{')
        while (true) {
            p.skipWs()
            p.value() // key (string)
            p.expect(':')
            p.skipWs()
            if (p.peek() == '[') {
                val save = p.pos()
                val spans = JsonParser(text, save).arrayWithSpans()
                if (spans.size == target.size && spans.all { it.first is Map<*, *> } && spans.map { jsonCompact(it.first) } == target.map { jsonCompact(it) }) return spans
            }
            p.value()
            p.skipWs()
            if (p.peek() == ',') { p.expect(','); continue }
            break
        }
        return target.map { Triple(it, 0, text.length) } // unreachable in practice; coarse fallback
    }
}
