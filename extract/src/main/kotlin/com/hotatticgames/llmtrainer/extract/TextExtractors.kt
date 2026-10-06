package com.hotatticgames.llmtrainer.extract

import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.Charset
import java.nio.charset.CodingErrorAction

internal class Decoded(val text: String, val encoding: String, val bomBytes: Int, val issues: List<IngestIssue>)

/**
 * Text decoding with a deliberately small, documented sniff: UTF-8 BOM, UTF-16 LE/BE BOM, BOM-less UTF-16 (NUL pattern),
 * strict UTF-8, then Windows-1252 (latin-1 superset) with a warning. Binary-looking data is `undecodable_text`.
 * Offsets everywhere are relative to the decoded text with the BOM removed.
 */
internal fun decodeText(raw: ByteArray, name: String): Decoded? {
    fun dec(cs: Charset, off: Int) = cs.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(raw, off, raw.size - off)).toString()
    fun b(i: Int) = raw[i].toInt() and 0xFF
    if (raw.size >= 3 && b(0) == 0xEF && b(1) == 0xBB && b(2) == 0xBF) {
        return try { Decoded(dec(Charsets.UTF_8, 3), "utf-8", 3, emptyList()) } catch (e: CharacterCodingException) { null }
    }
    if (raw.size >= 2 && b(0) == 0xFF && b(1) == 0xFE) return utf16(raw, 2, Charsets.UTF_16LE, "utf-16le")
    if (raw.size >= 2 && b(0) == 0xFE && b(1) == 0xFF) return utf16(raw, 2, Charsets.UTF_16BE, "utf-16be")
    val sample = minOf(raw.size, 4096)
    var evenNul = 0
    var oddNul = 0
    for (i in 0 until sample) if (raw[i].toInt() == 0) { if (i % 2 == 0) evenNul++ else oddNul++ }
    if (evenNul + oddNul > 0) {
        val half = sample / 2
        if (oddNul > half / 2 && evenNul == 0) return utf16(raw, 0, Charsets.UTF_16LE, "utf-16le")
        if (evenNul > half / 2 && oddNul == 0) return utf16(raw, 0, Charsets.UTF_16BE, "utf-16be")
        return null // binary
    }
    try {
        return Decoded(dec(Charsets.UTF_8, 0), "utf-8", 0, emptyList())
    } catch (e: CharacterCodingException) { /* fall through */ }
    val cs = try { Charset.forName("windows-1252") } catch (e: Exception) { Charsets.ISO_8859_1 }
    val text = String(raw, cs)
    if (controlRatio(text) > 0.1) return null
    return Decoded(text, cs.name().lowercase(), 0, listOf(IngestIssue("encoding_fallback", name,
        "not valid UTF-8; decoded as ${cs.name()} (single-byte fallback, accents may be wrong if the real encoding differs)", "warning")))
}

private fun utf16(raw: ByteArray, off: Int, cs: Charset, label: String): Decoded? {
    val text = try {
        cs.newDecoder().onMalformedInput(CodingErrorAction.REPORT).onUnmappableCharacter(CodingErrorAction.REPORT)
            .decode(ByteBuffer.wrap(raw, off, raw.size - off)).toString()
    } catch (e: CharacterCodingException) { return null }
    if (controlRatio(text) > 0.1) return null
    return Decoded(text, label, off, emptyList())
}

private fun controlRatio(s: String): Double {
    if (s.isEmpty()) return 0.0
    var bad = 0
    for (c in s) if ((c.code < 32 && c != '\n' && c != '\r' && c != '\t' && c != '\u000c') || c == '￾') bad++
    return bad.toDouble() / s.length
}

internal data class Ln(val body: String, val start: Int, val end: Int)

/** Split on \n, \r\n, \r. Offsets are exact; the terminator is excluded from [Ln.end]. */
internal fun scanLines(text: String): List<Ln> {
    val out = ArrayList<Ln>()
    var i = 0
    val n = text.length
    while (i < n) {
        var j = i
        while (j < n && text[j] != '\n' && text[j] != '\r') j++
        out.add(Ln(text.substring(i, j), i, j))
        i = if (j < n && text[j] == '\r' && j + 1 < n && text[j + 1] == '\n') j + 2 else j + 1
    }
    return out
}

private const val TEXT_VERSION = "1"

private fun undecodable(extractor: String, name: String) = extractedOf(extractor, TEXT_VERSION, listOf(IngestIssue(
    "undecodable_text", name, "not decodable text (binary data or invalid UTF-16); re-encode the file as UTF-8")))

private fun finish(name: String, extractor: String, text: String, blocks: List<RawBlock>, issues: MutableList<IngestIssue>): Extracted {
    if (blocks.isEmpty()) issues.add(IngestIssue("empty_content", name, "no text content"))
    return Extracted(text, blocks, issues, emptyList(), extractor, TEXT_VERSION, null)
}

/**
 * TXT. Encoding sniff (see [decodeText]); paragraphs are blank-line separated runs of lines with exact char offsets.
 * Headings are a heuristic (TXT has no structure): an isolated short line (<= 80 chars, no trailing period) that is
 * ALL CAPS or starts with a section number ("1.2 Title", "Chapter 3 ...", "Appendix A ..."). Expect misses and false positives.
 */
class PlainTextExtractor : Extractor {
    override fun supports(fileName: String, mime: String?) =
        extOf(fileName) in setOf(".txt", ".text") || (extOf(fileName) == "" && mime == "text/plain")

    private val dotted = Regex("""^(\d+(?:\.\d+)*)\.?\s+\S""")
    private val named = Regex("""^(?i:chapter|section|part)\s+\w+[.:]?\s+\S|^(?i:appendix)\s+[A-Z0-9]+[.:]?\s+\S""")

    private fun headingLevel(line: String): Int {
        val t = line.trim()
        if (t.isEmpty() || t.length > 80 || t.endsWith(".") || t.endsWith(",") || t.endsWith(";")) return 0
        if (t.count { it.isLetter() } >= 3 && t.none { it.isLowerCase() }) return 1
        dotted.find(t)?.let { return it.groupValues[1].count { c -> c == '.' } + 1 }
        if (named.containsMatchIn(t)) return 1
        return 0
    }

    override fun extract(fileName: String, bytes: ByteArray): Extracted {
        val d = decodeText(bytes, fileName) ?: return undecodable("plain_text", fileName)
        val lines = scanLines(d.text)
        val blocks = ArrayList<RawBlock>()
        val path = ArrayList<Pair<Int, String>>()
        val buf = ArrayList<Ln>()
        fun spath() = path.map { it.second }
        fun flush() {
            if (buf.isEmpty()) return
            blocks.add(RawBlock(buf.joinToString("\n") { it.body }, BlockKind.PARAGRAPH, null, spath(), buf.first().start, buf.last().end))
            buf.clear()
        }
        for ((i, ln) in lines.withIndex()) {
            if (ln.body.isBlank()) { flush(); continue }
            val isolated = buf.isEmpty() && (i + 1 >= lines.size || lines[i + 1].body.isBlank())
            val lvl = if (isolated) headingLevel(ln.body) else 0
            if (lvl > 0) {
                while (path.isNotEmpty() && path.last().first >= lvl) path.removeAt(path.size - 1)
                path.add(lvl to ln.body.trim())
                blocks.add(RawBlock(ln.body.trim(), BlockKind.HEADING, null, spath(), ln.start, ln.end))
            } else buf.add(ln)
        }
        flush()
        return finish(fileName, "plain_text", d.text, blocks, d.issues.toMutableList())
    }
}

/**
 * Markdown. ATX and setext headings maintain a section path; fenced code stays inside paragraphs; pipe tables become
 * one `table_row` block per row (header row first, cells joined with " | "). All offsets are exact slices of the decoded text.
 */
class MarkdownExtractor : Extractor {
    override fun supports(fileName: String, mime: String?) =
        extOf(fileName) in setOf(".md", ".markdown") || (extOf(fileName) == "" && mime == "text/markdown")

    private val atx = Regex("""^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$""")
    private val fence = Regex("""^ {0,3}(```|~~~)""")
    private val setext = Regex("""^ {0,3}(=+|-+)\s*$""")
    private val pipeSep = Regex("""^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$""")

    private fun splitRow(line: String): List<String> {
        var s = line.trim()
        if (s.startsWith("|")) s = s.substring(1)
        if (s.endsWith("|")) s = s.substring(0, s.length - 1)
        return s.split("|").map { it.trim() }
    }

    override fun extract(fileName: String, bytes: ByteArray): Extracted {
        val d = decodeText(bytes, fileName) ?: return undecodable("markdown", fileName)
        val lines = scanLines(d.text)
        val blocks = ArrayList<RawBlock>()
        val path = ArrayList<Pair<Int, String>>()
        val buf = ArrayList<Ln>()
        var inFence = false
        fun spath() = path.map { it.second }
        fun flush() {
            if (buf.isEmpty()) return
            blocks.add(RawBlock(buf.joinToString("\n") { it.body }, BlockKind.PARAGRAPH, null, spath(), buf.first().start, buf.last().end))
            buf.clear()
        }
        fun heading(level: Int, title: String, s: Int, e: Int) {
            while (path.isNotEmpty() && path.last().first >= level) path.removeAt(path.size - 1)
            path.add(level to title)
            blocks.add(RawBlock(title, BlockKind.HEADING, null, spath(), s, e))
        }
        var i = 0
        while (i < lines.size) {
            val ln = lines[i]
            val body = ln.body
            if (fence.containsMatchIn(body)) { inFence = !inFence; buf.add(ln); i++; continue }
            if (inFence) { buf.add(ln); i++; continue }
            if (body.isBlank()) { flush(); i++; continue }
            val m = atx.matchEntire(body)
            if (m != null && m.groupValues[2].isNotEmpty()) {
                flush(); heading(m.groupValues[1].length, m.groupValues[2], ln.start, ln.end); i++; continue
            }
            val ts = body.trimStart()
            if (i + 1 < lines.size && buf.isEmpty() && setext.matches(lines[i + 1].body) &&
                !ts.startsWith("-") && !ts.startsWith("*") && !ts.startsWith(">") && !ts.startsWith("|")) {
                val nxt = lines[i + 1]
                heading(if (nxt.body.trim().startsWith("=")) 1 else 2, body.trim(), ln.start, nxt.end); i += 2; continue
            }
            if ('|' in body && i + 1 < lines.size && buf.isEmpty() && pipeSep.matches(lines[i + 1].body)) {
                val sp = spath()
                blocks.add(RawBlock(splitRow(body).joinToString(" | "), BlockKind.TABLE_ROW, null, sp, ln.start, ln.end))
                var j = i + 2
                while (j < lines.size && '|' in lines[j].body && lines[j].body.isNotBlank()) {
                    blocks.add(RawBlock(splitRow(lines[j].body).joinToString(" | "), BlockKind.TABLE_ROW, null, sp, lines[j].start, lines[j].end))
                    j++
                }
                i = j; continue
            }
            buf.add(ln); i++
        }
        flush()
        return finish(fileName, "markdown", d.text, blocks, d.issues.toMutableList())
    }
}
