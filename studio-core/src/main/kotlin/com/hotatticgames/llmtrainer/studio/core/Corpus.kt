package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.extract.BlockKind
import com.hotatticgames.llmtrainer.extract.Extracted
import com.hotatticgames.llmtrainer.extract.RawBlock

/** A cleaned block. Offsets still index the extraction stream the extractor produced. */
data class CBlock(
    val kind: String, val text: String, val page: Int?, val sectionPath: List<String>,
    val charStart: Int, val charEnd: Int, val transforms: List<String> = emptyList(),
) {
    val isTabular: Boolean get() = kind == BlockKind.TABLE_ROW || kind == BlockKind.RECORD
}

class CleanReport(val transformations: List<Map<String, Any?>>, val removedNoise: List<Map<String, Any?>>)

/**
 * Corpus cleaning, ported from `factory/llmtrainer/extract/clean.py` onto the Kotlin extractor's paragraph-level blocks.
 * Every transformation is counted and returned for the provenance record; the original file is never touched.
 * Not ported (deferred): heading detection from line layout in PDFs beyond the heuristics below.
 */
object Cleaner {
    const val VERSION = "1"
    private val LIGATURES = linkedMapOf("ﬀ" to "ff", "ﬁ" to "fi", "ﬂ" to "fl", "ﬃ" to "ffi", "ﬄ" to "ffl", "ﬅ" to "st", "ﬆ" to "st")
    private val CTRL = Regex("[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f\\u200b\\u200c\\u200d\\u2060\\ufeff]")
    private val PAGE_NUM = Regex("^(page\\s+)?([0-9]{1,4}|[ivxlc]{1,6})(\\s*(of|/)\\s*[0-9]{1,4})?$", RegexOption.IGNORE_CASE)
    private val NUM_HEADING = Regex("^(\\d{1,2}(\\.\\d{1,2}){0,3})[.)]?\\s+[A-Z][^.!?;]{2,70}$")
    private val LIST_START = Regex("^\\s*([-*•●]|\\d{1,3}[.)])\\s+")
    private val HYPH_END = Regex("[A-Za-z]{2,}-$")
    private val WORDISH = Regex("[A-Za-z]+(?:-[A-Za-z]+)*")
    private const val MAX_SAMPLES = 8

    fun noiseKey(line: String): String =
        line.trim().lowercase().replace(Regex("\\s+"), " ").replace(Regex("\\d+"), "#")

    private fun headingLevel(line: String): Int {
        val s = line.trim()
        if (s.isEmpty() || s.length > 80 || s.endsWith(".") || s.endsWith(",") || s.endsWith(";") || s.endsWith(":")) return 0
        NUM_HEADING.find(s)?.let { return it.groupValues[1].count { c -> c == '.' } + 1 }
        val letters = s.filter { it.isLetter() }
        if (letters.length >= 4 && s.split(Regex("\\s+")).size <= 8 && letters.all { it.isUpperCase() }) return 1
        return 0
    }

    private class Ln(val block: Int, val idx: Int, var text: String)

    fun clean(ex: Extracted): Pair<List<CBlock>, CleanReport> {
        val raw = ex.blocks
        var lig = 0; var ctrl = 0; var nbsp = 0; var spaces = 0
        fun norm(s0: String): String {
            var s = s0
            for ((k, v) in LIGATURES) if (s.contains(k)) { lig += s.split(k).size - 1; s = s.replace(k, v) }
            val c = CTRL.findAll(s).count()
            if (c > 0) { ctrl += c; s = CTRL.replace(s, "") }
            if (s.contains(' ')) { nbsp += s.count { it == ' ' }; s = s.replace(' ', ' ') }
            val collapsed = s.replace(Regex("[ \\t]{2,}"), " ")
            if (collapsed != s) spaces++
            return collapsed.trim()
        }

        // ---- repeated header/footer detection over PAGE-addressed text blocks
        val lines = raw.mapIndexed { bi, b ->
            if (b.kind == BlockKind.PARAGRAPH || b.kind == BlockKind.HEADING) b.text.split("\n").mapIndexed { li, t -> Ln(bi, li, t) } else emptyList()
        }
        val byPage = java.util.TreeMap<Int, MutableList<Ln>>()
        raw.forEachIndexed { bi, b -> val pg = b.page; if (pg != null) byPage.getOrPut(pg) { ArrayList() }.addAll(lines[bi]) }
        val pagesWithText = byPage.filterValues { it.isNotEmpty() }
        val pagesTotal = ex.pageCount ?: pagesWithText.size
        val noiseKeys = HashSet<String>()
        var need = 0
        val edge = HashSet<Ln>()
        if (pagesWithText.size >= 3) {
            need = maxOf(3, Math.ceil(0.5 * pagesWithText.size).toInt())
            val seen = HashMap<String, MutableSet<Int>>()
            for ((p, ls) in pagesWithText) {
                val e = ls.take(2) + ls.takeLast(2)
                edge.addAll(e)
                for (l in e) if (l.text.isNotBlank()) seen.getOrPut(noiseKey(l.text)) { HashSet() }.add(p)
            }
            for ((k, ps) in seen) if (ps.size >= need) noiseKeys.add(k)
        }
        val vocab = HashMap<String, Int>()
        for (ls in lines) for (l in ls) for (m in WORDISH.findAll(l.text)) vocab.merge(m.value.lowercase(), 1, Int::plus)

        val removed = LinkedHashMap<String, Int>()
        val removedPages = HashMap<String, MutableSet<Int>>()
        var pageNumRemoved = 0
        val hyphFixed = ArrayList<String>()
        var hyphKept = 0
        var headingLines = 0
        var reflowTotal = 0
        val out = ArrayList<CBlock>()
        val applyNoise = pagesTotal >= 3

        raw.forEachIndexed { bi, b ->
            when (b.kind) {
                BlockKind.PARAGRAPH, BlockKind.HEADING -> {
                    if (b.kind == BlockKind.HEADING) {
                        val t = norm(b.text.replace('\n', ' '))
                        if (t.isNotEmpty()) out.add(CBlock(BlockKind.HEADING, t, b.page, b.sectionPath, b.charStart, b.charEnd))
                        return@forEachIndexed
                    }
                    val kept = ArrayList<Ln>()
                    for (ln in lines[bi]) {
                        val pg = b.page
                        if (pg != null && applyNoise) {
                            if (noiseKeys.contains(noiseKey(ln.text)) && ln.text.isNotBlank()) {
                                val t = ln.text.trim()
                                removed.merge(t, 1, Int::plus); removedPages.getOrPut(t) { HashSet() }.add(pg)
                                continue
                            }
                            if (ln in edge && PAGE_NUM.matches(ln.text.trim())) { pageNumRemoved++; continue }
                        }
                        kept.add(ln)
                    }
                    if (kept.isEmpty()) return@forEachIndexed
                    // hyphenation repair across line breaks inside the paragraph
                    val merged = ArrayList<String>()
                    for (ln in kept) {
                        val prev = merged.lastOrNull()
                        if (prev != null && HYPH_END.containsMatchIn(prev.trimEnd()) && ln.text.take(1).let { it.isNotEmpty() && it[0].isLowerCase() }) {
                            val stem = prev.trimEnd().dropLast(1)
                            val tail = WORDISH.matchAt(ln.text, 0)
                            val left = Regex("[A-Za-z]+(?:-[A-Za-z]+)*$").find(stem)
                            if (tail != null && left != null) {
                                val hyphenated = (left.value + "-" + tail.value).lowercase()
                                if ((vocab[hyphenated] ?: 0) > 0) { hyphKept++; merged[merged.size - 1] = prev.trimEnd() + ln.text; continue }   // real compound: keep the hyphen, drop the line break
                                hyphFixed.add("${left.value}-|${tail.value}")
                                merged[merged.size - 1] = stem + ln.text
                                continue
                            }
                        }
                        merged.add(ln.text)
                    }
                    // heuristic headings for page-addressed (PDF) text only
                    data class G(val heading: Boolean, val lines: List<String>, val level: Int)
                    val groups = ArrayList<G>()
                    var cur = ArrayList<String>()
                    for (t in merged) {
                        val lvl = if (b.page != null) headingLevel(t) else 0
                        if (lvl > 0) {
                            if (cur.isNotEmpty()) { groups.add(G(false, cur, 0)); cur = ArrayList() }
                            groups.add(G(true, listOf(t), lvl)); headingLines++
                        } else cur.add(t)
                    }
                    if (cur.isNotEmpty()) groups.add(G(false, cur, 0))
                    for (g in groups) {
                        if (g.heading) {
                            val t = norm(g.lines[0])
                            if (t.isNotEmpty()) out.add(CBlock(BlockKind.HEADING, t, b.page, b.sectionPath, b.charStart, b.charEnd, listOf("heading_heuristic:${g.level}")))
                            continue
                        }
                        if (g.lines.any { it.trimStart().startsWith("```") || it.trimStart().startsWith("~~~") }) {
                            out.add(CBlock(BlockKind.PARAGRAPH, g.lines.joinToString("\n") { it.trimEnd() }, b.page, b.sectionPath, b.charStart, b.charEnd, listOf("preformatted")))
                            continue
                        }
                        val parts = ArrayList<String>()
                        var reflowed = 0
                        for (ln in g.lines) {
                            val t = norm(ln)
                            if (t.isEmpty()) continue
                            if (parts.isNotEmpty() && !LIST_START.containsMatchIn(t)) { parts[parts.size - 1] = parts.last() + " " + t; reflowed++ } else parts.add(t)
                        }
                        if (parts.isEmpty()) continue
                        reflowTotal += reflowed
                        out.add(CBlock(BlockKind.PARAGRAPH, parts.joinToString("\n"), b.page, b.sectionPath, b.charStart, b.charEnd, if (reflowed > 0) listOf("reflow") else emptyList()))
                    }
                }
                else -> {
                    val t = b.text.split("\n").joinToString("\n") { norm(it) }
                    if (t.isNotBlank()) out.add(CBlock(b.kind, t, b.page, b.sectionPath, b.charStart, b.charEnd))
                }
            }
        }
        // section paths from heuristic headings (page-addressed text only)
        val finalOut = if (headingLines > 0) {
            val path = ArrayList<Pair<Int, String>>()
            out.map { b ->
                if (b.kind == BlockKind.HEADING && b.transforms.firstOrNull()?.startsWith("heading_heuristic") == true) {
                    val lvl = b.transforms.first().substringAfter(':').toInt()
                    while (path.isNotEmpty() && path.last().first >= lvl) path.removeAt(path.size - 1)
                    path.add(lvl to b.text)
                }
                if (path.isNotEmpty() && b.page != null) b.copy(sectionPath = path.map { it.second }) else b
            }
        } else out

        val tr = ArrayList<Map<String, Any?>>()
        fun add(name: String, count: Int, vararg extra: Pair<String, Any?>) {
            if (count > 0) tr.add(linkedMapOf<String, Any?>("name" to name, "version" to VERSION, "count" to count).also { m -> extra.forEach { m[it.first] = it.second } })
        }
        add("remove_repeated_header_footer", removed.values.sum(), "min_pages" to need, "distinct_lines" to removed.size)
        add("remove_page_number_lines", pageNumRemoved)
        add("repair_hyphenation", hyphFixed.size, "examples" to hyphFixed.take(MAX_SAMPLES), "kept_compound_hyphens" to hyphKept)
        add("heuristic_heading_detection", headingLines)
        add("reflow_soft_wrapped_lines", reflowTotal)
        add("normalize_ligatures", lig)
        add("strip_control_and_zero_width_chars", ctrl)
        add("normalize_nbsp", nbsp)
        add("collapse_whitespace", spaces)
        val noise = removed.entries.sortedWith(compareBy({ -it.value }, { it.key })).take(MAX_SAMPLES).map {
            linkedMapOf<String, Any?>("text" to it.key, "occurrences" to it.value, "pages" to (removedPages[it.key] ?: emptySet<Int>()).sorted())
        }
        return Pair(finalOut, CleanReport(tr, noise))
    }
}

object ChunkRoles { const val TRAIN = "train"; const val REFERENCE = "reference"; const val EXCLUDED = "excluded" }

data class Chunk(
    val id: String, val sectionPath: List<String>, val page: Int?, val charStart: Int, val charEnd: Int,
    val role: String, val kind: String, val excludeReason: String?, val duplicateOf: String?,
    val text: String, val sha256: String, val origin: String, val transforms: List<String>,
) {
    val section: String get() = if (sectionPath.isEmpty()) "Document" else sectionPath.joinToString(" > ")
}

/** Section-aware chunking, ported from `ingest.chunk_blocks`. Chunks never cross a section; tables are reference-only. */
object Chunker {
    const val NAME = "section_paragraph_chunker"
    const val VERSION = "1"
    const val TABLE_ROWS_PER_CHUNK = 25
    const val MIN_LETTER_RATIO = 0.5
    const val CHARS_PER_TOKEN = 4
    private val SENT = Regex("(?<=[.!?])\\s+")

    private fun splitOversize(text: String, maxChars: Int): List<String> {
        val parts = ArrayList<String>()
        var cur = ""
        for (s0 in text.split(SENT)) {
            var sent = s0
            while (sent.length > maxChars) {
                var cut = sent.lastIndexOf(' ', maxChars - 1)
                if (cut <= 0) cut = maxChars
                if (cur.isNotEmpty()) { parts.add(cur); cur = "" }
                parts.add(sent.substring(0, cut).trim())
                sent = sent.substring(cut).trim()
            }
            if (cur.isNotEmpty() && cur.length + 1 + sent.length > maxChars) { parts.add(cur); cur = sent } else cur = "$cur $sent".trim()
        }
        if (cur.isNotEmpty()) parts.add(cur)
        return parts.filter { it.isNotEmpty() }
    }

    private fun letterRatio(t: String): Double {
        var n = 0; var l = 0
        for (c in t) if (!c.isWhitespace()) { n++; if (c.isLetter()) l++ }
        return if (n == 0) 0.0 else l.toDouble() / n
    }

    private fun digest(text: String) = Hashing.sha256(text.replace(Regex("\\s+"), " ").lowercase())

    fun chunk(blocks: List<CBlock>, maxChars: Int = 400 * CHARS_PER_TOKEN, minChars: Int = 20): Pair<List<Chunk>, Map<String, Int>> {
        val chunks = ArrayList<Chunk>()
        val stats = linkedMapOf("dropped_short" to 0, "excluded_low_quality" to 0, "excluded_duplicate" to 0, "split_oversize_paragraphs" to 0)
        val seen = HashMap<String, String>()

        fun emit(text0: String, bl: List<CBlock>, role0: String, kind: String) {
            val text = text0.trim()
            if (text.length < minChars) { stats["dropped_short"] = stats.getValue("dropped_short") + 1; return }
            val id = "c%04d".format(chunks.size + 1)
            var role = role0
            var reason: String? = null
            var dupOf: String? = null
            if (role == ChunkRoles.TRAIN) {
                if (letterRatio(text) < MIN_LETTER_RATIO) {
                    role = ChunkRoles.EXCLUDED; reason = "low_extraction_quality"
                    stats["excluded_low_quality"] = stats.getValue("excluded_low_quality") + 1
                } else {
                    val d = digest(text)
                    val prev = seen[d]
                    if (prev != null) {
                        role = ChunkRoles.EXCLUDED; reason = "duplicate_chunk"; dupOf = prev
                        stats["excluded_duplicate"] = stats.getValue("excluded_duplicate") + 1
                    } else seen[d] = id
                }
            }
            val pages = bl.mapNotNull { it.page }
            chunks.add(Chunk(id, bl[0].sectionPath, pages.minOrNull(), bl.first().charStart, bl.last().charEnd, role, kind, reason, dupOf, text,
                Hashing.sha256(text), if (kind == "table") "table" else "extracted", bl.flatMap { it.transforms }.distinct().sorted()))
        }

        var buf = ArrayList<CBlock>()
        var bufSection: List<String> = emptyList()
        fun flush() {
            if (buf.isEmpty()) return
            val paras = buf; buf = ArrayList()
            var cur = ArrayList<CBlock>()
            var curLen = 0
            for (b in paras) {
                val pieces = if (b.text.length > maxChars) splitOversize(b.text, maxChars) else listOf(b.text)
                if (pieces.size > 1) {
                    stats["split_oversize_paragraphs"] = stats.getValue("split_oversize_paragraphs") + 1
                    if (cur.isNotEmpty()) { emit(cur.joinToString("\n\n") { it.text }, cur, ChunkRoles.TRAIN, "prose"); cur = ArrayList(); curLen = 0 }
                    for (pc in pieces) emit(pc, listOf(b), ChunkRoles.TRAIN, "prose")
                    continue
                }
                if (cur.isNotEmpty() && curLen + b.text.length + 2 > maxChars) { emit(cur.joinToString("\n\n") { it.text }, cur, ChunkRoles.TRAIN, "prose"); cur = ArrayList(); curLen = 0 }
                cur.add(b); curLen += b.text.length + 2
            }
            if (cur.isNotEmpty()) emit(cur.joinToString("\n\n") { it.text }, cur, ChunkRoles.TRAIN, "prose")
        }

        var i = 0
        while (i < blocks.size) {
            val b = blocks[i]
            if (b.isTabular) {
                flush()
                var j = i
                while (j < blocks.size && blocks[j].isTabular && blocks[j].sectionPath == b.sectionPath) j++
                val group = blocks.subList(i, j)
                var k = 0
                while (k < group.size) {
                    val part = group.subList(k, minOf(group.size, k + TABLE_ROWS_PER_CHUNK))
                    emit(part.joinToString("\n") { it.text }, part, ChunkRoles.REFERENCE, "table")
                    k += TABLE_ROWS_PER_CHUNK
                }
                i = j
                continue
            }
            if (b.kind == BlockKind.HEADING) { flush(); bufSection = b.sectionPath; i++; continue }
            if (buf.isNotEmpty() && b.sectionPath != bufSection) flush()
            bufSection = b.sectionPath
            buf.add(b)
            i++
        }
        flush()
        return Pair(chunks, stats)
    }
}

data class Term(val term: String, val tf: Int, val df: Int, val score: Double, val chunks: List<String>)

/** Deterministic domain-term candidates (TF-IDF-like), ported from `terms.py`. */
object Terms {
    private val STOP = ("a about above after again all also an and any are as at be because been before being below between both but by can could did do " +
        "does doing down during each few for from further had has have having he her here hers him his how i if in into is it its just may me more most " +
        "must my no nor not now of off on once only or other our out over own same she should so some such than that the their them then there these they " +
        "this those through to too under until up use used using very was we were what when where which while who whom why will with would you your " +
        "shall per via etc").split(" ").toSet()
    private val TOKEN = Regex("[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*")
    private fun ok(t: String) = t.length >= 3 && t !in STOP

    fun extract(rows: List<Pair<String, String>>, topK: Int = 50, minDf: Int = 1, minTf: Int = 2, maxChunks: Int = 10): List<Term> {
        val tf = HashMap<String, Int>()
        val df = HashMap<String, MutableSet<String>>()
        for ((ref, text) in rows) {
            val toks = TOKEN.findAll(text.lowercase()).map { it.value }.toList()
            val terms = ArrayList<String>()
            for (t in toks) if (ok(t)) terms.add(t)
            for (i in 0 until toks.size - 1) if (ok(toks[i]) && ok(toks[i + 1])) terms.add(toks[i] + " " + toks[i + 1])
            for (t in terms) { tf.merge(t, 1, Int::plus); df.getOrPut(t) { HashSet() }.add(ref) }
        }
        val n = rows.size
        val out = ArrayList<Term>()
        for ((t, c) in tf) {
            val d = df.getValue(t).size
            if (c < minTf || d < minDf || t.replace(" ", "").replace("-", "").all { it.isDigit() }) continue
            val idf = Math.log((1.0 + n) / (1.0 + d)) + 1
            val bonus = if (t.contains(' ')) 1.5 else 1.0
            out.add(Term(t, c, d, Math.round(c * idf * bonus * 1e4) / 1e4, df.getValue(t).sorted().take(maxChunks)))
        }
        out.sortWith(compareBy({ -it.score }, { it.term }))
        return out.take(topK)
    }
}
