package com.hotatticgames.llmtrainer.extract

import org.w3c.dom.Element
import org.w3c.dom.Node
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.zip.ZipException
import java.util.zip.ZipInputStream
import javax.xml.parsers.DocumentBuilderFactory

private const val W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
private const val MAX_XML = 64 * 1024 * 1024
private const val DOCX_VERSION = "1"

/**
 * DOCX via java.util.zip + javax.xml DOM. Headings come from paragraph styles (style name / id "HeadingN", "Title") or an
 * explicit outline level; paragraphs and tables (one `table_row` block per row, cells joined with " | ") are emitted in
 * document order, including content controls. Page numbers are not stored in DOCX, so [RawBlock.page] is null.
 * Not extracted: headers/footers, footnotes/endnotes, text boxes, comments, tracked-deletion text, images.
 * Encrypted (OLE-wrapped) and legacy .doc files are reported as `encrypted_or_legacy`.
 */
class DocxExtractor : Extractor {
    override fun supports(fileName: String, mime: String?) =
        extOf(fileName) == ".docx" ||
            (extOf(fileName) == "" && mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    private fun readParts(raw: ByteArray): Map<String, ByteArray> {
        val out = HashMap<String, ByteArray>()
        var total = 0L
        ZipInputStream(ByteArrayInputStream(raw)).use { zin ->
            while (true) {
                val e = zin.nextEntry ?: break
                if (e.name != "word/document.xml" && e.name != "word/styles.xml") continue
                val bos = ByteArrayOutputStream()
                val buf = ByteArray(16384)
                while (true) {
                    val n = zin.read(buf)
                    if (n < 0) break
                    total += n
                    if (total > MAX_XML) throw ZipException("${e.name} too large (zip bomb guard)")
                    bos.write(buf, 0, n)
                }
                out[e.name] = bos.toByteArray()
            }
        }
        return out
    }

    private fun parse(data: ByteArray): Element {
        val f = DocumentBuilderFactory.newInstance()
        f.isNamespaceAware = true
        for ((feat, v) in listOf("http://apache.org/xml/features/disallow-doctype-decl" to true,
            "http://xml.org/sax/features/external-general-entities" to false,
            "http://xml.org/sax/features/external-parameter-entities" to false)) {
            try { f.setFeature(feat, v) } catch (e: Exception) { /* not supported on this runtime */ }
        }
        try { f.isExpandEntityReferences = false } catch (e: Exception) { }
        val b = f.newDocumentBuilder()
        b.setEntityResolver { _, _ -> org.xml.sax.InputSource(java.io.StringReader("")) }
        return b.parse(ByteArrayInputStream(data)).documentElement
    }

    private fun Element.isW(local: String) = namespaceURI == W_NS && localName == local
    private fun Element.kids(): List<Element> {
        val out = ArrayList<Element>()
        var c: Node? = firstChild
        while (c != null) { if (c is Element) out.add(c); c = c.nextSibling }
        return out
    }
    private fun Element.wAttr(name: String): String = getAttributeNS(W_NS, name).ifEmpty { getAttribute("w:$name") }

    private fun paraText(p: Element): String {
        val sb = StringBuilder()
        fun walk(e: Element) {
            for (k in e.kids()) when {
                k.isW("t") -> sb.append(k.textContent)
                k.isW("tab") -> sb.append('\t')
                k.isW("br") || k.isW("cr") -> sb.append('\n')
                k.isW("noBreakHyphen") -> sb.append('-')
                k.isW("pPr") || k.isW("rPr") -> {}
                else -> walk(k)
            }
        }
        walk(p)
        return sb.toString()
    }

    private fun styleNames(stylesXml: ByteArray?): Map<String, String> {
        if (stylesXml == null) return emptyMap()
        val out = HashMap<String, String>()
        for (st in parse(stylesXml).kids().filter { it.isW("style") }) {
            val nm = st.kids().firstOrNull { it.isW("name") }?.wAttr("val") ?: continue
            out[st.wAttr("styleId")] = nm
        }
        return out
    }

    private fun headingLevel(p: Element, names: Map<String, String>): Int {
        val ppr = p.kids().firstOrNull { it.isW("pPr") } ?: return 0
        val sid = ppr.kids().firstOrNull { it.isW("pStyle") }?.wAttr("val") ?: ""
        for (cand in listOf(names[sid] ?: "", sid)) {
            val m = Regex("""(?i)heading\s*([1-9])""").matchEntire(cand.trim())
            if (m != null) return m.groupValues[1].toInt()
            if (cand.trim().equals("title", true)) return 1
        }
        val ol = ppr.kids().firstOrNull { it.isW("outlineLvl") }?.wAttr("val")?.toIntOrNull()
        if (ol != null && ol in 0..8) return ol + 1
        return 0
    }

    override fun extract(fileName: String, bytes: ByteArray): Extracted {
        fun fail(code: String, detail: String) = extractedOf("docx", DOCX_VERSION, listOf(IngestIssue(code, fileName, detail)))
        val isZip = bytes.size > 4 && bytes[0] == 'P'.code.toByte() && bytes[1] == 'K'.code.toByte()
        if (!isZip) {
            return if (bytes.size >= 4 && (bytes[0].toInt() and 0xFF) == 0xD0 && (bytes[1].toInt() and 0xFF) == 0xCF)
                fail("encrypted_or_legacy", "OLE container: an encrypted .docx or a legacy .doc, not an OOXML zip")
            else fail("corrupt_file", "not a valid zip/OOXML container")
        }
        val parts = try { readParts(bytes) } catch (e: Exception) { return fail("corrupt_file", "cannot read zip: ${e.message}") }
        val body = parts["word/document.xml"] ?: return fail("corrupt_file", "word/document.xml missing")
        val root: Element
        val names: Map<String, String>
        try {
            root = parse(body)
            names = try { styleNames(parts["word/styles.xml"]) } catch (e: Exception) { emptyMap() }
        } catch (e: Exception) { return fail("corrupt_file", "cannot parse document XML: ${e.message}") }
        val bodyEl = root.kids().firstOrNull { it.isW("body") } ?: return fail("corrupt_file", "no w:body")

        val sb = StringBuilder()
        val blocks = ArrayList<RawBlock>()
        val path = ArrayList<Pair<Int, String>>()
        fun spath() = path.map { it.second }
        fun emit(text: String): Pair<Int, Int> { val s = sb.length; sb.append(text).append('\n'); return s to s + text.length }

        fun handle(el: Element, depth: Int) {
            if (depth > 50) return
            when {
                el.isW("p") -> {
                    val text = paraText(el).trim()
                    if (text.isEmpty()) return
                    val lvl = headingLevel(el, names)
                    val (s, e) = emit(text)
                    if (lvl > 0) {
                        while (path.isNotEmpty() && path.last().first >= lvl) path.removeAt(path.size - 1)
                        path.add(lvl to text)
                        blocks.add(RawBlock(text, BlockKind.HEADING, null, spath(), s, e))
                    } else blocks.add(RawBlock(text, BlockKind.PARAGRAPH, null, spath(), s, e))
                }
                el.isW("tbl") -> {
                    fun rows(t: Element): List<Element> = t.kids().flatMap { if (it.isW("tr")) listOf(it) else emptyList() }
                    for (tr in rows(el)) {
                        val cells = tr.kids().filter { it.isW("tc") }.map { tc ->
                            val ps = ArrayList<String>()
                            fun collect(e: Element) { for (k in e.kids()) if (k.isW("p")) paraText(k).trim().takeIf { it.isNotEmpty() }?.let(ps::add) else collect(k) }
                            collect(tc)
                            ps.joinToString(" ")
                        }
                        if (cells.none { it.isNotEmpty() }) continue
                        val text = cells.joinToString(" | ")
                        val (s, e) = emit(text)
                        blocks.add(RawBlock(text, BlockKind.TABLE_ROW, null, spath(), s, e))
                    }
                }
                el.isW("sdt") -> el.kids().firstOrNull { it.isW("sdtContent") }?.kids()?.forEach { handle(it, depth + 1) }
                el.isW("ins") || el.isW("smartTag") || el.isW("customXml") -> el.kids().forEach { handle(it, depth + 1) }
            }
        }
        bodyEl.kids().forEach { handle(it, 0) }
        val issues = ArrayList<IngestIssue>()
        if (blocks.isEmpty()) issues.add(IngestIssue("empty_content", fileName, "document contains no text"))
        return Extracted(sb.toString(), blocks, issues, emptyList(), "docx", DOCX_VERSION, null)
    }
}
