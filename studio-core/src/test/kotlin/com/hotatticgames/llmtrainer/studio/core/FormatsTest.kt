package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import com.hotatticgames.llmtrainer.extract.IngestIssue as XIssue

/** Real PDF/DOCX bytes through the whole ingestion path (extractor -> clean -> chunk -> persist -> dataset). */
class FormatsTest {
    // ---- tiny PDF writer (classic xref, one content stream per page) --------------------------------------------------------
    private fun pdf(pages: List<List<String>>): ByteArray {
        val objs = LinkedHashMap<Int, String>()
        objs[1] = "<< /Type /Catalog /Pages 2 0 R >>"
        objs[2] = "<< /Type /Pages /Kids [${pages.indices.joinToString(" ") { "${10 + it} 0 R" }}] /Count ${pages.size} >>"
        objs[3] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
        pages.forEachIndexed { i, lines ->
            objs[10 + i] = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents ${100 + i} 0 R >>"
            val stream = "BT /F1 11 Tf 72 740 Td " + lines.joinToString(" 0 -14 Td ") { "(" + it.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") Tj" } + " ET"
            objs[100 + i] = "<< /Length ${stream.length} >>\nstream\n$stream\nendstream"
        }
        val bos = ByteArrayOutputStream()
        bos.write("%PDF-1.4\n".toByteArray())
        val offsets = HashMap<Int, Int>()
        for ((n, b) in objs) { offsets[n] = bos.size(); bos.write("$n 0 obj\n$b\nendobj\n".toByteArray(Charsets.ISO_8859_1)) }
        val xref = bos.size(); val max = objs.keys.max()
        val sb = StringBuilder("xref\n0 ${max + 1}\n0000000000 65535 f \n")
        for (i in 1..max) sb.append(offsets[i]?.let { "%010d 00000 n \n".format(it) } ?: "0000000000 65535 f \n")
        sb.append("trailer\n<< /Size ${max + 1} /Root 1 0 R >>\nstartxref\n$xref\n%%EOF\n")
        bos.write(sb.toString().toByteArray())
        return bos.toByteArray()
    }

    private fun docx(body: String): ByteArray {
        val ns = "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\""
        val styles = "<?xml version=\"1.0\"?><w:styles $ns><w:style w:type=\"paragraph\" w:styleId=\"Heading1\"><w:name w:val=\"heading 1\"/></w:style></w:styles>"
        val bos = ByteArrayOutputStream()
        ZipOutputStream(bos).use { z ->
            for ((n, c) in listOf("[Content_Types].xml" to "<Types/>", "word/document.xml" to "<?xml version=\"1.0\"?><w:document $ns><w:body>$body</w:body></w:document>", "word/styles.xml" to styles)) {
                z.putNextEntry(ZipEntry(n)); z.write(c.toByteArray()); z.closeEntry()
            }
        }
        return bos.toByteArray()
    }
    private fun p(t: String, style: String? = null) = "<w:p>" + (style?.let { "<w:pPr><w:pStyle w:val=\"$it\"/></w:pPr>" } ?: "") + "<w:r><w:t xml:space=\"preserve\">$t</w:t></w:r></w:p>"

    private fun page(n: Int, body: List<String>) = listOf("Acme Service Manual - Page $n") + body + listOf("Confidential - do not copy", "$n")

    @Test fun pdfHeadersFootersPageNumbersAndHyphenationAreCleanedWithProvenance() {
        val bodies = (1..7).map { n ->
            listOf("${n}.1 Brake service", "Inspect the caliper piston seals for wear and replace them when the rub-",
                "ber shows cracking on page $n of this procedure for the machine.", "Always bleed the system after reassembly number $n so that no air remains trapped.")
        }
        val rig = TK.rig(); val s = rig.open()
        val p = s.createProject(NewProject("PDF", "d", "p")).ok().id
        val item = s.ingest(p, listOf(TK.bytesInput("manual.pdf", pdf(bodies.mapIndexed { i, b -> page(i + 1, b) }), "application/pdf")), RightsStatus.OWNER_AUTHORED).ok().items.single()
        assertEquals(IngestStatus.INGESTED, item.status, item.issues.toString())
        assertEquals(7, item.provenance!!.pages)
        assertTrue(item.provenance!!.extractor.startsWith("pdf"), item.provenance!!.extractor)
        val dir = File(rig.dir, "projects/${p.value}/sources/${item.sourceId}")
        val prov = JSONObject(File(dir, "provenance.json").readText())
        val names = prov.getJSONObject("cleaning").getJSONArray("transformations").let { a -> (0 until a.length()).map { a.getJSONObject(it).getString("name") } }
        assertTrue("remove_repeated_header_footer" in names, names.toString())
        assertTrue("repair_hyphenation" in names, names.toString())
        val all = File(dir, "blocks.jsonl").readText()
        assertFalse(all.contains("Acme Service Manual"), "header must be removed")
        assertFalse(all.contains("Confidential"), "footer must be removed")
        assertTrue(all.contains("rubber shows cracking"), "hyphenation repaired")
        // raw PDF kept byte-identical
        assertEquals(Hashing.sha256(File(dir, "raw/manual.pdf").readBytes()), item.provenance!!.sha256)
    }

    @Test fun scannedAndEncryptedAndCorruptPdfsAreReportedNotIngested() {
        val s = TK.rig().open()
        val p = s.createProject(NewProject("PDF", "d", "p")).ok().id
        val scanned = pdf(listOf(listOf("text"))).let { String(it, Charsets.ISO_8859_1).replace("BT /F1 11 Tf 72 740 Td (text) Tj ET", "q 100 0 0 100 0 0 cm /Im0 Do Q                  ").toByteArray(Charsets.ISO_8859_1) }
        val encrypted = String(pdf(listOf(listOf("secret text"))), Charsets.ISO_8859_1).replace("/Root 1 0 R", "/Root 1 0 R /Encrypt 50 0 R").toByteArray(Charsets.ISO_8859_1)
        val rep = s.ingest(p, listOf(TK.bytesInput("scan.pdf", scanned, "application/pdf"), TK.bytesInput("enc.pdf", encrypted, "application/pdf"),
            TK.bytesInput("junk.pdf", "%PDF-1.4\nnot really a pdf at all".toByteArray(), "application/pdf")), RightsStatus.OWNER_AUTHORED).ok()
        assertEquals(setOf(IngestStatus.FAILED), rep.items.map { it.status }.toSet())
        assertTrue(rep.items[0].issues.any { it.code == IssueCode.NEEDS_OCR }, rep.items[0].issues.toString())
        assertTrue(rep.items[1].issues.any { it.code == IssueCode.CORRUPT_FILE && it.message.contains("[encrypted]") }, rep.items[1].issues.toString())
        assertTrue(rep.items[2].issues.any { it.code == IssueCode.CORRUPT_FILE }, rep.items[2].issues.toString())
        assertTrue(s.listSources(p).ok().isEmpty())
    }

    @Test fun docxHeadingsParagraphsAndTablesBecomeSectionedChunksAndReferenceTables() {
        val body = p("Brake Manual", "Heading1") +
            (1..4).joinToString("") { p("Step $it: bleed the front brake circuit until the fluid runs clear and free of any trapped air bubbles, then close the nipple number $it firmly.") } +
            "<w:tbl>" + (1..5).joinToString("") { r -> "<w:tr><w:tc>${p("Bolt $r")}</w:tc><w:tc>${p("${20 + r} Nm")}</w:tc></w:tr>" } + "</w:tbl>"
        val rig = TK.rig(); val s = rig.open()
        val p = s.createProject(NewProject("D", "d", "p")).ok().id
        s.ingest(p, TK.corpus(5), RightsStatus.OWNER_AUTHORED).ok()
        val item = s.ingest(p, listOf(TK.bytesInput("brakes.docx", docx(body), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")), RightsStatus.OWNER_AUTHORED).ok().items.single()
        assertEquals(IngestStatus.INGESTED, item.status, item.issues.toString())
        s.buildDataset(p).ok()
        val mine = s.reviewItems(p, ReviewFilter(sourceId = item.sourceId), 0, 100).ok().items
        assertTrue(mine.any { it.section == "Brake Manual" && it.role != ChunkRole.REFERENCE })
        val table = mine.filter { ReviewFlag.TABLE in it.flags }
        assertEquals(1, table.size); assertEquals(ChunkRole.REFERENCE, table.single().role)
        assertTrue(table.single().excerpt.contains("Bolt 1 | 21 Nm"))
    }

    @Test fun mixedLibraryOfFormatsProducesAValidJobPackage() {
        val rig = TK.rig(); val s = rig.open()
        val p = s.createProject(NewProject("Mixed", "motorcycle service", "mixed formats")).ok().id
        val pdfs = (1..3).map { d -> TK.bytesInput("doc$d.pdf", pdf((1..6).map { n -> page(n, listOf("${n}.1 Section $d$n", "The ${listOf("alpha", "bravo", "charlie")[d - 1]} assembly ${n * 11} must be inspected carefully before every long trip on rough roads, with attention to wear.",
            "Torque the retaining bolt to ${10 + d * n} N-m using a calibrated wrench and then mark it with paint to show it was done.")) }), "application/pdf") }
        s.ingest(p, pdfs + TK.corpus(3) + TK.input("t.csv", "part,torque\n" + (1..10).joinToString("\n") { "p$it,${it + 5}" }, "text/csv") + TK.input("n.json", """[{"q":"What oil?","a":"Use 10W-40 semi synthetic oil for this engine in normal climates."}]""", "application/json"), RightsStatus.OWNER_AUTHORED).ok().also {
            assertEquals(8, it.ingested.size, it.items.map { i -> i.name to i.issues }.toString())
        }
        s.buildDataset(p).ok(); s.approveDataset(p).ok(); TK.verifyLicense(rig, s); s.selectBaseModel(p, TK.MODEL, null).ok(); s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).ok()
        val out = ByteArrayOutputStream(); s.exportTrainingJobPackage(p, out).ok()
        assertEquals(emptyList(), JobValidator.validate(out.toByteArray()))
        val man = JSONObject(Zips.readAll(out.toByteArray().inputStream()).getValue("manifest.json").toString(Charsets.UTF_8))
        assertEquals(8, man.getJSONArray("sources").length())
        val extractors = (0 until 8).map { man.getJSONArray("sources").getJSONObject(it).getString("extractor") }.toSet()
        assertTrue(extractors.size >= 3, extractors.toString())
    }

    @Test fun issueMappingCoversEveryExtractorCode() {
        fun m(code: String, sev: String = "error") = IssueMap.map(XIssue(code, "f", "d", sev)).code
        assertEquals(IssueCode.NEEDS_OCR, m("needs_ocr", "warning"))
        assertEquals(IssueCode.CORRUPT_FILE, m("encrypted"))
        assertEquals(IssueCode.CORRUPT_FILE, m("corrupt_file"))
        assertEquals(IssueCode.CORRUPT_FILE, m("extractor_error"))
        assertEquals(IssueCode.CORRUPT_FILE, m("corrupt_structured"))
        assertEquals(IssueCode.LOW_TEXT_QUALITY, m("undecodable_glyphs", "warning"))
        assertEquals(IssueCode.LOW_TEXT_QUALITY, m("page_extract_failed", "warning"))
        assertEquals(IssueCode.LOW_TEXT_QUALITY, m("encoding_fallback", "warning"))
        assertEquals(IssueCode.LOW_TEXT_QUALITY, m("ragged_rows", "warning"))
        assertEquals(IssueCode.EMPTY_TEXT, m("empty_file"))
        assertEquals(IssueCode.EMPTY_TEXT, m("empty_content"))
        assertEquals(IssueCode.UNSUPPORTED_TYPE, m("unsupported_format"))
        assertEquals(IssueCode.CORRUPT_FILE, m("something_new"))                          // unknown errors fail closed
        assertEquals(IssueCode.LOW_TEXT_QUALITY, m("something_new", "warning"))
    }
}
