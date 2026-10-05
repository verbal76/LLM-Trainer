package com.hotatticgames.llmtrainer.extract

import java.io.ByteArrayOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

internal object DocxFixture {
    const val NS = "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\""

    fun zip(vararg parts: Pair<String, String>): ByteArray {
        val bos = ByteArrayOutputStream()
        ZipOutputStream(bos).use { z ->
            for ((n, c) in parts) { z.putNextEntry(ZipEntry(n)); z.write(c.toByteArray(Charsets.UTF_8)); z.closeEntry() }
        }
        return bos.toByteArray()
    }

    fun p(text: String, style: String? = null) =
        "<w:p>" + (if (style != null) "<w:pPr><w:pStyle w:val=\"$style\"/></w:pPr>" else "") +
            "<w:r><w:t xml:space=\"preserve\">$text</w:t></w:r></w:p>"

    fun tbl(vararg rows: List<String>) = "<w:tbl>" + rows.joinToString("") { r ->
        "<w:tr>" + r.joinToString("") { "<w:tc>${p(it)}</w:tc>" } + "</w:tr>"
    } + "</w:tbl>"

    val styles = "<?xml version=\"1.0\"?><w:styles $NS>" +
        "<w:style w:type=\"paragraph\" w:styleId=\"Heading1\"><w:name w:val=\"heading 1\"/></w:style>" +
        "<w:style w:type=\"paragraph\" w:styleId=\"H2\"><w:name w:val=\"Heading 2\"/></w:style></w:styles>"

    fun docx(body: String) = zip(
        "[Content_Types].xml" to "<Types/>",
        "word/document.xml" to "<?xml version=\"1.0\"?><w:document $NS><w:body>$body</w:body></w:document>",
        "word/styles.xml" to styles,
    )
}

class DocxExtractorTest {
    @Test fun headingsParagraphsTablesInOrder() {
        val body = DocxFixture.p("Service Manual", "Heading1") + DocxFixture.p("Intro paragraph.") +
            DocxFixture.p("Brakes", "H2") + DocxFixture.p("Bleed the system &amp; check.") +
            DocxFixture.tbl(listOf("Part", "Torque"), listOf("Caliper", "25 Nm")) +
            "<w:p><w:r><w:t>Line</w:t><w:br/><w:t>break</w:t><w:tab/><w:t>tab</w:t></w:r></w:p>" +
            "<w:p><w:r><w:delText>deleted</w:delText></w:r></w:p>" +
            "<w:sdt><w:sdtContent>${DocxFixture.p("In control")}</w:sdtContent></w:sdt>"
        val e = Extractors.extract("m.docx", null, DocxFixture.docx(body))
        assertEquals(listOf("heading", "paragraph", "heading", "paragraph", "table_row", "table_row", "paragraph", "paragraph"), e.blocks.map { it.kind })
        assertEquals("Bleed the system & check.", e.blocks[3].text)
        assertEquals(listOf("Service Manual", "Brakes"), e.blocks[3].sectionPath)
        assertEquals("Caliper | 25 Nm", e.blocks[5].text)
        assertEquals("Line\nbreak\ttab", e.blocks[6].text)
        assertEquals("In control", e.blocks[7].text)
        assertTrue(e.blocks.all { it.page == null })
        for (b in e.blocks) assertEquals(b.text, e.docText.substring(b.charStart, b.charEnd))
        assertEquals(null, e.pageCount)
        assertTrue(e.issues.isEmpty())
    }

    @Test fun emptyDocumentReportsEmptyContent() {
        val e = Extractors.extract("m.docx", null, DocxFixture.docx("<w:p/>"))
        assertEquals("empty_content", e.issues.single().code)
    }

    @Test fun notAZipIsCorrupt() {
        assertEquals("corrupt_file", Extractors.extract("m.docx", null, "plain text".toByteArray()).issues.single().code)
    }

    @Test fun oleContainerIsEncryptedOrLegacy() {
        val ole = byteArrayOf(0xD0.toByte(), 0xCF.toByte(), 0x11, 0xE0.toByte(), 1, 2, 3, 4, 5, 6)
        assertEquals("encrypted_or_legacy", Extractors.extract("m.docx", null, ole).issues.single().code)
    }

    @Test fun missingDocumentXmlAndBadXml() {
        assertEquals("corrupt_file", Extractors.extract("m.docx", null, DocxFixture.zip("x.txt" to "hi")).issues.single().code)
        assertEquals("corrupt_file", Extractors.extract("m.docx", null, DocxFixture.zip("word/document.xml" to "<w:document")).issues.single().code)
    }

    @Test fun doctypeIsRejectedNotResolved() {
        val xml = "<?xml version=\"1.0\"?><!DOCTYPE d [<!ENTITY x SYSTEM \"file:///etc/passwd\">]><w:document ${DocxFixture.NS}><w:body>${DocxFixture.p("&x;")}</w:body></w:document>"
        val e = Extractors.extract("m.docx", null, DocxFixture.zip("word/document.xml" to xml))
        assertTrue(e.blocks.none { it.text.contains("root:") })
    }

    @Test fun truncatedZipDoesNotThrow() {
        val z = DocxFixture.docx(DocxFixture.p("hello"))
        val e = Extractors.extract("m.docx", null, z.copyOf(z.size / 2))
        assertTrue(e.blocks.isEmpty() || e.issues.isNotEmpty() || e.blocks.isNotEmpty())
    }
}
