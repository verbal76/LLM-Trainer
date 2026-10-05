package com.hotatticgames.llmtrainer.extract

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TextExtractorsTest {
    private fun run(name: String, bytes: ByteArray) = Extractors.extract(name, null, bytes)
    private fun run(name: String, s: String) = run(name, s.toByteArray(Charsets.UTF_8))

    @Test fun plainTextParagraphsAndOffsetsAreExact() {
        val text = "First para line one.\nline two.\n\nSecond para.\n"
        val e = run("a.txt", text)
        assertEquals(2, e.blocks.size)
        assertEquals(text, e.docText)
        for (b in e.blocks) assertEquals(b.text, e.docText.substring(b.charStart, b.charEnd))
        assertEquals("First para line one.\nline two.", e.blocks[0].text)
        assertTrue(e.issues.isEmpty())
        assertEquals(null, e.blocks[0].page)
    }

    @Test fun headingsHeuristicAndSectionPath() {
        val e = run("a.txt", "ENGINE OVERHAUL\n\n1.1 Valve Clearance\n\nCheck the clearance with a gauge.\n\nTorque values follow here.\n")
        val kinds = e.blocks.map { it.kind }
        assertEquals(listOf("heading", "heading", "paragraph", "paragraph"), kinds)
        assertEquals(listOf("ENGINE OVERHAUL", "1.1 Valve Clearance"), e.blocks[2].sectionPath)
    }

    @Test fun sentenceLineIsNotAHeading() {
        val e = run("a.txt", "This is a plain sentence.\n\nAnother one here\n\nTORQUE.\n")
        assertTrue(e.blocks.none { it.kind == "heading" })
    }

    @Test fun utf8BomIsStrippedAndOffsetsAreRelativeToDecodedText() {
        val e = run("a.txt", byteArrayOf(0xEF.toByte(), 0xBB.toByte(), 0xBF.toByte()) + "hello world".toByteArray())
        assertEquals("hello world", e.docText)
        assertEquals(0, e.blocks[0].charStart)
    }

    @Test fun utf16WithBomLeAndBe() {
        val le = byteArrayOf(0xFF.toByte(), 0xFE.toByte()) + "Café au lait".toByteArray(Charsets.UTF_16LE)
        val be = byteArrayOf(0xFE.toByte(), 0xFF.toByte()) + "Café au lait".toByteArray(Charsets.UTF_16BE)
        assertEquals("Café au lait", run("a.txt", le).docText)
        assertEquals("Café au lait", run("a.txt", be).docText)
    }

    @Test fun utf16WithoutBomIsSniffed() {
        val e = run("a.txt", "Hello there, plain text.".toByteArray(Charsets.UTF_16LE))
        assertEquals("Hello there, plain text.", e.docText)
    }

    @Test fun latin1FallbackWarns() {
        val e = run("a.txt", "Café crème".toByteArray(Charsets.ISO_8859_1))
        assertEquals("Café crème", e.docText)
        assertEquals(listOf("encoding_fallback"), e.issues.map { it.code })
        assertEquals("warning", e.issues[0].severity)
    }

    @Test fun binaryIsUndecodable() {
        val bin = ByteArray(300) { (it * 7 % 256).toByte() }
        bin[0] = 0
        val e = run("a.txt", bin)
        assertEquals("undecodable_text", e.issues.single().code)
        assertTrue(e.blocks.isEmpty())
    }

    @Test fun emptyAndWhitespaceFile() {
        assertEquals("empty_file", run("a.txt", ByteArray(0)).issues.single().code)
        assertEquals("empty_file", run("a.md", "  \n\n").issues.single().code)
    }

    @Test fun unsupportedExtension() {
        val e = run("a.xyz", "data")
        assertEquals("unsupported_format", e.issues.single().code)
        assertTrue(e.blocks.isEmpty())
        assertEquals(null, Extractors.forFile("a.exe", null))
    }

    @Test fun markdownHeadingsPathsTablesAndFences() {
        val md = "# Manual\n\nIntro text.\n\n## Brakes\n\n| Part | Torque |\n| --- | --- |\n| Caliper | 25 Nm |\n| Pad | 10 Nm |\n\n```\n# not a heading\n```\n\nSetext Title\n------------\n\nBody.\n"
        val e = run("m.md", md)
        val h = e.blocks.filter { it.kind == "heading" }
        assertEquals(listOf("Manual", "Brakes", "Setext Title"), h.map { it.text })
        val rows = e.blocks.filter { it.kind == "table_row" }
        assertEquals(listOf("Part | Torque", "Caliper | 25 Nm", "Pad | 10 Nm"), rows.map { it.text })
        assertEquals(listOf("Manual", "Brakes"), rows[0].sectionPath)
        assertEquals("| Caliper | 25 Nm |", e.docText.substring(rows[1].charStart, rows[1].charEnd))
        val fence = e.blocks.first { it.text.contains("not a heading") }
        assertEquals("paragraph", fence.kind)
        assertEquals(listOf("Manual", "Setext Title"), e.blocks.last().sectionPath)
        for (b in e.blocks.filter { it.kind == "heading" && it.text == "Manual" }) assertEquals("# Manual", e.docText.substring(b.charStart, b.charEnd))
    }

    @Test fun markdownCrlfOffsets() {
        val md = "# T\r\n\r\npara one\r\nstill one\r\n"
        val e = run("m.md", md)
        assertEquals("# T", e.docText.substring(e.blocks[0].charStart, e.blocks[0].charEnd))
        assertEquals("para one\r\nstill one", e.docText.substring(e.blocks[1].charStart, e.blocks[1].charEnd))
    }

    @Test fun sha256Helper() {
        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", ContentHash.sha256Hex("abc".toByteArray()))
    }
}
