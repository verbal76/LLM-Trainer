package com.hotatticgames.llmtrainer.extract

import java.io.ByteArrayOutputStream
import java.util.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class PdfExtractorTest {
    private fun run(b: ByteArray, name: String = "t.pdf") = Extractors.extract(name, null, b)
    private fun texts(e: Extracted) = e.blocks.map { it.text }

    @Test fun basicTextWithOffsetsAndPages() {
        val e = run(Pdfs.text("Hello World", "Second line"))
        assertEquals(listOf("Hello World\nSecond line"), texts(e))
        assertEquals(1, e.pageCount)
        assertEquals(1, e.blocks[0].page)
        assertEquals("pdf_kotlin", e.extractor)
        for (b in e.blocks) assertEquals(b.text, e.docText.substring(b.charStart, b.charEnd))
        assertTrue(e.docText.endsWith("\u000c"))
        assertTrue(e.needsOcrPages.isEmpty())
        assertTrue(e.issues.isEmpty(), e.issues.toString())
    }

    @Test fun flateCompressedContentStream() {
        val b = Pdfs.pages(listOf("BT /F1 12 Tf 72 720 Td (Compressed text here) Tj ET"), compress = true).build()
        assertEquals(listOf("Compressed text here"), texts(run(b)))
    }

    @Test fun tjKerningCreatesWordGapsOnlyWhenLarge() {
        val b = Pdfs.pages(listOf("BT /F1 12 Tf 72 720 Td [(Hel) 20 (lo) -300 (Wor) -40 (ld)] TJ ET")).build()
        assertEquals(listOf("Hello World"), texts(run(b)))
    }

    @Test fun linesAndParagraphsFromTextMatrix() {
        val c = "BT /F1 12 Tf 72 720 Td (Line one) Tj 0 -14 Td (Line two) Tj 0 -40 Td (Next para) Tj ET"
        assertEquals(listOf("Line one\nLine two", "Next para"), texts(run(Pdfs.pages(listOf(c)).build())))
    }

    @Test fun sameBaselineRepositioningInsertsSpaceButContinuedTjDoesNot() {
        val c = "BT /F1 12 Tf 72 720 Td (foo) Tj (bar) Tj 100 0 Td (baz) Tj 1 0 0 1 72 710 Tm (qux) Tj ET"
        assertEquals(listOf("foobar baz\nqux"), texts(run(Pdfs.pages(listOf(c)).build())))
    }

    @Test fun quoteOperatorsAndTStarBreakLines() {
        val c = "BT /F1 12 Tf 14 TL 72 720 Td (one) Tj T* (two) Tj (three) ' ET"
        assertEquals(listOf("one\ntwo\nthree"), texts(run(Pdfs.pages(listOf(c)).build())))
    }

    @Test fun stringEscapesAndHexStrings() {
        val c = "BT /F1 12 Tf 72 720 Td (a\\(b\\)c\\\\ \\101\\102 x\\\ny) Tj <48656C6C6F> Tj ET"
        assertEquals(listOf("a(b)c\\ AB xyHello"), texts(run(Pdfs.pages(listOf(c)).build())))
    }

    @Test fun winAnsiAndMacRomanAndDifferences() {
        val fonts = listOf(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman /Encoding /MacRomanEncoding >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman /Encoding << /Type /Encoding /BaseEncoding /WinAnsiEncoding /Differences [65 /fi /Euro 90 /uni00E9 /g99] >> >>",
        )
        val c = "BT /F1 12 Tf 72 720 Td (caf\\351 \\223q\\224) Tj /F2 12 Tf 0 -20 Td (caf\\216) Tj /F3 12 Tf 0 -20 Td (AB Z) Tj ET"
        val e = run(Pdfs.pages(listOf(c), extraFonts = fonts).build())
        // F1 is WinAnsi: \351 = e-acute, \223/\224 = curly quotes. F2 MacRoman: \216 = e-acute. F3: fi ligature expands, Euro, uni00E9.
        assertEquals("café “q”\ncafé\nfi€ é", texts(run(Pdfs.pages(listOf(c), extraFonts = fonts).build())).joinToString("\n").replace("\n\n", "\n"))
        assertTrue(e.issues.none { it.code == "undecodable_glyphs" })
    }

    @Test fun macRomanTableHas128Entries() {
        assertEquals('é'.toString(), Encodings.macRoman[0x8E])
        assertEquals("ß", Encodings.macRoman[0xA7])
        assertEquals("…", Encodings.macRoman[0xC9])
    }

    @Test fun type0IdentityFontWithToUnicodeCMap() {
        val cmap = """/CIDInit /ProcSet findresource begin 12 dict begin begincmap
            |1 begincodespacerange <0000> <FFFF> endcodespacerange
            |2 beginbfchar <0001> <0048> <0002> <0069> endbfchar
            |2 beginbfrange <0010> <0012> <0061> <0020> <0021> [<0059> <006F0075>] endbfrange
            |endcmap end end""".trimMargin()
        val b = PdfBuilder()
        Pdfs.pages(listOf("BT /F2 12 Tf 72 720 Td <0001000200100011001200200021> Tj ET"), extraFonts = listOf(
            "<< /Type /Font /Subtype /Type0 /BaseFont /ABCDEF+X /Encoding /Identity-H /DescendantFonts [<< /Type /Font /Subtype /CIDFontType2 /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> >>] /ToUnicode 30 0 R >>"), b = b)
        b.stream(30, "", latin1(cmap))
        val e = run(b.build())
        assertEquals(listOf("HiabcYou"), texts(e))
        assertTrue(e.issues.isEmpty(), e.issues.toString())
    }

    @Test fun type0WithoutToUnicodeIsReportedNotGuessed() {
        val b = Pdfs.pages(listOf("BT /F2 12 Tf 72 720 Td <00010002> Tj ET"), extraFonts = listOf(
            "<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /DescendantFonts [<< /Subtype /CIDFontType2 >>] >>")).build()
        val e = run(b)
        assertTrue(e.blocks.isEmpty())
        assertEquals(listOf(1), e.needsOcrPages)
        assertTrue(e.issues.any { it.code == "undecodable_glyphs" && it.page == 1 })
        assertTrue(e.issues.any { it.code == "needs_ocr" && it.page == 1 && it.detail.contains("cannot be decoded") })
    }

    @Test fun multiPageOrderingNestedPageTreeAndInheritedResources() {
        val b = PdfBuilder()
        b.obj(1, "<< /Type /Catalog /Pages 2 0 R >>")
        b.obj(2, "<< /Type /Pages /Kids [5 0 R 12 0 R] /Count 3 >>")
        b.obj(3, Pdfs.HELV)
        b.obj(5, "<< /Type /Pages /Parent 2 0 R /Kids [10 0 R 11 0 R] /Count 2 /Resources << /Font << /F1 3 0 R >> >> >>")
        b.obj(10, "<< /Type /Page /Parent 5 0 R /Contents 100 0 R >>")
        b.obj(11, "<< /Type /Page /Parent 5 0 R /Contents [101 0 R 102 0 R] >>")
        b.obj(12, "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /Contents 103 0 R >>")
        b.stream(100, "", latin1("BT /F1 12 Tf 72 720 Td (Page one) Tj ET"))
        b.stream(101, "", latin1("BT /F1 12 Tf 72 720 Td (Page"))
        b.stream(102, "", latin1(" two) Tj ET"))
        b.stream(103, "", latin1("BT /F1 12 Tf 72 720 Td (Page three) Tj ET"))
        val e = run(b.build())
        assertEquals(listOf("Page one", "Page two", "Page three"), texts(e))
        assertEquals(listOf(1, 2, 3), e.blocks.map { it.page })
        assertEquals(3, e.pageCount)
        assertEquals(2, e.docText.count { it == '\u000c' } - 1)
        for (bl in e.blocks) assertEquals(bl.text, e.docText.substring(bl.charStart, bl.charEnd))
    }

    @Test fun formXObjectTextIsExtractedAndSelfReferenceTerminates() {
        val b = PdfBuilder()
        Pdfs.pages(listOf("q 1 0 0 1 0 0 cm /Fm0 Do Q"), b = b)
        b.obj(10, "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> /XObject << /Fm0 40 0 R >> >> /Contents 100 0 R >>")
        b.stream(40, "/Type /XObject /Subtype /Form /BBox [0 0 100 100] /Resources << /Font << /F1 3 0 R >> /XObject << /Fm0 40 0 R >> >>",
            latin1("BT /F1 12 Tf 10 50 Td (Inside form) Tj ET /Fm0 Do"))
        assertEquals(listOf("Inside form"), texts(run(b.build())))
    }

    @Test fun asciiHexAscii85AndRunLengthFilters() {
        val plain = "BT /F1 12 Tf 72 720 Td (Filtered) Tj ET"
        val hex = plain.toByteArray().joinToString("") { "%02X".format(it) } + ">"
        val b = Pdfs.pages(listOf(plain)).stream(100, "/Filter /ASCIIHexDecode", latin1(hex)).build()
        assertEquals(listOf("Filtered"), texts(run(b)))
        // ASCII85 of "BT /F1 12 Tf 72 720 Td (Filtered) Tj ET" produced by a tiny encoder
        val b2 = Pdfs.pages(listOf(plain)).stream(100, "/Filter [/ASCII85Decode]", latin1(a85(plain.toByteArray()))).build()
        assertEquals(listOf("Filtered"), texts(run(b2)))
    }

    private fun a85(d: ByteArray): String {
        val sb = StringBuilder("<~")
        var i = 0
        while (i < d.size) {
            val n = minOf(4, d.size - i)
            var v = 0L
            for (k in 0 until 4) v = (v shl 8) or (if (k < n) (d[i + k].toLong() and 0xFF) else 0L)
            val c = CharArray(5)
            for (k in 4 downTo 0) { c[k] = ('!'.code + (v % 85).toInt()).toChar(); v /= 85 }
            sb.append(String(c, 0, n + 1))
            i += 4
        }
        return sb.append("~>").toString()
    }

    @Test fun lzwSpecExample() {
        val enc = byteArrayOf(0x80.toByte(), 0x0B, 0x60, 0x50, 0x22, 0x0C, 0x0C, 0x85.toByte(), 0x01)
        assertEquals(listOf<Byte>(45, 45, 45, 45, 45, 65, 45, 45, 45, 66), PdfFilters.lzw(enc, 1).toList())
    }

    @Test fun encryptedIsReportedAndNotAttempted() {
        val b = Pdfs.pages(listOf("BT /F1 12 Tf 72 720 Td (secret) Tj ET")).obj(50, "<< /Filter /Standard /V 4 /R 4 /O (x) /U (y) /P -4 >>")
            .build(trailerExtra = "/Encrypt 50 0 R")
        val e = run(b)
        assertEquals(listOf("encrypted"), e.issues.map { it.code })
        assertTrue(e.blocks.isEmpty())
    }

    @Test fun scannedPageWithoutTextOperatorsNeedsOcr() {
        val b = PdfBuilder()
        Pdfs.pages(listOf("q 612 0 0 792 0 0 cm /Im0 Do Q"), b = b)
        b.obj(10, "<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Im0 41 0 R >> >> /Contents 100 0 R >>")
        b.stream(41, "/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode", byteArrayOf(1, 2, 3))
        val e = run(b.build())
        assertTrue(e.blocks.isEmpty())
        assertEquals(listOf(1), e.needsOcrPages)
        assertEquals(1, e.pageCount)
        val codes = e.issues.map { it.code to it.severity }
        assertTrue(("needs_ocr" to "warning") in codes)
        assertTrue(("needs_ocr" to "error") in codes) // whole-file summary, mirrors the Python extractor
    }

    @Test fun mixedDocumentKeepsTextPagesAndFlagsScannedOnes() {
        val b = Pdfs.pages(listOf("BT /F1 12 Tf 72 720 Td (Real text on page one) Tj ET", "q Q", "BT /F1 12 Tf 72 720 Td (Page three text) Tj ET")).build()
        val e = run(b)
        assertEquals(listOf(1, 3), e.blocks.map { it.page })
        assertEquals(listOf(2), e.needsOcrPages)
        assertEquals(3, e.pageCount)
        assertEquals("Real text on page one\n\u000c\u000cPage three text\n\u000c", e.docText)
        for (bl in e.blocks) assertEquals(bl.text, e.docText.substring(bl.charStart, bl.charEnd))
    }

    @Test fun perPageFailureIsAnIssueAndExtractionContinues() {
        val b = Pdfs.pages(listOf("x", "BT /F1 12 Tf 72 720 Td (Survivor page) Tj ET"))
        b.stream(100, "/Filter /DCTDecode", byteArrayOf(9, 9))
        val e = run(b.build())
        assertEquals(listOf("Survivor page"), texts(e))
        assertEquals(2, e.blocks[0].page)
        assertTrue(e.issues.any { it.code == "page_extract_failed" && it.page == 1 })
        assertEquals(listOf(1), e.needsOcrPages)
    }

    @Test fun missingXrefIsRebuiltByScanning() {
        val b = Pdfs.pages(listOf("BT /F1 12 Tf 72 720 Td (Recovered) Tj ET"), compress = true).build(withXref = false)
        val e = run(b)
        assertEquals(listOf("Recovered"), texts(e))
        assertTrue(e.issues.any { it.code == "xref_rebuilt" && it.severity == "warning" })
    }

    @Test fun wrongStartxrefAndBadLengthAreSurvived() {
        val good = String(Pdfs.pages(listOf("BT /F1 12 Tf 72 720 Td (Resilient) Tj ET")).build(), Charsets.ISO_8859_1)
        val bad = good.replace(Regex("startxref\\n\\d+"), "startxref\n7").replace("/Length 39", "/Length 9999")
        assertEquals(listOf("Resilient"), texts(run(latin1(bad))))
    }

    @Test fun garbageBeforeHeaderStillWorks() {
        val b = Pdfs.pages(listOf("BT /F1 12 Tf 72 720 Td (Offset junk) Tj ET")).build()
        assertEquals(listOf("Offset junk"), texts(run(latin1("JUNKJUNK\n") + b)))
    }

    @Test fun incrementalUpdateNewestObjectWins() {
        val base = Pdfs.pages(listOf("BT /F1 12 Tf 72 720 Td (Old text) Tj ET")).build()
        val prevStart = String(base, Charsets.ISO_8859_1).substringAfterLast("startxref\n").substringBefore("\n").toInt()
        val bos = ByteArrayOutputStream()
        bos.write(base)
        val newStream = "BT /F1 12 Tf 72 720 Td (New text) Tj ET"
        val off = bos.size()
        bos.write(latin1("100 0 obj\n<< /Length ${newStream.length} >>\nstream\n$newStream\nendstream\nendobj\n"))
        val xr = bos.size()
        bos.write(latin1("xref\n100 1\n${"%010d".format(off)} 00000 n \ntrailer\n<< /Size 101 /Root 1 0 R /Prev $prevStart >>\nstartxref\n$xr\n%%EOF\n"))
        val e = run(bos.toByteArray())
        assertEquals(listOf("New text"), texts(e))
        assertTrue(e.issues.isEmpty(), e.issues.toString())
    }

    @Test fun xrefStreamAndObjectStream() {
        val bodies = listOf(
            1 to "<< /Type /Catalog /Pages 2 0 R >>",
            2 to "<< /Type /Pages /Kids [10 0 R] /Count 1 >>",
            3 to Pdfs.HELV,
            10 to "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /Contents 11 0 R >>",
        )
        val offs = ArrayList<Int>()
        val body = StringBuilder()
        for ((_, s) in bodies) { offs.add(body.length); body.append(s).append('\n') }
        val header = bodies.indices.joinToString("") { "${bodies[it].first} ${offs[it]} " }
        val first = header.length
        val objStm = latin1(header + body)
        val out = ByteArrayOutputStream()
        out.write(latin1("%PDF-1.5\n"))
        val o11 = out.size()
        val content = latin1("BT /F1 12 Tf 72 720 Td (From object stream) Tj ET")
        val z = deflate(content)
        out.write(latin1("11 0 obj\n<< /Length ${z.size} /Filter /FlateDecode >>\nstream\n")); out.write(z); out.write(latin1("\nendstream\nendobj\n"))
        val o20 = out.size()
        val zs = deflate(objStm)
        out.write(latin1("20 0 obj\n<< /Type /ObjStm /N ${bodies.size} /First $first /Length ${zs.size} /Filter /FlateDecode >>\nstream\n")); out.write(zs); out.write(latin1("\nendstream\nendobj\n"))
        val o21 = out.size()
        // rows: type(1) offset/stm(3) gen/idx(1), PNG "Up" predictor
        fun row(t: Int, a: Int, c: Int) = byteArrayOf(t.toByte(), (a shr 16).toByte(), (a shr 8).toByte(), a.toByte(), c.toByte())
        val rows = ArrayList<ByteArray>()
        for (n in 0..21) rows.add(when (n) {
            1 -> row(2, 20, 0); 2 -> row(2, 20, 1); 3 -> row(2, 20, 2); 10 -> row(2, 20, 3)
            11 -> row(1, o11, 0); 20 -> row(1, o20, 0); 21 -> row(1, o21, 0)
            else -> row(0, 0, 0)
        })
        val enc = ByteArrayOutputStream()
        var prev = ByteArray(5)
        for (r in rows) { enc.write(2); enc.write(ByteArray(5) { (r[it] - prev[it]).toByte() }); prev = r }
        val zx = deflate(enc.toByteArray())
        out.write(latin1("21 0 obj\n<< /Type /XRef /Size 22 /W [1 3 1] /Root 1 0 R /Filter /FlateDecode /DecodeParms << /Predictor 12 /Columns 5 >> /Length ${zx.size} >>\nstream\n"))
        out.write(zx); out.write(latin1("\nendstream\nendobj\nstartxref\n$o21\n%%EOF\n"))
        val e = run(out.toByteArray())
        assertEquals(listOf("From object stream"), texts(e))
        assertTrue(e.issues.isEmpty(), e.issues.toString())
        // damaged startxref: rebuild must also find objects inside the object stream
        val damaged = String(out.toByteArray(), Charsets.ISO_8859_1).replace("startxref\n$o21", "startxref\n3")
        val e2 = run(latin1(damaged))
        assertEquals(listOf("From object stream"), texts(e2))
        assertTrue(e2.issues.any { it.code == "xref_rebuilt" })
    }

    @Test fun corruptAndNonPdfInput() {
        assertEquals("corrupt_file", run(latin1("%PDF-1.4\nthis is not really a pdf at all\n")).issues.single().code)
        assertEquals("corrupt_file", run(latin1("hello, definitely not a pdf, long enough text")).issues.single().code)
        assertEquals("empty_file", run(ByteArray(0)).issues.single().code)
        val e = run(latin1("%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"))
        assertEquals("corrupt_file", e.issues.single().code)
    }

    @Test fun pageTreeCycleTerminates() {
        val b = PdfBuilder()
        b.obj(1, "<< /Type /Catalog /Pages 2 0 R >>")
        b.obj(2, "<< /Type /Pages /Kids [2 0 R 5 0 R 2 0 R] /Count 1 >>")
        b.obj(3, Pdfs.HELV)
        b.obj(5, "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> /Contents 6 0 R >>")
        b.stream(6, "", latin1("BT /F1 12 Tf 72 720 Td (Cyclic ok) Tj ET"))
        assertEquals(listOf("Cyclic ok"), texts(run(b.build())))
    }

    @Test fun truncationAndByteFlipFuzzNeverThrowsOrHangs() {
        val base = Pdfs.pages(listOf("BT /F1 12 Tf 72 720 Td (Fuzz target text) Tj ET", "BT /F1 12 Tf 72 720 Td [(a) -300 (b)] TJ ET"), compress = true).build()
        val rnd = Random(42)
        val t0 = System.nanoTime()
        var cut = 0
        while (cut < base.size) { Extractors.extract("f.pdf", null, base.copyOf(cut)); cut += 7 }
        repeat(400) {
            val m = base.copyOf()
            repeat(1 + rnd.nextInt(8)) { m[rnd.nextInt(m.size)] = rnd.nextInt(256).toByte() }
            val e = Extractors.extract("f.pdf", null, m)
            assertNotNull(e)
        }
        assertTrue((System.nanoTime() - t0) / 1_000_000_000 < 60, "fuzz run too slow")
    }

    @Test fun unterminatedHugeNestingDoesNotOverflowStack() {
        val deep = "%PDF-1.4\n1 0 obj\n" + "[".repeat(200000) + "\nendobj\ntrailer\n<< /Root 1 0 R >>\n"
        val e = run(latin1(deep))
        assertTrue(e.blocks.isEmpty())
        assertTrue(e.issues.isNotEmpty())
    }

    @Test fun contentWithInlineImageIsSkippedSafely() {
        val c = "BT /F1 12 Tf 72 720 Td (Before) Tj ET q 10 0 0 10 0 0 cm BI /W 1 /H 1 /CS /G /BPC 8 ID \u0001\u0002 (EI) Tj\nEI Q BT /F1 12 Tf 72 600 Td (After) Tj ET"
        assertEquals(listOf("Before", "After"), texts(run(Pdfs.pages(listOf(c)).build())))
    }

    @Test fun fileWithPdfHeaderIsRoutedToPdfEvenWithOtherExtension() {
        val e = Extractors.extract("scan.bin", null, Pdfs.text("Routed by magic"))
        assertEquals(listOf("Routed by magic"), texts(e))
        assertEquals("sha", "sha".also { assertEquals(64, ContentHash.sha256Hex(Pdfs.text("x")).length) })
    }
}
