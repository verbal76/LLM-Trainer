package com.hotatticgames.llmtrainer.extract

private const val PDF_VERSION = "1"
private const val MIN_CHARS_FOR_TEXT_PAGE = 5

/**
 * Pure-Kotlin PDF text extractor (no third-party code, no native code, no OCR).
 *
 * Supported: classic xref tables, xref streams, hybrid files, incremental-update chains, object streams, a full-file rebuild scan
 * for damaged/missing xref data, Flate/LZW/ASCII85/ASCIIHex/RunLength with PNG/TIFF predictors, page-tree order and inherited
 * resources, Form XObjects, simple fonts (WinAnsi, MacRoman, Differences with standard Latin glyph names, ToUnicode), Type0/CID
 * fonts through ToUnicode (or Uni*-UCS2/UTF16 encodings), literal/hex strings, TJ-kerning word gaps.
 *
 * Honest limits:
 *  * No layout analysis: text comes out in content-stream order. Multi-column pages interleave, headers/footers are not removed,
 *    and tables are NOT reconstructed (cells arrive as ordinary lines/words). Use the structured formats for tabular data.
 *  * Glyph widths are not read, so spacing is inferred from position changes and TJ kerning (threshold: 180/1000 em); expect
 *    occasional missing or extra spaces. Rotated text is not handled specially. Ligature glyphs are expanded (fi, fl, ffi...).
 *  * Fonts without ToUnicode and without a known encoding (typical for subsetted CID/Type3 fonts) cannot be decoded: those glyphs are
 *    dropped, reported as `undecodable_glyphs`, and a page with nothing left is listed in needsOcrPages.
 *  * Encrypted PDFs (even with an empty user password) are reported as `encrypted`; no decryption is attempted.
 *  * Pages without a text layer are reported as `needs_ocr` and never silently skipped. No OCR is performed.
 *  * Image-only filters (DCT, JPX, CCITT, JBIG2) are not decoded; content streams using them fail per page.
 *  * Limits: 96 MB per decoded stream, 128 MB decoded content per page, 50 000 pages, 3M operators per page.
 *
 * A failing page becomes a `page_extract_failed` issue and extraction continues with the next page.
 */
class PdfExtractor : Extractor {
    override fun supports(fileName: String, mime: String?) =
        extOf(fileName) == ".pdf" || (extOf(fileName) == "" && mime == "application/pdf")

    override fun extract(fileName: String, bytes: ByteArray): Extracted {
        fun fail(code: String, detail: String) = extractedOf("pdf_kotlin", PDF_VERSION, listOf(IngestIssue(code, fileName, detail)))
        val doc = PdfDocument(bytes)
        val loaded = try { doc.load() } catch (e: Exception) { false } catch (e: StackOverflowError) { false }
        if (doc.isEncrypted) return fail("encrypted", "PDF is encrypted (password-protected or permission-restricted); decrypt it first, no decryption is attempted")
        if (!loaded) return fail("corrupt_file", "cannot read PDF: no usable objects or page tree found")
        val pages = try { doc.pages() } catch (e: Exception) { return fail("corrupt_file", "cannot read PDF page tree: ${e.message}") }
            catch (e: StackOverflowError) { return fail("corrupt_file", "cannot read PDF page tree: too deeply nested") }
        if (pages.isEmpty()) return fail("corrupt_file", "PDF has no pages")

        val issues = ArrayList<IngestIssue>()
        for (n in doc.notes) issues.add(IngestIssue(n, fileName, when (n) {
            "xref_rebuilt" -> "cross-reference data missing or damaged; objects were recovered by scanning the file"
            else -> "page tree damaged; pages were recovered by scanning for /Type /Page objects (order = object number)"
        }, "warning"))

        val fonts = FontFactory(doc)
        val sb = StringBuilder()
        val blocks = ArrayList<RawBlock>()
        val ocr = ArrayList<Int>()
        var pos = 0
        for ((idx, page) in pages.withIndex()) {
            val pno = idx + 1
            var lines: List<String> = emptyList()
            val stats = DecodeStats()
            var sawText = false
            try {
                val text = PageText()
                val ip = ContentInterpreter(doc, fonts, text, stats)
                val contents = doc.resolve(page.dict.map["Contents"])
                val streams: List<PStream> = when (contents) {
                    is PStream -> listOf(contents)
                    is PArr -> contents.items.mapNotNull { doc.resolve(it) as? PStream }
                    else -> emptyList()
                }
                val seen = HashSet<PStream>()
                val data = java.io.ByteArrayOutputStream()
                for (s in streams) {
                    val d = doc.decode(s)
                    if (d == null) { issues.add(IngestIssue("page_extract_failed", fileName, "content stream uses an unsupported filter", "warning", pno)); continue }
                    data.write(d); data.write('\n'.code)
                }
                ip.run(data.toByteArray(), page.resources, 0, seen)
                sawText = ip.sawTextOps
                lines = text.finish()
            } catch (e: Exception) {
                issues.add(IngestIssue("page_extract_failed", fileName, "${e.javaClass.simpleName}: ${e.message}", "warning", pno))
            } catch (e: StackOverflowError) {
                issues.add(IngestIssue("page_extract_failed", fileName, "content too deeply nested", "warning", pno))
            }
            val chars = lines.sumOf { l -> l.count { !it.isWhitespace() } }
            if (stats.unmapped > 0) issues.add(IngestIssue("undecodable_glyphs", fileName,
                "${stats.unmapped} of ${stats.total} glyph(s) could not be mapped to Unicode (font without ToUnicode/known encoding) and were dropped", "warning", pno))
            if (chars < MIN_CHARS_FOR_TEXT_PAGE) {
                ocr.add(pno)
                issues.add(IngestIssue("needs_ocr", fileName,
                    if (sawText && stats.unmapped > 0) "page has text operators but its glyphs cannot be decoded; NOT OCRed"
                    else "page has no extractable text layer (scanned/image-only?); NOT OCRed", "warning", pno))
                sb.append('\u000c'); pos += 1
                continue
            }
            var buf = ArrayList<Pair<String, Int>>()
            fun flush() {
                if (buf.isEmpty()) return
                blocks.add(RawBlock(buf.joinToString("\n") { it.first }, BlockKind.PARAGRAPH, pno, emptyList(), buf.first().second, buf.last().second + buf.last().first.length))
                buf = ArrayList()
            }
            for (ln in lines) {
                val start = pos
                pos += ln.length + 1
                if (ln.isNotBlank()) buf.add(ln to start) else flush()
            }
            flush()
            sb.append(lines.joinToString("\n")).append("\n\u000c")
            pos += 1
        }
        if (blocks.isEmpty()) issues.add(IngestIssue("needs_ocr", fileName,
            "none of the ${pages.size} page(s) has extractable text; the file looks scanned. Not OCRed: run an OCR tool and ingest its text output"))
        return Extracted(sb.toString(), blocks, issues, ocr, "pdf_kotlin", PDF_VERSION, pages.size)
    }
}
