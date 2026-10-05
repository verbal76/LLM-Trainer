package com.hotatticgames.llmtrainer.extract

import java.security.MessageDigest

/** Block kinds emitted by the extractors. */
object BlockKind {
    const val HEADING = "heading"
    const val PARAGRAPH = "paragraph"
    const val TABLE_ROW = "table_row"
    const val RECORD = "record"
}

/**
 * One addressable unit of extracted content. [charStart]/[charEnd] are offsets into [Extracted.docText]
 * (end exclusive). [page] is 1-based and null when the format has no stable pages (TXT, MD, DOCX, CSV, JSON).
 */
data class RawBlock(
    val text: String,
    val kind: String,
    val page: Int?,
    val sectionPath: List<String>,
    val charStart: Int,
    val charEnd: Int,
)

/** Non-fatal, structured problem for one file. Codes mirror factory/llmtrainer/extract (Python). */
data class IngestIssue(
    val code: String,
    val file: String,
    val detail: String,
    val severity: String = "error", // error: file not (fully) ingested; warning: ingested with a caveat
    val page: Int? = null,
)

/**
 * Extraction result. [docText] is the text stream the block offsets index: the decoded file for text formats
 * (BOM removed), a generated stream for DOCX/PDF (PDF pages are separated by a form feed, as in the Python extractor).
 */
data class Extracted(
    val docText: String,
    val blocks: List<RawBlock>,
    val issues: List<IngestIssue>,
    val needsOcrPages: List<Int>,
    val extractor: String,
    val extractorVersion: String,
    val pageCount: Int?,
)

interface Extractor {
    fun supports(fileName: String, mime: String?): Boolean
    fun extract(fileName: String, bytes: ByteArray): Extracted
}

/** Content hash helper (provenance: source hash). */
object ContentHash {
    fun sha256Hex(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}

internal fun extractedOf(extractor: String, version: String, issues: List<IngestIssue>) =
    Extracted("", emptyList(), issues, emptyList(), extractor, version, null)

internal fun extOf(fileName: String): String {
    val base = fileName.substringAfterLast('/').substringAfterLast('\\')
    val i = base.lastIndexOf('.')
    return if (i < 0) "" else base.substring(i).lowercase()
}

/**
 * Dispatch entry point. [extract] never throws: every failure becomes an [IngestIssue] with empty blocks.
 * Selection is by extension, then MIME type; a `%PDF-` header wins over the name (as in the Python extractor).
 */
object Extractors {
    private val all: List<Extractor> = listOf(
        MarkdownExtractor(), PlainTextExtractor(),
    )

    fun forFile(fileName: String, mime: String?): Extractor? = all.firstOrNull { it.supports(fileName, mime) }

    fun extract(fileName: String, mime: String?, bytes: ByteArray): Extracted {
        val ex: Extractor? = forFile(fileName, mime)
        if (ex == null) {
            return extractedOf("none", "0", listOf(IngestIssue("unsupported_format", fileName,
                "unsupported file type (supported: .txt .md .docx .pdf .csv .tsv .json .jsonl)")))
        }
        if (bytes.all { it.toInt().let { b -> b == 32 || b in 9..13 } }) {
            return extractedOf(ex.javaClass.simpleName, "0", listOf(IngestIssue("empty_file", fileName, "file is empty")))
        }
        return try {
            ex.extract(fileName, bytes)
        } catch (t: Throwable) { // defensive: one bad file must never kill a batch (includes StackOverflow/OOM)
            extractedOf(ex.javaClass.simpleName, "0",
                listOf(IngestIssue("extractor_error", fileName, "${t.javaClass.simpleName}: ${t.message}")))
        }
    }

    private fun looksLikePdf(b: ByteArray): Boolean {
        val lim = minOf(b.size, 1024)
        for (i in 0..lim - 5) if (b[i] == '%'.code.toByte() && b[i + 1] == 'P'.code.toByte() && b[i + 2] == 'D'.code.toByte() &&
            b[i + 3] == 'F'.code.toByte() && b[i + 4] == '-'.code.toByte()) return true
        return false
    }
}
