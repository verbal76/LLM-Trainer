package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.extract.Extractors
import com.hotatticgames.llmtrainer.studio.api.IngestIssue
import com.hotatticgames.llmtrainer.studio.api.IngestItem
import com.hotatticgames.llmtrainer.studio.api.IngestReport
import com.hotatticgames.llmtrainer.studio.api.IngestStatus
import com.hotatticgames.llmtrainer.studio.api.IssueCode
import com.hotatticgames.llmtrainer.studio.api.ProjectId
import com.hotatticgames.llmtrainer.studio.api.Progress
import com.hotatticgames.llmtrainer.studio.api.ProvenanceSummary
import com.hotatticgames.llmtrainer.studio.api.RightsStatus
import com.hotatticgames.llmtrainer.studio.api.SourceInput
import com.hotatticgames.llmtrainer.studio.api.SourceRecord
import java.io.File
import java.io.IOException
import com.hotatticgames.llmtrainer.extract.IngestIssue as XIssue

/** One ingested source document as persisted in `sources/<id>/source.json`. */
data class SourceDoc(
    val sourceId: String, val name: String, val mime: String, val sizeBytes: Long, val sha256: String, val ingestedAt: Long,
    val rights: RightsStatus, val extractor: String, val extractorVersion: String, val pages: Int?, val chunkCount: Int,
    val issues: List<IngestIssue>, val needsOcrPages: List<Int>, val rawFile: String, val title: String,
) {
    fun summary() = ProvenanceSummary(sourceId, sha256, mime, sizeBytes, ingestedAt, pages, chunkCount, "$extractor/$extractorVersion", rights)
    fun record() = SourceRecord(sourceId, name, summary(), issues)
    val trainable: Boolean get() = rights == RightsStatus.OWNER_AUTHORED || rights == RightsStatus.LICENSED_FOR_TRAINING || rights == RightsStatus.PERMISSION_GRANTED
}

object IssueMap {
    private val OCR = setOf("needs_ocr", "no_text_layer", "scanned")
    private val UNSUPPORTED = setOf("unsupported_format", "unsupported")
    private val EMPTY = setOf("empty_file", "empty_content", "no_usable_chunks", "no_extractable_content")
    private val CORRUPT = setOf("encrypted", "corrupt_file", "corrupt_structured", "extractor_error", "unreadable", "undecodable_text", "not_found")
    private val QUALITY = setOf("encoding_fallback", "ragged_rows", "undecodable_glyphs", "page_extract_failed")

    /** Extractor issue codes are mapped explicitly; unknown ones fall back by severity. The original code always stays in the message. */
    fun map(x: XIssue): IngestIssue {
        val c = x.code.lowercase()
        val code = when {
            c in OCR || c.contains("ocr") -> IssueCode.NEEDS_OCR
            c in UNSUPPORTED -> IssueCode.UNSUPPORTED_TYPE
            c.contains("too_large") -> IssueCode.TOO_LARGE
            c in EMPTY -> IssueCode.EMPTY_TEXT
            c in QUALITY -> IssueCode.LOW_TEXT_QUALITY
            c in CORRUPT -> IssueCode.CORRUPT_FILE
            x.severity == "warning" -> IssueCode.LOW_TEXT_QUALITY
            else -> IssueCode.CORRUPT_FILE
        }
        val page = if (x.page != null) " (page ${x.page})" else ""
        return IngestIssue(code, "${x.detail}$page [${x.code}]")
    }
}

/** Source ingestion and its on-disk layout (see [SourceStore]). */
class IngestService(private val clock: Clock) {
    companion object {
        const val MAX_SOURCE_BYTES = 50L * 1024 * 1024
        const val STORE_VERSION = 1
    }

    /**
     * Ingests [inputs] into [store]. A bad file yields a FAILED item with issue codes; the batch always completes.
     * [onItem] receives the report-so-far after every file (persisted by the caller so a crash keeps per-file status).
     */
    fun ingest(
        projectId: ProjectId, store: SourceStore, inputs: List<SourceInput>, rights: RightsStatus,
        onProgress: (Progress) -> Unit, onItem: (IngestReport) -> Unit,
    ): IngestReport {
        val items = ArrayList<IngestItem>()
        val seen = HashMap<String, String>()          // sha -> sourceId (existing + this batch)
        store.all().forEach { seen[it.sha256] = it.sourceId }
        inputs.forEachIndexed { i, input ->
            onProgress(Progress(i.toLong(), inputs.size.toLong(), "files"))
            val item = try { ingestOne(store, input, rights, seen) } catch (t: Throwable) {
                if (t is OutOfMemoryError || t is StackOverflowError) IngestItem(input.name, IngestStatus.FAILED, null, null,
                    listOf(IngestIssue(IssueCode.TOO_LARGE, "The file is too large or complex to process on this device")), null)
                else IngestItem(input.name, IngestStatus.FAILED, null, null, listOf(IngestIssue(IssueCode.CORRUPT_FILE, "Unexpected error while ingesting: ${t.javaClass.simpleName}: ${t.message}")), null)
            }
            items.add(item)
            onItem(IngestReport(projectId, items.toList(), clock.nowMs()))
        }
        onProgress(Progress(inputs.size.toLong(), inputs.size.toLong(), "files"))
        return IngestReport(projectId, items.toList(), clock.nowMs())
    }

    private fun fail(name: String, code: IssueCode, msg: String, extra: List<IngestIssue> = emptyList()) =
        IngestItem(name, IngestStatus.FAILED, null, null, listOf(IngestIssue(code, msg)) + extra, null)

    private fun ingestOne(store: SourceStore, input: SourceInput, rights: RightsStatus, seen: MutableMap<String, String>): IngestItem {
        if (input.sizeBytes > MAX_SOURCE_BYTES) return fail(input.name, IssueCode.TOO_LARGE, "File exceeds ${MAX_SOURCE_BYTES / 1024 / 1024} MB")
        val bytes = try { input.open().use { Fs.readLimited(it, MAX_SOURCE_BYTES) } } catch (e: Exception) {
            return fail(input.name, IssueCode.CORRUPT_FILE, "Could not read the file: ${e.message ?: e.javaClass.simpleName}")
        } ?: return fail(input.name, IssueCode.TOO_LARGE, "File exceeds ${MAX_SOURCE_BYTES / 1024 / 1024} MB")
        if (bytes.isEmpty()) return fail(input.name, IssueCode.EMPTY_TEXT, "The file is empty")
        val sha = Hashing.sha256(bytes)
        val sid = "src-" + sha.take(12)
        seen[sha]?.let { dup ->
            return IngestItem(input.name, IngestStatus.DUPLICATE, null, dup, listOf(IngestIssue(IssueCode.DUPLICATE_CONTENT, "Identical content to an existing source ($dup)")), null)
        }
        val ex = Extractors.extract(input.name, input.mime, bytes)
        val issues = ex.issues.map(IssueMap::map).toMutableList()
        if (ex.needsOcrPages.isNotEmpty() && issues.none { it.code == IssueCode.NEEDS_OCR })
            issues.add(IngestIssue(IssueCode.NEEDS_OCR, "Pages without a text layer were skipped (OCR is not available): ${ex.needsOcrPages.take(20).joinToString(", ")}" + if (ex.needsOcrPages.size > 20) ", ..." else ""))
        if (ex.blocks.isEmpty()) {
            if (issues.isEmpty()) issues.add(IngestIssue(IssueCode.EMPTY_TEXT, "No extractable content"))
            return IngestItem(input.name, IngestStatus.FAILED, null, null, issues, null)
        }
        val (cleaned, rep) = Cleaner.clean(ex)
        val (chunks, cstats) = Chunker.chunk(cleaned)
        if (chunks.isEmpty()) {
            issues.add(IngestIssue(IssueCode.EMPTY_TEXT, "Content is too short or unusable after cleaning ($cstats)"))
            return IngestItem(input.name, IngestStatus.FAILED, null, null, issues, null)
        }
        val lowQ = chunks.count { it.excludeReason == "low_extraction_quality" }
        if (lowQ > 0 && lowQ * 2 >= chunks.size) issues.add(IngestIssue(IssueCode.LOW_TEXT_QUALITY, "$lowQ of ${chunks.size} chunks look like extraction noise (mostly non-letters)"))
        else if (lowQ > 0) issues.add(IngestIssue(IssueCode.LOW_TEXT_QUALITY, "$lowQ chunk(s) look like extraction noise and are excluded by default"))
        if (rights == RightsStatus.UNSET) issues.add(IngestIssue(IssueCode.RIGHTS_UNSET, "Set usage rights before building a dataset"))
        val doc = SourceDoc(sid, input.name, input.mime, bytes.size.toLong(), sha, clock.nowMs(), rights, ex.extractor, ex.extractorVersion, ex.pageCount,
            chunks.size, issues, ex.needsOcrPages, Fs.safeName(input.name), input.name.substringBeforeLast('.', input.name))
        try {
            store.commit(doc, bytes, cleaned, ex.docText, rep, cstats, ex.issues)
        } catch (e: IOException) {
            return fail(input.name, IssueCode.CORRUPT_FILE, "Could not store the source on this device: ${e.message}")
        }
        seen[sha] = sid
        return IngestItem(input.name, IngestStatus.INGESTED, sid, null, issues, doc.summary())
    }
}

/**
 * Per-project source storage:
 * ```
 * sources/<id>/source.json        record (written LAST: its presence commits the source)
 * sources/<id>/raw/<file>         byte-identical raw copy (hash verified after writing)
 * sources/<id>/blocks.jsonl       cleaned blocks with offsets/pages/section path
 * sources/<id>/stream.txt         extraction stream the char offsets index
 * sources/<id>/provenance.json    extractor, every cleaning transformation, issues, chunking stats
 * ```
 * Directories named `.tmp-*` are interrupted ingests and are deleted on open.
 */
class SourceStore(private val dir: File, private val problems: MutableList<String>) {
    private val docs = LinkedHashMap<String, SourceDoc>()

    init { load() }

    private fun load() {
        dir.mkdirs()
        dir.listFiles()?.forEach { if (it.isDirectory && it.name.startsWith(".tmp-")) it.deleteRecursively() }
        val ds = dir.listFiles { f -> f.isDirectory && !f.name.startsWith(".") }?.sortedBy { it.name } ?: return
        for (d in ds) {
            val o = Fs.readJson(File(d, "source.json")) { problems.add("source ${d.name}: $it") }
            if (o == null) { problems.add("source ${d.name} has no readable record and was ignored"); continue }
            try {
                docs[d.name] = SourceDoc(
                    d.name, o.str("name")!!, o.str("mime") ?: "application/octet-stream", o.lng("size")!!, o.str("sha256")!!, o.lng("ingested_at")!!,
                    runCatching { RightsStatus.valueOf(o.str("rights")!!) }.getOrDefault(RightsStatus.UNSET), o.str("extractor") ?: "unknown", o.str("extractor_version") ?: "0",
                    o.int("pages"), o.int("chunks") ?: 0,
                    o.objList("issues").mapNotNull { i -> runCatching { IngestIssue(IssueCode.valueOf(i.str("code")!!), i.str("message") ?: "") }.getOrNull() },
                    o.arr("needs_ocr_pages")?.let { a -> (0 until a.length()).map { a.getInt(it) } } ?: emptyList(), o.str("raw_file") ?: "file", o.str("title") ?: o.str("name")!!)
            } catch (e: Exception) { problems.add("source ${d.name} record is malformed and was ignored") }
        }
    }

    /** Oldest first (ingest time, then id): the same order before and after a restart. */
    @Synchronized fun all(): List<SourceDoc> = docs.values.sortedWith(compareBy({ it.ingestedAt }, { it.sourceId }))
    @Synchronized fun get(id: String): SourceDoc? = docs[id]
    fun dirOf(id: String) = File(dir, id)

    private fun writeDoc(d: SourceDoc, into: File = dirOf(d.sourceId)) {
        Fs.writeJson(File(into, "source.json"), linkedMapOf(
            "schema" to STORE_SCHEMA, "name" to d.name, "mime" to d.mime, "size" to d.sizeBytes, "sha256" to d.sha256, "ingested_at" to d.ingestedAt,
            "rights" to d.rights.name, "extractor" to d.extractor, "extractor_version" to d.extractorVersion, "pages" to d.pages, "chunks" to d.chunkCount,
            "issues" to d.issues.map { mapOf("code" to it.code.name, "message" to it.message) }, "needs_ocr_pages" to d.needsOcrPages,
            "raw_file" to d.rawFile, "title" to d.title))
    }

    @Synchronized fun commit(
        doc: SourceDoc, raw: ByteArray, blocks: List<CBlock>, stream: String, rep: CleanReport, chunkStats: Map<String, Int>, extractIssues: List<XIssue>,
    ) {
        val tmp = File(dir, ".tmp-${doc.sourceId}-${System.nanoTime()}")
        try {
            tmp.mkdirs()
            val rawFile = File(File(tmp, "raw"), doc.rawFile)
            Fs.writeBytes(rawFile, raw)
            if (Hashing.sha256File(rawFile) != doc.sha256) throw IOException("raw copy hash mismatch")
            Fs.writeAtomic(File(tmp, "blocks.jsonl")) { out ->
                for (b in blocks) out.write((J.dump(linkedMapOf("kind" to b.kind, "text" to b.text, "page" to b.page, "section_path" to b.sectionPath,
                    "char_start" to b.charStart, "char_end" to b.charEnd, "transforms" to b.transforms)) + "\n").toByteArray(Charsets.UTF_8))
            }
            Fs.writeBytes(File(tmp, "stream.txt"), stream.toByteArray(Charsets.UTF_8))
            Fs.writeJson(File(tmp, "provenance.json"), linkedMapOf(
                "source_id" to doc.sourceId, "original_filename" to doc.name, "sha256" to Hashing.prefixed(doc.sha256), "size_bytes" to doc.sizeBytes,
                "extractor" to mapOf("name" to doc.extractor, "version" to doc.extractorVersion), "pages_total" to doc.pages, "needs_ocr_pages" to doc.needsOcrPages,
                "stream_sha256" to Hashing.prefixed(Hashing.sha256(stream)),
                "offset_note" to "char offsets index the stored extraction stream (stream.txt)",
                "issues" to extractIssues.map { mapOf("code" to it.code, "detail" to it.detail, "severity" to it.severity, "page" to it.page) },
                "cleaning" to mapOf("transformations" to rep.transformations, "removed_noise" to rep.removedNoise),
                "chunking" to (mapOf("name" to Chunker.NAME, "version" to Chunker.VERSION, "max_chars" to 400 * Chunker.CHARS_PER_TOKEN) + chunkStats),
                "raw_copy" to mapOf("path" to "raw/${doc.rawFile}", "sha256" to Hashing.prefixed(doc.sha256), "verified" to true)))
            writeDoc(doc, tmp)     // written last: the directory rename below is the commit point
            val target = dirOf(doc.sourceId)
            if (target.exists()) target.deleteRecursively()
            if (!tmp.renameTo(target)) throw IOException("could not move the source into place")
            docs[doc.sourceId] = doc
        } catch (e: IOException) {
            tmp.deleteRecursively()
            throw e
        }
    }

    @Synchronized fun setRights(id: String, rights: RightsStatus): SourceDoc? {
        val d = docs[id] ?: return null
        val issues = d.issues.filter { it.code != IssueCode.RIGHTS_UNSET } +
            (if (rights == RightsStatus.UNSET) listOf(IngestIssue(IssueCode.RIGHTS_UNSET, "Set usage rights before building a dataset")) else emptyList())
        val nd = d.copy(rights = rights, issues = issues)
        docs[id] = nd
        writeDoc(nd)
        return nd
    }

    @Synchronized fun remove(id: String): Boolean {
        if (docs.remove(id) == null) return false
        dirOf(id).deleteRecursively()
        return true
    }

    fun loadBlocks(id: String): List<CBlock> {
        val f = File(dirOf(id), "blocks.jsonl")
        return Fs.readLines(f).mapNotNull { line ->
            val o = J.parseOrNull(line) ?: return@mapNotNull null
            CBlock(o.str("kind") ?: "paragraph", o.str("text") ?: "", o.int("page"), o.strList("section_path"), o.int("char_start") ?: 0, o.int("char_end") ?: 0, o.strList("transforms"))
        }
    }

    fun provenanceJson(id: String): org.json.JSONObject? = Fs.readJson(File(dirOf(id), "provenance.json"))
    fun rawFile(id: String): File? = get(id)?.let { File(File(dirOf(id), "raw"), it.rawFile) }.takeIf { it?.isFile == true }

    companion object { const val STORE_SCHEMA = 1; const val STORE_VERSION = 1 }
}
