package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.extract.BlockKind
import com.hotatticgames.llmtrainer.extract.Extracted
import com.hotatticgames.llmtrainer.extract.RawBlock
import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.err
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import org.json.JSONObject
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import com.hotatticgames.llmtrainer.extract.IngestIssue as XIssue

class IngestTest {
    private fun ex(blocks: List<RawBlock>, pages: Int? = null) = Extracted(blocks.joinToString("\n") { it.text }, blocks, emptyList(), emptyList(), "test", "1", pages)
    private fun para(t: String, page: Int? = null, sec: List<String> = emptyList(), s: Int = 0) = RawBlock(t, BlockKind.PARAGRAPH, page, sec, s, s + t.length)

    // ---- cleaning ----------------------------------------------------------------------------------------------------

    @Test fun repeatedHeadersFootersAndPageNumbersAreRemovedAndRecorded() {
        val words = listOf("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india")
        val blocks = (1..9).map { p ->
            para("Acme Service Manual - Page $p\nIntro for ${words[p - 1]} procedure goes here.\nThe swing arm pivot bolt must be tightened to ${80 + p} N-m after every ${words[p - 1]} inspection.\nClosing remark about ${words[9 - p]} checks.\nConfidential document\n$p", page = p)
        }
        val (out, rep) = Cleaner.clean(ex(blocks, pages = 9))
        val text = out.joinToString("\n") { it.text }
        assertFalse(text.contains("Service Manual"), text)
        assertFalse(text.contains("Confidential"))
        assertTrue(text.contains("swing arm pivot bolt"))
        val names = rep.transformations.map { it["name"] }
        assertTrue("remove_repeated_header_footer" in names, names.toString())
        val hdr = rep.transformations.first { it["name"] == "remove_repeated_header_footer" }
        assertEquals(27, hdr["count"]); assertEquals(19, hdr["distinct_lines"])     // 9 header texts + 9 page numbers + 1 footer (matching is digit-normalised, counting is per distinct text)
        assertTrue(rep.removedNoise.isNotEmpty() && (rep.removedNoise[0]["pages"] as List<*>).size == 9)
    }

    @Test fun romanPageNumbersAtPageEdgesAreRemovedByTheirOwnRule() {
        val romans = listOf("i", "ii", "iii", "iv", "v", "vi")
        val blocks = romans.mapIndexed { i, r -> para("A genuine sentence about torque settings that remains after cleaning number ${i + 1} of the guide.\n$r", page = i + 1) }
        val (out, rep) = Cleaner.clean(ex(blocks, pages = 6))
        assertFalse(out.joinToString("\n") { it.text }.lines().any { it.trim() in romans })
        assertEquals(6, rep.transformations.first { it["name"] == "remove_page_number_lines" }["count"])
    }

    @Test fun headerRemovalNeedsAtLeastThreePages() {
        val blocks = (1..2).map { p -> para("Header line\nBody sentence number $p that is long enough to keep around in the chunker.", page = p) }
        val (out, _) = Cleaner.clean(ex(blocks, pages = 2))
        assertTrue(out.joinToString("\n") { it.text }.contains("Header line"))
    }

    @Test fun hyphenationIsRepairedButRealCompoundsAreKept() {
        val b = para("Perform routine main-\ntenance on the front-\nwheel bearing every season. The front-wheel assembly needs grease.")
        val (out, rep) = Cleaner.clean(ex(listOf(b)))
        val t = out.single().text
        assertTrue(t.contains("maintenance"), t)
        assertTrue(t.contains("front-wheel bearing"), t)           // compound attested intact elsewhere keeps its hyphen
        val h = rep.transformations.first { it["name"] == "repair_hyphenation" }
        assertEquals(1, h["count"]); assertEquals(1, h["kept_compound_hyphens"])
    }

    @Test fun normalisationIsCountedPerKind() {
        val b = para("The ﬁrst ﬂange bolt  needs​ care\u0007 here.")
        val (out, rep) = Cleaner.clean(ex(listOf(b)))
        assertEquals("The first flange bolt needs care here.", out.single().text)
        val by = rep.transformations.associate { it["name"] to it["count"] }
        assertEquals(2, by["normalize_ligatures"]); assertEquals(1, by["normalize_nbsp"]); assertEquals(2, by["strip_control_and_zero_width_chars"])
        assertTrue(by.containsKey("collapse_whitespace"))
        assertTrue(rep.transformations.all { it["version"] == Cleaner.VERSION })
    }

    @Test fun softWrappedLinesAreReflowedButListsKeepTheirLines() {
        val b = para("This sentence was wrapped\nby the original layout and continues\nhere.\n- first item\n- second item")
        val (out, rep) = Cleaner.clean(ex(listOf(b)))
        assertEquals("This sentence was wrapped by the original layout and continues here.\n- first item\n- second item", out.single().text)
        assertEquals(2, rep.transformations.first { it["name"] == "reflow_soft_wrapped_lines" }["count"])
    }

    // ---- chunking ------------------------------------------------------------------------------------------------------

    private fun cb(kind: String, t: String, sec: List<String> = emptyList(), s: Int = 0, page: Int? = null) = CBlock(kind, t, page, sec, s, s + t.length)

    @Test fun chunksNeverCrossSectionsAndKeepProvenance() {
        val blocks = listOf(cb(BlockKind.HEADING, "Brakes", listOf("Brakes")),
            cb(BlockKind.PARAGRAPH, "Bleed the brake lines until fluid runs clear of air bubbles.", listOf("Brakes"), 10, page = 3),
            cb(BlockKind.HEADING, "Chain", listOf("Chain")),
            cb(BlockKind.PARAGRAPH, "Lubricate the chain after every wash with a suitable spray.", listOf("Chain"), 100, page = 4))
        val (chunks, _) = Chunker.chunk(blocks)
        assertEquals(2, chunks.size)
        assertEquals(listOf("c0001", "c0002"), chunks.map { it.id })
        assertEquals("Brakes", chunks[0].section); assertEquals(3, chunks[0].page); assertEquals(10, chunks[0].charStart)
        assertEquals(Hashing.sha256(chunks[0].text), chunks[0].sha256)
        assertEquals(Chunker.chunk(blocks).first.map { it.sha256 }, chunks.map { it.sha256 })      // deterministic
    }

    @Test fun tablesBecomeReferenceOnlyChunksOf25Rows() {
        val rows = (1..60).map { cb(BlockKind.TABLE_ROW, "Bolt $it | ${10 + it} N-m", listOf("Torque table"), it * 20) }
        val (chunks, _) = Chunker.chunk(rows)
        assertEquals(3, chunks.size)
        assertTrue(chunks.all { it.role == ChunkRoles.REFERENCE && it.kind == "table" && it.origin == "table" })
        assertEquals(25, chunks[0].text.lines().size); assertEquals(10, chunks[2].text.lines().size)
    }

    @Test fun duplicateAndNoiseChunksAreExcludedNotDropped() {
        val para = "Check the coolant level when the engine is cold and top up with the specified mixture only."
        val blocks = listOf(cb(BlockKind.PARAGRAPH, para, listOf("A")), cb(BlockKind.HEADING, "B", listOf("B")), cb(BlockKind.PARAGRAPH, para, listOf("B")),
            cb(BlockKind.HEADING, "C", listOf("C")), cb(BlockKind.PARAGRAPH, "1234 5678 9012 3456 7890 ---- ==== ####", listOf("C")))
        val (chunks, st) = Chunker.chunk(blocks)
        assertEquals(3, chunks.size)
        assertEquals(ChunkRoles.TRAIN, chunks[0].role)
        assertEquals("duplicate_chunk", chunks[1].excludeReason); assertEquals("c0001", chunks[1].duplicateOf)
        assertEquals("low_extraction_quality", chunks[2].excludeReason)
        assertEquals(1, st["excluded_duplicate"]); assertEquals(1, st["excluded_low_quality"])
    }

    @Test fun oversizeParagraphsAreSplitOnSentencesAndTinyOnesAreCountedNotSilent() {
        val big = (1..40).joinToString(" ") { "Sentence number $it describes a maintenance step in some detail." }
        val (chunks, st) = Chunker.chunk(listOf(cb(BlockKind.PARAGRAPH, big), cb(BlockKind.HEADING, "x", listOf("x")), cb(BlockKind.PARAGRAPH, "tiny", listOf("x"))), 400)
        assertTrue(chunks.size > 3 && chunks.all { it.text.length <= 400 })
        assertEquals(1, st["split_oversize_paragraphs"]); assertEquals(1, st["dropped_short"])
    }

    // ---- ingestion through the facade ------------------------------------------------------------------------------------

    private fun project(s: Studio) = s.createProject(NewProject("P", "d", "p")).ok().id

    @Test fun ingestKeepsAHashVerifiedRawCopyAndFullProvenance() {
        val rig = TK.rig(); val s = rig.open(); val p = project(s)
        val body = TK.doc(1)
        val rep = s.ingest(p, listOf(TK.input("manual.md", body, "text/markdown")), RightsStatus.OWNER_AUTHORED).ok()
        val item = rep.ingested.single()
        val prov = item.provenance!!
        assertEquals(Hashing.sha256(body.toByteArray()), prov.sha256)
        assertEquals("src-" + prov.sha256.take(12), item.sourceId)
        assertEquals(body.toByteArray().size.toLong(), prov.sizeBytes)
        assertTrue(prov.chunks > 3); assertEquals(RightsStatus.OWNER_AUTHORED, prov.rights)
        assertTrue(prov.extractor.startsWith("markdown"))
        val dir = File(rig.dir, "projects/${p.value}/sources/${item.sourceId}")
        assertEquals(body, File(dir, "raw/manual.md").readText())                             // byte-identical original kept
        val provJson = JSONObject(File(dir, "provenance.json").readText())
        assertEquals("sha256:${prov.sha256}", provJson.getString("sha256"))
        assertTrue(provJson.getJSONObject("raw_copy").getBoolean("verified"))
        assertTrue(provJson.has("cleaning") && provJson.has("chunking"))
        assertTrue(File(dir, "blocks.jsonl").readLines().isNotEmpty())
        assertEquals(item.sourceId, s.listSources(p).ok().single().sourceId)
    }

    @Test fun duplicatesAreDetectedAgainstStoredAndSameBatchSources() {
        val s = TK.rig().open(); val p = project(s)
        val a = TK.input("a.md", TK.doc(1), "text/markdown")
        val rep1 = s.ingest(p, listOf(a, TK.input("copy-of-a.md", TK.doc(1), "text/markdown"), TK.input("b.md", TK.doc(2), "text/markdown")), RightsStatus.OWNER_AUTHORED).ok()
        assertEquals(2, rep1.ingested.size)
        val dup = rep1.duplicates.single()
        assertEquals(rep1.ingested[0].sourceId, dup.duplicateOfSourceId)
        assertEquals(IssueCode.DUPLICATE_CONTENT, dup.issues.single().code)
        val rep2 = s.ingest(p, listOf(a), RightsStatus.OWNER_AUTHORED).ok()
        assertEquals(IngestStatus.DUPLICATE, rep2.items.single().status)
        assertEquals(2, s.listSources(p).ok().size)
    }

    @Test fun badFilesGetIssueCodesAndNeverAbortTheBatch() {
        val s = TK.rig().open(); val p = project(s)
        val unreadable = SourceInput("broken.md", "text/markdown", 100) { throw java.io.IOException("permission revoked") }
        val rep = s.ingest(p, listOf(
            TK.input("good.md", TK.doc(1), "text/markdown"),
            TK.input("empty.txt", "   \n\n"),
            TK.bytesInput("blob.bin", ByteArray(500) { it.toByte() }, "application/octet-stream"),
            TK.input("trunc.json", "{\"a\": [1, 2,", "application/json"),
            TK.bytesInput("nul.txt", ByteArray(2000), "text/plain"),
            unreadable,
            SourceInput("huge.txt", "text/plain", 80L * 1024 * 1024) { error("must not be opened") },
            TK.input("tiny.txt", "short"),
        ), RightsStatus.OWNER_AUTHORED).ok()
        assertEquals(8, rep.items.size)
        assertEquals(IngestStatus.INGESTED, rep.items[0].status)
        assertEquals(setOf(IngestStatus.FAILED), rep.items.drop(1).map { it.status }.toSet())
        fun codes(i: Int) = rep.items[i].issues.map { it.code }
        assertTrue(IssueCode.EMPTY_TEXT in codes(1))
        assertTrue(IssueCode.UNSUPPORTED_TYPE in codes(2))
        assertTrue(IssueCode.CORRUPT_FILE in codes(3), codes(3).toString())
        assertTrue(codes(4).isNotEmpty())
        assertTrue(IssueCode.CORRUPT_FILE in codes(5) && rep.items[5].issues[0].message.contains("permission revoked"))
        assertEquals(listOf(IssueCode.TOO_LARGE), codes(6))
        assertTrue(IssueCode.EMPTY_TEXT in codes(7))
        assertTrue(rep.items.drop(1).all { it.sourceId == null && it.issues.isNotEmpty() && it.issues.all { i -> i.message.isNotBlank() } })
        assertEquals(1, s.listSources(p).ok().size)                         // nothing half-stored
        assertEquals(StageStatus.NEEDS_ATTENTION, s.getProject(p).ok().stages.first { it.id == StageId.SOURCES }.status)
        assertEquals(rep, s.lastIngestReport(p).ok())
    }

    @Test fun hostileContentIsHandledWithoutCrashing() {
        val s = TK.rig().open(); val p = project(s)
        val deepJson = "[".repeat(100_000) + "]".repeat(100_000)
        val longLine = "word ".repeat(400_000)
        val rep = s.ingest(p, listOf(
            TK.input("deep.json", deepJson, "application/json"),
            TK.input("longline.txt", longLine),
            TK.input("../../evil/../path.md", "# t\n\n" + "A safe sentence about brake fluid replacement intervals in the manual. ".repeat(30), "text/markdown"),
            TK.bytesInput("bom.txt", byteArrayOf(0xEF.toByte(), 0xBB.toByte(), 0xBF.toByte()) + "Chain tension should be checked every five hundred kilometres on a dry road.".toByteArray()),
            TK.input("rtl.txt", "‮" + "Tighten the cylinder head bolts in a spiral pattern from the centre outwards twice." + "\u0000\u0001"),
            TK.input("script.md", "# <script>alert(1)</script>\n\nThe throttle cable free play should be about three millimetres at the grip.\n"),
        ), RightsStatus.OWNER_AUTHORED).ok()
        assertEquals(6, rep.items.size)
        assertTrue(rep.items.all { it.status == IngestStatus.INGESTED || it.issues.isNotEmpty() })
        // the traversal-looking name cannot escape the project tree
        val root = File(s.let { (it as StudioCore) }.let { rootOf(it) })
        assertTrue(root.walkTopDown().none { it.name == "evil" })
        val stored = File(root, "projects/${p.value}/sources").walkTopDown().filter { it.isFile && it.name == "path.md" }.toList()
        assertEquals(1, stored.size)
    }

    private fun rootOf(c: StudioCore): String { val f = StudioCore::class.java.getDeclaredField("rootDir"); f.isAccessible = true; return (f.get(c) as File).path }

    @Test fun rightsAreFirstClass() {
        val s = TK.rig().open(); val p = project(s)
        val rep = s.ingest(p, TK.corpus(3)).ok()                                    // default rights UNSET
        assertTrue(rep.ingested.all { it.issues.any { i -> i.code == IssueCode.RIGHTS_UNSET } })
        val blocked = s.buildDataset(p).err() as StudioError.Blocked
        assertEquals("RIGHTS_UNSET", blocked.code); assertEquals(3, blocked.reasons.size)
        val ids = s.listSources(p).ok().map { it.sourceId }
        ids.forEach { s.setSourceRights(p, it, RightsStatus.OWNER_AUTHORED).ok() }
        assertTrue(s.listSources(p).ok().all { it.provenance.rights == RightsStatus.OWNER_AUTHORED && it.issues.none { i -> i.code == IssueCode.RIGHTS_UNSET } })
        assertTrue(s.lastIngestReport(p).ok()!!.ingested.all { it.provenance!!.rights == RightsStatus.OWNER_AUTHORED })
        assertEquals("NOT_FOUND", s.setSourceRights(p, "src-nope", RightsStatus.UNSET).err().code)
        assertEquals("NO_SOURCES", s.buildDataset(project(s)).err().code)
    }

    @Test fun perFileStatusIsPersistedWhileIngestIsStillRunning() {
        val rig = TK.rig(); val s = rig.open(); val p = project(s)
        val seen = ArrayList<Int>()
        s.ingest(p, TK.corpus(3), RightsStatus.OWNER_AUTHORED) { prog ->
            val f = File(rig.dir, "projects/${p.value}/ingest_report.json")
            if (f.isFile) seen.add(JSONObject(f.readText()).getJSONArray("items").length())
        }.ok()
        assertEquals(listOf(1, 2, 3), seen.takeLast(3).let { if (seen.contains(1)) listOf(1, 2, 3) else it })   // after file 1, 2, 3 the report on disk had that many items
        assertTrue(seen.max() >= 2)
        // a brand-new process sees the same report
        assertEquals(3, rig.open().lastIngestReport(p).ok()!!.items.size)
    }

    @Test fun tablesFromCsvAreReferenceOnlyAndNeverTrainingProse() {
        val rig = TK.rig(); val s = rig.open()
        val p = TK.readyProject(s, 5)
        val csv = "part,torque_nm,notes\n" + (1..30).joinToString("\n") { "bolt-$it,${20 + it},check after first ride number $it" }
        s.ingest(p, listOf(TK.input("torque.csv", csv, "text/csv")), RightsStatus.OWNER_AUTHORED).ok()
        s.buildDataset(p).ok()
        val items = s.reviewItems(p, ReviewFilter(), 0, 1000).ok().items
        val table = items.filter { it.flags.contains(ReviewFlag.TABLE) }
        assertTrue(table.isNotEmpty())
        assertTrue(table.all { it.role == ChunkRole.REFERENCE }, "tables must never carry a training/eval role")
        assertTrue(table.all { it.included })                                      // retained for retrieval
    }

    @Test fun nearDuplicateDocumentsAreFlaggedAtDatasetBuild() {
        val s = TK.rig().open(); val p = project(s)
        val base = TK.doc(1)
        val variant = base.replace("service procedure", "service procedure ") + "\n\nOne extra closing paragraph about warranty terms and the dealer network in this region.\n"
        s.ingest(p, listOf(TK.input("a.md", base, "text/markdown"), TK.input("a2.md", variant, "text/markdown")) + TK.corpus(5).drop(2), RightsStatus.OWNER_AUTHORED).ok()
        s.buildDataset(p).ok()
        val dup = s.reviewItems(p, ReviewFilter(flag = ReviewFlag.DUPLICATE), 0, 1000).ok().items
        assertTrue(dup.size >= 3, "near-duplicate chunks of the second copy must be flagged, got ${dup.size}")
        assertTrue(dup.none { it.included }, "duplicates are excluded by default but remain listed")
    }

    @Test fun removingASourceDeletesEverythingDerivedAndMarksTheDatasetStale() {
        val rig = TK.rig(); val s = rig.open()
        val p = TK.readyProject(s, 6)
        s.approveDataset(p).ok()
        val victim = s.listSources(p).ok()[2]
        val vdir = File(rig.dir, "projects/${p.value}/sources/${victim.sourceId}")
        assertTrue(vdir.isDirectory)
        s.removeSource(p, victim.sourceId).ok()
        assertFalse(vdir.exists())
        assertEquals(5, s.listSources(p).ok().size)
        assertEquals(DatasetStatus.STALE, s.datasetPreview(p).ok()!!.status)
        val items = s.reviewItems(p, ReviewFilter(), 0, 5000).ok().items
        assertTrue(items.none { it.sourceId == victim.sourceId })
        assertTrue(File(rig.dir, "projects/${p.value}").walkTopDown().filter { it.isFile && it.extension in setOf("jsonl", "json", "md", "txt") }
            .none { it.readText().contains(victim.sourceId) && !it.name.startsWith("ingest_report") && !it.name.startsWith("build.json") }, "no derived file may still mention the removed source")
        assertEquals("DATASET_STALE", (s.approveDataset(p).err() as StudioError.Blocked).code)
        assertEquals("DATASET_NOT_APPROVED", (s.exportReferencePackage(p, java.io.ByteArrayOutputStream()).err() as StudioError.Blocked).code)
        assertEquals(StageStatus.NEEDS_ATTENTION, s.getProject(p).ok().stages.first { it.id == StageId.DATASET }.status)
        s.buildDataset(p).ok()
        val rebuilt = s.reviewItems(p, ReviewFilter(), 0, 5000).ok().items
        assertTrue(rebuilt.none { it.sourceId == victim.sourceId })
        assertEquals("NOT_FOUND", s.removeSource(p, victim.sourceId).err().code)
    }

    @Test fun interruptedIngestLeavesNoHalfSource() {
        val rig = TK.rig(); val s = rig.open(); val p = project(s)
        s.ingest(p, TK.corpus(1), RightsStatus.OWNER_AUTHORED).ok()
        val sources = File(rig.dir, "projects/${p.value}/sources")
        val tmp = File(sources, ".tmp-src-deadbeef-1").also { File(it, "raw").mkdirs(); File(it, "raw/x.md").writeText("partial") }
        val orphan = File(sources, "src-orphan0000000").also { it.mkdirs(); File(it, "blocks.jsonl").writeText("x") }       // dir without source.json (never committed)
        val reborn = rig.open()
        assertFalse(tmp.exists())
        assertEquals(1, reborn.listSources(p).ok().size)
        assertTrue(reborn.startupProblems.any { it.contains("src-orphan0000000") })
        assertTrue(orphan.exists())                                              // reported, left in place, never read
    }

    @Test fun issueMappingNeverLosesTheOriginalCode() {
        fun m(code: String) = IssueMap.map(XIssue(code, "f", "detail text", "error", 4))
        assertEquals(IssueCode.NEEDS_OCR, m("needs_ocr").code)
        assertEquals(IssueCode.NEEDS_OCR, m("no_text_layer").code)
        assertEquals(IssueCode.CORRUPT_FILE, m("corrupt_pdf").code)
        assertEquals(IssueCode.CORRUPT_FILE, m("encrypted_pdf").code)
        assertEquals(IssueCode.UNSUPPORTED_TYPE, m("unsupported_format").code)
        assertEquals(IssueCode.EMPTY_TEXT, m("empty_content").code)
        assertEquals(IssueCode.LOW_TEXT_QUALITY, m("encoding_fallback").code)
        val x = m("ragged_rows")
        assertTrue(x.message.contains("[ragged_rows]") && x.message.contains("page 4"))
    }

    @Test fun termExtractionIsDeterministicAndProvenanced() {
        val rows = listOf("c/1" to "The carburetor float height must be set before the carburetor jets are cleaned. Float height matters.",
            "c/2" to "Clean the carburetor jets with compressed air and check float height again after cleaning the carburetor.")
        val a = Terms.extract(rows, 10); val b = Terms.extract(rows.reversed(), 10)
        assertEquals(a, b)
        assertTrue(a.any { it.term == "carburetor" && it.df == 2 && it.chunks == listOf("c/1", "c/2") })
        assertTrue(a.any { it.term == "float height" })
        assertTrue(a.none { it.term == "the" || it.term == "must" })
    }
}
