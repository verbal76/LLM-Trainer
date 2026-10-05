package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.err
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class DatasetTest {
    private fun items(s: Studio, p: ProjectId, f: ReviewFilter = ReviewFilter()) = s.reviewItems(p, f, 0, 100_000).ok().items

    private fun fresh(nDocs: Int = 6, options: DatasetOptions = DatasetOptions(), rights: RightsStatus = RightsStatus.OWNER_AUTHORED): Triple<TK.Rig, StudioCore, ProjectId> {
        val rig = TK.rig(); val s = rig.open()
        return Triple(rig, s, TK.readyProject(s, nDocs, rights, options = options))
    }

    @Test fun buildIsDeterministicAndIdChangesWithInputsAndConfig() {
        fun sha(opts: DatasetOptions, docs: Int = 6): String { val (_, s, p) = fresh(docs, opts); return s.datasetPreview(p).ok()!!.datasetSha256 }
        val a = sha(DatasetOptions()); val b = sha(DatasetOptions())
        assertEquals(a, b)
        assertEquals(64, a.length)
        assertNotEquals(a, sha(DatasetOptions(seed = 7)))
        assertNotEquals(a, sha(DatasetOptions(heldOutFraction = 0.2)))
        assertNotEquals(a, sha(DatasetOptions(chunkTargetTokens = 150)))
        assertNotEquals(a, sha(DatasetOptions(), docs = 7))
    }

    @Test fun splitsFollowDocumentGroupsAndRolesPartitionTheChunks() {
        val (_, s, p) = fresh(8)
        val all = items(s, p)
        val prev = s.datasetPreview(p).ok()!!
        assertEquals(all.size, prev.stats.totalChunks)
        assertEquals(all.size, prev.stats.byRole.values.sum())
        assertEquals(all.size, prev.stats.byOrigin.values.sum())
        assertTrue(prev.stats.byRole.getValue(ChunkRole.TRAIN) > prev.stats.byRole.getValue(ChunkRole.HELD_OUT_EVAL))
        // every document lands wholly in ONE split
        val roleBySource = all.groupBy { it.sourceId }.mapValues { e -> e.value.map { it.role }.filter { it != ChunkRole.REFERENCE }.toSet() }
        assertTrue(roleBySource.values.all { it.size <= 1 }, roleBySource.toString())
        assertEquals(setOf(ChunkRole.TRAIN, ChunkRole.VALIDATION, ChunkRole.HELD_OUT_EVAL), roleBySource.values.flatten().toSet())
        assertEquals(DatasetStatus.NEEDS_REVIEW, prev.status)
        assertEquals(StageStatus.NEEDS_ATTENTION, s.getProject(p).ok().stages.first { it.id == StageId.DATASET }.status)
    }

    @Test fun fewDocumentsFallBackToSectionGroupingWithANote() {
        val (_, s, p) = fresh(2)
        val prev = s.datasetPreview(p).ok()!!
        assertTrue(prev.warnings.any { it.contains("grouping by section") }, prev.warnings.toString())
        // sections of one document never straddle splits
        val bySection = items(s, p).filter { it.role != ChunkRole.REFERENCE }.groupBy { it.sourceId to it.section }.mapValues { e -> e.value.map { it.role }.toSet() }
        assertTrue(bySection.values.all { it.size == 1 })
    }

    @Test fun tooFewGroupsMeansNoSplitAndNoTrainingExportButReferenceStillWorks() {
        val rig = TK.rig(); val s = rig.open()
        val p = s.createProject(NewProject("Tiny", "d", "p")).ok().id
        s.ingest(p, listOf(TK.input("one.md", TK.doc(1, sections = 2), "text/markdown")), RightsStatus.OWNER_AUTHORED).ok()
        val prev = s.buildDataset(p).ok()
        assertTrue(prev.warnings.any { it.contains("fewer than 3 independent groups") }, prev.warnings.toString())
        assertTrue(items(s, p).all { it.role == ChunkRole.REFERENCE })
        s.approveDataset(p).ok()
        TK.verifyLicense(rig, s); s.selectBaseModel(p, TK.MODEL, null).ok()
        val m = s.methodOptions(p).ok().first { it.id == MethodIds.ADAPTER_DESKTOP }
        assertFalse(m.available); assertTrue(m.whyNotAvailable!!.contains("Fewer than 3"))
        assertEquals("TOO_FEW_GROUPS", (s.exportHeldOutEvalSet(p, ByteArrayOutputStream()).err() as StudioError.Blocked).code)
        val ref = ByteArrayOutputStream()
        s.exportReferencePackage(p, ref).ok()
        assertTrue(ref.size() > 500)
    }

    @Test fun optionsAreValidated() {
        val (_, s, p) = fresh(4)
        for (bad in listOf(DatasetOptions(heldOutFraction = 0.0), DatasetOptions(validationFraction = -1.0), DatasetOptions(heldOutFraction = 0.5, validationFraction = 0.5), DatasetOptions(chunkTargetTokens = 3)))
            assertEquals("BAD_OPTIONS", s.buildDataset(p, bad).err().code)
    }

    @Test fun referenceOnlySourcesNeverFeedTraining() {
        val rig = TK.rig(); val s = rig.open()
        val p = s.createProject(NewProject("R", "d", "p")).ok().id
        s.ingest(p, TK.corpus(6), RightsStatus.OWNER_AUTHORED).ok()
        s.ingest(p, listOf(TK.input("licensed-manual.md", TK.doc(77), "text/markdown")), RightsStatus.REFERENCE_ONLY).ok()
        val refSrc = s.listSources(p).ok().first { it.name == "licensed-manual.md" }.sourceId
        s.buildDataset(p).ok()
        val mine = items(s, p).filter { it.sourceId == refSrc }
        assertTrue(mine.isNotEmpty() && mine.all { it.role == ChunkRole.REFERENCE && !it.included })        // default: excluded entirely
        s.buildDataset(p, DatasetOptions(excludeReferenceOnlySources = false)).ok()
        val mine2 = items(s, p).filter { it.sourceId == refSrc }
        assertTrue(mine2.all { it.role == ChunkRole.REFERENCE && it.included }, "listed for retrieval, still never training")
        s.approveDataset(p).ok(); TK.verifyLicense(rig, s); s.selectBaseModel(p, TK.MODEL, null).ok(); s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).ok()
        val out = ByteArrayOutputStream()
        s.exportTrainingJobPackage(p, out).ok()
        assertEquals(emptyList(), JobValidator.validate(out.toByteArray()))
        val files = Zips.readAll(out.toByteArray().inputStream())
        val trainChunks = JobValidator.lines(files["chunks.jsonl"]).filter { it.getString("source_id") == refSrc }
        assertTrue(trainChunks.isNotEmpty() && trainChunks.all { it.getString("role") == "reference" })
        assertTrue(files.filterKeys { it.startsWith("dataset/") }.values.none { it.toString(Charsets.UTF_8).contains(refSrc) })
        val man = org.json.JSONObject(files.getValue("manifest.json").toString(Charsets.UTF_8))
        val r = man.getJSONArray("sources").let { a -> (0 until a.length()).map { a.getJSONObject(it) } }.first { it.getString("id") == refSrc }.getJSONObject("rights")
        assertEquals("no", r.getString("permitted_training"))
    }

    @Test fun reviewIncludeExcludeChangesIdentityAndResetsApproval() {
        val (_, s, p) = fresh(6)
        val before = s.datasetPreview(p).ok()!!
        s.approveDataset(p).ok()
        assertNotNull(s.datasetPreview(p).ok()!!.approvedAt)
        val victim = items(s, p).first { it.included && it.role == ChunkRole.TRAIN }
        val after = s.setIncluded(p, listOf(victim.id), false).ok()
        assertEquals(DatasetStatus.NEEDS_REVIEW, after.status)
        assertEquals(null, after.approvedAt)
        assertNotEquals(before.datasetSha256, after.datasetSha256)
        assertEquals(before.stats.included - 1, after.stats.included)
        assertFalse(items(s, p).first { it.id == victim.id }.included)
        val back = s.setIncluded(p, listOf(victim.id), true).ok()
        assertEquals(before.datasetSha256, back.datasetSha256)                        // identity is a pure function of content
        assertEquals("NOT_FOUND", s.setIncluded(p, listOf("src-x/c9999"), true).err().code)
        assertEquals("NO_DATASET", (s.setIncluded(s.createProject(NewProject("n", "", "")).ok().id, listOf("x"), true).err() as StudioError.Blocked).code)
    }

    @Test fun reviewDecisionsSurviveARestartAndARebuildWithUnchangedText() {
        val rig = TK.rig(); val s = rig.open()
        val p = TK.readyProject(s, 6)
        val victim = items(s, p).first { it.included && it.role == ChunkRole.TRAIN }
        s.setIncluded(p, listOf(victim.id), false).ok()
        val s2 = rig.open()
        assertFalse(items(s2, p).first { it.id == victim.id }.included)
        s2.buildDataset(p).ok()                                                       // same sources, same options
        assertFalse(items(s2, p).first { it.id == victim.id }.included, "owner decision must survive a rebuild when the chunk text is unchanged")
    }

    @Test fun reviewListingFiltersAndPages() {
        val (_, s, p) = fresh(6)
        val all = items(s, p)
        val page = s.reviewItems(p, ReviewFilter(), 5, 7).ok()
        assertEquals(all.size, page.total); assertEquals(5, page.offset); assertEquals(all.subList(5, 12).map { it.id }, page.items.map { it.id })
        assertTrue(items(s, p, ReviewFilter(role = ChunkRole.HELD_OUT_EVAL)).all { it.role == ChunkRole.HELD_OUT_EVAL })
        val src = all.first().sourceId
        assertTrue(items(s, p, ReviewFilter(sourceId = src)).all { it.sourceId == src })
        assertEquals(all.count { it.included }, items(s, p, ReviewFilter(included = true)).size)
        assertTrue(items(s, p).all { it.excerpt.length <= 241 && it.origin == ChunkOrigin.SOURCE_DERIVED })
        assertEquals(0, s.reviewItems(s.createProject(NewProject("n", "", "")).ok().id).ok().total)
    }

    private fun chunkRefOf(s: Studio, p: ProjectId, textPart: String): List<ReviewItem> = items(s, p).filter { it.excerpt.contains(textPart) }

    @Test fun crossSplitCopiesAreFlaggedAsLeakageExcludedByDefaultAndBlockApprovalWhenReincluded() {
        var found = false
        for (seed in 1L..40L) {
            val rig = TK.rig(); val s = rig.open()
            val p = s.createProject(NewProject("L", "d", "p")).ok().id
            val shared = "The unmistakably distinctive gearbox countershaft retaining circlip must be replaced with a genuine part whenever the casing is split for any reason."
            val docs = (1..6).map { i -> TK.doc(i) + (if (i == 1 || i == 2) "\n\n## Shared note\n\n$shared\n" else "") }
            s.ingest(p, docs.mapIndexed { i, d -> TK.input("m$i.md", d, "text/markdown") }, RightsStatus.OWNER_AUTHORED).ok()
            s.buildDataset(p, DatasetOptions(seed = seed)).ok()
            val copies = chunkRefOf(s, p, "unmistakably distinctive")
            if (copies.size != 2 || copies.map { it.role }.toSet().size != 2) continue
            found = true
            val (a, b) = copies
            val prio = mapOf(ChunkRole.TRAIN to 0, ChunkRole.VALIDATION to 1, ChunkRole.HELD_OUT_EVAL to 2)
            val victim = if (prio.getValue(a.role) < prio.getValue(b.role)) a else b
            val keeper = if (victim === a) b else a
            assertTrue(ReviewFlag.POSSIBLE_LEAKAGE in victim.flags, "the less protected side is flagged: $copies")
            assertFalse(victim.included); assertTrue(keeper.included, "the more protected side is never dropped")
            assertEquals(1, s.datasetPreview(p).ok()!!.stats.byFlag[ReviewFlag.POSSIBLE_LEAKAGE])
            s.approveDataset(p).ok()                                          // resolved by default
            s.setIncluded(p, listOf(victim.id), true).ok()
            val blocked = s.approveDataset(p).err() as StudioError.Blocked
            assertEquals("LEAKAGE_UNRESOLVED", blocked.code)
            assertEquals(1, s.datasetPreview(p).ok()!!.stats.leakageSuspects)
            s.setIncluded(p, listOf(victim.id), false).ok()
            s.approveDataset(p).ok()
            break
        }
        assertTrue(found, "no seed placed the two copies in different splits")
    }

    @Test fun shortChunkCopiedFromAHeldOutChunkIsCaughtByContainment() {
        var found = false
        for (seed in 1L..60L) {
            val rig = TK.rig(); val s = rig.open()
            val p = s.createProject(NewProject("C", "d", "p")).ok().id
            val para = "The remarkable sprocket carrier shim stack sets the chain line and must be measured with a calibrated micrometer before reassembly of the rear hub assembly."
            val docs = (1..6).map { i -> TK.doc(i) + (if (i == 1) "\n\n## Special\n\n$para\n\nAnd a second unrelated paragraph that keeps the chunk longer than the copied sentence, concerning drain plugs and washers in detail.\n" else "") +
                (if (i == 2) "\n\n## Copy\n\n$para\n" else "") }
            s.ingest(p, docs.mapIndexed { i, d -> TK.input("m$i.md", d, "text/markdown") }, RightsStatus.OWNER_AUTHORED).ok()
            s.buildDataset(p, DatasetOptions(seed = seed)).ok()
            val copies = chunkRefOf(s, p, "remarkable sprocket carrier")
            if (copies.size != 2 || copies.map { it.role }.toSet().size != 2) continue
            found = true
            val less = copies.minByOrNull { listOf(ChunkRole.TRAIN, ChunkRole.VALIDATION, ChunkRole.HELD_OUT_EVAL).indexOf(it.role) }!!
            assertTrue(ReviewFlag.POSSIBLE_LEAKAGE in less.flags && !less.included, "containment/near-duplicate guard must catch it: $copies")
            break
        }
        assertTrue(found)
    }

    @Test fun exportedExamplesNeverCrossSplitsAcrossManySeedsAndSizes() {
        for ((seed, docs) in listOf(1L to 4, 2L to 5, 3L to 7, 11L to 6, 99L to 9)) {
            val rig = TK.rig(); val s = rig.open()
            val p = TK.readyProject(s, docs, options = DatasetOptions(seed = seed))
            s.approveDataset(p).ok(); TK.verifyLicense(rig, s); s.selectBaseModel(p, TK.MODEL, null).ok(); s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).ok()
            val out = ByteArrayOutputStream()
            s.exportTrainingJobPackage(p, out).ok()
            assertEquals(emptyList(), JobValidator.validate(out.toByteArray()), "seed=$seed docs=$docs")
        }
    }

    @Test fun syntheticExamplesAreLabelledOptInTrainOnlyAndNeverInTest() {
        val rig = TK.rig(); val s = rig.open()
        val p = TK.readyProject(s, 7, options = DatasetOptions(includeSynthetic = true))
        val syn = items(s, p).filter { it.origin == ChunkOrigin.SYNTHETIC }
        assertTrue(syn.isNotEmpty())
        assertTrue(syn.all { it.role == ChunkRole.TRAIN && !it.included && ReviewFlag.LOW_CONFIDENCE in it.flags && it.excerpt.startsWith("Q: ") })
        s.setIncluded(p, syn.take(5).map { it.id }, true).ok()
        s.approveDataset(p).ok(); TK.verifyLicense(rig, s); s.selectBaseModel(p, TK.MODEL, null).ok(); s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).ok()
        val out = ByteArrayOutputStream()
        s.exportTrainingJobPackage(p, out).ok()
        assertEquals(emptyList(), JobValidator.validate(out.toByteArray()))
        val files = Zips.readAll(out.toByteArray().inputStream())
        assertEquals(5, JobValidator.lines(files["dataset/train.jsonl"]).count { it.getString("origin") == "synthetic" })
        assertTrue(listOf("dataset/validation.jsonl", "dataset/test.jsonl").all { f -> JobValidator.lines(files[f]).none { it.getString("origin") == "synthetic" } })
        assertTrue(JobValidator.lines(files["dataset/train.jsonl"]).filter { it.getString("origin") == "synthetic" }.all { it.getString("task") == "qa_template" })
    }

    @Test fun datasetStalesWhenSourcesOrRightsChangeAndSurvivesRestart() {
        val rig = TK.rig(); val s = rig.open()
        val p = TK.readyProject(s, 5)
        s.approveDataset(p).ok()
        assertEquals(DatasetStatus.APPROVED, rig.open().datasetPreview(p).ok()!!.status)           // approval persists
        s.ingest(p, listOf(TK.input("late.md", TK.doc(42), "text/markdown")), RightsStatus.OWNER_AUTHORED).ok()
        assertEquals(DatasetStatus.STALE, s.datasetPreview(p).ok()!!.status)
        assertTrue(s.datasetPreview(p).ok()!!.warnings.any { it.contains("rebuild required") })
        s.buildDataset(p).ok()
        assertEquals(2, s.datasetPreview(p).ok()!!.version)
        s.approveDataset(p).ok()
        s.setSourceRights(p, s.listSources(p).ok()[0].sourceId, RightsStatus.REFERENCE_ONLY).ok()
        assertEquals(DatasetStatus.STALE, rig.open().datasetPreview(p).ok()!!.status)
    }

    @Test fun rebuildClearsExportFlagsAndEvaluationBecauseTheyDescribeAnOldDataset() {
        val rig = TK.rig(); val s = rig.open()
        val p = TK.readyProject(s, 6)
        s.approveDataset(p).ok(); TK.verifyLicense(rig, s); s.selectBaseModel(p, TK.MODEL, null).ok(); s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).ok()
        val out = ByteArrayOutputStream(); s.exportTrainingJobPackage(p, out).ok()
        val job = org.json.JSONObject(Zips.readAll(out.toByteArray().inputStream()).getValue("manifest.json").toString(Charsets.UTF_8)).getString("job_id")
        s.importEvaluation(p, ResultsKit.results(p.value, job).inputStream()).ok()
        assertNotNull(s.evaluation(p).ok())
        s.buildDataset(p, DatasetOptions(seed = 5)).ok()
        assertEquals(null, s.evaluation(p).ok())
        val stages = s.getProject(p).ok().stages
        assertEquals(StageStatus.NEEDS_ATTENTION, stages.first { it.id == StageId.DATASET }.status)
        assertEquals(StageStatus.BLOCKED, stages.first { it.id == StageId.TRAINING_PACKAGE }.status)
    }

    @Test fun datasetBuildReportsProgressAndNoSourcesOrRightsBlocks() {
        val rig = TK.rig(); val s = rig.open()
        val p = s.createProject(NewProject("x", "", "")).ok().id
        s.ingest(p, TK.corpus(4), RightsStatus.OWNER_AUTHORED).ok()
        val ticks = ArrayList<Progress>()
        s.buildDataset(p, DatasetOptions()) { ticks.add(it) }.ok()
        assertTrue(ticks.size >= 2 && ticks.last().done == ticks.last().total)
        assertTrue(ticks.all { it.unit == "sources" })
    }

    @Test fun heldOutSetComesOnlyFromHeldOutChunks() {
        val rig = TK.rig(); val s = rig.open()
        val p = TK.readyProject(s, 8)
        s.approveDataset(p).ok()
        val out = ByteArrayOutputStream()
        val e = s.exportHeldOutEvalSet(p, out).ok()
        assertEquals("heldout-eval", e.kind)
        val files = Zips.readAll(out.toByteArray().inputStream())
        assertEquals(emptyList(), Zips.verifyChecksums(files))
        val itemsJson = JobValidator.lines(files["eval/heldout.jsonl"])
        assertTrue(itemsJson.isNotEmpty())
        val heldChunks = items(s, p).filter { it.role == ChunkRole.HELD_OUT_EVAL }.map { it.id }.toSet()
        for (it in itemsJson) for (r in it.getJSONArray("gold_refs").let { a -> (0 until a.length()).map { i -> a.getString(i) } }) assertTrue(r in heldChunks, "gold ref $r is not a held-out chunk")
        assertTrue(itemsJson.all { it.getString("origin") == "source_derived" })
        assertTrue(itemsJson.any { it.getString("kind") == "fact" && it.getJSONObject("expected").getString("unit") == "n-m" })
        assertTrue(e.warnings.any { it.contains("statistically weak") })
    }
}
