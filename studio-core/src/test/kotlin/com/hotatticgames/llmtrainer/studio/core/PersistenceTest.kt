package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.err
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class PersistenceTest {
    @Test fun atomicWriteLeavesTheOldFileIntactWhenTheWriterFails() {
        val d = TK.tmp(); val f = File(d, "x.json")
        Fs.writeJson(f, mapOf("v" to 1))
        assertFailsWith<IllegalStateException> { Fs.writeAtomic(f, keepBackup = true) { it.write("{\"v\":".toByteArray()); error("power cut") } }
        assertEquals(1, JSONObject(f.readText()).getInt("v"))
        assertEquals(listOf("x.json"), d.list()!!.filter { !it.endsWith(".bak") }.sorted(), "no temp file may remain")
    }

    @Test fun jsonReadFallsBackToThePreviousGenerationAndQuarantinesGarbage() {
        val d = TK.tmp(); val f = File(d, "x.json")
        Fs.writeJson(f, mapOf("v" to 1)); Fs.writeJson(f, mapOf("v" to 2))
        assertEquals(2, JSONObject(f.readText()).getInt("v")); assertEquals(1, JSONObject(File(d, "x.json.bak").readText()).getInt("v"))
        f.writeText("{\"v\": 3, trunca")                                   // torn write
        val notes = ArrayList<String>()
        assertEquals(1, Fs.readJson(f) { notes.add(it) }!!.getInt("v"))
        assertTrue(notes.any { it.contains("set aside") } && notes.any { it.contains("restored") })
        assertTrue(d.list()!!.any { it.startsWith("x.json.corrupt-") })          // evidence kept, not deleted
        assertNull(Fs.readJson(File(d, "missing.json")))
    }

    private fun populated(): Triple<TK.Rig, ProjectId, StudioCore> {
        val rig = TK.rig(); val s = rig.open()
        val p = TK.readyProject(s, 6)
        s.approveDataset(p).ok(); TK.verifyLicense(rig, s); s.selectBaseModel(p, TK.MODEL, null).ok(); s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).ok()
        val job = ByteArrayOutputStream(); s.exportTrainingJobPackage(p, job).ok()
        val jid = JSONObject(Zips.readAll(job.toByteArray().inputStream()).getValue("manifest.json").toString(Charsets.UTF_8)).getString("job_id")
        s.importEvaluation(p, ResultsKit.results(p.value, jid, kind = "adapter").inputStream()).ok()
        s.exportSpecialistPackage(p, ByteArrayOutputStream()).ok()
        return Triple(rig, p, s)
    }

    @Test fun everythingSurvivesProcessDeathAndRestart() {
        val (rig, p, s) = populated()
        val reborn = rig.open()
        assertTrue(reborn.startupProblems.isEmpty(), reborn.startupProblems.toString())
        val before = s.getProject(p).ok(); val after = reborn.getProject(p).ok()
        assertEquals(before.stages, after.stages)
        assertEquals(before.baseModelId, after.baseModelId)
        assertEquals(s.listProjects(), reborn.listProjects())
        assertEquals(s.datasetPreview(p).ok(), reborn.datasetPreview(p).ok())
        assertEquals(s.reviewItems(p, ReviewFilter(), 0, 10_000).ok(), reborn.reviewItems(p, ReviewFilter(), 0, 10_000).ok())
        assertEquals(s.evaluation(p).ok(), reborn.evaluation(p).ok())
        assertEquals(s.listSources(p).ok(), reborn.listSources(p).ok())
        assertEquals(s.lastIngestReport(p).ok(), reborn.lastIngestReport(p).ok())
        assertEquals(s.methodOptions(p).ok(), reborn.methodOptions(p).ok())
        assertEquals(s.model(TK.MODEL).ok().license, reborn.model(TK.MODEL).ok().license)
        // and the reborn instance keeps working (exports still validate, identity unchanged)
        val a = ByteArrayOutputStream(); reborn.exportTrainingJobPackage(p, a).ok()
        assertEquals(emptyList(), JobValidator.validate(a.toByteArray()))
        val o = ByteArrayOutputStream(); val e = reborn.exportSpecialistPackage(p, o).ok()
        assertTrue(String(Zips.readAll(o.toByteArray().inputStream()).getValue("specialist_manifest.json")).contains("\"specialist_version\":\"1.0.1\""))   // seq persisted
        assertTrue(e.sizeBytes > 0)
    }

    @Test fun aTornProjectJsonIsRestoredFromTheBackupGeneration() {
        val (rig, p, s) = populated()
        val f = File(rig.dir, "projects/${p.value}/project.json")
        assertTrue(File(f.path + ".bak").isFile)
        f.writeText("{\"schema\":1,\"id\":\"${p.value}\",\"na")
        val reborn = rig.open()
        assertTrue(reborn.startupProblems.any { it.contains("restored from the previous generation") }, reborn.startupProblems.toString())
        val got = reborn.getProject(p).ok()
        assertEquals("Motorcycle Mechanic", got.name)
        assertEquals(s.listSources(p).ok().size, reborn.listSources(p).ok().size)
    }

    @Test fun anUnreadableProjectIsSkippedLoudlyAndItsDataIsLeftAlone() {
        val (rig, p, _) = populated()
        val dir = File(rig.dir, "projects/${p.value}")
        File(dir, "project.json").writeText("garbage"); File(dir, "project.json.bak").delete()
        val other = rig.open().let { it.createProject(NewProject("Other", "", "")).ok().id }
        val reborn = rig.open()
        assertEquals(listOf(other), reborn.listProjects().map { it.id })
        assertTrue(reborn.startupProblems.any { it.contains(p.value) && it.contains("data left in place") }, reborn.startupProblems.toString())
        assertTrue(File(dir, "sources").isDirectory && File(dir, "dataset").isDirectory)
        assertEquals("NOT_FOUND", reborn.getProject(p).err().code)
    }

    @Test fun projectsFromANewerSchemaAreNotTouched() {
        val (rig, p, _) = populated()
        val f = File(rig.dir, "projects/${p.value}/project.json")
        val o = JSONObject(f.readText()); o.put("schema", 99); f.writeText(o.toString())
        val reborn = rig.open()
        assertTrue(reborn.listProjects().isEmpty())
        assertTrue(reborn.startupProblems.any { it.contains("newer app version") })
        assertEquals(99, JSONObject(f.readText()).getInt("schema"))
    }

    @Test fun interruptedDatasetSwapIsRolledBackOrCompleted() {
        val (rig, p, s) = populated()
        val dir = File(rig.dir, "projects/${p.value}")
        val sha = s.datasetPreview(p).ok()!!.datasetSha256
        // crash after the old dataset was moved aside but before the new one was installed
        File(dir, "dataset").renameTo(File(dir, "dataset.old"))
        File(dir, "dataset.tmp").also { it.mkdirs(); File(it, "build.json").writeText("{half written") }
        val reborn = rig.open()
        assertEquals(sha, reborn.datasetPreview(p).ok()!!.datasetSha256)
        assertFalse(File(dir, "dataset.tmp").exists()); assertFalse(File(dir, "dataset.old").exists())
        // crash while the tmp dataset was still being written (old dataset untouched)
        File(dir, "dataset.tmp").also { it.mkdirs(); File(it, "chunks.jsonl").writeText("{") }
        assertEquals(sha, rig.open().datasetPreview(p).ok()!!.datasetSha256)
        assertFalse(File(dir, "dataset.tmp").exists())
    }

    @Test fun aCorruptDatasetDegradesToNoDatasetAndCanBeRebuilt() {
        val (rig, p, _) = populated()
        val dir = File(rig.dir, "projects/${p.value}/dataset")
        File(dir, "chunks.jsonl").writeText("{ broken\n"); File(dir, "build.json").writeText("nope"); File(dir, "build.json.bak").delete()
        val reborn = rig.open()
        assertNull(reborn.datasetPreview(p).ok())
        assertEquals(6, reborn.listSources(p).ok().size)                  // sources are untouched
        reborn.buildDataset(p).ok()
        assertEquals(DatasetStatus.NEEDS_REVIEW, reborn.datasetPreview(p).ok()!!.status)
    }

    @Test fun staleTempFilesFromKilledWritesAreIgnored() {
        val (rig, p, _) = populated()
        val dir = File(rig.dir, "projects/${p.value}")
        File(dir, ".project.json.tmp-12345").writeText("{")
        File(rig.dir, "workspace/.licenses.json.tmp-1").writeText("{")
        val reborn = rig.open()
        assertTrue(reborn.startupProblems.isEmpty(), reborn.startupProblems.toString())
        assertEquals(1, reborn.listProjects().size)
    }

    @Test fun anInterruptedIngestIsReportedOnNextStartWithWhatFinished() {
        val rig = TK.rig(); val s = rig.open()
        val p = s.createProject(NewProject("x", "", "")).ok().id
        s.ingest(p, TK.corpus(3), RightsStatus.OWNER_AUTHORED).ok()
        val f = File(rig.dir, "projects/${p.value}/ingest_report.json")
        val o = JSONObject(f.readText()); o.put("finished", false); f.writeText(o.toString())          // what a process killed mid-batch leaves behind
        val reborn = rig.open()
        assertTrue(reborn.startupProblems.any { it.contains("interrupted after 3 file(s)") }, reborn.startupProblems.toString())
        assertEquals(3, reborn.lastIngestReport(p).ok()!!.items.size)                                  // finished files are kept and visible
        assertEquals(3, reborn.listSources(p).ok().size)
    }

    @Test fun crashBetweenSourceRemovalAndDatasetPurgeIsRepairedOnNextStart() {
        val (rig, p, s) = populated()
        val victim = s.listSources(p).ok()[1].sourceId
        File(rig.dir, "projects/${p.value}/sources/$victim").deleteRecursively()          // the process died right after deleting the source
        val reborn = rig.open()
        assertTrue(reborn.startupProblems.any { it.contains("sources changed after the dataset was built") && it.contains("purged") }, reborn.startupProblems.toString())
        assertEquals(DatasetStatus.STALE, reborn.datasetPreview(p).ok()!!.status)
        assertTrue(reborn.reviewItems(p, ReviewFilter(), 0, 10_000).ok().items.none { it.sourceId == victim })
        assertEquals(DatasetStatus.STALE, rig.open().datasetPreview(p).ok()!!.status)           // and the repair was persisted
        assertTrue(File(rig.dir, "projects/${p.value}/dataset").walkTopDown().filter { it.isFile }.none { it.readText().contains("\"source_id\":\"$victim\"") })
    }

    @Test fun crashBetweenRightsChangeAndStaleMarkIsRepairedOnNextStart() {
        val (rig, p, s) = populated()
        val sid = s.listSources(p).ok()[0].sourceId
        val f = File(rig.dir, "projects/${p.value}/sources/$sid/source.json")
        f.writeText(f.readText().replace("OWNER_AUTHORED", "REFERENCE_ONLY"))                  // rights changed on disk, dataset not yet marked
        val reborn = rig.open()
        assertEquals(DatasetStatus.STALE, reborn.datasetPreview(p).ok()!!.status)
        assertEquals(RightsStatus.REFERENCE_ONLY, reborn.listSources(p).ok().first { it.sourceId == sid }.provenance.rights)
        assertEquals("DATASET_STALE", (reborn.approveDataset(p).err() as StudioError.Blocked).code)
    }

    @Test fun finishedOperationsAreForgottenAfterAMonth() {
        val rig = TK.rig(); val s = rig.open()
        val ops = File(rig.dir, "workspace/operations").also { it.mkdirs() }
        val old = rig.clock.now - 40L * 24 * 3600 * 1000
        File(ops, "op-old.json").writeText("""{"schema":1,"id":"op-old","kind":"DOWNLOAD","state":"SUCCEEDED","subject":"v","done":1,"total":1,"message":"","resumable":false,"created_at":$old,"updated_at":$old,"variant_id":"v#a","file_name":"f.gguf"}""")
        File(ops, "op-new.json").writeText("""{"schema":1,"id":"op-new","kind":"DOWNLOAD","state":"FAILED","subject":"v","done":1,"total":9,"message":"x","resumable":true,"created_at":$old,"updated_at":$old,"variant_id":"v#a","file_name":"f.gguf"}""")
        val reborn = rig.open()
        assertEquals(listOf("op-new"), reborn.operations().map { it.id })               // a resumable failure is never auto-forgotten
        assertFalse(File(ops, "op-old.json").exists())
        assertNotNull(s)
    }

    @Test fun deletedProjectsStayDeletedAndOthersAreUntouched() {
        val rig = TK.rig(); val s = rig.open()
        val a = TK.readyProject(s, 4, name = "A"); val b = TK.readyProject(s, 4, name = "B")
        s.deleteProject(a).ok()
        assertFalse(File(rig.dir, "projects/${a.value}").exists())
        val reborn = rig.open()
        assertEquals(listOf("B"), reborn.listProjects().map { it.name })
        assertEquals(DatasetStatus.NEEDS_REVIEW, reborn.datasetPreview(b).ok()!!.status)
    }

    @Test fun projectIdsAreSafeForFileNamesAndPackageIdentifiers() {
        val s = TK.rig().open()
        for (name in listOf("Motorcycle Mechanic", "日本語 プロジェクト", "../../etc/passwd", "a".repeat(120), "  spaced  ", "emoji 😀 test", "HVAC/Tech\\nician")) {
            val p = s.createProject(NewProject(name, "d", "p")).ok()
            assertTrue(Regex(Packages.ID_PATTERN).matches(p.id.value), "${p.id.value} for '$name'")
            assertEquals(name.trim(), p.name)
        }
        assertEquals("NAME_REQUIRED", s.createProject(NewProject("   ", "", "")).err().code)
        assertEquals("TOO_LONG", s.createProject(NewProject("x".repeat(500), "", "")).err().code)
        assertEquals(7, s.listProjects().size)
        assertEquals(7, s.listProjects().map { it.id }.toSet().size)
    }

    @Test fun concurrentUseOfDifferentProjectsAndTheCatalogIsSafe() {
        val rig = TK.rig(); val s = rig.open()
        val ids = (1..3).map { s.createProject(NewProject("P$it", "d", "p")).ok().id }
        val pool = Executors.newFixedThreadPool(6)
        val start = CountDownLatch(1)
        val errors = java.util.concurrent.ConcurrentLinkedQueue<Throwable>()
        fun go(f: () -> Unit) = pool.submit { try { start.await(); f() } catch (t: Throwable) { errors.add(t) } }
        ids.forEachIndexed { i, p -> go { s.ingest(p, TK.corpus(5), RightsStatus.OWNER_AUTHORED).ok(); s.buildDataset(p).ok(); s.approveDataset(p).ok() } }
        go { repeat(20) { s.listProjects(); s.catalog(); s.recommendations(); s.operations() } }
        go { repeat(20) { ids.forEach { p -> s.getProject(p).ok(); s.datasetPreview(p).ok() } } }
        go { repeat(5) { s.createProject(NewProject("Extra$it", "", "")).ok() } }
        start.countDown(); pool.shutdown()
        assertTrue(pool.awaitTermination(120, TimeUnit.SECONDS))
        assertEquals(emptyList(), errors.toList().map { it.toString() })
        ids.forEach { assertEquals(DatasetStatus.APPROVED, s.datasetPreview(it).ok()!!.status) }
        assertEquals(8, rig.open().listProjects().size)
    }

    @Test fun noPublicCallThrowsOnHostileArguments() {
        val s = TK.rig().open()
        val bogus = ProjectId("nope")
        val out = ByteArrayOutputStream()
        val results = listOf<StudioResult<*>>(
            s.getProject(bogus), s.deleteProject(bogus), s.selectBaseModel(bogus, "m", null), s.listSources(bogus), s.ingest(bogus, emptyList()), s.lastIngestReport(bogus),
            s.setSourceRights(bogus, "s", RightsStatus.UNSET), s.removeSource(bogus, "s"), s.buildDataset(bogus), s.datasetPreview(bogus), s.reviewItems(bogus), s.setIncluded(bogus, emptyList(), true),
            s.approveDataset(bogus), s.methodOptions(bogus), s.selectMethod(bogus, "m"), s.exportTrainingJobPackage(bogus, out), s.exportReferencePackage(bogus, out), s.exportHeldOutEvalSet(bogus, out),
            s.importEvaluation(bogus, ByteArray(0).inputStream()), s.evaluation(bogus), s.exportSpecialistPackage(bogus, out), s.importSpecialistPackage(ByteArray(0).inputStream()),
            s.model(""), s.fetchLicenseText(""), s.importLicenseText("", "f", ByteArray(0).inputStream()), s.attestLicense("", LicenseAttestation("", emptyMap(), false)),
            s.planAcquisition("", IntendedUse()), s.startDownload("", IntendedUse(), true), s.importModelFile("", SourceInput("n", "m", 0) { ByteArray(0).inputStream() }),
            s.operation(""), s.cancelOperation(""), s.resumeOperation(""))
        assertTrue(results.all { it is StudioResult.Err }, results.indexOfFirst { it is StudioResult.Ok }.toString())
        val p = s.createProject(NewProject("x", "", "")).ok().id
        s.reviewItems(p, ReviewFilter(), -5, -1).ok()
        s.reviewItems(p, ReviewFilter(), Int.MAX_VALUE, Int.MAX_VALUE).ok()
    }

    @Test fun hostCallsNeverClaimWhatTheCoreDoesNotHave() {
        val s = TK.rig().open()
        var got: UpdateStatus? = null
        s.host.checkForUpdates { got = it }
        assertEquals("error", got!!.state)
        assertTrue(s.host.diagnosticsJson().contains("studio-core"))
        assertEquals(TK.snapshot(), s.host.deviceSnapshotJson())
        s.host.restartApp()
        var injected: UpdateStatus? = null
        val h = object : HostHooks { override fun diagnosticsJson() = "{\"x\":1}"; override fun deviceSnapshotJson() = "{}"; override fun checkForUpdates(callback: (UpdateStatus) -> Unit) = callback(UpdateStatus("up-to-date", "ok")); override fun restartApp() {} }
        TK.rig().open(host = h).host.checkForUpdates { injected = it }
        assertEquals("up-to-date", injected!!.state)
    }

    @Test fun factoryReturnsTheRealCoreAndKeepsItsSignature() {
        val dir = TK.tmp()
        val s: Studio = StudioFactory.create(dir) { TK.snapshot() }
        assertTrue(s is StudioCore)
        assertTrue(s.listProjects().isEmpty())
        s.createProject(NewProject("F", "", "")).ok()
        assertEquals(1, StudioFactory.create(dir) { TK.snapshot() }.listProjects().size)
        assertNotNull(StudioFactory.createWithHost(dir, { TK.snapshot() }, s.host))
    }
}
