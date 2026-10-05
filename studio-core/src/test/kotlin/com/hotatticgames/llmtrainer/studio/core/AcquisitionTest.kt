package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.err
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import java.io.File
import java.util.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class AcquisitionTest {
    private val url = "https://models.example.org/qwen3-1.7b-q4.gguf"
    private val use = IntendedUse()

    private class Env(val rig: TK.Rig, var core: StudioCore, val vid: String, val data: ByteArray, val sha: String)

    private fun env(size: Int = 200_000, runner: TaskRunner = InlineTaskRunner, withSha: Boolean = true, verified: Boolean = true, sizeKnown: Boolean = true, dir: File = TK.tmp()): Env {
        val rig = TK.rig(runner, dir)
        val data = ByteArray(size).also { Random(3).nextBytes(it) }
        val sha = Hashing.sha256(data)
        val ws = File(rig.dir, "workspace").also { it.mkdirs() }
        File(ws, "catalog_overrides.json").writeText("""{"variants":{"${TK.MODEL}":[{"variant_id":"gguf-test","format":"GGUF","quantization":"Q4_K_M","source_url":"$url",
            ${if (sizeKnown) "\"size_bytes\":$size," else ""} ${if (withSha) "\"sha256\":\"$sha\"," else ""} "size_evidence":"test"}]}}""")
        rig.http.resources[url] = data
        val core = rig.open()
        if (verified) TK.verifyLicense(rig, core)
        return Env(rig, core, "${TK.MODEL}#gguf-test", data, sha)
    }

    @Test fun downloadIsBlockedUntilLicenseVerifiedAndNeverTouchesTheNetwork() {
        val e = env(verified = false)
        val plan = e.core.planAcquisition(e.vid, use).ok()
        assertFalse(plan.allowed)
        assertTrue(plan.blocking.any { it.code == LicenseCodes.UNVERIFIED })
        val err = e.core.startDownload(e.vid, use, true).err() as StudioError.Blocked
        assertEquals("DOWNLOAD_BLOCKED", err.code)
        assertTrue(err.reasons.any { it.code == LicenseCodes.UNVERIFIED })
        assertTrue(e.rig.http.requests.none { it == url })
        assertTrue(e.core.operations().isEmpty())
    }

    @Test fun downloadNeedsExplicitConfirmation() {
        val e = env()
        val plan = e.core.planAcquisition(e.vid, use).ok()
        assertTrue(plan.allowed, plan.blocking.toString()); assertTrue(plan.requiresExplicitConfirmation)
        val err = e.core.startDownload(e.vid, use, false).err() as StudioError.Blocked
        assertEquals("CONFIRMATION_REQUIRED", err.code)
        assertTrue(e.rig.http.requests.none { it == url })
    }

    @Test fun planReportsSizeStorageImpactDeviceCompatAndLicense() {
        val e = env()
        val p = e.core.planAcquisition(e.vid, use).ok()
        assertEquals(200_000L, p.sizeBytes)
        assertEquals(url, p.url)
        assertEquals(LicenseState.VERIFIED, p.licenseState)
        assertEquals(p.freeStorageBytes - p.sizeBytes, p.storageAfterBytes)
        assertTrue(p.storageReserveBytes >= 2048L * 1024 * 1024)       // qualifier's storage reserve
        assertNotNull(p.deviceCompat.verdict)
        assertEquals(use, p.intendedUse)
    }

    @Test fun repositoryPagesAreNotDownloadable() {
        val e = env()
        val hf = e.core.model(TK.MODEL).ok().variants.first { it.id.endsWith("#hf-weights") }
        val plan = e.core.planAcquisition(hf.id, use).ok()
        assertFalse(plan.allowed)
        assertNull(plan.url)
        assertTrue(plan.blocking.any { it.code == "NO_DOWNLOAD_URL" })
        // size unknown: the plan carries an estimate, never zero, and says so
        assertTrue(plan.sizeBytes >= 1L shl 30)
        assertTrue(plan.deviceCompat.reasons.any { it.contains("ESTIMATE") })
    }

    @Test fun happyPathVerifiesHashAndRecordsAcquisition() {
        val e = env()
        val op = e.core.startDownload(e.vid, use, true).ok()
        val done = e.core.operation(op.id).ok()
        assertEquals(OperationState.SUCCEEDED, done.state, done.message)
        assertEquals(200_000L, done.progress.done); assertEquals(200_000L, done.progress.total); assertFalse(done.resumable)
        val v = e.core.model(TK.MODEL).ok().variants.first { it.id == e.vid }
        assertTrue(v.acquired)
        val rec = File(e.rig.dir, "workspace/models").walkTopDown().first { it.name == "qwen3-1.7b-q4.gguf" }
        assertEquals(e.sha, Hashing.sha256File(rec))
        assertTrue(File(e.rig.dir, "workspace/models").walkTopDown().none { it.name.endsWith(".part") })
        assertEquals("Conflict", e.core.startDownload(e.vid, use, true).err().let { if (it is StudioError.Conflict) "Conflict" else it.code })
    }

    @Test fun checksumMismatchRemovesTheFileAndFails() {
        val e = env()
        e.rig.http.resources[url] = e.data.copyOf().also { it[10] = (it[10] + 1).toByte() }
        val op = e.core.startDownload(e.vid, use, true).ok()
        val done = e.core.operation(op.id).ok()
        assertEquals(OperationState.FAILED, done.state)
        assertEquals("HASH_MISMATCH", done.error!!.code)
        assertFalse(done.resumable)
        assertTrue(File(e.rig.dir, "workspace/models").walkTopDown().none { it.isFile && it.name.startsWith("qwen3") })
        assertFalse(e.core.model(TK.MODEL).ok().variants.first { it.id == e.vid }.acquired)
    }

    @Test fun interruptedDownloadResumesWithARangeRequestFromThePartFile() {
        val e = env()
        e.rig.http.failAfterBytes = 70_000
        val op = e.core.startDownload(e.vid, use, true).ok()
        val failed = e.core.operation(op.id).ok()
        assertEquals(OperationState.FAILED, failed.state)
        assertEquals("NETWORK_ERROR", failed.error!!.code)
        assertTrue(failed.resumable)
        val part = File(e.rig.dir, "workspace/models").walkTopDown().first { it.name.endsWith(".part") }
        val partial = part.length()
        assertTrue(partial in 70_000L..71_999L, "partial=$partial")
        val resumed = e.core.resumeOperation(op.id).ok()
        val done = e.core.operation(resumed.id).ok()
        assertEquals(OperationState.SUCCEEDED, done.state, done.message)
        assertEquals(listOf(0L, partial), e.rig.http.rangesSeen)                     // second request resumed at the partial length
        assertEquals(e.sha, Hashing.sha256File(File(e.rig.dir, "workspace/models").walkTopDown().first { it.name == "qwen3-1.7b-q4.gguf" }))
    }

    @Test fun serverThatIgnoresRangeRestartsFromZeroAndStillVerifies() {
        val e = env()
        e.rig.http.failAfterBytes = 50_000
        val op = e.core.startDownload(e.vid, use, true).ok()
        e.rig.http.ignoreRange = true
        e.core.resumeOperation(op.id).ok()
        assertEquals(OperationState.SUCCEEDED, e.core.operation(op.id).ok().state)
    }

    @Test fun processDeathMidDownloadReloadsPausedAndResumes() {
        val dir = TK.tmp()
        val runner = ManualRunner()
        val e = env(runner = runner, dir = dir)
        val op = e.core.startDownload(e.vid, use, true).ok()
        assertEquals(OperationState.QUEUED, op.state)
        // simulate a half-finished download that was running when the process died
        e.rig.http.failAfterBytes = 90_000
        runner.runAll()
        val part = File(dir, "workspace/models").walkTopDown().first { it.name.endsWith(".part") }
        val partial = part.length()
        assertTrue(partial in 90_000L..91_999L, "partial=$partial")
        // rewrite the persisted state exactly as a killed process would have left it: RUNNING
        val f = File(dir, "workspace/operations/${op.id}.json")
        f.writeText(f.readText().replace("\"state\":\"FAILED\"", "\"state\":\"RUNNING\""))
        val reborn = e.rig.open()                       // new process, same directory
        val r = reborn.operation(op.id).ok()
        assertEquals(OperationState.PAUSED, r.state)
        assertTrue(r.resumable)
        assertEquals(partial, r.progress.done)
        assertTrue(r.message.contains("Interrupted"))
        assertEquals(1, reborn.operations().size)
        reborn.resumeOperation(op.id).ok()
        runner.runAll()
        assertEquals(OperationState.SUCCEEDED, reborn.operation(op.id).ok().state)
        assertEquals(partial, e.rig.http.rangesSeen.last())
    }

    @Test fun queuedDownloadSurvivesRestartAsPaused() {
        val dir = TK.tmp(); val runner = ManualRunner()
        val e = env(runner = runner, dir = dir)
        val op = e.core.startDownload(e.vid, use, true).ok()
        val reborn = e.rig.open()
        assertEquals(OperationState.PAUSED, reborn.operation(op.id).ok().state)
        reborn.resumeOperation(op.id).ok()
        runner.queue.clear()                       // the old process's queued task died with it; the resumed one is the last submitted
        e.rig.runner.let { }
        val last = ManualRunner()
        val rig2 = TK.Rig(dir, e.rig.http, e.rig.clock, e.rig.storage, last, e.rig.ids) { TK.snapshot() }
        val again = rig2.open()
        again.resumeOperation(op.id).ok()
        last.runAll()
        assertEquals(OperationState.SUCCEEDED, again.operation(op.id).ok().state)
    }

    @Test fun cancelStopsADownloadAndRemovesThePartialFile() {
        val e = env()
        e.rig.http.failAfterBytes = 40_000
        val op = e.core.startDownload(e.vid, use, true).ok()             // fails with a resumable partial
        assertTrue(File(e.rig.dir, "workspace/models").walkTopDown().any { it.name.endsWith(".part") })
        val c = e.core.cancelOperation(op.id).ok()
        assertEquals(OperationState.CANCELLED, c.state)
        assertFalse(c.resumable)
        assertTrue(File(e.rig.dir, "workspace/models").walkTopDown().none { it.name.endsWith(".part") })
        assertTrue(e.core.cancelOperation(op.id).err() is StudioError.Conflict)
        assertTrue(e.core.resumeOperation(op.id).err() is StudioError.Conflict)
    }

    @Test fun cancelWhileRunningIsObservedByTheTransfer() {
        val rig = TK.rig(InlineTaskRunner)
        val cancelAfter = Random(1)
        val data = ByteArray(300_000).also { cancelAfter.nextBytes(it) }
        File(rig.dir, "workspace").mkdirs()
        File(rig.dir, "workspace/catalog_overrides.json").writeText("""{"variants":{"${TK.MODEL}":[{"variant_id":"gguf-test","format":"GGUF","source_url":"$url","size_bytes":300000}]}}""")
        val http = object : Http {
            lateinit var core: StudioCore
            override fun getBytes(url: String, maxBytes: Long, timeoutMs: Int) = rig.http.getBytes(url, maxBytes, timeoutMs)
            override fun downloadToFile(url: String, part: File, maxBytes: Long, cancelled: () -> Boolean, onProgress: (Long, Long) -> Unit, timeoutMs: Int): DownloadOutcome {
                java.io.RandomAccessFile(part, "rw").use { raf ->
                    var pos = 0
                    while (pos < data.size) {
                        if (cancelled()) throw HttpException(HttpException.Kind.CANCELLED, "Cancelled")
                        raf.write(data, pos, 1024); pos += 1024
                        onProgress(pos.toLong(), data.size.toLong())
                        if (pos == 10_240) core.cancelOperation(core.operations().single().id)       // user taps Cancel mid-transfer
                    }
                }
                return DownloadOutcome(data.size.toLong(), data.size.toLong())
            }
        }
        rig.http.resources["https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/LICENSE"] = "lic".toByteArray()
        val core = StudioCore(rig.dir, { TK.snapshot() }, null, http, rig.clock, rig.storage, InlineTaskRunner, rig.ids, "t", com.hotatticgames.llmtrainer.qualify.SafetyPolicy())
        http.core = core
        TK.verifyLicense(rig, core)
        val vid = "${TK.MODEL}#gguf-test"
        val op = core.startDownload(vid, use, true).ok()
        val done = core.operation(op.id).ok()
        assertEquals(OperationState.CANCELLED, done.state, done.message)
        assertTrue(File(rig.dir, "workspace/models").walkTopDown().none { it.isFile && (it.name.endsWith(".part") || it.name.endsWith(".gguf")) })
    }

    @Test fun storageHeadroomIsEnforcedAtPlanStartAndResume() {
        val e = env()
        e.rig.storage.free = 2048L * 1024 * 1024 + 100_000                 // reserve (2 GiB min) + less than the file
        val plan = e.core.planAcquisition(e.vid, use).ok()
        assertFalse(plan.allowed)
        assertTrue(plan.blocking.any { it.code == "INSUFFICIENT_STORAGE" && it.message.contains("safety reserve") })
        assertEquals("DOWNLOAD_BLOCKED", e.core.startDownload(e.vid, use, true).err().code)
        // enough at start, then the disk fills up before a resume
        e.rig.storage.free = 100L * 1024 * 1024 * 1024
        e.rig.http.failAfterBytes = 30_000
        val op = e.core.startDownload(e.vid, use, true).ok()
        e.rig.storage.free = 2048L * 1024 * 1024 + 10_000
        val r = e.core.resumeOperation(op.id)
        assertTrue(r.err() is StudioError.Blocked)
    }

    @Test fun nonHttpsSourceIsRefused() {
        val rig = TK.rig()
        File(rig.dir, "workspace").mkdirs()
        File(rig.dir, "workspace/catalog_overrides.json").writeText("""{"variants":{"${TK.MODEL}":[{"variant_id":"plain","format":"GGUF","source_url":"http://models.example.org/x.gguf","size_bytes":1000}]}}""")
        val core = rig.open(); TK.verifyLicense(rig, core)
        val plan = core.planAcquisition("${TK.MODEL}#plain", use).ok()
        assertFalse(plan.allowed)
        assertTrue(plan.blocking.any { it.code == "NOT_HTTPS" })
        assertTrue(core.startDownload("${TK.MODEL}#plain", use, true).err() is StudioError.Blocked)
    }

    @Test fun onlyOneActiveDownloadPerVariant() {
        val runner = ManualRunner()
        val e = env(runner = runner)
        e.core.startDownload(e.vid, use, true).ok()
        assertTrue(e.core.startDownload(e.vid, use, true).err() is StudioError.Conflict)
    }

    @Test fun licenseIsRecheckedWhenResuming() {
        val e = env()
        e.rig.http.failAfterBytes = 20_000
        val op = e.core.startDownload(e.vid, use, true).ok()
        // owner re-reads the text and attests "no fine-tuning": a paused download must not silently continue
        val lic = e.core.model(TK.MODEL).ok().license
        e.core.attestLicense(TK.MODEL, LicenseAttestation(lic.textSha256!!, TK.allYes() + (Permission.FINE_TUNE to Tri.NO), true)).ok()
        val err = e.core.resumeOperation(op.id).err() as StudioError.Blocked
        assertTrue(err.reasons.any { it.code == LicenseCodes.DISALLOWED })
    }

    @Test fun unknownVariantsAreNotFound() {
        val e = env()
        assertEquals("NOT_FOUND", e.core.planAcquisition("nope#x", use).err().code)
        assertEquals("NOT_FOUND", e.core.resumeOperation("op-x").err().code)
        assertEquals("NOT_FOUND", e.core.cancelOperation("op-x").err().code)
        assertEquals("NOT_FOUND", e.core.operation("op-x").err().code)
    }

    // ---- import from a stream -----------------------------------------------------------------------------------------

    @Test fun importVerifiesAgainstTheRecordedChecksum() {
        val e = env()
        val op = e.core.importModelFile(e.vid, TK.bytesInput("model.gguf", e.data, "application/octet-stream")).ok()
        val done = e.core.operation(op.id).ok()
        assertEquals(OperationState.SUCCEEDED, done.state, done.message)
        assertTrue(e.core.model(TK.MODEL).ok().variants.first { it.id == e.vid }.acquired)
        val bad = e.core.importModelFile(e.vid, TK.bytesInput("other.gguf", e.data.copyOf().also { it[0] = 1 }, "application/octet-stream")).ok()
        assertEquals("HASH_MISMATCH", e.core.operation(bad.id).ok().error!!.code)
    }

    @Test fun importWithoutAKnownHashRecordsTheComputedOne() {
        val e = env(withSha = false)
        val op = e.core.importModelFile(e.vid, TK.bytesInput("m.gguf", e.data, "application/octet-stream")).ok()
        assertEquals(OperationState.SUCCEEDED, e.core.operation(op.id).ok().state)
        assertTrue(e.core.operation(op.id).ok().message.contains("no registry checksum"))
        val rec = File(e.rig.dir, "workspace/models").walkTopDown().first { it.name == "acquired.json" }.readText()
        assertTrue(rec.contains(e.sha))
    }

    @Test fun importRefusesEmptyHugeAndDisallowed() {
        val e = env()
        assertEquals("EMPTY_FILE", e.core.importModelFile(e.vid, TK.bytesInput("e.gguf", ByteArray(0))).err().code)
        e.rig.storage.free = 1024L * 1024 * 1024
        assertEquals("INSUFFICIENT_STORAGE", e.core.importModelFile(e.vid, TK.bytesInput("m.gguf", e.data)).err().code)
        e.rig.storage.free = 100L * 1024 * 1024 * 1024
        val lic = e.core.model(TK.MODEL).ok().license
        e.core.attestLicense(TK.MODEL, LicenseAttestation(lic.textSha256!!, TK.allYes() + (Permission.ADAPTER to Tri.NO), true)).ok()
        assertEquals("LICENSE_DISALLOWED", e.core.importModelFile(e.vid, TK.bytesInput("m.gguf", e.data)).err().code)
        assertEquals("NOT_FOUND", e.core.importModelFile("zzz#a", TK.bytesInput("m.gguf", e.data)).err().code)
    }

    @Test fun readFailureDuringImportFailsCleanlyWithoutLeavingPartFiles() {
        val e = env()
        val bad = SourceInput("m.gguf", "application/octet-stream", 100_000) { object : java.io.InputStream() {
            var n = 0
            override fun read(): Int { if (n++ > 5000) throw java.io.IOException("device unplugged"); return 1 }
        } }
        val op = e.core.importModelFile(e.vid, bad).ok()
        val done = e.core.operation(op.id).ok()
        assertEquals(OperationState.FAILED, done.state)
        assertEquals("IO_ERROR", done.error!!.code)
        assertTrue(File(e.rig.dir, "workspace/models").walkTopDown().none { it.name.endsWith(".part") })
    }

    @Test fun interruptedImportReloadsAsFailedNotResumable() {
        val dir = TK.tmp(); val runner = ManualRunner()
        val e = env(runner = runner, dir = dir)
        val op = e.core.importModelFile(e.vid, TK.bytesInput("m.gguf", e.data)).ok()
        val reborn = e.rig.open()
        val r = reborn.operation(op.id).ok()
        assertEquals(OperationState.FAILED, r.state); assertFalse(r.resumable)
        assertTrue(r.message.contains("choose the file again"))
        assertTrue(reborn.resumeOperation(op.id).err() is StudioError.Conflict)
    }

    @Test fun corruptOperationFileIsIgnoredNotFatal() {
        val e = env()
        e.core.startDownload(e.vid, use, true).ok()
        val files = File(e.rig.dir, "workspace/operations").listFiles()!!
        files.forEach { it.writeText("{ not json"); File(it.path + ".bak").delete() }
        val reborn = e.rig.open()
        assertTrue(reborn.operations().isEmpty())
        assertTrue(reborn.startupProblems.any { it.contains("unreadable") || it.contains("set aside") }, reborn.startupProblems.toString())
    }
}
