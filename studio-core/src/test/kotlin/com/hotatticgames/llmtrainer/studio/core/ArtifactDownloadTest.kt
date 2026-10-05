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

/** Downloader/store hardening for registry artifacts: exact URL + hash, license evidence, recovery, accounting, uninstall, manual import. */
class ArtifactDownloadTest {
    private val base = "Qwen/Qwen3-1.7B"
    private val repo = "Qwen/Qwen3-1.7B-GGUF"
    private val file = "Qwen3-1.7B-Q8_0.gguf"
    private val url = "https://huggingface.co/$repo/resolve/${ArtifactKit.REV}/$file"
    private val vid = "${TK.MODEL}#qwen3-1.7b-q8_0"
    private val use = IntendedUse()

    private class Env(val rig: TK.Rig, val core: StudioCore, val data: ByteArray, val sha: String) {
        val models get() = core.models
        fun modelFiles() = ModelFiles(File(rig.dir, "workspace/models"))
    }

    private fun env(size: Int = 300_000, runner: TaskRunner = InlineTaskRunner, verified: Boolean = true, serve: Boolean = true, dataOverride: ByteArray? = null,
                    configure: (TK.Rig) -> Unit = {}, artifactExtra: Map<String, Any?> = emptyMap(), match: String = "exact", cardAgrees: Boolean = true,
                    canonicalSha: String = ArtifactKit.APACHE_SHA, textSha: String = ArtifactKit.APACHE_SHA): Env {
        val rig = TK.rig(runner)
        configure(rig)
        val data = ByteArray(size).also { Random(5).nextBytes(it) }
        val sha = Hashing.sha256(data)
        val art = ArtifactKit.refreshed("qwen3-1.7b-q8_0", base, repo, file, size.toLong(), sha, licenseVerified = verified, match = match, cardAgrees = cardAgrees,
            canonicalSha = canonicalSha, textSha = textSha, extra = artifactExtra)
        if (serve) rig.http.resources[url] = dataOverride ?: data
        return Env(rig, ArtifactKit.open(rig, listOf("qwen3-1.7b-q8_0.json" to art)), data, sha)
    }

    // ---- shipped (unrefreshed) registry ----------------------------------------------------------------------------

    @Test fun shippedUnrefreshedArtifactsAreNeverDownloadable() {
        val rig = TK.rig()
        val core = rig.open()
        val arts = core.models.modelChoices().artifacts
        assertTrue(arts.size >= 10)
        for (a in arts) {
            assertEquals("unrefreshed", a.catalogState, a.artifactId); assertFalse(a.downloadable, a.artifactId); assertFalse(a.canDownload, a.artifactId)
            val v = core.model(a.modelId).ok().variants.first { it.id == a.variantId }
            assertNull(v.downloadUrl); assertNull(v.sha256)
            assertTrue(v.provenance.note!!.contains("not been refreshed"), v.provenance.note)
            val plan = core.planAcquisition(a.variantId, use).ok()
            assertFalse(plan.allowed); assertTrue(plan.blocking.any { it.code == "ARTIFACT_UNREFRESHED" }, a.artifactId)
            val e = core.startDownload(a.variantId, use, true).err() as StudioError.Blocked
            assertEquals("DOWNLOAD_BLOCKED", e.code)
        }
        assertTrue(rig.http.requests.isEmpty(), "nothing may be fetched for an unrefreshed artifact")
        val st = core.models.catalogStatus()
        assertEquals(0, st.downloadableCount); assertEquals(st.artifactCount, st.unrefreshedCount)
        assertEquals(LicenseState.UNVERIFIED, core.model(TK.MODEL).ok().license.state)    // nothing is VERIFIED before the CI refresh
        assertTrue(core.startupProblems.isEmpty(), core.startupProblems.toString())
    }

    // ---- license rule ----------------------------------------------------------------------------------------------

    @Test fun ciVerifiedApacheLicenseUnlocksDownloadWithCiEvidence() {
        val e = env()
        val lic = e.core.model(TK.MODEL).ok().license
        assertEquals(LicenseState.VERIFIED, lic.state); assertEquals(Evidence.CI_CANONICAL, lic.evidenceLevel)
        assertTrue(lic.scopeText!!.contains("byte-identical") || lic.scopeText!!.contains("exact"))
        assertEquals(ArtifactKit.APACHE_SHA, lic.textSha256)
        val plan = e.core.planAcquisition(vid, use).ok()
        assertTrue(plan.allowed, plan.blocking.toString()); assertEquals(url, plan.url); assertEquals(300_000L, plan.sizeBytes)
        assertTrue(plan.requiresExplicitConfirmation)
    }

    @Test fun licenseStaysUnverifiedUnlessEveryConditionHolds() {
        val cases = listOf(
            "text differs from canonical" to env(match = "mismatch"),
            "model card disagrees" to env(cardAgrees = false),
            "canonical hash is not the one this build ships" to env(canonicalSha = "cd".repeat(32)),
            "exact match claimed but text hash differs" to env(textSha = "cd".repeat(32)),
            "CI says unverified" to env(verified = false),
            "unknown match mode" to env(match = "fuzzy"),
        )
        for ((why, e) in cases) {
            assertEquals(LicenseState.UNVERIFIED, e.core.model(TK.MODEL).ok().license.state, why)
            val plan = e.core.planAcquisition(vid, use).ok()
            assertFalse(plan.allowed, why); assertTrue(plan.blocking.any { it.code == LicenseCodes.UNVERIFIED }, why)
        }
    }

    @Test fun oneUnverifiedArtifactOfTheRepoKeepsTheWholeModelUnverified() {
        val rig = TK.rig()
        val d = ByteArray(1000)
        val a1 = ArtifactKit.refreshed("a1", base, repo, "a1.gguf", 1000, Hashing.sha256(d))
        val a2 = ArtifactKit.refreshed("a2", base, repo, "a2.gguf", 1000, Hashing.sha256(d), cardAgrees = false)
        val core = ArtifactKit.open(rig, listOf("a1.json" to a1, "a2.json" to a2))
        assertEquals(LicenseState.UNVERIFIED, core.model(TK.MODEL).ok().license.state)
    }

    @Test fun ownerAttestationAndDisallowedStillWinOverCiEvidence() {
        val e = env()
        // a registry DISALLOWED entry is never upgraded by CI evidence
        val dis = EmbeddedRegistry.files.map { (n, j) -> if (n.startsWith("qwen3-1.7b")) n to j.replace(Regex("\"state\"\\s*:\\s*\"UNVERIFIED\""), "\"state\":\"DISALLOWED\",\"disallowed_reason\":\"test\"") else n to j }
        val core = StudioCore(e.rig.dir, { e.rig.snap() }, null, e.rig.http, e.rig.clock, e.rig.storage, e.rig.runner, e.rig.ids, "test",
            com.hotatticgames.llmtrainer.qualify.SafetyPolicy(), dis,
            listOf("x.json" to ArtifactKit.refreshed("qwen3-1.7b-q8_0", base, repo, file, 300_000, e.sha)), EmbeddedArtifacts.canonicalLicenses)
        assertEquals(LicenseState.DISALLOWED, core.model(TK.MODEL).ok().license.state)
    }

    // ---- download verification and recovery ------------------------------------------------------------------------

    @Test fun happyPathWritesExactUrlVerifiesHashAndRecordsTheInstallManifest() {
        val e = env()
        val op = e.core.startDownload(vid, use, true).ok()
        val done = e.core.operation(op.id).ok()
        assertEquals(OperationState.SUCCEEDED, done.state, done.message)
        assertEquals(listOf(url), e.rig.http.requests.filter { it == url })
        val inst = e.models.installedModels().single()
        assertEquals(vid, inst.variantId); assertEquals("qwen3-1.7b-q8_0", inst.artifactId); assertEquals(e.sha, inst.sha256); assertEquals(300_000L, inst.sizeBytes)
        assertEquals("download", inst.source); assertEquals(InstallProvenance.VERIFIED_DOWNLOAD, inst.provenance); assertEquals(url, inst.url)
        assertEquals(ArtifactKit.REV, inst.revision); assertEquals(LicenseState.VERIFIED, inst.licenseStateAtInstall)
        assertTrue(File(inst.path).isFile && File(inst.path).length() == 300_000L)
        assertEquals(inst.path, e.models.installedPath(vid))
        val manifest = Fs.readJson(e.modelFiles().acquiredRecord(vid))!!
        assertEquals(2, manifest.int("schema")); assertEquals(Evidence.CI_CANONICAL, manifest.str("license_evidence_level")); assertEquals(true, manifest.bool("checksum_verified_against_registry"))
        assertTrue(manifest.str("path")!!.endsWith(file)); assertEquals(e.sha, manifest.str("expected_sha256"))
        assertTrue(e.modelFiles().partialFiles().isEmpty())
    }

    @Test fun hashMismatchDeletesTheFileAndNothingBecomesInstalled() {
        val e = env(dataOverride = ByteArray(300_000) { 1 })
        val op = e.core.startDownload(vid, use, true).ok()
        val done = e.core.operation(op.id).ok()
        assertEquals(OperationState.FAILED, done.state)
        assertEquals("HASH_MISMATCH", (done.error as StudioError.Invalid).code)
        assertFalse(done.resumable)
        assertTrue(e.models.installedModels().isEmpty()); assertNull(e.models.installedPath(vid))
        assertTrue(File(e.rig.dir, "workspace/models").walkTopDown().none { it.isFile && (it.name.endsWith(".part") || it.name == file) })
    }

    @Test fun truncatedServerFileIsASizeMismatchNotAnInstall() {
        val e = env(dataOverride = ByteArray(250_000).also { Random(5).nextBytes(it) })
        val done = e.core.operation(e.core.startDownload(vid, use, true).ok().id).ok()
        assertEquals(OperationState.FAILED, done.state)
        assertEquals("SIZE_MISMATCH", (done.error as StudioError.Invalid).code)
        assertTrue(e.models.installedModels().isEmpty())
    }

    @Test fun corruptResumedPartialIsDiscardedAndTheDownloadRestartsOnceFromZero() {
        val e = env()
        val dir = e.modelFiles().dirOf(vid).also { it.mkdirs() }
        File(dir, "$file.part").writeBytes(ByteArray(50_000) { 9 })           // garbage that a previous run left behind
        val done = e.core.operation(e.core.startDownload(vid, use, true).ok().id).ok()
        assertEquals(OperationState.SUCCEEDED, done.state, done.message)
        assertEquals(listOf(50_000L, 0L), e.rig.http.rangesSeen)               // resumed first, then restarted from zero
        assertEquals(e.sha, Hashing.sha256File(File(e.models.installedPath(vid)!!)))
    }

    @Test fun aFreshDownloadThatFailsVerificationIsNotRetriedForever() {
        val e = env(dataOverride = ByteArray(300_000) { 2 })
        e.core.startDownload(vid, use, true).ok()
        assertEquals(1, e.rig.http.rangesSeen.size, "no restart loop for a download that was not resumed")
    }

    @Test fun oversizedPartialFileIsDeletedBeforeAnyRangeRequest() {
        val e = env()
        File(e.modelFiles().dirOf(vid).also { it.mkdirs() }, "$file.part").writeBytes(ByteArray(400_000))
        val done = e.core.operation(e.core.startDownload(vid, use, true).ok().id).ok()
        assertEquals(OperationState.SUCCEEDED, done.state, done.message)
        assertEquals(listOf(0L), e.rig.http.rangesSeen)
    }

    @Test fun aCompletePartialFileOnlyNeedsVerificationAndNoNetwork() {
        val e = env()
        File(e.modelFiles().dirOf(vid).also { it.mkdirs() }, "$file.part").writeBytes(e.data)
        val done = e.core.operation(e.core.startDownload(vid, use, true).ok().id).ok()
        assertEquals(OperationState.SUCCEEDED, done.state, done.message)
        assertTrue(e.rig.http.rangesSeen.isEmpty() && e.rig.http.requests.none { it == url })
    }

    @Test fun aVerifiedFinalFileWithoutManifestIsAdoptedAfterACrash() {
        val e = env()
        File(e.modelFiles().dirOf(vid).also { it.mkdirs() }, file).writeBytes(e.data)       // moved into place, then the process died
        assertFalse(e.core.models.installedModels().isNotEmpty())
        val done = e.core.operation(e.core.startDownload(vid, use, true).ok().id).ok()
        assertEquals(OperationState.SUCCEEDED, done.state, done.message)
        assertTrue(e.rig.http.requests.none { it == url })
        assertEquals(1, e.models.installedModels().size)
    }

    @Test fun aWrongFinalFileWithoutManifestIsNotAdopted() {
        val e = env()
        File(e.modelFiles().dirOf(vid).also { it.mkdirs() }, file).writeBytes(ByteArray(300_000))      // right size, wrong bytes
        val done = e.core.operation(e.core.startDownload(vid, use, true).ok().id).ok()
        assertEquals(OperationState.SUCCEEDED, done.state, done.message)        // downloaded properly over it
        assertEquals(e.sha, Hashing.sha256File(File(e.models.installedPath(vid)!!)))
    }

    private class StorageFullHttp(val inner: FakeHttp, val failAt: Long) : Http by inner {
        var armed = true
        override fun downloadToFile(url: String, part: File, maxBytes: Long, cancelled: () -> Boolean, onProgress: (Long, Long) -> Unit, timeoutMs: Int): DownloadOutcome {
            if (!armed) return inner.downloadToFile(url, part, maxBytes, cancelled, onProgress, timeoutMs)
            armed = false
            val b = inner.resources.getValue(url)
            part.parentFile?.mkdirs()
            part.writeBytes(b.copyOf(failAt.toInt()))
            throw HttpException(HttpException.Kind.STORAGE, "No space left on device")
        }
    }

    @Test fun diskFullKeepsThePartialFileReportsStorageNotNetworkAndResumes() {
        val rig = TK.rig()
        val data = ByteArray(300_000).also { Random(5).nextBytes(it) }
        val sha = Hashing.sha256(data)
        val full = StorageFullHttp(rig.http, 120_000)
        rig.http.resources[url] = data
        val art = ArtifactKit.refreshed("qwen3-1.7b-q8_0", base, repo, file, 300_000, sha)
        val core = StudioCore(rig.dir, { rig.snap() }, null, full, rig.clock, rig.storage, rig.runner, rig.ids, "test", com.hotatticgames.llmtrainer.qualify.SafetyPolicy(),
            EmbeddedRegistry.files, listOf("a.json" to art), EmbeddedArtifacts.canonicalLicenses)
        val op = core.startDownload(vid, use, true).ok()
        val failed = core.operation(op.id).ok()
        assertEquals(OperationState.FAILED, failed.state)
        assertTrue(failed.error is StudioError.Io && failed.error!!.message.contains("out of storage"), failed.error.toString())
        assertTrue(failed.resumable)
        val part = File(ModelFiles(File(rig.dir, "workspace/models")).dirOf(vid), "$file.part")
        assertEquals(120_000L, part.length())
        assertEquals(120_000L, core.models.storage().partialBytes)
        assertEquals(OperationState.SUCCEEDED, core.resumeOperation(op.id).ok().let { core.operation(op.id).ok().state })
        assertEquals(listOf(120_000L), rig.http.rangesSeen)
    }

    private class ShrinkingStorage(val big: Long, val small: Long, val bigCalls: Int) : StorageProbe {
        var calls = 0
        override fun freeBytes(dir: File): Long = if (calls++ < bigCalls) big else small
    }

    @Test fun storageCollapsingMidTransferStopsTheDownloadAtTheSafetyFloorAndKeepsThePartial() {
        val size = 200L * 1024 * 1024
        val rig = TK.rig()
        val sparse = ArtifactKit.SparseHttp(url, size)
        val art = ArtifactKit.refreshed("qwen3-1.7b-q8_0", base, repo, file, size, "ab".repeat(32))
        val probe = ShrinkingStorage(500L * 1024 * 1024 * 1024, 10L * 1024 * 1024, 2)
        val core = StudioCore(rig.dir, { rig.snap() }, null, sparse, rig.clock, probe, rig.runner, rig.ids, "test", com.hotatticgames.llmtrainer.qualify.SafetyPolicy(),
            EmbeddedRegistry.files, listOf("a.json" to art), EmbeddedArtifacts.canonicalLicenses)
        val failed = core.operation(core.startDownload(vid, use, true).ok().id).ok()
        assertEquals(OperationState.FAILED, failed.state, failed.message)
        assertEquals("Out of storage", failed.message)
        assertTrue(core.models.installedModels().isEmpty())
    }

    @Test fun severalGigabytesAreHashedByStreamingNeverBuffered() {
        val size = 2_300_000_000L                                       // > Int.MAX_VALUE: any int/array shortcut would break
        val zeroSha = run {
            val md = java.security.MessageDigest.getInstance("SHA-256"); val buf = ByteArray(1 shl 20); var left = size
            while (left > 0) { val n = minOf(left, buf.size.toLong()).toInt(); md.update(buf, 0, n); left -= n }
            Hashing.hex(md.digest())
        }
        val rig = TK.rig()
        val sparse = ArtifactKit.SparseHttp(url, size)
        val art = ArtifactKit.refreshed("qwen3-1.7b-q8_0", base, repo, file, size, zeroSha)
        val core = StudioCore(rig.dir, { rig.snap() }, null, sparse, rig.clock, rig.storage, rig.runner, rig.ids, "test", com.hotatticgames.llmtrainer.qualify.SafetyPolicy(),
            EmbeddedRegistry.files, listOf("a.json" to art), EmbeddedArtifacts.canonicalLicenses)
        val done = core.operation(core.startDownload(vid, use, true).ok().id).ok()
        assertEquals(OperationState.SUCCEEDED, done.state, done.message)
        val inst = core.models.installedModels().single()
        assertEquals(size, inst.sizeBytes); assertEquals(zeroSha, inst.sha256)
    }

    // ---- never load a partial -------------------------------------------------------------------------------------

    @Test fun partialAndTruncatedFilesAreNeverReportedInstalled() {
        val e = env()
        val dir = e.modelFiles().dirOf(vid).also { it.mkdirs() }
        File(dir, "$file.part").writeBytes(ByteArray(1000))
        assertNull(e.models.installedPath(vid)); assertTrue(e.models.installedModels().isEmpty())
        e.core.startDownload(vid, use, true).ok()
        val path = File(e.models.installedPath(vid)!!)
        assertTrue(path.isFile)
        java.io.RandomAccessFile(path, "rw").use { it.setLength(1234) }                       // truncated after install
        assertNull(e.models.installedPath(vid)); assertTrue(e.models.installedModels().isEmpty())
        assertFalse(e.core.model(TK.MODEL).ok().variants.first { it.id == vid }.acquired)
    }

    // ---- accounting + uninstall ------------------------------------------------------------------------------------

    @Test fun uninstallFreesTheFileAndAccountingTracksInstalledAndPartialBytes() {
        val e = env()
        val before = e.models.storage()
        assertEquals(0L, before.installedBytes)
        assertTrue(before.reserveBytes >= 2048L * 1024 * 1024); assertEquals(before.freeBytes - before.reserveBytes, before.headroomBytes)
        e.core.startDownload(vid, use, true).ok()
        assertEquals(300_000L, e.models.storage().installedBytes)
        val after = e.models.uninstall(vid).ok()
        assertEquals(0L, after.installedBytes); assertNull(e.models.installedPath(vid))
        assertTrue(File(e.rig.dir, "workspace/models").walkTopDown().none { it.isFile && it.name.endsWith(".gguf") })
        assertEquals(StudioError.NotFound("installed model $vid").message, e.models.uninstall(vid).err().message)
        // can be downloaded again after uninstall
        assertEquals(OperationState.SUCCEEDED, e.core.operation(e.core.startDownload(vid, use, true).ok().id).ok().state)
    }

    @Test fun uninstallIsRefusedWhileATransferIsRunningAndCancelsAPausedOne() {
        val runner = ManualRunner()
        val e = env(runner = runner)
        val op = e.core.startDownload(vid, use, true).ok()                         // queued, not running yet
        assertTrue(e.models.uninstall(vid).err() is StudioError.Conflict)
        runner.runAll()
        assertEquals(OperationState.SUCCEEDED, e.core.operation(op.id).ok().state)
    }

    // ---- manual import of a user-supplied GGUF ---------------------------------------------------------------------

    @Test fun userSuppliedGgufIsImportedWithHashAndLabelledUnverifiedProvenance() {
        val e = env()
        val bytes = ArtifactKit.gguf(layers = 6, heads = 8, kvHeads = 2, embd = 64, vocab = 50)
        val op = e.models.importUserModel(TK.bytesInput("my-model.Q4_K_M.gguf", bytes)).ok()
        val done = e.core.operation(op.id).ok()
        assertEquals(OperationState.SUCCEEDED, done.state, done.message)
        assertTrue(done.message.contains("unverified provenance"))
        val m = e.models.installedModels().single()
        assertEquals(Hashing.sha256(bytes), m.sha256); assertEquals(bytes.size.toLong(), m.sizeBytes)
        assertEquals("user_import", m.source); assertEquals(InstallProvenance.UNVERIFIED, m.provenance); assertEquals(LicenseState.UNVERIFIED, m.licenseStateAtInstall)
        assertEquals("llama", m.architecture); assertEquals(6, m.layers); assertEquals(64L * 50 + 64L * 64, m.parameterCount); assertEquals(true, m.chatTemplatePresent)
        assertNull(m.artifactId); assertTrue(File(m.path).isFile)
        assertEquals(m.path, e.models.installedPath(m.variantId))
        // the same bytes again are a duplicate
        val dup = e.core.operation(e.models.importUserModel(TK.bytesInput("copy.gguf", bytes)).ok().id).ok()
        assertEquals(OperationState.FAILED, dup.state); assertTrue(dup.error is StudioError.Conflict)
        assertEquals(1, e.models.installedModels().size)
        // and it can be removed again
        e.models.uninstall(m.variantId).ok(); assertTrue(e.models.installedModels().isEmpty())
    }

    @Test fun manualImportRefusesNonGgufEmptyAndHugeFiles() {
        val e = env()
        assertEquals("NOT_GGUF", (e.models.importUserModel(TK.bytesInput("model.bin", ByteArray(100))).err() as StudioError.Invalid).code)
        assertEquals("EMPTY_FILE", (e.models.importUserModel(TK.bytesInput("m.gguf", ByteArray(0))).err() as StudioError.Invalid).code)
        e.rig.storage.free = 1024L * 1024 * 1024
        assertEquals("INSUFFICIENT_STORAGE", (e.models.importUserModel(TK.bytesInput("m.gguf", ByteArray(100))).err() as StudioError.Blocked).code)
        e.rig.storage.free = 500L * 1024 * 1024 * 1024
        val bad = e.core.operation(e.models.importUserModel(TK.bytesInput("fake.gguf", ByteArray(5000) { 3 })).ok().id).ok()
        assertEquals(OperationState.FAILED, bad.state); assertEquals("NOT_GGUF", (bad.error as StudioError.Invalid).code)
        assertTrue(File(e.rig.dir, "workspace/models").walkTopDown().none { it.isFile && (it.name.endsWith(".part") || it.name.endsWith(".gguf")) })
    }

    @Test fun importingACatalogFileVerifiesItAgainstTheCatalogHash() {
        val e = env()
        val wrong = e.core.operation(e.core.importModelFile(vid, TK.bytesInput(file, ByteArray(300_000) { 4 })).ok().id).ok()
        assertEquals("HASH_MISMATCH", (wrong.error as StudioError.Invalid).code)
        val good = e.core.operation(e.core.importModelFile(vid, TK.bytesInput(file, e.data)).ok().id).ok()
        assertEquals(OperationState.SUCCEEDED, good.state, good.message)
        val m = e.models.installedModels().single()
        assertEquals(InstallProvenance.VERIFIED_IMPORT, m.provenance); assertEquals("import", m.source)
    }

    @Test fun readFailureDuringAcquisitionNeverLeavesAFinalFile() {
        val e = env(serve = false)                                                  // 404 from the server
        val done = e.core.operation(e.core.startDownload(vid, use, true).ok().id).ok()
        assertEquals(OperationState.FAILED, done.state)
        assertTrue(e.models.installedModels().isEmpty())
        assertNotNull(done.error)
    }
}
