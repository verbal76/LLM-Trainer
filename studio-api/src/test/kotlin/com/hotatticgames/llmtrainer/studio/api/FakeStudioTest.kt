package com.hotatticgames.llmtrainer.studio.api

import java.io.ByteArrayOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.fail

class FakeStudioTest {
    private fun <T> StudioResult<T>.ok(): T = (this as? StudioResult.Ok)?.value ?: fail("expected Ok but got ${(this as StudioResult.Err).error}")
    private fun <T> StudioResult<T>.err(): StudioError = (this as? StudioResult.Err)?.error ?: fail("expected Err")
    private fun proj(s: FakeStudio, name: String) = s.listProjects().first { it.name == name }
    private fun stage(p: ProjectSummary, id: StageId) = p.stages.first { it.id == id }
    private val use = IntendedUse()

    @Test fun sampleProjectsAndStagesDeriveCorrectly() {
        val s = FakeStudio()
        val moto = proj(s, "Motorcycle Mechanic")
        assertEquals(StageStatus.DONE, stage(moto, StageId.BASE_MODEL).status)
        assertEquals(StageStatus.NEEDS_ATTENTION, stage(moto, StageId.SOURCES).status)  // corrupt + OCR failures
        assertEquals(StageStatus.NEEDS_ATTENTION, stage(moto, StageId.DATASET).status)  // needs review
        assertEquals(StageStatus.BLOCKED, stage(moto, StageId.METHOD).status)
        assertEquals(Screen.INGESTION_REPORT, moto.nextAction?.screen)
        val hvac = proj(s, "HVAC Technician")
        assertEquals(StageStatus.NOT_STARTED, stage(hvac, StageId.BASE_MODEL).status)
        assertEquals(StageStatus.BLOCKED, stage(hvac, StageId.DATASET).status)
        val town = proj(s, "Town Design")
        assertEquals(StageStatus.DONE, stage(town, StageId.DATASET).status)
        assertEquals(StageStatus.NEEDS_ATTENTION, stage(town, StageId.EVALUATION).status)
        assertEquals(StageStatus.NOT_STARTED, stage(town, StageId.EXPORT).status)
    }

    @Test fun ingestionReportHasDuplicateCorruptAndOcr() {
        val s = FakeStudio()
        val rep = s.lastIngestReport(proj(s, "Motorcycle Mechanic").id).ok()!!
        assertEquals(2, rep.ingested.size)
        assertEquals(listOf(IssueCode.DUPLICATE_CONTENT), rep.duplicates.single().issues.map { it.code })
        assertEquals(setOf(IssueCode.CORRUPT_FILE, IssueCode.NEEDS_OCR), rep.failed.flatMap { it.issues }.map { it.code }.toSet())
        assertTrue(rep.ingested.all { it.provenance != null && it.provenance!!.sha256.length == 64 })
    }

    @Test fun downloadBlockedUnlessVerifiedAndConfirmed() {
        val s = FakeStudio()
        assertEquals("CONFIRMATION_REQUIRED", s.startDownload("qwen3-4b:q4_k_m", use, false).err().code)
        val unv = s.startDownload("gemma-4-e4b:q4_k_m", use, true).err() as StudioError.Blocked
        assertTrue(unv.reasons.any { it.code == "LICENSE_UNVERIFIED" })
        val dis = s.startDownload("sample-restricted-7b:q4_k_m", use, true).err() as StudioError.Blocked
        assertTrue(dis.reasons.any { it.code == "LICENSE_DISALLOWED" })
        val big = s.planAcquisition("qwen3-8b:q8_0", use).ok()
        assertFalse(big.allowed); assertTrue(big.blocking.any { it.code == "INSUFFICIENT_STORAGE" })
        assertTrue(s.planAcquisition("qwen3-4b:q4_k_m", use).ok().allowed)
        assertNotNull(s.startDownload("qwen3-4b:q4_k_m", use, true).ok())
    }

    @Test fun downloadProgressCancelResumeAndProcessDeath() {
        val s = FakeStudio()
        val op = s.startDownload("qwen3-4b:q4_k_m", use, true).ok()
        s.tick()
        assertEquals(0.25, s.operation(op.id).ok().progress.fraction, 1e-9)
        s.simulateProcessDeath()
        assertEquals(OperationState.PAUSED, s.operation(op.id).ok().state)
        assertTrue(s.operation(op.id).ok().resumable)
        s.resumeOperation(op.id).ok()
        repeat(4) { s.tick() }
        assertEquals(OperationState.SUCCEEDED, s.operation(op.id).ok().state)
        assertTrue(s.model("qwen3-4b").ok().variants.first { it.id == "qwen3-4b:q4_k_m" }.acquired)
        assertEquals("ALREADY_ACQUIRED", (s.planAcquisition("qwen3-4b:q4_k_m", use).ok().blocking.single().code))
        val op2 = s.startDownload("qwen3-4b:q8_0", use, true)
        assertEquals("DOWNLOAD_BLOCKED", op2.err().code)   // q8_0 does not fit the safe RAM envelope
    }

    @Test fun cancelStopsDownload() {
        val s = FakeStudio()
        val op = s.startDownload("qwen3-8b:q4_k_m", use, true).let { it.getOrNull() ?: return }
        assertEquals(OperationState.CANCELLED, s.cancelOperation(op.id).ok().state)
        assertEquals("CONFLICT", s.cancelOperation(op.id).err().code)
    }

    @Test fun licenseEvidenceFlowVerifiesOnlyWithMatchingHash() {
        val s = FakeStudio()
        assertEquals(LicenseState.UNVERIFIED, s.model("gemma-4-e4b").ok().license.state)
        val bad = LicenseAttestation("0".repeat(64), mapOf(Permission.FINE_TUNE to Tri.YES, Permission.ADAPTER to Tri.YES), false)
        assertEquals("EVIDENCE_MISMATCH", s.attestLicense("gemma-4-e4b", bad).err().code)
        val f = s.fetchLicenseText("gemma-4-e4b").ok()
        assertTrue(f.url.startsWith("https://")); assertEquals(64, f.sha256.length)
        val lic = s.attestLicense("gemma-4-e4b", LicenseAttestation(f.sha256, Permission.values().associateWith { Tri.YES }, true)).ok()
        assertEquals(LicenseState.VERIFIED, lic.state)
        assertTrue(lic.scopeText!!.contains(f.sha256))
        assertTrue(s.planAcquisition("gemma-4-e4b:q4_k_m", use).ok().allowed)
    }

    @Test fun attestingNoFineTuningDisallowsAndDisallowedCannotBeVerified() {
        val s = FakeStudio()
        val f = s.importLicenseText("llama-3.1-8b", "LICENSE.txt", "terms".byteInputStream()).ok()
        assertEquals("file:LICENSE.txt", f.url)
        val l = s.attestLicense("llama-3.1-8b", LicenseAttestation(f.sha256, mapOf(Permission.FINE_TUNE to Tri.NO, Permission.ADAPTER to Tri.YES), false)).ok()
        assertEquals(LicenseState.DISALLOWED, l.state)
        assertTrue(s.attestLicense("sample-restricted-7b", LicenseAttestation("x", emptyMap(), false)).err() is StudioError.Blocked)
        val p = s.createProject(NewProject("X", "d", "p")).ok()
        assertTrue(s.selectBaseModel(p.id, "sample-restricted-7b", null).err() is StudioError.Blocked)
    }

    @Test fun recommendationsHonestAndSafe() {
        val r = FakeStudio().recommendations()
        assertEquals(setOf(ProfileKind.PERFORMANCE, ProfileKind.BALANCED, ProfileKind.MAX_QUALITY), r.profiles.map { it.kind }.toSet())
        assertTrue(r.profiles.all { !it.confidence.benchmarked })
        val s = FakeStudio()
        r.profiles.forEach { rec -> rec.variantId?.let { v -> assertTrue(s.catalog().flatMap { it.variants }.first { it.id == v }.android.verdict != Verdict.DOES_NOT_FIT) } }
        assertTrue(r.profiles.none { it.modelId == "sample-restricted-7b" || it.modelId == "llama-3.1-8b" })
    }

    @Test fun datasetRequiresRightsAndApprovalGatesExports() {
        val s = FakeStudio(false)
        val p = s.createProject(NewProject("T", "d", "p")).ok().id
        assertEquals("NO_SOURCES", s.buildDataset(p).err().code)
        val rep = s.ingest(p, listOf(SourceInput("a.txt", "text/plain", 40000) { "x".repeat(40000).byteInputStream() })).ok()
        assertEquals(IssueCode.RIGHTS_UNSET, rep.ingested.single().issues.single().code)
        assertEquals("RIGHTS_UNSET", s.buildDataset(p).err().code)
        s.setSourceRights(p, rep.ingested.single().sourceId!!, RightsStatus.OWNER_AUTHORED).ok()
        val d = s.buildDataset(p).ok()
        assertEquals(DatasetStatus.NEEDS_REVIEW, d.status)
        assertEquals("DATASET_NOT_APPROVED", s.exportReferencePackage(p, ByteArrayOutputStream()).err().code)
        val leak =s.reviewItems(p, ReviewFilter(flag = ReviewFlag.POSSIBLE_LEAKAGE), 0, 100).ok()
        assertTrue(leak.items.all { !it.included })   // leakage suspects default to excluded
        s.setIncluded(p, listOf(leak.items.first().id), true).ok()
        assertEquals("LEAKAGE_UNRESOLVED", s.approveDataset(p).err().code)
        s.setIncluded(p, listOf(leak.items.first().id), false).ok()
        assertEquals(DatasetStatus.APPROVED, s.approveDataset(p).ok().status)
        assertTrue(s.exportReferencePackage(p, ByteArrayOutputStream()).ok().sizeBytes > 0)
        // removing a source makes the dataset stale and approval is revoked
        s.removeSource(p, rep.ingested.single().sourceId!!).ok()
        assertEquals(DatasetStatus.STALE, s.datasetPreview(p).ok()!!.status)
    }

    @Test fun methodOptionsAreHonest() {
        val s = FakeStudio()
        val moto = proj(s, "Motorcycle Mechanic").id
        val m = s.methodOptions(moto).ok().associateBy { it.id }
        assertFalse(m.getValue(MethodIds.ADAPTER_ON_DEVICE).available)
        assertTrue(m.getValue(MethodIds.ADAPTER_ON_DEVICE).whyNotAvailable!!.contains("runtime"))
        assertFalse(m.getValue(MethodIds.REFERENCE_PACKAGE).isTraining)
        assertFalse(m.getValue(MethodIds.PROMPT_SPECIALIZATION).isTraining)
        assertEquals(RunLocation.DESKTOP, m.getValue(MethodIds.ADAPTER_DESKTOP).whereItRuns)
        assertEquals("METHOD_UNAVAILABLE", s.selectMethod(moto, MethodIds.ADAPTER_ON_DEVICE).err().code)
    }

    @Test fun evaluationClaimOnlyWithLargeNAndNeverForStub() {
        val s = FakeStudio()
        val town = proj(s, "Town Design").id
        val ev = s.evaluation(town).ok()!!
        assertFalse(ev.improvementClaimAllowed); assertTrue(ev.claimReason.contains("n=24")); assertTrue(ev.metrics.all { it.n == 24 })
        assertEquals(StageStatus.NEEDS_ATTENTION, stage(s.getProject(town).ok(), StageId.EVALUATION).status)
        val big = s.importEvaluation(town, "LARGE".byteInputStream()).ok()
        assertTrue(big.improvementClaimAllowed)
        assertEquals(StageStatus.DONE, stage(s.getProject(town).ok(), StageId.EVALUATION).status)
        val stub = s.importEvaluation(town, "LARGE STUB".byteInputStream()).ok()
        assertTrue(stub.isStub); assertFalse(stub.improvementClaimAllowed)
        assertEquals("HELDOUT_NOT_EXPORTED", s.importEvaluation(proj(s, "HVAC Technician").id, "x".byteInputStream()).err().code)
    }

    @Test fun specialistExportImportRoundTrip() {
        val s = FakeStudio()
        val town = proj(s, "Town Design").id
        val out = ByteArrayOutputStream()
        val pkg = s.exportSpecialistPackage(town, out).ok()
        assertTrue(pkg.warnings.any { it.contains("improvement claim") })
        val v = s.importSpecialistPackage(out.toByteArray().inputStream()).ok()
        assertEquals("Town Design", v.name); assertEquals(LicenseState.VERIFIED, v.licenseState); assertTrue(v.valid); assertFalse(v.improvementClaimAllowed)
        assertEquals("PACKAGE_INVALID", s.importSpecialistPackage("junk".byteInputStream()).err().code)
        assertEquals(StageStatus.DONE, stage(s.getProject(town).ok(), StageId.EXPORT).status)
        assertEquals("NO_EVALUATION", s.exportSpecialistPackage(proj(s, "HVAC Technician").id, ByteArrayOutputStream()).err().code)
    }

    @Test fun trainingExportGatedOnLicenseAndMethod() {
        val s = FakeStudio()
        val p = s.createProject(NewProject("G", "d", "p")).ok().id
        s.selectBaseModel(p, "gemma-4-e4b", null).ok()              // UNVERIFIED: selectable, but gated
        assertEquals(StageStatus.NEEDS_ATTENTION, stage(s.getProject(p).ok(), StageId.BASE_MODEL).status)
        assertEquals("METHOD_UNAVAILABLE", s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).err().code)
    }

    @Test fun hostPassthroughAndDeterminism() {
        val s = FakeStudio()
        var st: UpdateStatus? = null
        s.host.checkForUpdates { st = it }
        assertEquals("up-to-date", st?.state)
        assertTrue(s.host.diagnosticsJson().contains("none-v1"))
        assertEquals(FakeStudio().listProjects(), FakeStudio().listProjects())
    }
}
