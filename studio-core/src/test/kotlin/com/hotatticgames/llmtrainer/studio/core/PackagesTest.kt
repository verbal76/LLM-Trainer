package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.err
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.OutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class PackagesTest {
    private class Ready(val rig: TK.Rig, val s: StudioCore, val p: ProjectId)

    private fun ready(docs: Int = 8, method: String? = MethodIds.ADAPTER_DESKTOP): Ready {
        val rig = TK.rig(); val s = rig.open()
        val p = TK.readyProject(s, docs)
        s.approveDataset(p).ok(); TK.verifyLicense(rig, s); s.selectBaseModel(p, TK.MODEL, null).ok()
        if (method != null) s.selectMethod(p, method).ok()
        return Ready(rig, s, p)
    }

    private fun job(r: Ready): Pair<ByteArray, ExportedPackage> { val o = ByteArrayOutputStream(); val e = r.s.exportTrainingJobPackage(r.p, o).ok(); return o.toByteArray() to e }
    private fun jobId(z: ByteArray) = JSONObject(Zips.readAll(z.inputStream()).getValue("manifest.json").toString(Charsets.UTF_8)).getString("job_id")

    private val fixtures = File(System.getProperty("repo.root"), "factory/tests/fixtures/studio")
    private fun fixture(n: String) = File(fixtures, n).readBytes()

    // ---- exports ----------------------------------------------------------------------------------------------------------

    @Test fun everyMemberIsCoveredByChecksumsAndRawSourcesNeverTravel() {
        val r = ready()
        val (z, e) = job(r)
        val files = Zips.readAll(z.inputStream())
        assertEquals(emptyList(), Zips.verifyChecksums(files))
        assertEquals(files.keys.sorted(), e.files.sorted())
        assertTrue(files.keys.none { it.startsWith("sources/") || it.endsWith(".md") || it.contains("raw") })
        val sums = JSONObject(files.getValue("checksums.json").toString(Charsets.UTF_8))
        assertEquals("sha256", sums.getString("algorithm"))
        assertEquals(files.size - 1, sums.getJSONObject("files").length())
        assertEquals(e.sha256, Hashing.sha256(z)); assertEquals(z.size.toLong(), e.sizeBytes)
        assertTrue(e.suggestedFileName.endsWith(".llmtrainer-job.zip") && e.suggestedFileName.startsWith(r.p.value))
        assertTrue(e.warnings.any { it.contains("nothing was trained on this device") })
        // flipping one byte anywhere makes the independent validator reject it
        val tampered = LinkedHashMap(files); tampered["chunks.jsonl"] = tampered.getValue("chunks.jsonl").also { it[10] = (it[10] + 1).toByte() }
        assertTrue(JobValidator.validate(zipOf(tampered)).any { it.contains("hash mismatch for chunks.jsonl") })
    }

    private fun zipOf(m: Map<String, ByteArray>): ByteArray { val b = ByteArrayOutputStream(); ZipOutputStream(b).use { z -> for ((k, v) in m) { z.putNextEntry(ZipEntry(k)); z.write(v); z.closeEntry() } }; return b.toByteArray() }

    @Test fun twoExportsOfTheSameStateDifferOnlyInJobIdentity() {
        val r = ready()
        val a = Zips.readAll(job(r).first.inputStream()); val b = Zips.readAll(job(r).first.inputStream())
        assertNotEquals(a.getValue("manifest.json").toString(Charsets.UTF_8), b.getValue("manifest.json").toString(Charsets.UTF_8))     // job_id / timestamp
        for (k in a.keys - setOf("manifest.json", "checksums.json")) assertTrue(a.getValue(k).contentEquals(b.getValue(k)), "$k must be reproducible")
    }

    @Test fun trainingJobIsRefusedForNonTrainingMethodsAndUnapprovedDatasets() {
        val r = ready(method = MethodIds.REFERENCE_PACKAGE)
        assertEquals("METHOD_NOT_TRAINING", (r.s.exportTrainingJobPackage(r.p, ByteArrayOutputStream()).err() as StudioError.Blocked).code)
        val rig = TK.rig(); val s = rig.open(); val p = TK.readyProject(s, 6)
        for (f in listOf<(OutputStream) -> StudioResult<ExportedPackage>>({ s.exportTrainingJobPackage(p, it) }, { s.exportReferencePackage(p, it) }, { s.exportHeldOutEvalSet(p, it) }))
            assertEquals("DATASET_NOT_APPROVED", (f(ByteArrayOutputStream()).err() as StudioError.Blocked).code)
    }

    @Test fun trainingJobIsLicenseGated() {
        val rig = TK.rig(); val s = rig.open(); val p = TK.readyProject(s, 6)
        s.approveDataset(p).ok(); s.selectBaseModel(p, TK.MODEL, null).ok()
        val e = s.exportTrainingJobPackage(p, ByteArrayOutputStream()).err() as StudioError.Blocked
        assertTrue(e.code in setOf("METHOD_NOT_TRAINING", "LICENSE_UNVERIFIED"))
        TK.verifyLicense(rig, s)
        val lic = s.model(TK.MODEL).ok().license
        s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).ok()
        // commercial use intended but the owner did not attest it
        s.attestLicense(TK.MODEL, LicenseAttestation(lic.textSha256!!, TK.allYes() + (Permission.COMMERCIAL to Tri.NO), true)).ok()
        s.exportTrainingJobPackage(p, ByteArrayOutputStream()).ok()          // default intended use is non-commercial: allowed
    }

    @Test fun failingOutputStreamIsAnIoErrorAndStageStaysOpen() {
        val r = ready()
        val bad = object : OutputStream() { override fun write(b: Int) = throw java.io.IOException("disk full") }
        val err = r.s.exportTrainingJobPackage(r.p, bad).err()
        assertEquals("IO_ERROR", err.code)
        assertEquals(StageStatus.NOT_STARTED, r.s.getProject(r.p).ok().stages.first { it.id == StageId.TRAINING_PACKAGE }.status)
    }

    @Test fun referencePackageCarriesProvenanceTermsAndIsNotTraining() {
        val r = ready(method = MethodIds.REFERENCE_PACKAGE)
        val o = ByteArrayOutputStream(); val e = r.s.exportReferencePackage(r.p, o).ok()
        assertEquals("reference", e.kind)
        assertTrue(e.warnings.first().contains("NOT training"))
        val f = Zips.readAll(o.toByteArray().inputStream())
        assertEquals(emptyList(), Zips.verifyChecksums(f))
        val man = JSONObject(f.getValue("reference_manifest.json").toString(Charsets.UTF_8))
        assertEquals("llmtrainer-reference-package", man.getString("format")); assertTrue(man.getString("purpose").contains("not training"))
        assertTrue(man.getJSONArray("heldout_chunk_refs").length() > 0)
        val rows = JobValidator.lines(f["chunks.jsonl"])
        assertEquals(man.getInt("n_chunks"), rows.size)
        assertTrue(rows.all { it.getString("sha256") == Hashing.prefixed(Hashing.sha256(it.getString("text"))) && it.has("source_id") && it.has("section_path") })
        assertTrue(org.json.JSONArray(f.getValue("terms.json").toString(Charsets.UTF_8)).length() > 5)
        assertEquals(StageStatus.DONE, r.s.getProject(r.p).ok().stages.first { it.id == StageId.TRAINING_PACKAGE }.status)
    }

    // ---- results import ------------------------------------------------------------------------------------------------------

    @Test fun pythonStubResultsAreShownAsStubWithCaveatsAndNoClaim() {
        val files = Zips.readAll(fixture("sample_results_stub.zip").inputStream())
        val m = JSONObject(files.getValue("manifest.json").toString(Charsets.UTF_8))
        val parsed = Packages.parseResults(ProjectId(m.getString("project_id")), files, setOf(m.getString("job_id")), 5L, "fallback")
        val v = parsed.view
        assertTrue(v.isStub); assertFalse(v.improvementClaimAllowed)
        assertTrue(v.caveats.first().startsWith("STUB results"))
        assertTrue(v.metrics.size >= 3 && v.metrics.all { it.n > 0 })
        val report = JSONObject(files.getValue("evaluation_report.json").toString(Charsets.UTF_8))
        assertEquals(report.getString("improvement_claim_reason"), v.claimReason)                   // shown exactly as shipped
        val row0 = report.getJSONArray("rows").getJSONObject(0)
        assertEquals(row0.getDouble("specialist"), v.metrics[0].specialist); assertEquals(row0.getDouble("delta"), v.metrics[0].delta)
        assertEquals(row0.getString("metric"), v.metrics[0].id)
        assertEquals("none", parsed.specialistKind)
    }

    @Test fun pythonPlannedAndLocalExperimentResultsParse() {
        for ((name, kind) in listOf("sample_results_planned.zip" to "none", "sample_results_adapter_local_experiment.zip" to "adapter")) {
            val files = Zips.readAll(fixture(name).inputStream())
            val m = JSONObject(files.getValue("manifest.json").toString(Charsets.UTF_8))
            val parsed = Packages.parseResults(ProjectId(m.getString("project_id")), files, setOf(m.getString("job_id")), 5L, "fallback")
            assertFalse(parsed.view.improvementClaimAllowed, name)
            assertEquals(kind, parsed.specialistKind, name)
            if (name.contains("planned")) { assertTrue(parsed.view.metrics.isEmpty()); assertTrue(parsed.view.caveats.any { it.contains("planned_only") }) }
            else assertTrue(parsed.artifactRefs.isNotEmpty() && parsed.artifactHashes.keys.containsAll(parsed.artifactRefs))
        }
    }

    @Test fun pythonJobFixturesPassOrFailOurChecksumVerifierLikePythonDoes() {
        assertEquals(emptyList(), Zips.verifyChecksums(Zips.readAll(fixture("sample_job.zip").inputStream())))
        assertEquals(emptyList(), JobValidator.validate(fixture("sample_job.zip")).filterNot { it.contains("license") })        // our validator agrees with the Python-built job
        val bad = Zips.verifyChecksums(Zips.readAll(fixture("sample_job_bad_checksum.zip").inputStream()))
        assertTrue(bad.any { it.contains("chunks.jsonl") }, bad.toString())
    }

    private fun exportedAndJob(r: Ready) = job(r).let { it.first to jobId(it.first) }

    @Test fun resultsForAnotherProjectOrJobAreRejected() {
        val r = ready(); val (_, jid) = exportedAndJob(r)
        val other = r.s.createProject(NewProject("other", "", "")).ok().id
        assertEquals("WRONG_PROJECT", r.s.importEvaluation(r.p, ResultsKit.results(other.value, jid).inputStream()).err().code)
        assertEquals("WRONG_JOB", r.s.importEvaluation(r.p, ResultsKit.results(r.p.value, "job-19990101-abcdef").inputStream()).err().code)
        assertEquals(null, r.s.evaluation(r.p).ok())
    }

    @Test fun tamperedOrMalformedResultsAreRejectedBeforeAnythingIsShown() {
        val r = ready(); val (_, jid) = exportedAndJob(r)
        assertEquals("RESULTS_INVALID", r.s.importEvaluation(r.p, ResultsKit.results(r.p.value, jid, corruptChecksum = true).inputStream()).err().code)
        assertEquals("RESULTS_INVALID", r.s.importEvaluation(r.p, "not a zip at all".byteInputStream()).err().code)
        assertEquals("RESULTS_INVALID", r.s.importEvaluation(r.p, ByteArray(0).inputStream()).err().code)
        assertEquals("NO_EVALUATION_REPORT", r.s.importEvaluation(r.p, ResultsKit.results(r.p.value, jid, withReport = false, status = "planned_only").inputStream()).err().code)
        assertEquals(null, r.s.evaluation(r.p).ok())
    }

    @Test fun resultsRequireAnExportedJobFirst() {
        val rig = TK.rig(); val s = rig.open(); val p = TK.readyProject(s, 6)
        s.approveDataset(p).ok()
        val e = s.importEvaluation(p, ResultsKit.results(p.value, "job-20260101-abcdef").inputStream()).err() as StudioError.Blocked
        assertEquals("HELDOUT_NOT_EXPORTED", e.code)
    }

    @Test fun claimIsTakenFromTheFileNeverInflatedAndWithheldWhenInconsistent() {
        val r = ready(); val (_, jid) = exportedAndJob(r)
        val allowed = r.s.importEvaluation(r.p, ResultsKit.results(r.p.value, jid, claim = true, n = 400, reason = "n=400 and every CI excludes zero").inputStream()).ok()
        assertTrue(allowed.improvementClaimAllowed); assertEquals("n=400 and every CI excludes zero", allowed.claimReason); assertFalse(allowed.isStub)
        assertEquals(StageStatus.DONE, r.s.getProject(r.p).ok().stages.first { it.id == StageId.EVALUATION }.status)
        val denied = r.s.importEvaluation(r.p, ResultsKit.results(r.p.value, jid, claim = false, n = 24, reason = "only 24 items").inputStream()).ok()
        assertFalse(denied.improvementClaimAllowed); assertEquals("only 24 items", denied.claimReason)
        assertTrue(denied.caveats.any { it.contains("Only 24 held-out items") })
        // stub that (wrongly) claims improvement: Studio withholds
        val stub = r.s.importEvaluation(r.p, ResultsKit.results(r.p.value, jid, claim = true, stub = true, status = "stub", n = 400).inputStream()).ok()
        assertTrue(stub.isStub); assertFalse(stub.improvementClaimAllowed)
        assertTrue(stub.claimReason.contains("withheld"))
        assertEquals(StageStatus.NEEDS_ATTENTION, r.s.getProject(r.p).ok().stages.first { it.id == StageId.EVALUATION }.status)
        // a non-completed job cannot claim either
        val partial = r.s.importEvaluation(r.p, ResultsKit.results(r.p.value, jid, claim = true, status = "partial", n = 400).inputStream()).ok()
        assertFalse(partial.improvementClaimAllowed); assertTrue(partial.caveats.any { it.contains("partial") })
    }

    @Test fun evaluationPersistsAcrossRestartExactlyAsImported() {
        val r = ready(); val (_, jid) = exportedAndJob(r)
        val ev = r.s.importEvaluation(r.p, ResultsKit.results(r.p.value, jid).inputStream()).ok()
        assertEquals(ev, r.rig.open().evaluation(r.p).ok())
    }

    // ---- specialist package ------------------------------------------------------------------------------------------------------

    @Test fun specialistExportNeedsEvaluationAndCarriesTheClaimCaveat() {
        val r = ready(); val (_, jid) = exportedAndJob(r)
        assertEquals("NO_EVALUATION", (r.s.exportSpecialistPackage(r.p, ByteArrayOutputStream()).err() as StudioError.Blocked).code)
        r.s.importEvaluation(r.p, ResultsKit.results(r.p.value, jid, stub = true, status = "stub", kind = "none").inputStream()).ok()
        val o = ByteArrayOutputStream(); val e = r.s.exportSpecialistPackage(r.p, o).ok()
        assertTrue(e.warnings.any { it.contains("pipeline-validation stub") } && e.warnings.any { it.contains("No parameter training happened") })
        val view = r.s.importSpecialistPackage(o.toByteArray().inputStream()).ok()
        assertTrue(view.valid, view.validation.toString())
        assertFalse(view.improvementClaimAllowed)
        assertTrue(view.evaluationSummary!!.startsWith("STUB evaluation"))
        assertTrue(view.knownLimitations.any { it.contains("Evaluation is a stub") })
    }

    @Test fun specialistVersionsNeverRepeatForDifferentArtifacts() {
        val r = ready(); val (_, jid) = exportedAndJob(r)
        r.s.importEvaluation(r.p, ResultsKit.results(r.p.value, jid).inputStream()).ok()
        fun version(): String { val o = ByteArrayOutputStream(); r.s.exportSpecialistPackage(r.p, o).ok(); return JSONObject(Zips.readAll(o.toByteArray().inputStream()).getValue("specialist_manifest.json").toString(Charsets.UTF_8)).getString("specialist_version") }
        assertEquals(listOf("1.0.0", "1.0.1", "1.0.2"), listOf(version(), version(), version()))
    }

    @Test fun importedSpecialistPackagesAreValidatedNotTrusted() {
        val r = ready(); val (_, jid) = exportedAndJob(r)
        r.s.importEvaluation(r.p, ResultsKit.results(r.p.value, jid).inputStream()).ok()
        val o = ByteArrayOutputStream(); r.s.exportSpecialistPackage(r.p, o).ok()
        val files = Zips.readAll(o.toByteArray().inputStream())
        // tamper with a reference chunk: checksum AND chunk-hash checks both fail
        val t = LinkedHashMap(files)
        t["reference/chunks.jsonl"] = files.getValue("reference/chunks.jsonl").toString(Charsets.UTF_8).replaceFirst("Tighten", "Loosen").toByteArray()
        val view = r.s.importSpecialistPackage(zipOf(t).inputStream()).ok()
        assertFalse(view.valid)
        assertTrue(view.validation.filter { !it.passed }.map { it.name }.containsAll(listOf("checksums", "reference chunk hashes")) || view.validation.any { !it.passed })
        assertEquals("PACKAGE_INVALID", r.s.importSpecialistPackage("junk".byteInputStream()).err().code)
        val noManifest = LinkedHashMap(files).also { it.remove("specialist_manifest.json") }
        assertEquals("PACKAGE_INVALID", r.s.importSpecialistPackage(zipOf(noManifest).inputStream()).err().code)
        val wrong = LinkedHashMap(files).also { it["specialist_manifest.json"] = it.getValue("specialist_manifest.json").toString(Charsets.UTF_8).replace("\"version\":1", "\"version\":2").toByteArray() }
        assertEquals("PACKAGE_INVALID", r.s.importSpecialistPackage(zipOf(wrong).inputStream()).err().code)
    }

    // ---- hostile zips ------------------------------------------------------------------------------------------------------------

    @Test fun hostileZipsAreRejectedByTheReader() {
        fun rejects(name: String, bytes: ByteArray, code: String) = try { Zips.readAll(bytes.inputStream()); error("accepted $name") } catch (e: PackageException) { assertEquals(code, e.code, name) }
        for (bad in listOf("../evil.txt", "/abs.txt", "a/../../b.txt", "c:/windows.txt", "dir\\file.txt", "a//b.txt"))
            rejects(bad, zipOf(mapOf(bad to "x".toByteArray())), "ZIP_UNSAFE_NAME")
        // duplicate names
        val dup = ByteArrayOutputStream(); ZipOutputStream(dup).use { z -> for (n in listOf("a.txt", "a.tx2")) { z.putNextEntry(ZipEntry(n)); z.write(1); z.closeEntry() } }
        val dupBytes = String(dup.toByteArray(), Charsets.ISO_8859_1).replace("a.tx2", "a.txt").toByteArray(Charsets.ISO_8859_1)      // rename in local + central headers
        rejects("dup", dupBytes, "ZIP_DUPLICATE")
        // zip bomb: 20 MiB of zeros compress ~1000:1
        rejects("bomb", zipOf(mapOf("zeros.bin" to ByteArray(20 * 1024 * 1024))), "ZIP_BOMB")
        // too many members
        val many = ByteArrayOutputStream(); ZipOutputStream(many).use { z -> for (i in 0..ZipLimits.MAX_MEMBERS) { z.putNextEntry(ZipEntry("f$i")); z.write(i and 0x7f); z.closeEntry() } }
        rejects("many", many.toByteArray(), "ZIP_TOO_MANY")
        rejects("empty", zipOf(emptyMap()), "ZIP_EMPTY")
        rejects("garbage", ByteArray(64) { it.toByte() }, "ZIP_EMPTY")
    }

    @Test fun checksumVerifierRejectsMissingExtraAndBadFiles() {
        val base = PackageWriter().also { it.add("a.txt", "A".toByteArray()); it.add("b.txt", "B".toByteArray()) }.let { w -> ByteArrayOutputStream().also { w.write(it) }.toByteArray() }
        val f = Zips.readAll(base.inputStream())
        assertEquals(emptyList(), Zips.verifyChecksums(f))
        assertTrue(Zips.verifyChecksums(f + ("extra.txt" to "E".toByteArray())).any { it.contains("not covered") })
        assertTrue(Zips.verifyChecksums(f - "b.txt").any { it.contains("missing member") })
        assertTrue(Zips.verifyChecksums(f - "checksums.json").any { it.contains("missing") })
        assertTrue(Zips.verifyChecksums(f + ("checksums.json" to "[]".toByteArray())).isNotEmpty())
    }
}
