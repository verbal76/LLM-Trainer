package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.err
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * FakeStudio is the executable oracle for the Studio contract: the same scenario, run against the fake and against the real
 * core, must produce the same stage vectors, the same refusal codes and the same availability matrix at every checkpoint.
 * (Content differs by design: the fake invents data; the core derives it.)
 */
class ParityTest {
    private class Trace {
        val lines = ArrayList<String>()
        fun add(key: String, value: Any?) { lines.add("$key = $value") }
    }

    private fun stages(s: Studio, p: ProjectId) = s.getProject(p).ok().stages.map { "${it.id}:${it.status}" }

    private fun code(r: StudioResult<*>): String = (r.err() as StudioError).code

    private fun scenario(
        s: Studio, name: String, corpus: List<SourceInput>, prepareLicense: (Studio, String) -> Unit, results: (ByteArray) -> ByteArray,
    ): Trace {
        val t = Trace()
        val p = s.createProject(NewProject(name, "motorcycle service", "diagnose")).ok().id
        t.add("A create", stages(s, p))
        t.add("A2 nextAction", s.getProject(p).ok().nextAction?.screen)
        t.add("A3 project fields", s.getProject(p).ok().let { listOf(it.name, it.domain, it.purpose, it.baseModelId) })
        t.add("B3 build without sources", code(s.buildDataset(p)))
        val rep = s.ingest(p, corpus, RightsStatus.UNSET).ok()
        t.add("B ingest", listOf(rep.ingested.size, rep.failed.size, rep.duplicates.size))
        t.add("B2 stages", stages(s, p))
        t.add("B4 build with rights unset", code(s.buildDataset(p)))
        val blockedReasons = (s.buildDataset(p).err() as StudioError.Blocked).reasons.map { it.code }.toSet()
        t.add("B5 reasons", blockedReasons)
        s.listSources(p).ok().forEach { s.setSourceRights(p, it.sourceId, RightsStatus.OWNER_AUTHORED).ok() }
        t.add("C0 approve before build", code(s.approveDataset(p)))
        t.add("C0b setIncluded before build", code(s.setIncluded(p, listOf("x"), true)))
        s.buildDataset(p).ok()
        t.add("C build", stages(s, p))
        t.add("C2 preview status", s.datasetPreview(p).ok()!!.status)
        // resolve any leakage suspects the way a user would, then approve
        s.reviewItems(p, ReviewFilter(flag = ReviewFlag.POSSIBLE_LEAKAGE), 0, 10_000).ok().items.let { l -> if (l.isNotEmpty()) s.setIncluded(p, l.map { it.id }, false).ok() }
        val prev = s.approveDataset(p).ok()
        t.add("D approve", listOf(prev.status, stages(s, p)))
        t.add("D2 export before method", code(s.exportTrainingJobPackage(p, ByteArrayOutputStream())))
        // base model + license
        val m = s.catalog().first { it.license.state == LicenseState.UNVERIFIED && it.variants.isNotEmpty() }
        t.add("E0 download blocked while unverified", (s.startDownload(m.variants[0].id, IntendedUse(), true).err() as StudioError.Blocked).reasons.any { it.code == "LICENSE_UNVERIFIED" })
        prepareLicense(s, m.id)
        t.add("E license", s.model(m.id).ok().license.let { listOf(it.state, it.evidenceLevel, it.permissions[Permission.FINE_TUNE]) })
        s.selectBaseModel(p, m.id, null).ok()
        t.add("E2 base", stages(s, p))
        val ms = s.methodOptions(p).ok()
        t.add("F methods", ms.map { listOf(it.id, it.available, it.isTraining, it.whereItRuns) })
        t.add("F2 on-device blocked", code(s.selectMethod(p, MethodIds.ADAPTER_ON_DEVICE)))
        t.add("F3 unknown method", code(s.selectMethod(p, "nope")))
        s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).ok()
        val jobOut = ByteArrayOutputStream()
        val job = s.exportTrainingJobPackage(p, jobOut).ok()
        t.add("I export", listOf(job.kind, stages(s, p)))
        t.add("I2 eval import before heldout", stages(s, p).last { it.startsWith("EVALUATION") })
        val held = s.exportHeldOutEvalSet(p, ByteArrayOutputStream()).ok()
        t.add("J heldout", listOf(held.kind, stages(s, p)))
        val ev = s.importEvaluation(p, results(jobOut.toByteArray()).inputStream()).ok()
        t.add("K eval", listOf(ev.isStub, ev.improvementClaimAllowed, ev.metrics.isNotEmpty(), stages(s, p)))
        t.add("K2 eval readback", s.evaluation(p).ok() == ev)
        val spec = ByteArrayOutputStream()
        val sp = s.exportSpecialistPackage(p, spec).ok()
        t.add("L specialist", listOf(sp.kind, stages(s, p), sp.warnings.any { it.contains("claim", ignoreCase = true) }))
        val view = s.importSpecialistPackage(spec.toByteArray().inputStream()).ok()
        t.add("L2 view", listOf(view.valid, view.licenseState, view.improvementClaimAllowed))
        t.add("L3 junk import", code(s.importSpecialistPackage("junk".byteInputStream())))
        // source removal invalidates everything derived
        s.removeSource(p, s.listSources(p).ok().first().sourceId).ok()
        t.add("M removed", stages(s, p))
        t.add("M2 dataset", s.datasetPreview(p).ok()!!.status)
        t.add("M3 approve stale", code(s.approveDataset(p)))
        t.add("M4 evaluation cleared", s.evaluation(p).ok() == null)
        t.add("N list", s.listProjects().map { it.name })
        s.deleteProject(p).ok()
        t.add("N2 deleted", listOf(code(s.getProject(p)), code(s.deleteProject(p)), s.listProjects().size))
        t.add("N3 bad create", code(s.createProject(NewProject("  ", "", ""))).let { if (it == "NAME_REQUIRED") "invalid" else it })
        return t
    }

    @Test fun fakeAndCoreAgreeOnTheWholeWorkflow() {
        val fake = FakeStudio(seedSampleData = false)
        val rig = TK.rig(); val core = rig.open()
        for (m in core.catalog()) if (m.license.authoritativeUrl.isNotBlank()) rig.http.resources[m.license.authoritativeUrl] = ("license text of " + m.id).toByteArray()
        val corpus = TK.corpus(6)
        val a = scenario(fake, "Parity", corpus, { s, id ->
            val f = s.fetchLicenseText(id).ok()
            s.attestLicense(id, LicenseAttestation(f.sha256, TK.allYes(), true, "ok")).ok()
        }, { "RESULTS n=24".toByteArray() })
        val b = scenario(core, "Parity", corpus, { s, id ->
            val f = s.fetchLicenseText(id).ok()
            s.attestLicense(id, LicenseAttestation(f.sha256, TK.allYes(), true, "ok")).ok()
        }, { job ->
            val jid = JSONObject(Zips.readAll(job.inputStream()).getValue("manifest.json").toString(Charsets.UTF_8)).getString("job_id")
            ResultsKit.results(core.listProjects().first().id.value, jid, claim = false, n = 24)
        })
        // the fake invents a different number of "sources failed"/chunks, so compare line by line on the contract-level facts
        val skip = setOf("B ingest", "B2 stages")        // fake rejects some mime types the core can read; compared separately below
        val la = a.lines.filterNot { l -> skip.any { l.startsWith(it) } }
        val lb = b.lines.filterNot { l -> skip.any { l.startsWith(it) } }
        assertEquals(la.size, lb.size)
        val diffs = la.zip(lb).filter { it.first != it.second }
        // evaluation stages differ only in the fake's invented metric ids; everything else must be identical
        assertTrue(diffs.isEmpty(), "behavioural differences between FakeStudio and StudioCore:\n" + diffs.joinToString("\n") { "  fake: ${it.first}\n  core: ${it.second}" })
    }

    @Test fun ingestVocabularyMatches() {
        // the fake only knows mime types text/*, pdf, json, docx; the core also sniffs by extension. Same report shape and codes either way.
        val files = TK.corpus(3)
        val fake = FakeStudio(seedSampleData = false); val core = TK.rig().open()
        for (s in listOf<Studio>(fake, core)) {
            val p = s.createProject(NewProject("x", "", "")).ok().id
            val r = s.ingest(p, files + TK.input("empty.txt", ""), RightsStatus.OWNER_AUTHORED).ok()
            assertEquals(3, r.ingested.size)
            assertEquals(1, r.failed.size)
            assertEquals(IssueCode.EMPTY_TEXT, r.failed.single().issues.single().code)
            assertEquals(StageStatus.NEEDS_ATTENTION, s.getProject(p).ok().stages.first { it.id == StageId.SOURCES }.status)
            val again = s.ingest(p, listOf(files[0]), RightsStatus.OWNER_AUTHORED).ok()
            assertEquals(IssueCode.DUPLICATE_CONTENT, again.duplicates.single().issues.single().code)
        }
    }

    @Test fun downloadVocabularyMatchesForUnverifiedAndUnconfirmed() {
        val fake = FakeStudio(seedSampleData = false); val core = TK.rig().open()
        for (s in listOf<Studio>(fake, core)) {
            val v = s.catalog().first { it.license.state == LicenseState.UNVERIFIED && it.variants.isNotEmpty() }.variants[0].id
            val e = s.startDownload(v, IntendedUse(), true).err() as StudioError.Blocked
            assertEquals("DOWNLOAD_BLOCKED", e.code)
            assertTrue(e.reasons.any { it.code == "LICENSE_UNVERIFIED" })
            assertEquals("NOT_FOUND", s.planAcquisition("zzz", IntendedUse()).err().code)
            assertEquals("NOT_FOUND", s.operation("nope").err().code)
            assertTrue(s.operations().isEmpty())
        }
    }

    @Test fun licenseAttestationVocabularyMatches() {
        val fake = FakeStudio(seedSampleData = false); val rig = TK.rig(); val core = rig.open()
        for (m in core.catalog()) if (m.license.authoritativeUrl.isNotBlank()) rig.http.resources[m.license.authoritativeUrl] = "text".toByteArray()
        for (s in listOf<Studio>(fake, core)) {
            val id = s.catalog().first { it.license.state == LicenseState.UNVERIFIED }.id
            assertEquals("EVIDENCE_MISMATCH", s.attestLicense(id, LicenseAttestation("0".repeat(64), TK.allYes(), true)).err().code)
            assertEquals("NOT_FOUND", s.fetchLicenseText("zzz").err().code)
            val f = s.fetchLicenseText(id).ok()
            assertEquals(64, f.sha256.length)
            val no = s.attestLicense(id, LicenseAttestation(f.sha256, TK.allYes() + (Permission.FINE_TUNE to Tri.NO), true)).ok()
            assertEquals(LicenseState.DISALLOWED, no.state)
            val p = s.createProject(NewProject("x", "", "")).ok().id
            assertEquals("LICENSE_DISALLOWED", s.selectBaseModel(p, id, null).err().code)
        }
    }
}
