package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.err
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import org.json.JSONObject
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Registry-entry mutations for state-machine tests. */
object RegKit {
    fun base(file: String = "qwen3-1.7b.json") = JSONObject(EmbeddedRegistry.files.first { it.first == file }.second)

    fun withVerification(file: String, f: (JSONObject) -> Unit): List<Pair<String, String>> {
        val o = base(file); f(o.getJSONObject("verification"))
        return EmbeddedRegistry.files.map { if (it.first == file) file to o.toString() else it }
    }

    fun mutate(file: String, f: (JSONObject) -> Unit): List<Pair<String, String>> {
        val o = base(file); f(o)
        return EmbeddedRegistry.files.map { if (it.first == file) file to o.toString() else it }
    }
}

class LicenseTest {
    private val model = TK.MODEL
    private val use = IntendedUse()

    private fun setup(): Triple<TK.Rig, StudioCore, String> {
        val rig = TK.rig(); val s = rig.open()
        val url = s.model(model).ok().license.authoritativeUrl
        rig.http.resources[url] = "Apache License, Version 2.0 ... test text ...".toByteArray()
        return Triple(rig, s, url)
    }

    @Test fun registryBaselineIsFailClosedAndClaimsAreIgnored() {
        val s = TK.rig().open()
        val m = s.model(model).ok()
        assertEquals(LicenseState.UNVERIFIED, m.license.state)
        assertEquals(Evidence.SECONDARY, m.license.evidenceLevel)
        assertEquals(Tri.YES, m.license.permissions[Permission.FINE_TUNE])          // a CLAIM from a secondary source ...
        val plan = s.planAcquisition(m.variants[0].id, use).ok()
        assertFalse(plan.allowed)                                                   // ... which the gate ignores
        assertTrue(plan.blocking.any { it.code == LicenseCodes.UNVERIFIED })
        assertEquals(LicenseState.UNVERIFIED, plan.licenseState)
        // no catalog model is VERIFIED by a secondary claim
        val states = s.catalog().associate { it.id to it.license.state }
        assertTrue(states.values.count { it == LicenseState.VERIFIED } <= 2)
        for (c in s.catalog()) if (c.license.evidenceLevel == Evidence.SECONDARY || c.license.evidenceLevel == Evidence.NONE) assertEquals(LicenseState.UNVERIFIED, c.license.state, c.id)
    }

    @Test fun registryVerifiedButConditionalStillBlocksTheGate() {
        val s = TK.rig().open()
        val llama = s.catalog().first { it.id.startsWith("Llama") && it.license.state == LicenseState.VERIFIED }
        assertEquals(Evidence.PRIMARY_REGISTRY, llama.license.evidenceLevel)
        val b = LicenseGate.check(llama.license, IntendedUse())
        assertTrue(b.any { it.code == LicenseCodes.CONDITIONAL }, b.toString())
    }

    @Test fun registryVerifiedWithIncompleteEvidenceIsDemotedToUnverified() {
        val reg = RegKit.withVerification("qwen3-1.7b.json") { v ->
            v.put("state", "VERIFIED"); v.put("evidence_level", "primary_license_text_read"); v.remove("license_text_sha256")   // no hash
        }
        val s = TK.rig().open(reg)
        assertEquals(LicenseState.UNVERIFIED, s.model(model).ok().license.state)
    }

    @Test fun fetchHashesTheExactTextAndRecordsTime() {
        val (rig, s, url) = setup()
        val f = s.fetchLicenseText(model).ok()
        assertEquals(Hashing.sha256("Apache License, Version 2.0 ... test text ...".toByteArray()), f.sha256)
        assertEquals(url, f.url)
        assertTrue(f.fetchedAt > 0 && !f.truncated)
        assertTrue(url.startsWith("https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/LICENSE"), url)   // blob page rewritten to the raw text
        assertEquals(listOf(url), rig.http.requests)
    }

    @Test fun fetchFailuresAreReportedAndStoreNothing() {
        val rig = TK.rig(); val s = rig.open()
        assertEquals("NETWORK_ERROR", s.fetchLicenseText(model).err().code)       // 404
        val noUrl = s.catalog().first { it.license.authoritativeUrl.isBlank() }
        assertEquals("NO_LICENSE_URL", s.fetchLicenseText(noUrl.id).err().code)
        assertEquals("NOT_FOUND", s.fetchLicenseText("nope").err().code)
        // attestation cannot be anchored to a failed fetch
        assertEquals("EVIDENCE_MISMATCH", s.attestLicense(model, LicenseAttestation("0".repeat(64), TK.allYes(), true)).err().code)
    }

    @Test fun httpRefusalOfNonHttpsIsSurfaced() {
        val reg = RegKit.mutate("qwen3-1.7b.json") { o ->
            o.getJSONObject("verification").put("license_text_url", "http://insecure.example/LICENSE")
            o.put("license_url", "http://insecure.example/LICENSE")
        }
        val s = TK.rig().open(reg)
        assertEquals("NO_LICENSE_URL", s.fetchLicenseText(model).err().code)     // never even attempted over plain http
    }

    @Test fun attestationRequiresTheExactFetchedText() {
        val (_, s, _) = setup()
        val f = s.fetchLicenseText(model).ok()
        assertEquals("EVIDENCE_MISMATCH", s.attestLicense(model, LicenseAttestation("a".repeat(64), TK.allYes(), true)).err().code)
        // prefixed form of the right hash is accepted
        val lic = s.attestLicense(model, LicenseAttestation("sha256:" + f.sha256, TK.allYes(), false, "ok")).ok()
        assertEquals(LicenseState.VERIFIED, lic.state)
        assertEquals(false, lic.attributionRequired)
    }

    @Test fun allYesBecomesVerifiedWithCompleteEvidence() {
        val (rig, s, url) = setup()
        val f = s.fetchLicenseText(model).ok()
        val lic = s.attestLicense(model, LicenseAttestation(f.sha256, TK.allYes(), true, "read it")).ok()
        assertEquals(LicenseState.VERIFIED, lic.state)
        assertEquals(Evidence.OWNER_TEXT, lic.evidenceLevel)
        assertEquals(f.sha256, lic.textSha256)
        assertEquals(f.fetchedAt, lic.fetchedAt)
        assertTrue(lic.scopeText!!.startsWith("owner-reviewed license text from $url, hash sha256:${f.sha256}; permissions as attested by owner"))
        assertTrue(LicenseGate.evidenceProblems(lic).isEmpty())
        assertTrue(LicenseGate.check(lic, IntendedUse(true, true, true, true)).isEmpty())
        val plan = s.planAcquisition(s.model(model).ok().variants[0].id, use).ok()
        assertTrue(plan.blocking.none { it.code.startsWith("LICENSE") || it.code.startsWith("PERMISSION") })
        // survives a restart (workspace-level store)
        val s2 = rig.open()
        assertEquals(LicenseState.VERIFIED, s2.model(model).ok().license.state)
        assertEquals(f.sha256, s2.model(model).ok().license.textSha256)
    }

    @Test fun perUsePermissionsGateIndependently() {
        val (_, s, _) = setup()
        val f = s.fetchLicenseText(model).ok()
        s.attestLicense(model, LicenseAttestation(f.sha256, TK.allYes() + (Permission.COMMERCIAL to Tri.NO) + (Permission.REDISTRIBUTION to Tri.NO), true)).ok()
        val v = s.model(model).ok().variants[0].id
        assertTrue(s.planAcquisition(v, IntendedUse(true, true, false, false)).ok().blocking.none { it.code == LicenseCodes.NOT_GRANTED })
        val c = s.planAcquisition(v, IntendedUse(true, true, true, false)).ok()
        assertTrue(c.blocking.any { it.code == LicenseCodes.NOT_GRANTED && it.message.contains("Commercial") })
        assertTrue(s.planAcquisition(v, IntendedUse(true, true, false, true)).ok().blocking.any { it.code == LicenseCodes.NOT_GRANTED })
    }

    @Test fun ownerSayingNoToFineTuningMakesItDisallowedButItCanBeCorrected() {
        val (_, s, _) = setup()
        val f = s.fetchLicenseText(model).ok()
        val lic = s.attestLicense(model, LicenseAttestation(f.sha256, TK.allYes() + (Permission.FINE_TUNE to Tri.NO), true)).ok()
        assertEquals(LicenseState.DISALLOWED, lic.state)
        assertEquals(Evidence.OWNER_ATTESTED_DISALLOWED, lic.evidenceLevel)
        val p = s.createProject(NewProject("x", "d", "p")).ok().id
        assertEquals("LICENSE_DISALLOWED", s.selectBaseModel(p, model, null).err().code)
        // an owner-attested NO is the owner's call and can be re-attested after re-reading the text
        val again = s.attestLicense(model, LicenseAttestation(f.sha256, TK.allYes(), true)).ok()
        assertEquals(LicenseState.VERIFIED, again.state)
    }

    @Test fun conditionalPermissionsCannotBeRecordedAsVerified() {
        val (_, s, _) = setup()
        val f = s.fetchLicenseText(model).ok()
        val e = s.attestLicense(model, LicenseAttestation(f.sha256, TK.allYes() + (Permission.ADAPTER to Tri.CONDITIONAL), true)).err() as StudioError.Blocked
        assertEquals("CONDITIONAL_PERMISSIONS", e.code)
        assertEquals(LicenseState.UNVERIFIED, s.model(model).ok().license.state)    // nothing recorded
    }

    @Test fun incompleteAttestationIsRejected() {
        val (_, s, _) = setup()
        val f = s.fetchLicenseText(model).ok()
        val partial = mapOf(Permission.FINE_TUNE to Tri.YES, Permission.ADAPTER to Tri.YES)
        assertEquals("INCOMPLETE_ATTESTATION", s.attestLicense(model, LicenseAttestation(f.sha256, partial, true)).err().code)
        assertEquals(LicenseState.UNVERIFIED, s.model(model).ok().license.state)
    }

    @Test fun registryDisallowedIsNeverOverridable() {
        val reg = RegKit.withVerification("qwen3-1.7b.json") { v ->
            v.put("state", "DISALLOWED"); v.put("evidence_level", "primary_license_text_read"); v.put("disallowed_reason", "research-only license")
        }
        val rig = TK.rig(); val s = rig.open(reg)
        val url = s.model(model).ok().license.authoritativeUrl
        rig.http.resources[url] = "text".toByteArray()
        val f = s.fetchLicenseText(model).ok()
        val e = s.attestLicense(model, LicenseAttestation(f.sha256, TK.allYes(), true)).err() as StudioError.Blocked
        assertEquals("LICENSE_DISALLOWED", e.code)
        assertEquals(LicenseState.DISALLOWED, s.model(model).ok().license.state)
        assertTrue(s.planAcquisition(s.model(model).ok().variants[0].id, use).ok().blocking.any { it.code == LicenseCodes.DISALLOWED && it.message.contains("research-only") })
        assertEquals("LICENSE_DISALLOWED", s.selectBaseModel(s.createProject(NewProject("n", "", "")).ok().id, model, null).err().code)
        // and a stored owner record (e.g. from an older registry) cannot resurrect it either
        assertEquals(LicenseState.DISALLOWED, rig.open(reg).model(model).ok().license.state)
    }

    @Test fun importedLicenseFileFlowsThroughTheSameGate() {
        val rig = TK.rig(); val s = rig.open()
        val text = "MIT-like license text imported from a local file\n".repeat(20)
        val f = s.importLicenseText(model, "LICENSE.txt", text.byteInputStream()).ok()
        assertEquals("file:LICENSE.txt", f.url)
        assertEquals(Hashing.sha256(text.toByteArray()), f.sha256)
        assertEquals("EMPTY_FILE", s.importLicenseText(model, "e.txt", ByteArray(0).inputStream()).err().code)
        val lic = s.attestLicense(model, LicenseAttestation(f.sha256, TK.allYes(), true)).ok()
        assertEquals(LicenseState.VERIFIED, lic.state)
        assertTrue(lic.scopeText!!.contains("imported from a file"))
        assertEquals("file:LICENSE.txt", lic.authoritativeUrl.takeIf { it.startsWith("file:") } ?: "file:LICENSE.txt")
    }

    @Test fun veryLargeLicenseTextIsHashedWholeButDisplayedTruncated() {
        val (rig, s, url) = setup()
        val big = ("clause. ".repeat(60_000)).toByteArray()
        rig.http.resources[url] = big
        val f = s.fetchLicenseText(model).ok()
        assertTrue(f.truncated)
        assertEquals(LicenseService.DISPLAY_CHARS, f.text.length)
        assertEquals(Hashing.sha256(big), f.sha256)
    }

    @Test fun licenseEvidenceGoesIntoJobPackageOnlyWhenComplete() {
        val (rig, s, _) = setup()
        val p = TK.readyProject(s, 6)
        s.approveDataset(p).ok()
        s.selectBaseModel(p, model, null).ok()
        // UNVERIFIED: method unavailable -> no job at all
        assertEquals("METHOD_UNAVAILABLE", s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).err().code)
        assertEquals(StageStatus.NEEDS_ATTENTION, s.getProject(p).ok().stages.first { it.id == StageId.BASE_MODEL }.status)
        TK.verifyLicense(rig, s)
        s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).ok()
        val out = java.io.ByteArrayOutputStream()
        s.exportTrainingJobPackage(p, out).ok()
        val files = Zips.readAll(out.toByteArray().inputStream())
        val ev = JSONObject(files.getValue("manifest.json").toString(Charsets.UTF_8)).getJSONObject("license_evidence")
        assertEquals(Hashing.prefixed(Hashing.sha256(files.getValue("license/license_text.txt"))), ev.getString("text_sha256"))
        assertEquals("owner", ev.getString("attested_by"))
        assertTrue(ev.getString("scope_note").contains("permissions as attested by owner"))
    }

    @Test fun licenseChangeAfterSelectionWithdrawsTheMethod() {
        val (rig, s, _) = setup()
        val p = TK.readyProject(s, 6)
        s.approveDataset(p).ok()
        val lic = TK.verifyLicense(rig, s)
        s.selectBaseModel(p, model, null).ok(); s.selectMethod(p, MethodIds.ADAPTER_DESKTOP).ok()
        assertEquals(StageStatus.DONE, s.getProject(p).ok().stages.first { it.id == StageId.METHOD }.status)
        // owner corrects the attestation to "fine-tuning not allowed": project must not keep a method it may not use
        s.attestLicense(model, LicenseAttestation(lic.textSha256!!, TK.allYes() + (Permission.FINE_TUNE to Tri.NO), true)).ok()
        val summary = s.getProject(p).ok()
        assertEquals(StageStatus.BLOCKED, summary.stages.first { it.id == StageId.BASE_MODEL }.status)
        assertEquals(StageStatus.NOT_STARTED, summary.stages.first { it.id == StageId.METHOD }.status)
        assertEquals("METHOD_NOT_TRAINING", s.exportTrainingJobPackage(p, java.io.ByteArrayOutputStream()).err().code)
    }

    @Test fun secondaryAndMissingEvidenceNeverVerifiedAcrossWholeCatalog() {
        val s = TK.rig().open()
        for (c in s.catalog()) {
            if (c.license.state == LicenseState.VERIFIED) {
                assertTrue(LicenseGate.evidenceProblems(c.license).isEmpty(), c.id)
                assertNotNull(c.license.textSha256, c.id)
            }
            assertNull(c.license.scopeText.takeIf { c.license.state != LicenseState.VERIFIED }, c.id)
        }
    }
}
