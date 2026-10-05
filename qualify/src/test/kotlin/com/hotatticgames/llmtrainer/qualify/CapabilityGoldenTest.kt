package com.hotatticgames.llmtrainer.qualify

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.fail

/** Runs the SAME golden vectors the Python spec generates (factory/tests/golden/device_capability.v1.json). */
class CapabilityGoldenTest {
    private val doc: Map<String, Any?> by lazy {
        val path = System.getProperty("golden.capability.path") ?: fail("golden.capability.path system property missing")
        MiniJson.parse(File(path).readText()).asObj()!!
    }
    private val cases: List<Map<String, Any?>> get() = doc.list("cases")!!.map { it.asObj()!! }

    private fun near(a: Double?, b: Double?, what: String) {
        if (a == null || b == null) { assertEquals(a, b, what); return }
        assertTrue(Math.abs(a - b) <= 1e-9 * maxOf(1.0, Math.abs(b)), "$what: $a != $b")
    }

    private fun strings(l: List<Any?>?): List<String> = l.orEmpty().map { it as String }

    private fun specsOf(c: Map<String, Any?>): List<ArtifactSpec> {
        val set = doc.obj("spec_sets")!!.list(c.str("spec_set")!!)!!.map { it.asObj()!! }
        val patch = c.obj("spec_patch").orEmpty()
        return set.map { s ->
            val over = patch.obj(s.str("artifact_id")!!)
            CapabilityJson.spec(if (over == null) s else LinkedHashMap(s).also { m -> m.putAll(over) })
        }
    }

    private fun checkOption(w: String, got: TrainingOption?, e: Map<String, Any?>?) {
        if (e == null) { assertEquals(null, got, w); return }
        val o = assertNotNull(got, w)
        assertEquals(e.str("mode"), o.mode, "$w mode")
        assertEquals(e.lng("trainable_last_layers")!!.toInt(), o.trainableLastLayers, "$w layers")
        near(o.trainableParamsM, e.dbl("trainable_params_m"), "$w params")
        near(o.ramMb, e.dbl("ram_mb"), "$w ram")
        near(o.utilization, e.dbl("utilization"), "$w util")
        near(o.estTokensPerS, e.dbl("est_tokens_per_s"), "$w tps")
        near(o.estMinutesForReferenceTokens, e.dbl("est_minutes_for_reference_tokens"), "$w minutes")
        near(o.checkpointMb, e.dbl("checkpoint_mb"), "$w ckpt")
        near(o.storageRequiredMb, e.dbl("storage_required_mb"), "$w storage")
        assertEquals(e.bool("fits"), o.fits, "$w fits")
        assertEquals(e.bool("measured"), o.measured, "$w measured")
    }

    private fun checkReport(name: String, rep: CapabilityReport, exp: Map<String, Any?>) {
        val ec = exp.list("capabilities")!!
        assertEquals(ec.size, rep.capabilities.size, "$name capability count")
        for ((c, e0) in rep.capabilities.zip(ec)) {
            val e = e0.asObj()!!
            val w = "$name/${c.artifactId}"
            assertEquals(e.str("artifact_id"), c.artifactId, w)
            assertEquals(e.bool("can_download"), c.canDownload, "$w download")
            assertEquals(e.bool("can_load"), c.canLoad, "$w load")
            assertEquals(e.bool("can_infer"), c.canInfer, "$w infer")
            assertEquals(e.bool("can_evaluate"), c.canEvaluate, "$w evaluate")
            assertEquals(e.bool("can_specialize_full"), c.canSpecializeFull, "$w full")
            assertEquals(e.bool("can_specialize_partial"), c.canSpecializePartial, "$w partial")
            assertEquals(e.bool("external_compute_required"), c.externalComputeRequired, "$w external")
            assertEquals(e.lng("infer_context")?.toInt(), c.inferContext, "$w ctx")
            near(c.estTokensPerS, e.dbl("est_tokens_per_s"), "$w tps")
            near(c.inferRamMb, e.dbl("infer_ram_mb"), "$w infer ram")
            near(c.inferUtilization, e.dbl("infer_utilization"), "$w infer util")
            near(c.loadRamMb, e.dbl("load_ram_mb"), "$w load ram")
            near(c.qualityScore, e.dbl("quality_score"), "$w quality")
            assertEquals(e.lng("trainable_last_layers")?.toInt(), c.trainableLastLayers, "$w k")
            checkOption("$w full", c.full, e.obj("full"))
            checkOption("$w partial", c.partial, e.obj("partial"))
            assertEquals(e.str("specialize_via_artifact_id"), c.specializeViaArtifactId, "$w via")
            val conds = e.list("training_conditions")!!.map { it.asObj()!!.let { o -> o.str("name")!! to o.str("status")!! } }
            assertEquals(conds, c.trainingConditions.map { it.name to it.status }, "$w conditions")
            assertEquals(e.bool("ready_to_train_now"), c.readyToTrainNow, "$w ready")
            assertEquals(e.str("confidence_infer"), c.confidenceInfer, "$w conf infer")
            assertEquals(e.str("confidence_train"), c.confidenceTrain, "$w conf train")
            assertEquals(strings(e.list("reasons")), c.reasons, "$w reasons")
            assertEquals(strings(e.list("ignored_measurements")), c.ignoredMeasurements, "$w ignored")
        }
        val ech = exp.list("choices")!!
        assertEquals(ech.size, rep.choices.size, "$name choice count")
        for ((p, e0) in rep.choices.zip(ech)) {
            val e = e0.asObj()!!
            val w = "$name/${p.choice}"
            assertEquals(e.str("choice"), p.choice, w)
            assertEquals(e.str("label"), p.label, "$w label")
            assertEquals(e.str("artifact_id"), p.artifactId, "$w artifact")
            assertEquals(e.str("tier"), p.tier, "$w tier")
            assertEquals(e.str("confidence"), p.confidence, "$w confidence")
            assertEquals(e.lng("context_tokens")?.toInt(), p.contextTokens, "$w ctx")
            assertEquals(e.lng("trainable_last_layers")?.toInt(), p.trainableLastLayers, "$w layers")
            assertEquals(e.str("license_state"), p.licenseState, "$w license")
            assertEquals(e.bool("downloadable"), p.downloadable, "$w downloadable")
            assertEquals(e.str("reason"), p.reason, "$w reason")
        }
    }

    @Test
    fun formatAndAlgorithmVersionMatch() {
        assertEquals("device_capability.v1", doc.str("format"))
        assertEquals(Capability.ALGORITHM_VERSION, doc.str("algorithm_version"))
    }

    @Test
    fun allCapabilityCasesMatchPython() {
        var n = 0
        for (c in cases.filter { it.str("kind") == "capability" || it.str("kind") == "capability_snapshot" }) {
            val name = c.str("name")!!
            val safety = Codec.policy(c.obj("safety").orEmpty())
            val capPolicy = CapabilityJson.capPolicy(c.obj("cap_policy").orEmpty())
            val records = c.list("records").orEmpty().map { CapabilityJson.record(it.asObj()!!) }
            val device: DeviceProfile
            val state: DeviceState?
            if (c.str("kind") == "capability_snapshot") {
                val json = MiniJsonWriter.write(c["snapshot"])
                device = assertNotNull(DeviceSnapshot.profileFromSnapshot(json).profile, name)
                state = DeviceSnapshot.stateFromSnapshot(json)
                assertEquals(strings(c.obj("expected")!!.list("notes")), DeviceSnapshot.profileFromSnapshot(json).notes, "$name notes")
                assertEquals(CapabilityJson.state(c.obj("state")!!), state, "$name state")
            } else {
                device = Codec.device(c.obj("device")!!)
                state = c.obj("state")?.let(CapabilityJson::state)
            }
            val rep = Capability.choose(device, specsOf(c), safety, capPolicy, state, records)
            checkReport(name, rep, c.obj("expected")!!)
            n++
        }
        assertTrue(n >= 30, "expected many capability cases, ran $n")
    }

    @Test
    fun mergeCasesMatchPython() {
        var n = 0
        for (c in cases.filter { it.str("kind") == "merge" }) {
            val ex = c.list("existing").orEmpty().map { CapabilityJson.record(it.asObj()!!) }
            val nw = c.list("new").orEmpty().map { CapabilityJson.record(it.asObj()!!) }
            assertEquals(strings(c.obj("expected")!!.list("record_ids")), Capability.mergeRecords(ex, nw).map { it.recordId }, c.str("name"))
            n++
        }
        assertTrue(n >= 4)
    }

    @Test
    fun estimatesAreNeverRecommendedAndUnrefreshedAreOnlyPreviews() {
        for (c in cases.filter { it.str("kind") == "capability_snapshot" }) {
            for (ch in c.obj("expected")!!.list("choices")!!) {
                val tier = ch.asObj()!!.str("tier")
                assertTrue(tier != "recommended", c.str("name")!!)
                if (c.str("spec_set") == "unrefreshed") assertTrue(tier == "preview" || tier == "none", c.str("name")!!)
            }
        }
    }

    @Test
    fun recordJsonRoundTrip() {
        val r = MeasurementRecord("r1", "a", "sha256:x", "dev", null, null, "2026-10-06T00:00:00Z", "training", trainableLastLayers = 4,
            trainPeakRamMb = 1.5, trainCompleted = true)
        val back = CapabilityJson.record(MiniJson.parse(MiniJsonWriter.write(CapabilityJson.recordToMap(r))).asObj()!!)
        assertEquals(r, back)
    }
}
