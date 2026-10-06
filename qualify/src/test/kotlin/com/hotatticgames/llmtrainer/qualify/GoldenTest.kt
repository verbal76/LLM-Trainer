package com.hotatticgames.llmtrainer.qualify

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.fail

/** Runs the SAME golden vectors the Python spec generates (factory/tests/golden/device_qualification.v1.json). */
class GoldenTest {
    private val doc: Map<String, Any?> by lazy {
        val path = System.getProperty("golden.path") ?: fail("golden.path system property missing")
        MiniJson.parse(File(path).readText()).asObj()!!
    }
    private val cases: List<Map<String, Any?>> get() = doc.list("cases")!!.map { it.asObj()!! }

    private fun near(a: Double?, b: Double?, what: String) {
        if (a == null || b == null) { assertEquals(a, b, what); return }
        assertTrue(Math.abs(a - b) <= 1e-9 * maxOf(1.0, Math.abs(b)), "$what: $a != $b")
    }

    private fun strings(l: List<Any?>?): List<String> = l.orEmpty().map { it as String }

    private fun checkPicks(name: String, got: List<ProfilePick>, exp: List<Any?>) {
        assertEquals(exp.size, got.size, name)
        for ((g, e0) in got.zip(exp)) {
            val e = e0.asObj()!!
            val w = "$name/${g.profile}"
            assertEquals(e.str("profile"), g.profile, w)
            assertEquals(e.str("label"), g.label, w)
            assertEquals(e.str("config_id"), g.config?.configId, "$w config")
            assertEquals(e.str("tier"), g.tier, "$w tier")
            assertEquals(e.bool("recommended"), g.recommended, w)
            assertEquals(e.str("confidence"), g.confidence, "$w confidence")
            assertEquals(e.str("reason"), g.reason, "$w reason")
            assertEquals(strings(e.list("warnings")), g.warnings, "$w warnings")
            assertEquals(e.str("verified_fallback_config_id"), g.verifiedFallbackConfigId, "$w fallback")
        }
    }

    private fun checkAssessed(name: String, got: List<CandidateAssessment>, exp: List<Any?>) {
        assertEquals(exp.size, got.size, "$name assessed count")
        for ((a, e0) in got.zip(exp)) {
            val e = e0.asObj()!!
            val w = "$name/${a.config.configId}"
            assertEquals(e.str("config_id"), a.config.configId, w)
            assertEquals(e.str("storage_verdict"), a.storageVerdict, "$w storage")
            assertEquals(e.str("ram_verdict"), a.ramVerdict, "$w ram")
            assertEquals(e.str("sustained_verdict"), a.sustainedVerdict, "$w sustained")
            assertEquals(e.str("confidence"), a.confidence, w)
            near(a.qualityScore, e.dbl("quality_score"), "$w quality")
            assertEquals(e.bool("safe_to_deploy"), a.safeToDeploy, "$w safe")
            assertEquals(e.bool("eligible"), a.eligible, "$w eligible")
            assertEquals(strings(e.list("blocking_reasons")), a.blockingReasons, "$w blocking")
            near(a.storage.headroomMb, e.dbl("storage_headroom_mb"), "$w storage headroom")
            near(a.ram.weightsMb, e.dbl("ram_weights_mb"), "$w weights")
            near(a.ram.runtimeMb, e.dbl("ram_runtime_mb"), "$w runtime")
            near(a.ram.kvCacheMb, e.dbl("ram_kv_cache_mb"), "$w kv")
            near(a.ram.requiredModelSideMb, e.dbl("ram_required_mb"), "$w required")
            near(a.ram.ramBudgetMb, e.dbl("ram_budget_mb"), "$w budget")
            near(a.ram.headroomMb, e.dbl("ram_headroom_mb"), "$w headroom")
            near(a.ram.utilization, e.dbl("ram_utilization"), "$w utilization")
            assertEquals(e.bool("ram_measured_override"), a.ram.measuredOverride, "$w override")
            near(a.sustained.estTokensPerS, e.dbl("est_tps"), "$w est tps")
            near(a.sustained.estTtftMs.toDouble(), e.dbl("est_ttft_ms"), "$w est ttft")
            assertEquals(strings(e.list("missing_measurements")), a.sustained.missingMeasurements, "$w missing")
            assertEquals(strings(e.list("sustained_reasons")), a.sustained.reasons, "$w sustained reasons")
        }
    }

    @Test
    fun formatAndAlgorithmVersionMatch() {
        assertEquals("device_qualification.v1", doc.str("format"))
        assertEquals(Qualifier.ALGORITHM_VERSION, doc.str("algorithm_version"))
    }

    @Test
    fun embeddedReferenceCatalogEqualsGolden() {
        val golden = doc.obj("reference_catalog")!!
        val embedded = MiniJson.parse(ReferenceCatalog.JSON)
        assertEquals(golden, embedded, "ReferenceCatalog.JSON drifted from the Python spec; regenerate it")
    }

    @Test
    fun allQualifyCasesMatchPython() {
        var n = 0
        for (c in cases.filter { it.str("kind") == "qualify" }) {
            val name = c.str("name")!!
            val device = Codec.device(c.obj("device")!!)
            val policy = Codec.policy(c.obj("policy").orEmpty())
            val cands: List<CandidateConfig> = c.obj("catalog")?.let { ref ->
                val over = ref.list("retrieval")?.mapNotNull { it.asObj()?.let(Codec::retrieval) }
                ReferenceCatalog.load().candidates(over)
            } ?: c.list("candidates")!!.map { Codec.candidate(it.asObj()!!) }
            val rec = Qualifier.recommend(device, cands, policy, c.bool("require_retrieval") == true)
            val exp = c.obj("expected")!!
            checkPicks(name, rec.picks, exp.list("picks")!!)
            assertEquals(strings(exp.list("eligible_ids")), rec.assessed.filter { it.eligible }.map { it.config.configId }, "$name eligible")
            exp.list("assessed")?.let { checkAssessed(name, rec.assessed, it) }
            n++
        }
        assertTrue(n >= 30, "expected many qualify cases, ran $n")
    }

    @Test
    fun allSnapshotCasesMatchPython() {
        val cands = ReferenceCatalog.load().candidates()
        var n = 0
        for (c in cases.filter { it.str("kind") == "snapshot" }) {
            val name = c.str("name")!!
            val exp = c.obj("expected")!!
            val res = DeviceSnapshot.qualify(MiniJsonWriter.write(c["snapshot"]), cands)
            assertEquals(strings(exp.list("withheld")), res.withheld, "$name withheld")
            assertEquals(strings(exp.list("notes")), res.notes, "$name notes")
            val ep = exp.obj("profile")
            if (ep == null) {
                assertEquals(null, res.profile, name)
            } else {
                val p = assertNotNull(res.profile, name)
                assertEquals(ep.lng("total_ram_mb"), p.totalRamMb, "$name total")
                assertEquals(ep.lng("typical_available_ram_mb"), p.typicalAvailableRamMb, "$name typical")
                assertEquals(ep.lng("os_reserve_mb"), p.osReserveMb, "$name os")
                assertEquals(ep.lng("background_reserve_mb"), p.backgroundReserveMb, "$name bg")
                assertEquals(ep.lng("host_app_mb"), p.hostAppMb, "$name host")
                assertEquals(ep.lng("storage_total_mb"), p.storageTotalMb, "$name storage total")
                assertEquals(ep.lng("storage_free_mb"), p.storageFreeMb, "$name storage free")
                near(p.memBandwidthGbps, ep.dbl("mem_bandwidth_gbps"), "$name bw")
            }
            checkPicks(name, res.picks, exp.list("picks")!!)
            n++
        }
        assertTrue(n >= 15, "expected many snapshot cases, ran $n")
    }

    @Test
    fun snapshotsAreNeverRecommendedOnEstimates() {
        val cands = ReferenceCatalog.load().candidates()
        for (c in cases.filter { it.str("kind") == "snapshot" }) {
            val res = DeviceSnapshot.qualify(MiniJsonWriter.write(c["snapshot"]), cands)
            assertTrue(res.picks.none { it.tier == Tier.RECOMMENDED }, c.str("name")!!)
        }
    }
}

/** Test-only serializer so snapshot cases go through the same string entry point the bundle uses. */
object MiniJsonWriter {
    fun write(v: Any?): String = when (v) {
        null -> "null"
        is String -> "\"" + v.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
        is Map<*, *> -> v.entries.joinToString(",", "{", "}") { write(it.key as String) + ":" + write(it.value) }
        is List<*> -> v.joinToString(",", "[", "]") { write(it) }
        else -> v.toString()
    }
}
