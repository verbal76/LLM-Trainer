package com.hotatticgames.llmtrainer.qualify

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

class UnitTest {
    // ---- MiniJson -----------------------------------------------------------------------------------
    @Test
    fun parsesBasicsAndNumberKinds() {
        val v = MiniJson.parse("""{"a":1,"b":-2.5,"c":1e3,"d":[true,false,null,"x\n\u0041"],"e":{}}""").asObj()!!
        assertEquals(1L, v["a"]); assertEquals(-2.5, v["b"]); assertEquals(1000.0, v["c"])
        assertEquals(listOf(true, false, null, "x\nA"), v["d"])
        assertEquals(emptyMap<String, Any?>(), v["e"])
        assertEquals(12345678901234L, MiniJson.parse("12345678901234"))
    }

    @Test
    fun rejectsMalformed() {
        for (bad in listOf("", "{", "[1,]", "{\"a\"}", "01", "1.", "\"abc", "tru", "{} x", "[1 2]", "\"\\q\"", "-", "{\"a\":1,}")) {
            assertFailsWith<JsonException>(bad) { MiniJson.parse(bad) }
        }
    }

    private fun conv(e: JsonElement): Any? = when (e) {
        is JsonNull -> null
        is JsonObject -> e.mapValues { conv(it.value) }
        is JsonArray -> e.map { conv(it) }
        is JsonPrimitive -> when {
            e.isString -> e.content
            e.content == "true" -> true
            e.content == "false" -> false
            else -> e.content.toLongOrNull() ?: e.content.toDouble()
        }
    }

    @Test
    fun agreesWithKotlinxSerializationOnGoldenFile() {
        val text = File(System.getProperty("golden.path")).readText()
        assertEquals(conv(Json.parseToJsonElement(text)), MiniJson.parse(text))
    }

    // ---- PyFormat -----------------------------------------------------------------------------------
    @Test
    fun pythonCompatibleFormatting() {
        assertEquals("2", PyFormat.fixed(2.5, 0)); assertEquals("4", PyFormat.fixed(3.5, 0))
        assertEquals("0.1", PyFormat.fixed(0.125, 1)); assertEquals("0.12", PyFormat.fixed(0.125, 2))
        assertEquals("-0.0", PyFormat.fixed(-0.04, 1))
        assertEquals(0.12, PyFormat.round(0.125, 2)); assertEquals(2L, PyFormat.roundInt(2.5)); assertEquals(4L, PyFormat.roundInt(3.5))
        assertEquals("3.0", PyFormat.repr(3.0)); assertEquals("0.7", PyFormat.repr(0.7)); assertEquals("29.5043", PyFormat.repr(29.5043))
        assertEquals("92%", PyFormat.percent0(0.92)); assertEquals("1.0625", PyFormat.repr(1.0625))
    }

    // ---- Qualifier behavior (semantics, independent of the golden file) ----------------------------
    private fun flagship(typical: Long? = null) =
        DeviceProfile("f", "f", "c", null, "android", 12288, typical, 3000, 1500, 600, 256000, 120000, 68.0)

    @Test
    fun kvCacheFormula() {
        assertEquals(512.0, Qualifier.kvCacheMb(32, 8, 128, 4096, "f16"), 1e-9)
        assertTrue(Qualifier.kvCacheMb(32, 8, 128, 4096, "q8_0") < 512 * 0.6)
    }

    @Test
    fun neverRecommendedOnEstimatesAndSevenBFitsOnlyInsideEnvelope() {
        val cands = ReferenceCatalog.load().candidates()
        val rec = Qualifier.recommend(flagship(), cands)
        assertTrue(rec.picks.all { it.tier == Tier.PROVISIONAL && !it.recommended && it.confidence == Confidence.ESTIMATED })
        assertTrue(rec.picks.first { it.profile == "balanced" }.config!!.model.paramsB >= 7)
        val tight = Qualifier.recommend(flagship(typical = 5000), cands)
        assertTrue(tight.picks.all { it.config!!.model.paramsB < 7 }, "limited available RAM must exclude 7B")
        val none = Qualifier.recommend(flagship(typical = 0), cands)
        assertTrue(none.picks.all { it.tier == Tier.NONE })
    }

    @Test
    fun explanationsMentionWhyLargerModelsWereExcluded() {
        val cands = ReferenceCatalog.load().candidates()
        val rec = Qualifier.recommend(flagship(typical = 5000), cands)
        val bal = rec.picks.first { it.profile == "balanced" }
        val lines = Explain.exclusionLines(rec.assessed, bal, rec.policy)
        assertTrue(lines.any { it.startsWith("ref-7b:") && "RAM" in it }, lines.toString())
        assertTrue(Explain.pickLines(rec.assessed.first { it.config.configId == bal.config!!.configId }).size == 3)
        assertTrue(Explain.budgetLines(flagship(typical = 5000), rec.policy).any { it.startsWith("= model-side budget") })
    }

    // ---- Snapshot tolerance -----------------------------------------------------------------------
    @Test
    fun hostV1SnapshotWithNewOptionalFieldsParses() {
        val json = """{"manufacturer":"G","model":"P","sdkInt":35,"abis":"arm64-v8a","abis64":true,"totalRamBytes":12000000000,
            "availRamBytes":8000000000,"lowMemoryThresholdBytes":300000000,"lowMemory":false,"isLowRamDevice":false,"memoryClassMb":512,
            "largeMemoryClassMb":1024,"freeStorageBytes":100000000000,"totalStorageBytes":256000000000,"cpuCores":8,"socModel":"T",
            "socManufacturer":"Google","thermalStatus":1,"powerSaveMode":false,"batteryPct":80,"vulkanLevel":1,"vulkanVersion":4202496,
            "hasVulkanCompute":true,"cpuFeatures":["asimd","i8mm"],"procMemAvailableBytes":7000000000,"swapTotalBytes":0,
            "swapFreeBytes":0,"cachedBytes":1,"future":{"x":[1]}}"""
        val r = DeviceSnapshot.profileFromSnapshot(json)
        assertTrue(r.withheld.isEmpty()); assertTrue(r.profile != null)
        assertEquals("thermal-ok", if ("thermal_unknown" in r.notes) "unknown" else "thermal-ok")
    }

    @Test
    fun garbageSnapshotIsWithheldNotCrash() {
        assertEquals(listOf("missing_core_fields"), DeviceSnapshot.profileFromSnapshot("not json").withheld)
        assertNull(DeviceSnapshot.profileFromSnapshot("{}").profile)
        val res = DeviceSnapshot.qualify("{}", ReferenceCatalog.load().candidates())
        assertTrue(res.picks.all { it.tier == Tier.NONE })
    }
}
