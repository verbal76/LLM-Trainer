package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.err
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ModelsServiceTest {
    private val gb = 1024L * 1024 * 1024

    /** The owner's physical-test device: 15.2 GB Pixel 10 Pro XL reporting only ~2.1 GB 'available'. */
    private fun pixel(avail: Double = 2.1, extra: String = "") = """{"manufacturer":"Google","model":"Pixel 10 Pro XL","sdkInt":37,
        "abis":"arm64-v8a","totalRamBytes":${(15.2 * gb).toLong()},"availRamBytes":${(avail * gb).toLong()},
        "lowMemoryThresholdBytes":${400L * 1024 * 1024},"lowMemory":false,"isLowRamDevice":false,"memoryClassMb":512,
        "freeStorageBytes":${(54.5 * gb).toLong()},"totalStorageBytes":${(228.4 * gb).toLong()},"cpuCores":8,"thermalStatus":0$extra}"""

    private val PID = "Google-Pixel 10 Pro XL"

    /** A small but realistic CI-refreshed catalog: SmolLM2-360M (F16 + Q8_0) and Qwen3-4B Q4_K_M + Qwen3-0.6B (F16 + Q8_0). */
    private fun refreshedCatalog(): List<Pair<String, String>> {
        fun a(id: String, base: String, repo: String, file: String, size: Long, q: String, prec: String, nominal: Double, params: Long, cls: String, src: String? = null, layers: Int = 28) =
            "$id.json" to ArtifactKit.refreshed(id, base, repo, file, size, Hashing.sha256(id), q, prec, nominalB = nominal, params = params, training = cls, trainingSource = src, layers = layers)
        return listOf(
            a("smollm2-360m-instruct-f16", "HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", "smollm2-360m-f16.gguf", 724_000_000, "F16", "f16", 0.36, 362_000_000, "local_full", layers = 32),
            a("smollm2-360m-instruct-q8_0", "HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", "smollm2-360m-q8.gguf", 385_000_000, "Q8_0", "quantized", 0.36, 362_000_000, "inference_only", "smollm2-360m-instruct-f16", layers = 32),
            a("qwen3-0.6b-f16", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-f16.gguf", 1_190_000_000, "F16", "f16", 0.6, 596_000_000, "local_partial"),
            a("qwen3-0.6b-q8_0", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", 640_000_000, "Q8_0", "quantized", 0.6, 596_000_000, "inference_only", "qwen3-0.6b-f16"),
            a("qwen3-4b-q4_k_m", "Qwen/Qwen3-4B", "Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", 2_500_000_000, "Q4_K_M", "quantized", 4.0, 4_020_000_000, "inference_only", layers = 36),
        )
    }

    private fun open(snap: String = pixel(), artifacts: List<Pair<String, String>> = refreshedCatalog(), dir: File = TK.tmp()): StudioCore {
        val rig = TK.rig(dir = dir)
        rig.snap = { snap }
        return ArtifactKit.open(rig, artifacts)
    }

    // ---- choices ---------------------------------------------------------------------------------------------------

    @Test fun beforeTheFirstCatalogRefreshEveryChoiceIsAPreviewAndNothingIsDownloadable() {
        val rig = TK.rig(); rig.snap = { pixel() }
        val c = rig.open().models.modelChoices()
        assertEquals(setOf("fastest", "balanced", "best_quality", "best_specialize"), c.choices.map { it.choice }.toSet())
        for (ch in c.choices) { assertTrue(ch.tier == "preview" || ch.tier == "none", "${ch.choice}: ${ch.tier}"); assertFalse(ch.downloadable) }
        assertTrue(c.artifacts.none { it.downloadable || it.canDownload })
        assertTrue(c.artifacts.all { it.confidenceInfer == "low" && it.confidenceTrain == "low" })
        assertTrue(c.catalog.note.contains("not been refreshed"), c.catalog.note)
        // sizes of unrefreshed files are estimates and say so
        assertTrue(c.artifacts.all { it.sizeIsEstimate && it.catalogState == "unrefreshed" })
    }

    @Test fun pixelRegressionRefreshedCatalogGivesSensibleChoicesWithEstimatesLabelledLowConfidence() {
        val c = open().models.modelChoices()
        val by = c.choices.associateBy { it.choice }
        assertTrue(by.values.all { it.variantId != null && it.downloadable }, by.toString())
        assertTrue(by.values.all { it.tier == "provisional" && it.confidence == "low" }, "estimates are never 'recommended'")
        assertEquals("qwen3-4b-q4_k_m", by.getValue("best_quality").artifactId)
        assertTrue(by.getValue("best_specialize").artifactId in setOf("smollm2-360m-instruct-f16", "qwen3-0.6b-f16"))
        // inference is feasible for the quantized files, but they can NOT be trained: that needs the full-precision sibling
        val q8 = c.artifacts.first { it.artifactId == "smollm2-360m-instruct-q8_0" }
        assertTrue(q8.canInfer && !q8.canSpecializeFull && !q8.canSpecializePartial && q8.externalComputeRequired)
        assertEquals(c.artifacts.first { it.artifactId == "smollm2-360m-instruct-f16" }.variantId, q8.specializeViaVariantId)
        val f16 = c.artifacts.first { it.artifactId == "smollm2-360m-instruct-f16" }
        assertTrue(f16.canSpecializePartial && f16.trainableLastLayers!! >= 2 && f16.partial!!.ramMb > f16.estPeakRamMb!!, "training needs more RAM than inference")
        assertTrue(f16.reasons.none { it.contains("not been refreshed") })
        assertTrue(f16.conditions.any { it.name == "charging" && it.status == "unknown" } && !f16.readyToTrainNow)
        assertEquals(LicenseState.VERIFIED, f16.licenseState)
    }

    @Test fun lowMemoryStateAndHotDeviceWithholdChoices() {
        val low = open(pixel(0.2).replace("\"lowMemory\":false", "\"lowMemory\":true")).models.modelChoices()
        assertTrue(low.artifacts.none { it.canLoad }, "a device in low-memory state cannot load anything")
        val hot = open(pixel(extra = ",\"x\":1").replace("\"thermalStatus\":0", "\"thermalStatus\":3")).models.modelChoices()
        assertTrue(hot.choices.all { it.tier == "none" && it.reason.contains("thermally throttled") })
        val unknown = open("{}").models.modelChoices()
        assertTrue(unknown.choices.all { it.tier == "none" } && unknown.artifacts.isEmpty())
    }

    @Test fun disallowedLicenseModelsAreNeverOffered() {
        val rig = TK.rig(); rig.snap = { pixel() }
        val dis = EmbeddedRegistry.files.map { (n, j) -> if (n == "qwen3-4b.json") n to j.replace(Regex("\"state\"\\s*:\\s*\"UNVERIFIED\""), "\"state\":\"DISALLOWED\",\"disallowed_reason\":\"test\"") else n to j }
        val core = StudioCore(rig.dir, { rig.snap() }, null, rig.http, rig.clock, rig.storage, rig.runner, rig.ids, "test", com.hotatticgames.llmtrainer.qualify.SafetyPolicy(),
            dis, refreshedCatalog(), EmbeddedArtifacts.canonicalLicenses)
        assertTrue(core.models.modelChoices().choices.none { it.artifactId == "qwen3-4b-q4_k_m" })
    }

    @Test fun catalogVariantViewsCarryPerArtifactFeasibilityAndNeverClaimOnDeviceTrainingYet() {
        val core = open()
        val v = core.model("Qwen3@Qwen3-4B (2504 generation, hybrid thinking)").ok().variants.first { it.id.endsWith("#qwen3-4b-q4_k_m") }
        assertEquals(2_500_000_000L, v.sizeBytes); assertNotNull(v.downloadUrl); assertNotNull(v.sha256)
        assertTrue(v.android.verdict == Verdict.FITS_SAFELY || v.android.verdict == Verdict.TIGHT, v.android.toString())
        assertTrue(v.android.reasons.first().contains("ESTIMATE"))
        assertEquals(RunLocation.DESKTOP, v.training.where)                        // no on-device trainer in this app version
        val small = core.model("SmolLM2@SmolLM2-360M-Instruct").ok().variants.first { it.id.endsWith("#smollm2-360m-instruct-f16") }
        assertEquals(RunLocation.DESKTOP, small.training.where)
        assertTrue(small.training.reasons.any { it.contains("cannot train on the phone yet") }, small.training.reasons.toString())
        assertEquals("GGUF", v.format); assertTrue(v.architecture.contains("qwen3"))
    }

    // ---- measurements ----------------------------------------------------------------------------------------------

    private fun rec(artifact: String, extra: String = "", device: String = PID, id: String = "r1", at: String = "2026-10-06T00:00:00Z") =
        """{"schema_id":"device_measurement.v1","record_id":"$id","artifact_id":"$artifact","device_id":"$device","recorded_at":"$at","kind":"inference","context_tokens":2048,
            "ttft_ms":1800,"tokens_per_s":12.0,"peak_ram_mb":3200,"sustained_ram_mb":3100,"thermal_throttle_ratio":0.9,"ui_jank_pct":1.0,"crashes":0,"anrs":0,"background_kills":0,
            "sustained_minutes":15$extra}"""

    @Test fun recordedMeasurementsReplaceEstimatesRaiseConfidenceAndSurviveRestart() {
        val dir = TK.tmp()
        val core = open(dir = dir)
        val before = core.models.capabilities(core.models.modelChoices().artifacts.first { it.artifactId == "qwen3-4b-q4_k_m" }.variantId).ok()
        assertEquals("low", before.confidenceInfer)
        assertEquals(1, core.models.recordMeasurement(rec("qwen3-4b-q4_k_m")).ok())
        val after = core.models.capabilities(before.variantId).ok()
        assertTrue(after.confidenceInfer != "low", after.confidenceInfer)
        // restart: the ledger is on disk
        val again = open(dir = dir)
        assertEquals(1, again.models.measurementRecords().size)
        assertEquals(after.confidenceInfer, again.models.capabilities(before.variantId).ok().confidenceInfer)
        // latest wins, other keys are independent
        assertEquals(1, again.models.recordMeasurement(rec("qwen3-4b-q4_k_m", id = "r2", at = "2026-10-07T00:00:00Z")).ok())
        assertEquals(2, again.models.recordMeasurement(rec("qwen3-4b-q4_k_m", id = "r3").replace("\"context_tokens\":2048", "\"context_tokens\":4096")).ok())
        assertEquals(2, again.models.recordMeasurement(rec("qwen3-4b-q4_k_m", id = "old", at = "2026-01-01T00:00:00Z")).ok())     // an older record never replaces a newer one
        assertTrue(again.models.measurementRecords().none { it.contains("\"old\"") })
    }

    @Test fun measurementsForOtherDevicesWrongFilesOrUnknownArtifactsAreRejected() {
        val m = open().models
        assertEquals("OTHER_DEVICE", (m.recordMeasurement(rec("qwen3-4b-q4_k_m", device = "Acme-Phone")).err() as StudioError.Invalid).code)
        assertEquals("ARTIFACT_HASH_MISMATCH", (m.recordMeasurement(rec("qwen3-4b-q4_k_m", ",\"artifact_sha256\":\"sha256:${"cd".repeat(32)}\"")).err() as StudioError.Invalid).code)
        assertTrue(m.recordMeasurement(rec("no-such-artifact")).err() is StudioError.NotFound)
        assertEquals("BAD_RECORD", (m.recordMeasurement("{\"kind\":\"nope\"}").err() as StudioError.Invalid).code)
        assertEquals("BAD_RECORD", (m.recordMeasurement("not json").err() as StudioError.Invalid).code)
        assertTrue(m.measurementRecords().isEmpty())
    }

    @Test fun aMeasuredFailureTurnsAnEstimatedYesIntoNo() {
        val core = open()
        val vid = core.models.modelChoices().artifacts.first { it.artifactId == "qwen3-4b-q4_k_m" }.variantId
        assertTrue(core.models.capabilities(vid).ok().canInfer)
        for ((i, ctx) in listOf(512, 1024, 2048, 4096, 8192).withIndex())
            core.models.recordMeasurement(rec("qwen3-4b-q4_k_m", id = "c$i", extra = ",\"crashes\":1").replace("\"context_tokens\":2048", "\"context_tokens\":$ctx")).ok()
        assertFalse(core.models.capabilities(vid).ok().canInfer)
    }

    @Test fun measuredTrainingRecordCapsOrExtendsTheLayerChoice() {
        val core = open()
        val f16 = core.models.modelChoices().artifacts.first { it.artifactId == "qwen3-0.6b-f16" }
        val est = f16.trainableLastLayers!!
        val oom = """{"schema_id":"device_measurement.v1","record_id":"t1","artifact_id":"qwen3-0.6b-f16","device_id":"$PID","recorded_at":"2026-10-06T00:00:00Z","kind":"training",
            "trainable_last_layers":${est},"train_peak_ram_mb":99999,"train_oom":true,"train_completed":false}"""
        core.models.recordMeasurement(oom).ok()
        val capped = core.models.capabilities(f16.variantId).ok()
        assertTrue(capped.trainableLastLayers!! < est, "an OOM at $est layers must cap the choice below it")
        assertEquals("medium", capped.confidenceTrain)
    }

    // ---- installed models seen by the capability layer -------------------------------------------------------------

    @Test fun installedFlagAndPathFollowTheStore() {
        val rig = TK.rig(); rig.snap = { pixel() }
        val data = ArtifactKit.gguf()
        val sha = Hashing.sha256(data)
        val repo = "Qwen/Qwen3-0.6B-GGUF"
        val art = ArtifactKit.refreshed("qwen3-0.6b-q8_0", "Qwen/Qwen3-0.6B", repo, "m.gguf", data.size.toLong(), sha, nominalB = 0.6, params = 596_000_000L)
        rig.http.resources["https://huggingface.co/$repo/resolve/${ArtifactKit.REV}/m.gguf"] = data
        val core = ArtifactKit.open(rig, listOf("a.json" to art))
        val vid = core.models.modelChoices().artifacts.single().variantId
        assertFalse(core.models.capabilities(vid).ok().installed)
        val use = IntendedUse()
        assertEquals(OperationState.SUCCEEDED, core.operation(core.startDownload(vid, use, true).ok().id).ok().state)
        assertTrue(core.models.capabilities(vid).ok().installed)
        assertEquals(Files.size(File(core.models.installedPath(vid)!!).toPath()), data.size.toLong())
    }

    @Test fun startupHasNoProblemsWithTheShippedRegistry() {
        val rig = TK.rig()
        assertEquals(emptyList(), rig.open().startupProblems)
        assertNotNull(ModelsFactory.of(rig.open()))
    }
}
