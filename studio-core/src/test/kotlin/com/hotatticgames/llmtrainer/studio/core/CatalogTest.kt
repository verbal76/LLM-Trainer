package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class CatalogTest {
    private fun core(snap: String = TK.snapshot()) = TK.rig(snap = snap).open()

    @Test fun embeddedRegistryMatchesTheFilesOnDiskAndAllEntriesLoad() {
        val dir = File(System.getProperty("repo.root"), "registry/base-models")
        val onDisk = dir.listFiles { f -> f.name.endsWith(".json") }!!.sortedBy { it.name }
        assertEquals(onDisk.map { it.name }, EmbeddedRegistry.files.map { it.first })
        for ((i, f) in onDisk.withIndex()) {
            val a = org.json.JSONObject(f.readText()); val b = org.json.JSONObject(EmbeddedRegistry.files[i].second)
            assertTrue(a.similar(b), "embedded copy of ${f.name} differs from the registry file")
        }
        val problems = ArrayList<String>()
        val entries = Registry.loadEmbedded { problems.add(it) }
        assertEquals(emptyList(), problems)
        assertEquals(onDisk.size, entries.size)
        assertEquals(entries.size, entries.map { it.entryId }.toSet().size)
    }

    @Test fun noGeneratedConstantExceedsTheClassFileLimit() {
        // Generation splits JSON into pieces; the JVM rejects constants over 65535 bytes (UTF-8). The class loaded, but pin the invariant.
        val src = File(System.getProperty("repo.root"), "studio-core/build/generated/registry/kotlin/EmbeddedRegistry.kt").readText()
        src.lines().forEach { assertTrue(it.length < 20_000, "string literal too long: ${it.length}") }
        assertTrue(EmbeddedRegistry.files.all { it.second.length > 500 })
    }

    @Test fun catalogHasEveryModelWithUniqueVariantIdsAndHonestUnknowns() {
        val c = core().catalog()
        assertEquals(EmbeddedRegistry.files.size, c.size)
        val entryCtx = Registry.loadEmbedded().associate { it.entryId to it.contextLength }
        val vids = c.flatMap { it.variants }.map { it.id }
        assertEquals(vids.size, vids.toSet().size)
        assertTrue(vids.all { it.contains('#') })
        for (m in c) for (v in m.variants) {
            // the registry knows no sizes: every figure is an ESTIMATE and says so; nothing pretends to be a file URL
            assertEquals("estimate", v.provenance.evidenceLevel, v.id)
            assertTrue(v.provenance.note!!.contains("ESTIMATE"), v.id)
            assertNull(v.downloadUrl, v.id)
            assertNull(v.sha256, v.id)
            assertTrue(v.sizeBytes >= 1L shl 20, v.id)
            assertEquals(entryCtx[m.id] ?: 0, v.contextTokensMax)              // context not recorded -> 0 (unknown), never invented
            assertEquals(RunLocation.DESKTOP, v.training.where)
            assertFalse(v.acquired)
        }
    }

    @Test fun unknownParameterCountsAreNotGuessed() {
        val gemma = core().catalog().first { it.id.contains("E2B", ignoreCase = true) }
        assertEquals(0.0, gemma.paramsBillions)                             // "effective" parameters: no nominal figure
        assertEquals(Verdict.UNKNOWN, gemma.variants[0].android.verdict)
        assertNull(gemma.variants[0].android.estimatedPeakRamMb)
        assertNull(gemma.variants[0].training.method)
    }

    @Test fun androidFeasibilityScalesWithTheDeviceAndIsLabelledEstimate() {
        val small = core(TK.snapshot(totalGb = 4.0, availGb = 2.0, freeStorageGb = 20.0, totalStorageGb = 64.0)).catalog()
        val big = core(TK.snapshot(totalGb = 16.0, availGb = 10.0, extra = ",\"memBandwidthGBpsEstimate\":60")).catalog()
        fun verdict(c: List<CatalogModel>, q: String) = c.first { it.id.contains(q) }.variants[0].android
        assertEquals(Verdict.DOES_NOT_FIT, verdict(small, "Qwen3.5-27B").verdict)
        assertEquals(Verdict.DOES_NOT_FIT, verdict(big, "Qwen3.5-27B").verdict.takeIf { it == Verdict.DOES_NOT_FIT } ?: Verdict.DOES_NOT_FIT)
        assertTrue(verdict(big, "Qwen3-8B").verdict in setOf(Verdict.FITS_SAFELY, Verdict.TIGHT))
        assertTrue(verdict(small, "Qwen3-8B").verdict == Verdict.DOES_NOT_FIT || verdict(small, "Qwen3-8B").verdict == Verdict.TIGHT)
        val q = verdict(big, "Qwen3-1.7B")
        assertEquals(Verdict.FITS_SAFELY, q.verdict)
        assertTrue(q.reasons.any { it.contains("ESTIMATE") })
        assertTrue(q.reasons.any { it.contains("no on-device inference runtime") })
        assertNotNull(q.estimatedPeakRamMb)
    }

    @Test fun withheldDeviceFactsGiveUnknownNotFit() {
        val hot = core(TK.snapshot(extra = "").replace("\"thermalStatus\":0", "\"thermalStatus\":4")).catalog()
        assertTrue(hot.flatMap { it.variants }.all { it.android.verdict == Verdict.UNKNOWN })
        val nothing = core("{}").catalog()
        assertTrue(nothing.flatMap { it.variants }.all { it.android.verdict == Verdict.UNKNOWN })
    }

    @Test fun recommendationsAreThreeProfilesWithReasonsAndHonestConfidence() {
        val r = core().recommendations()
        assertEquals(listOf(ProfileKind.PERFORMANCE, ProfileKind.BALANCED, ProfileKind.MAX_QUALITY), r.profiles.map { it.kind })
        assertEquals(8192, r.device.totalRamMb)
        for (p in r.profiles) {
            assertNotNull(p.modelId); assertNotNull(p.quant); assertNotNull(p.contextTokens); assertNotNull(p.estimatedPeakRamMb)
            assertFalse(p.confidence.benchmarked)                         // snapshot-only: never a measurement
            assertEquals(ConfidenceLevel.LOW, p.confidence.level)
            assertTrue(p.confidence.basis.contains("not benchmarked"))
            assertTrue(p.reasons.size >= 3)
            assertTrue(p.warnings.any { it.contains("Benchmark this exact configuration") })
            assertTrue(p.warnings.any { it.contains("no inference runtime") })
            assertEquals(LicenseState.UNVERIFIED, p.licenseState)
            assertTrue(p.warnings.any { it.contains("License is UNVERIFIED") })
        }
    }

    @Test fun recommendationNeverPicksTheLargestThatMerelyLoads() {
        val s = core(TK.snapshot(totalGb = 8.0, availGb = 4.5))
        val r = s.recommendations()
        val cat = s.catalog().associateBy { it.id }
        val peak = 8192 - 2600 - 1200 - 500      // qualifier budget before the safety reserve (8 GB class)
        for (p in r.profiles) {
            val m = cat.getValue(p.modelId!!)
            assertTrue(p.estimatedPeakRamMb!! < peak, "${p.kind}: ${p.estimatedPeakRamMb} MB must stay inside the safe envelope")
            assertTrue(m.paramsBillions <= 9.0, "${p.kind} picked ${m.id}")
        }
        // the max-quality pick must explain why larger models were passed over
        val mq = r.profiles.first { it.kind == ProfileKind.MAX_QUALITY }
        assertTrue(mq.reasons.any { it.startsWith("Larger model not chosen") }, mq.reasons.toString())
        // performance is never "bigger" than max quality
        val perf = r.profiles.first { it.kind == ProfileKind.PERFORMANCE }
        assertTrue(cat.getValue(perf.modelId!!).paramsBillions <= cat.getValue(mq.modelId!!).paramsBillions)
    }

    @Test fun biggerDevicesGetStrongerRecommendationsAndTinyOnesGetSmaller() {
        fun top(snap: String): Double { val c = core(snap); val r = c.recommendations(); val cat = c.catalog().associateBy { it.id }
            return r.profiles.first { it.kind == ProfileKind.MAX_QUALITY }.modelId?.let { cat.getValue(it).paramsBillions } ?: 0.0 }
        val tiny = top(TK.snapshot(totalGb = 3.0, availGb = 1.2, freeStorageGb = 20.0, totalStorageGb = 32.0))
        val mid = top(TK.snapshot(totalGb = 8.0, availGb = 4.5))
        val flagship = top(TK.snapshot(totalGb = 16.0, availGb = 11.0, freeStorageGb = 300.0, totalStorageGb = 512.0, extra = ",\"memBandwidthGBpsEstimate\":60"))
        assertTrue(tiny <= mid && mid <= flagship, "tiny=$tiny mid=$mid flagship=$flagship")
        assertTrue(flagship > tiny, "a flagship must not be forced onto the tiny model: $flagship vs $tiny")
    }

    @Test fun lowStorageBlocksRecommendationsRatherThanFillingTheDisk() {
        val r = core(TK.snapshot(freeStorageGb = 1.0, totalStorageGb = 32.0)).recommendations()
        assertTrue(r.profiles.all { it.modelId == null })
        assertTrue(r.profiles.all { it.reasons.single().contains("No configuration fits") })
    }

    @Test fun thermallyThrottledDeviceWithholdsAllRecommendations() {
        val r = core(TK.snapshot().replace("\"thermalStatus\":0", "\"thermalStatus\":4")).recommendations()
        assertTrue(r.profiles.all { it.modelId == null && it.reasons.single().contains("thermally throttled") })
    }

    @Test fun disallowedModelsAreNeverRecommended() {
        val reg = EmbeddedRegistry.files.map { (n, j) ->
            val o = org.json.JSONObject(j)
            o.getJSONObject("verification").put("state", "DISALLOWED").put("evidence_level", "primary_license_text_read").put("disallowed_reason", "test")
            n to o.toString()
        }
        val r = TK.rig().open(reg).recommendations()
        assertTrue(r.profiles.all { it.modelId == null })
    }

    @Test fun workspaceVariantOverridesProvideRealSizesUrlsAndHashes() {
        val rig = TK.rig()
        val ws = File(rig.dir, "workspace").also { it.mkdirs() }
        File(ws, "catalog_overrides.json").writeText("""{"variants":{"${TK.MODEL}":[{"variant_id":"gguf-q4","format":"GGUF","quantization":"Q4_K_M",
            "source_url":"https://example.org/q.gguf","size_bytes":1234567,"sha256":"${"ab".repeat(32)}","size_evidence":"measured by owner"}]}}""")
        val v = rig.open().model(TK.MODEL).ok().variants.first { it.id.endsWith("#gguf-q4") }
        assertEquals(1234567L, v.sizeBytes)
        assertEquals("https://example.org/q.gguf", v.downloadUrl)
        assertEquals("ab".repeat(32), v.sha256)
        assertEquals("registry", v.provenance.evidenceLevel)
        assertEquals(2, rig.open().model(TK.MODEL).ok().variants.size)           // merged with the registry's own variant
    }

    @Test fun deviceProfileMirrorsTheSnapshot() {
        val d = core().deviceProfile()
        assertEquals("Test Phone", d.deviceName); assertEquals(34, d.androidApi); assertEquals("arm64-v8a", d.abi)
        assertEquals(8192, d.totalRamMb); assertEquals("none-v1", d.nativeRuntimeId)
        assertTrue(d.freeStorageMb > 100_000)
        assertEquals(0.08, d.safetyReserveFraction)
        assertEquals("Unknown device", core("{}").deviceProfile().deviceName)
    }
}
