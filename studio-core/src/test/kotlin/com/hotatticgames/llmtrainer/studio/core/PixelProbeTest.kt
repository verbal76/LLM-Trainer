package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.ProfileKind
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Regression for the owner's physical test: a 15.2 GB Pixel reporting only ~2.1 GB 'available' (cached apps are
 * reclaimable) must NOT be told that nothing fits, and 7B-class must stay reachable on capable hardware.
 */
class PixelProbeTest {
    private val gb = 1024L * 1024 * 1024
    private fun snapshot(avail: Double, extra: String = "") = """{"manufacturer":"Google","model":"Pixel 10 Pro XL","sdkInt":37,
        "abis":"arm64-v8a","totalRamBytes":${(15.2 * gb).toLong()},"availRamBytes":${(avail * gb).toLong()},
        "lowMemoryThresholdBytes":${400L * 1024 * 1024},"lowMemory":false,"isLowRamDevice":false,"memoryClassMb":512,
        "freeStorageBytes":${(54.5 * gb).toLong()},"totalStorageBytes":${(228.4 * gb).toLong()},"cpuCores":8,"thermalStatus":0$extra}"""

    private fun recs(snap: String) =
        StudioFactory.create(Files.createTempDirectory("px").toFile()) { snap }.recommendations().profiles.associateBy { it.kind }

    @Test fun capablePhoneWithLowMomentaryFreeRamStillGetsRecommendations() {
        val r = recs(snapshot(2.1))
        for (k in ProfileKind.values()) assertNotNull(r[k]?.modelId, "$k must have a pick on a 15 GB phone")
        assertTrue((r[ProfileKind.MAX_QUALITY]?.estimatedPeakRamMb ?: 0) > (r[ProfileKind.PERFORMANCE]?.estimatedPeakRamMb ?: 0))
        assertTrue(r.values.all { it.confidence.benchmarked == false }, "estimates must never claim a benchmark")
    }

    @Test fun reallyLowMemoryStateIsStillRespected() {
        val low = snapshot(0.2).replace("\"lowMemory\":false", "\"lowMemory\":true")
        assertTrue(recs(low).values.all { it.modelId == null }, "a device in low-memory state gets no recommendation")
    }
}
