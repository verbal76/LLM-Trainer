package com.hotatticgames.llmtrainer.host

import android.view.View
import androidx.test.platform.app.InstrumentationRegistry
import com.hotatticgames.hag.runtime.EngineStatus
import com.hotatticgames.llmtrainer.BuildConfig
import com.hotatticgames.llmtrainer.hostapi.Capabilities
import com.hotatticgames.llmtrainer.hostapi.HOST_API_LEVEL
import com.hotatticgames.llmtrainer.hostapi.UpdateStatus
import com.hotatticgames.llmtrainer.ota.BundleManifest
import com.hotatticgames.llmtrainer.ota.CHANNEL_SCHEMA
import com.hotatticgames.llmtrainer.ota.ChannelEntry
import com.hotatticgames.llmtrainer.ota.ChannelIndex
import com.hotatticgames.llmtrainer.ota.Fetcher
import com.hotatticgames.llmtrainer.ota.Hashing
import com.hotatticgames.llmtrainer.ota.OtaJson
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.zip.ZipInputStream

/**
 * Host <-> engine wiring on a real device/emulator: the REAL engine is advertised only when it really initialised,
 * and the host still boots (and refuses engine-dependent OTAs) when it did not.
 */
class HostEngineTest {
    private val instr = InstrumentationRegistry.getInstrumentation()
    private val target = instr.targetContext
    private fun root() = File(target.cacheDir, "hosteng-${System.nanoTime()}").also { it.mkdirs() }

    private class OneBundleFetcher(private val name: String, private val bytes: ByteArray) : Fetcher {
        val downloads = mutableListOf<String>()
        private val index: ByteArray
        init {
            val m = ZipInputStream(bytes.inputStream()).use { z ->
                generateSequence { z.nextEntry }.first { it.name == "manifest.json" }
                OtaJson.decodeFromString(BundleManifest.serializer(), z.readBytes().toString(Charsets.UTF_8))
            }
            index = OtaJson.encodeToString(
                ChannelIndex.serializer(),
                ChannelIndex(
                    CHANNEL_SCHEMA, "llmtrainer-main", "stable", "now",
                    listOf(ChannelEntry(m.bundleVersion, m.bundleVersionName, "https://test.invalid/$name.hagb", Hashing.sha256Hex(bytes), bytes.size.toLong(), m.requires)),
                ),
            ).toByteArray()
        }
        override fun getBytes(url: String, maxBytes: Long) = index
        override fun download(url: String, dest: File, maxBytes: Long) { downloads += url; dest.writeBytes(bytes) }
    }

    private fun fixture(name: String) = instr.context.assets.open("ota-fixtures/$name.hagb").readBytes()

    private fun check(rt: HostRuntime): UpdateStatus {
        var s: UpdateStatus? = null
        val latch = CountDownLatch(1)
        instr.runOnMainSync { rt.checkForUpdates { s = it; latch.countDown() } }
        assertTrue("update check timed out", latch.await(60, TimeUnit.SECONDS))
        return s!!
    }

    private fun unavailable() = HostRuntime(
        target, root(), engineStatusProvider = { EngineStatus.Unavailable("cpuGate", "simulated: CPU lacks dotprod") },
    )

    @Test fun nativeAbiAndApiLevelAreTheV2Numbers() {
        assertEquals(2, BuildConfig.NATIVE_ABI)
        assertEquals(2, HOST_API_LEVEL)
        assertTrue("builtin bundle must supersede v1-era OTA slots (#1,#2)", BuildConfig.BUILTIN_BUNDLE_VERSION > 2)
        assertTrue(BuildConfig.GIT_SHA.isNotBlank())
    }

    @Test fun realEngineInitialisesAndIsAdvertised() {
        val rt = HostRuntime(target, root())
        val svc = rt.services
        val status = NativeEngine.status
        assertTrue("engine must come up on the CI emulator: " + (status as? EngineStatus.Unavailable)?.let { "${it.stage}: ${it.reason}" }, status is EngineStatus.Ready)
        assertNotNull(svc.engine)
        assertNull(svc.engineUnavailableReason)
        assertTrue(rt.hostInfo.capabilities.containsAll(setOf(Capabilities.INFERENCE_GGUF_V1, Capabilities.TRAINING_PATCH_V1, Capabilities.CORE_V1)))
        assertTrue(svc.nativeRuntimeId, svc.nativeRuntimeId.startsWith("hag-engine"))
        assertEquals(svc.nativeRuntimeId, svc.nativeVersion)
        assertEquals(BuildConfig.GIT_SHA, svc.buildSha)
        val d = JSONObject(rt.diagnosticsJson())
        assertEquals("ready", d.getString("engineStatus"))
        assertEquals(BuildConfig.GIT_SHA, d.getString("buildSha"))
        assertEquals(2, d.getInt("nativeAbi"))
        assertEquals(svc.nativeRuntimeId, d.getString("nativeRuntimeId"))
        assertTrue(JSONObject(svc.engine!!.systemInfoJson()).has("abi"))
    }

    @Test fun unavailableEngineStillBootsAndAdvertisesNothing() {
        val rt = unavailable()
        var v: View? = null
        val r = rt.boot { app -> instr.runOnMainSync { v = app.createContentView(target) }; v!! }
        assertNotNull("host must boot without an engine", r)
        assertEquals("builtin", r!!.source)
        val svc = rt.services
        assertNull(svc.engine)
        assertTrue(svc.engineUnavailableReason!!.contains("cpuGate"))
        assertTrue(svc.nativeVersion.startsWith("unavailable"))
        assertFalse(Capabilities.INFERENCE_GGUF_V1 in rt.hostInfo.capabilities)
        assertFalse(Capabilities.TRAINING_PATCH_V1 in rt.hostInfo.capabilities)
        assertTrue(Capabilities.CORE_V1 in rt.hostInfo.capabilities)
        val d = JSONObject(rt.diagnosticsJson())
        assertTrue(d.getString("engineStatus"), d.getString("engineStatus").startsWith("unavailable: cpuGate"))
    }

    @Test fun engineDependentBundleIsRefusedWithoutAnEngineAndStagedWithOne() {
        val bytes = fixture("bundle-9-needs-engine")
        val noEngine = unavailable()
        val f1 = OneBundleFetcher("eng", bytes)
        assertEquals("NEEDS_NEW_APK", check(HostRuntime(target, root(), f1, engineStatusProvider = { EngineStatus.Unavailable("loadEngine", "x") })).kind)
        assertTrue("must not download a bundle whose capabilities this host lacks", f1.downloads.isEmpty())
        assertNull(noEngine.store.load().pending)

        val f2 = OneBundleFetcher("eng", bytes)
        assertEquals("STAGED", check(HostRuntime(target, root(), f2)).kind) // real engine -> capabilities present
    }
}
