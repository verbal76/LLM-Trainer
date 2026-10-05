package com.hotatticgames.llmtrainer.host

import android.view.View
import androidx.test.platform.app.InstrumentationRegistry
import com.hotatticgames.llmtrainer.BuildConfig
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
 * OTA qualification on a real Android runtime: real DexClassLoader, real filesystem, real signature
 * verification against the key embedded in this APK. Fixture bundles are produced by CI with the same
 * key (see .github/workflows/android.yml):
 *   bundle-2-good (v2), bundle-3-selftest (v3, selfTest fails), bundle-4-abi2 (v4, needs native ABI 2),
 *   bundle-5-throws (v5, entry constructor throws).
 */
class OtaQualificationTest {
    private val instr = InstrumentationRegistry.getInstrumentation()
    private val target = instr.targetContext

    private fun fixture(name: String) = instr.context.assets.open("ota-fixtures/$name.hagb").readBytes()

    private fun freshRoot() = File(target.cacheDir, "otatest-${System.nanoTime()}").also { it.mkdirs() }

    private class ChannelFetcher(bundles: Map<String, ByteArray>) : Fetcher {
        val downloads = mutableListOf<String>()
        private val files = bundles.mapKeys { "https://test.invalid/${it.key}.hagb" }
        private val index: ByteArray

        init {
            val entries = bundles.map { (name, bytes) ->
                val m = manifestOf(bytes)
                ChannelEntry(m.bundleVersion, m.bundleVersionName, "https://test.invalid/$name.hagb", Hashing.sha256Hex(bytes), bytes.size.toLong(), m.requires)
            }
            index = OtaJson.encodeToString(
                ChannelIndex.serializer(),
                ChannelIndex(CHANNEL_SCHEMA, "llmtrainer-main", "stable", "now", entries),
            ).toByteArray()
        }

        override fun getBytes(url: String, maxBytes: Long) = index
        override fun download(url: String, dest: File, maxBytes: Long) {
            downloads += url
            dest.writeBytes(files[url] ?: throw IOException("404 $url"))
        }

        companion object {
            fun manifestOf(bytes: ByteArray): BundleManifest {
                ZipInputStream(bytes.inputStream()).use { z ->
                    while (true) {
                        val e = z.nextEntry ?: break
                        if (e.name == "manifest.json") return OtaJson.decodeFromString(BundleManifest.serializer(), z.readBytes().toString(Charsets.UTF_8))
                    }
                }
                error("no manifest")
            }
        }
    }

    private fun runtime(root: File, fetcher: Fetcher = ChannelFetcher(emptyMap())) = HostRuntime(target, root, fetcher)

    private fun boot(rt: HostRuntime) = rt.boot { app ->
        var v: View? = null
        instr.runOnMainSync { v = app.createContentView(target) }
        v!!
    }

    private fun check(rt: HostRuntime): UpdateStatus {
        var s: UpdateStatus? = null
        val latch = CountDownLatch(1)
        instr.runOnMainSync { rt.checkForUpdates { s = it; latch.countDown() } }
        assertTrue("update check timed out", latch.await(60, TimeUnit.SECONDS))
        return s!!
    }

    @Test fun builtinBundleBootsWithBakedInVersion() {
        val rt = runtime(freshRoot())
        val r = boot(rt)
        assertNotNull(r)
        assertEquals("builtin", r!!.source)
        assertEquals(BuildConfig.BUILTIN_BUNDLE_VERSION, r.version)
        val d = JSONObject(rt.diagnosticsJson())
        assertEquals(BuildConfig.VERSION_NAME, d.getString("hostVersionName"))
        assertEquals(BuildConfig.NATIVE_ABI, d.getInt("nativeAbi"))
        assertEquals("builtin", d.getString("runningSource"))
    }

    @Test fun studioLogoAssetIsTheExactCanonicalFile() {
        val bytes = target.assets.open("branding/studio-logo.png").readBytes()
        assertEquals("e3d9bb5653eafb783eede827606e7ac73a4e45564a1c25b1ed13ad1429f48c4e", Hashing.sha256Hex(bytes))
    }

    @Test fun fullOtaCycleStagesRestartsAndPromotes() {
        val root = freshRoot()
        val fetcher = ChannelFetcher(mapOf("good" to fixture("bundle-2-good")))
        assertEquals("STAGED", check(runtime(root, fetcher)).kind)

        val rt2 = runtime(root, fetcher) // simulated app restart: new runtime, same on-disk state
        val r = boot(rt2)!!
        assertEquals("ota", r.source)
        assertEquals(2, r.version)
        rt2.onFirstFrame()
        val st = rt2.store.load()
        assertEquals(r.slotId, st.active); assertEquals(r.slotId, st.lastKnownGood); assertNull(st.pending)
        assertEquals("UP_TO_DATE", check(rt2).kind)
    }

    @Test fun badUpdatesRollBackToLastKnownGood() {
        val root = freshRoot()
        val good = ChannelFetcher(mapOf("good" to fixture("bundle-2-good")))
        check(runtime(root, good))
        val rtA = runtime(root, good); val a = boot(rtA)!!; rtA.onFirstFrame()

        for ((name, version) in listOf("bundle-3-selftest" to 3, "bundle-5-throws" to 5)) {
            val f = ChannelFetcher(mapOf(name to fixture(name)))
            assertEquals("STAGED", check(runtime(root, f)).kind)
            val rt = runtime(root, f)
            val r = boot(rt)!!
            assertEquals("must fall back to v2 after v$version fails", 2, r.version)
            assertEquals(a.slotId, r.slotId)
            assertTrue(rt.store.load().quarantined.keys.any { it.startsWith("v$version-") })
        }
    }

    @Test fun crashLoopingTrialIsQuarantinedAndBuiltinTakesOver() {
        val root = freshRoot()
        val f = ChannelFetcher(mapOf("good" to fixture("bundle-2-good")))
        check(runtime(root, f))
        // Two starts that never reach a first frame (process died): budget for a trial is 2.
        repeat(2) { assertEquals("ota", boot(runtime(root, f))!!.source) }
        val r = boot(runtime(root, f))!!
        assertEquals("builtin", r.source)
    }

    @Test fun incompatibleNativeBundleIsRefusedAndNeverInstalled() {
        val root = freshRoot()
        val f = ChannelFetcher(mapOf("abi2" to fixture("bundle-4-abi2")))
        val s = check(runtime(root, f))
        assertEquals("NEEDS_NEW_APK", s.kind)
        assertTrue(f.downloads.isEmpty())
        assertNull(runtime(root, f).store.load().pending)
    }

    @Test fun tamperedBundleWithMatchingIndexHashIsRejected() {
        val bytes = fixture("bundle-2-good").copyOf()
        bytes[bytes.size / 2] = (bytes[bytes.size / 2] + 1).toByte()
        val root = freshRoot()
        // The index lies and even carries the tampered file's hash; the signed manifest must still catch it.
        val f = object : Fetcher {
            val idx = OtaJson.encodeToString(
                ChannelIndex.serializer(),
                ChannelIndex(
                    CHANNEL_SCHEMA, "llmtrainer-main", "stable", "now",
                    listOf(ChannelEntry(2, "1.0.2", "https://test.invalid/t.hagb", Hashing.sha256Hex(bytes), bytes.size.toLong(), ChannelFetcher.manifestOf(fixture("bundle-2-good")).requires)),
                ),
            ).toByteArray()
            override fun getBytes(url: String, maxBytes: Long) = idx
            override fun download(url: String, dest: File, maxBytes: Long) = dest.writeBytes(bytes)
        }
        val s = check(runtime(root, f))
        assertEquals("FAILED", s.kind)
        assertFalse(File(root, "bundles").exists() && File(root, "bundles").list()!!.isNotEmpty())
    }
}
