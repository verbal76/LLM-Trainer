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
 * key (tools/ci/build-fixtures.sh; the built-in bundle of this APK is #3 / "2.0" on native ABI 2):
 *   bundle-4-good (#4, "2.1"), bundle-5-selftest (#5, selfTest fails), bundle-6-abi3 (#6, needs native ABI 3),
 *   bundle-7-throws (#7, entry constructor throws), bundle-8-abi1 (#8, built for native ABI 1: obsolete here),
 *   bundle-9-needs-engine (#9, requires inference.gguf.v1 + training.patch.v1).
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

    @Test fun diagnosticsAreTruthfulWhileTheBundleBuildsItsView() {
        // The bundle reads diagnostics inside createContentView (and in entry.create / selfTest via the host services).
        val rt = runtime(freshRoot())
        val seen = mutableListOf<JSONObject>()
        val r = rt.boot { app ->
            var v: View? = null
            instr.runOnMainSync {
                seen += JSONObject(rt.diagnosticsJson())
                v = app.createContentView(target)
                seen += JSONObject(rt.diagnosticsJson())
            }
            v!!
        }
        assertNotNull(r)
        assertEquals(2, seen.size)
        for (d in seen) {
            assertEquals("builtin", d.getString("runningSource"))
            assertEquals(BuildConfig.BUILTIN_BUNDLE_VERSION, d.getInt("runningBundleVersion"))
            assertFalse(d.getString("runningBundleName") == "-")
            val id = d.getJSONObject("identity")
            assertEquals(BuildConfig.VERSION_NAME, id.getString("nativeVersion"))
            assertEquals(r!!.versionName, id.getString("appVersion"))
            assertEquals(BuildConfig.BUILTIN_BUNDLE_VERSION, id.getInt("otaSequence"))
            assertEquals(BuildConfig.NATIVE_ABI, id.getInt("nativeAbi"))
            // the live runtime id is the engine's own version string once the engine is up; BuildConfig holds only the fallback
            assertTrue(id.getString("nativeRuntimeId").isNotBlank())
            assertEquals(BuildConfig.GIT_SHA, id.getString("sourceSha"))
            val block = d.getString("identityBlock")
            assertFalse("identity block must never say none/v0: $block", block.contains("safe mode") || block.contains("#0") || block.contains("v0"))
        }
        // Native version convention: application major == native version.
        assertEquals(BuildConfig.VERSION_NAME, r!!.versionName.substringBefore('.'))
    }

    @Test fun identityBlockHasAllFiveIdentities() {
        val rt = runtime(freshRoot()); boot(rt)
        val b = rt.identityBlock()
        for (label in listOf("Native version: ", "Application version: ", "OTA sequence: #", "Runtime: ABI ", "Source: ")) {
            assertTrue("missing '$label' in\n$b", b.contains(label))
        }
    }

    @Test fun missingChannelIsACleanUpToDate() {
        val notFound = object : Fetcher {
            override fun getBytes(url: String, maxBytes: Long): ByteArray = throw com.hotatticgames.llmtrainer.ota.HttpStatusException(404, "HTTP 404")
            override fun download(url: String, dest: File, maxBytes: Long) = throw com.hotatticgames.llmtrainer.ota.HttpStatusException(404, "HTTP 404")
        }
        val s = check(runtime(freshRoot(), notFound))
        assertEquals("UP_TO_DATE", s.kind)
        assertTrue(s.message, s.message.contains("No update channel is published yet"))
    }

    @Test fun networkAndServerFailuresAreDistinctFailures() {
        fun failing(t: () -> Throwable) = object : Fetcher {
            override fun getBytes(url: String, maxBytes: Long): ByteArray = throw t()
            override fun download(url: String, dest: File, maxBytes: Long) = throw t()
        }
        val net = check(runtime(freshRoot(), failing { java.net.UnknownHostException("github.com") }))
        assertEquals("FAILED", net.kind); assertTrue(net.message, net.message.contains("offline or network", ignoreCase = true))
        val srv = check(runtime(freshRoot(), failing { com.hotatticgames.llmtrainer.ota.HttpStatusException(503, "HTTP 503") }))
        assertEquals("FAILED", srv.kind); assertTrue(srv.message, srv.message.contains("server", ignoreCase = true))
    }

    @Test fun obsoleteAbi1BundleIsIgnoredByThisAbi2Host() {
        assertEquals(2, BuildConfig.NATIVE_ABI)
        val root = freshRoot()
        val f = ChannelFetcher(mapOf("abi1" to fixture("bundle-8-abi1")))
        val s = check(runtime(root, f))
        assertEquals("UP_TO_DATE", s.kind)
        assertTrue(f.downloads.isEmpty())
        assertNull(runtime(root, f).store.load().pending)
    }

    @Test fun abi1SlotLeftByAV1ApkIsDroppedOnFirstBoot() {
        // Simulate in-place upgrade v1 -> v2: v1 left a staged abi-1 slot (pre-ABI-field state.json).
        val root = freshRoot()
        val seed = com.hotatticgames.llmtrainer.ota.UpdateStore(root)
        val bytes = fixture("bundle-8-abi1")
        val v1Host = com.hotatticgames.llmtrainer.ota.HostInfo(1, "1", 1, 1, "none-v1", setOf("core.v1", "device.snapshot.v1", "update.check.v1"), 34, "llmtrainer-main", "stable", 1)
        val tmp = File(root, "seed.hagb").also { it.writeBytes(bytes) }
        val ok = (com.hotatticgames.llmtrainer.ota.BundleFormat.verify(tmp, listOf(com.hotatticgames.llmtrainer.ota.TrustedKey(BuildConfig.OTA_KEY_ID, BuildConfig.OTA_PUBLIC_KEY)), v1Host)
            as com.hotatticgames.llmtrainer.ota.BundleFormat.Result.Ok).bundle
        seed.stage(ok, Hashing.sha256Hex(bytes), 1)
        assertNotNull(seed.load().pending)
        val r = boot(runtime(root))!!
        assertEquals("builtin", r.source)
        assertTrue(runtime(root).store.load().slots.isEmpty())
    }

    @Test fun studioLogoAssetIsTheExactCanonicalFile() {
        val bytes = target.assets.open("branding/studio-logo.png").readBytes()
        assertEquals("e3d9bb5653eafb783eede827606e7ac73a4e45564a1c25b1ed13ad1429f48c4e", Hashing.sha256Hex(bytes))
    }

    @Test fun fullOtaCycleStagesRestartsAndPromotes() {
        val root = freshRoot()
        val fetcher = ChannelFetcher(mapOf("good" to fixture("bundle-4-good")))
        assertEquals("STAGED", check(runtime(root, fetcher)).kind)

        val rt2 = runtime(root, fetcher) // simulated app restart: new runtime, same on-disk state
        val r = boot(rt2)!!
        assertEquals("ota", r.source)
        assertEquals(4, r.version)
        rt2.onFirstFrame()
        val st = rt2.store.load()
        assertEquals(r.slotId, st.active); assertEquals(r.slotId, st.lastKnownGood); assertNull(st.pending)
        assertEquals("UP_TO_DATE", check(rt2).kind)
    }

    @Test fun badUpdatesRollBackToLastKnownGood() {
        val root = freshRoot()
        val good = ChannelFetcher(mapOf("good" to fixture("bundle-4-good")))
        check(runtime(root, good))
        val rtA = runtime(root, good); val a = boot(rtA)!!; rtA.onFirstFrame()

        for ((name, version) in listOf("bundle-5-selftest" to 5, "bundle-7-throws" to 7)) {
            val f = ChannelFetcher(mapOf(name to fixture(name)))
            assertEquals("STAGED", check(runtime(root, f)).kind)
            val rt = runtime(root, f)
            val r = boot(rt)!!
            assertEquals("must fall back to v4 after v$version fails; history=" + rt.store.load().history.takeLast(10).joinToString(" | ") { it.event + ":" + it.detail }, 4, r.version)
            assertEquals(a.slotId, r.slotId)
            assertTrue(rt.store.load().quarantined.keys.any { it.startsWith("v$version-") })
        }
    }

    @Test fun crashLoopingTrialIsQuarantinedAndBuiltinTakesOver() {
        val root = freshRoot()
        val f = ChannelFetcher(mapOf("good" to fixture("bundle-4-good")))
        check(runtime(root, f))
        // Two starts that never reach a first frame (process died): budget for a trial is 2.
        repeat(2) { assertEquals("ota", boot(runtime(root, f))!!.source) }
        val r = boot(runtime(root, f))!!
        assertEquals("builtin", r.source)
    }

    @Test fun incompatibleNativeBundleIsRefusedAndNeverInstalled() {
        val root = freshRoot()
        val f = ChannelFetcher(mapOf("abi3" to fixture("bundle-6-abi3")))
        val s = check(runtime(root, f))
        assertEquals("NEEDS_NEW_APK", s.kind)
        assertTrue(f.downloads.isEmpty())
        assertNull(runtime(root, f).store.load().pending)
    }

    @Test fun tamperedBundleWithMatchingIndexHashIsRejected() {
        val bytes = fixture("bundle-4-good").copyOf()
        bytes[bytes.size / 2] = (bytes[bytes.size / 2] + 1).toByte()
        val root = freshRoot()
        // The index lies and even carries the tampered file's hash; the signed manifest must still catch it.
        val f = object : Fetcher {
            val idx = OtaJson.encodeToString(
                ChannelIndex.serializer(),
                ChannelIndex(
                    CHANNEL_SCHEMA, "llmtrainer-main", "stable", "now",
                    listOf(ChannelEntry(4, "2.1", "https://test.invalid/t.hagb", Hashing.sha256Hex(bytes), bytes.size.toLong(), ChannelFetcher.manifestOf(fixture("bundle-4-good")).requires)),
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
