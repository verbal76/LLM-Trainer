package com.hotatticgames.llmtrainer.ota

import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.IOException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Exact-ABI semantics across native generations (v1 = abi 1, v2 = abi 2), channel fetch semantics, identity. */
class NativeAbiAndIdentityTest {
    private fun check(fetcher: Fetcher, host: HostInfo, store: UpdateStore = UpdateStore(TestKit.tmp())) =
        UpdateCoordinator(host, TestKit.trusted, store, fetcher, TestKit.tmp(), "https://x/channel.json").check()

    private fun abiBundle(version: Int, abi: Int) = TestKit.bundle(manifest = TestKit.manifest(version = version, abi = abi))

    private fun channel(vararg b: Pair<Int, Int>): Pair<ByteArray, Map<String, ByteArray>> =
        TestKit.channelFor(b.associate { (v, abi) -> v to abiBundle(v, abi) }) { v ->
            Requires(IntRangeSpec(1, 1), b.first { it.first == v }.second, listOf("core.v1"))
        }

    @Test fun abi2BundleIsNeverStagedByAbi1Host() {
        val (idx, files) = channel(5 to 2)
        val f = TestKit.FakeFetcher(idx, files)
        val store = UpdateStore(TestKit.tmp())
        assertIs<CheckResult.NeedsNewApk>(check(f, TestKit.host(abi = 1), store))
        assertTrue(f.downloads.isEmpty())
        assertNull(store.load().pending)
    }

    @Test fun abi2BundleWithLyingIndexIsStillRejectedByAbi1Host() {
        val b = abiBundle(5, 2)
        val (idx, files) = TestKit.channelFor(mapOf(5 to b)) // index claims abi 1
        val store = UpdateStore(TestKit.tmp())
        val r = check(TestKit.FakeFetcher(idx, files), TestKit.host(abi = 1), store)
        assertTrue(assertIs<CheckResult.Failed>(r).reasons.any { it.code == RejectCode.NATIVE_ABI_MISMATCH })
        assertNull(store.load().pending)
    }

    @Test fun storeItselfRefusesWrongAbiBundle() {
        val v = TestKit.verified(2, TestKit.host(abi = 1)) // abi-1 bundle
        val s = UpdateStore(TestKit.tmp())
        val r = s.stage(v, "ab".repeat(32), builtinVersion = 1, hostNativeAbi = 2)
        assertEquals(RejectCode.NATIVE_ABI_MISMATCH, (r.exceptionOrNull() as UpdateStore.RejectedException).reject.code)
        assertNull(s.load().pending)
    }

    @Test fun abi1BundleIsIgnoredByAbi2Host() {
        // Host abi 2, builtin #3. A later abi-1 OTA (#4, published for v1 users) is obsolete here: not offered, not "needs new APK".
        val (idx, files) = channel(4 to 1)
        val f = TestKit.FakeFetcher(idx, files)
        val store = UpdateStore(TestKit.tmp())
        val r = check(f, TestKit.host(abi = 2, builtin = 3), store)
        assertIs<CheckResult.UpToDate>(r)
        assertTrue(f.downloads.isEmpty())
        assertNull(store.load().pending)
    }

    @Test fun mixedChannelEachHostGetsItsOwnGeneration() {
        val (idx, files) = channel(2 to 1, 3 to 1, 4 to 2, 5 to 2, 6 to 1)
        val v1 = check(TestKit.FakeFetcher(idx, files), TestKit.host(abi = 1, builtin = 1))
        assertEquals(6, assertIs<CheckResult.Staged>(v1).slot.bundleVersion)
        val v2 = check(TestKit.FakeFetcher(idx, files), TestKit.host(abi = 2, builtin = 3))
        val staged = assertIs<CheckResult.Staged>(v2).slot
        assertEquals(5, staged.bundleVersion)
        assertEquals(2, staged.nativeAbi)
    }

    @Test fun abi1SlotLeftByV1ApkIsDroppedWhenAbi2HostStarts() {
        val s = UpdateStore(TestKit.tmp())
        val v = TestKit.verified(7, TestKit.host(abi = 1))
        val slot = s.stage(v, "cd".repeat(32), builtinVersion = 1).getOrThrow()
        assertEquals(1, slot.nativeAbi)
        s.reconcileBuiltin(builtinVersion = 3, hostNativeAbi = 2)
        assertTrue(s.load().slots.isEmpty())
        assertTrue(s.planBoot() is BootPlan.Builtin)
    }

    @Test fun sameAbiNewerSlotSurvivesReconcile() {
        val s = UpdateStore(TestKit.tmp())
        s.stage(TestKit.verified(7, TestKit.host(abi = 1)), "cd".repeat(32), builtinVersion = 1)
        s.reconcileBuiltin(builtinVersion = 3, hostNativeAbi = 1)
        assertEquals(1, s.load().slots.size)
    }

    // ---- channel fetch semantics -------------------------------------------------------------------------------
    private fun failing(t: () -> Throwable) = object : Fetcher {
        override fun getBytes(url: String, maxBytes: Long): ByteArray = throw t()
        override fun download(url: String, dest: java.io.File, maxBytes: Long) = throw t()
    }

    @Test fun http404IsUpToDateWithChannelUnpublishedFlag() {
        val r = assertIs<CheckResult.UpToDate>(check(failing { HttpStatusException(404, "HTTP 404") }, TestKit.host()))
        assertFalse(r.channelPublished)
    }

    @Test fun publishedChannelWithNothingNewIsUpToDateAndPublished() {
        val (idx, files) = TestKit.channelFor(mapOf(1 to TestKit.bundle(1)))
        assertTrue(assertIs<CheckResult.UpToDate>(check(TestKit.FakeFetcher(idx, files), TestKit.host())).channelPublished)
    }

    @Test fun networkFailureIsAFailureNotUpToDate() {
        val r = assertIs<CheckResult.Failed>(check(failing { java.net.UnknownHostException("github.com") }, TestKit.host()))
        assertEquals(RejectCode.NETWORK_UNAVAILABLE, r.reasons.single().code)
        val t = assertIs<CheckResult.Failed>(check(failing { java.net.SocketTimeoutException("timeout") }, TestKit.host()))
        assertEquals(RejectCode.NETWORK_UNAVAILABLE, t.reasons.single().code)
    }

    @Test fun http5xxIsServerErrorFailure() {
        for (status in listOf(500, 502, 503)) {
            val r = assertIs<CheckResult.Failed>(check(failing { HttpStatusException(status, "HTTP $status") }, TestKit.host()))
            assertEquals(RejectCode.SERVER_ERROR, r.reasons.single().code)
        }
    }

    @Test fun otherHttpErrorsAreFailuresNotUpToDate() {
        for (status in listOf(401, 403, 410, 429)) {
            val r = assertIs<CheckResult.Failed>(check(failing { HttpStatusException(status, "HTTP $status") }, TestKit.host()))
            assertEquals(RejectCode.CHANNEL_HTTP_ERROR, r.reasons.single().code)
        }
    }

    @Test fun ioErrorIsNotConfusedWithHttp404() {
        val r = check(failing { IOException("HTTP 404 in text only") }, TestKit.host())
        assertIs<CheckResult.Failed>(r)
    }

    // ---- identity ---------------------------------------------------------------------------------------------
    private val host2 = TestKit.host(abi = 2, builtin = 3).copy(hostVersionName = "2", nativeRuntimeId = "hag-engine 1; llama.cpp 0c1e570", sourceSha = "a1b2c3d")

    @Test fun identityBlockShowsAllFiveIdentitiesAndNeverNoneV0WhileRunning() {
        val d = Diagnostics.build(host2, StoreState(), "builtin", 3, "2.0", null)
        assertEquals("2", d.identity.nativeVersion)
        assertEquals("2.0", d.identity.appVersion)
        assertEquals(3, d.identity.otaSequence)
        assertEquals("a1b2c3d", d.identity.sourceSha)
        val b = d.identityBlock
        for (frag in listOf("Native version: 2", "Application version: 2.0", "OTA sequence: #3 (builtin)", "ABI 2, hag-engine 1; llama.cpp 0c1e570", "Source: a1b2c3d")) {
            assertTrue(b.contains(frag), "missing '$frag' in:\n$b")
        }
        assertFalse(b.contains("v0") || b.contains("Running none"))
        val root = kotlinx.serialization.json.Json.parseToJsonElement(Diagnostics.toJson(d)).jsonObject
        assertEquals("2", root["identity"]!!.jsonObject["nativeVersion"]!!.jsonPrimitive.content)
    }

    @Test fun safeModeBlockSaysSoExplicitly() {
        val b = Diagnostics.build(host2, StoreState(), "none", 0, "-", null).identityBlock
        assertTrue(b.contains("none (safe mode)"))
    }

    @Test fun appVersionMajorConventionHelper() {
        assertEquals("2", VersionIdentity.majorOf("2.0"))
        assertEquals("1", VersionIdentity.majorOf("1.1"))
    }
}
