package com.hotatticgames.llmtrainer.ota

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * The native v2 round's compatibility contract, expressed with the REAL numbers:
 * native v1 host = (api 1, abi 1, builtin 1), native v2 host = (api 2, abi 2, builtin 3).
 * An OTA that needs the new runtime must never reach an old host and stale v1-era OTAs must never run on the v2 host.
 */
class NativeV2RulesTest {
    private val v1Caps = setOf("core.v1", "device.snapshot.v1", "update.check.v1")
    private val v2Caps = v1Caps + setOf("inference.gguf.v1", "training.patch.v1")
    private val v1Host = TestKit.host(api = 1, abi = 1, caps = v1Caps, builtin = 1)
    private val v2Host = TestKit.host(api = 2, abi = 2, caps = v2Caps, builtin = 3)
    private val v2NoEngineHost = TestKit.host(api = 2, abi = 2, caps = v1Caps, builtin = 3)

    private fun m(version: Int, abi: Int, caps: List<String> = listOf("core.v1"), api: IntRangeSpec = IntRangeSpec(1, 99)) =
        TestKit.manifest(version = version, abi = abi, caps = caps, api = api)

    private fun verify(manifest: BundleManifest, host: HostInfo) =
        BundleFormat.verify(TestKit.write(TestKit.bundle(manifest = manifest)), TestKit.trusted, host)

    private fun rejects(r: BundleFormat.Result) = (r as BundleFormat.Result.Rejected).reasons.map { it.code }

    @Test fun abi1BundleIsRejectedByAnAbi2Host() {
        assertEquals(listOf(RejectCode.NATIVE_ABI_MISMATCH), rejects(verify(m(version = 2, abi = 1), v2Host)))
    }

    @Test fun abi2BundleIsRejectedByAnAbi1Host() {
        // The old native v1 APK must never accept the new-runtime application layer.
        assertEquals(listOf(RejectCode.NATIVE_ABI_MISMATCH), rejects(verify(m(version = 4, abi = 2), v1Host)))
    }

    @Test fun abi2BundleIsAcceptedByTheAbi2Host() {
        assertIs<BundleFormat.Result.Ok>(verify(m(version = 4, abi = 2), v2Host))
    }

    @Test fun abiGateIsExactNotMinimum() {
        // A future abi-3 bundle must not run on an abi-2 host either.
        assertEquals(listOf(RejectCode.NATIVE_ABI_MISMATCH), rejects(verify(m(version = 9, abi = 3), v2Host)))
    }

    @Test fun inferenceBundleNeedsTheEngineCapabilityNotJustTheAbi() {
        val needsEngine = m(version = 4, abi = 2, caps = listOf("core.v1", "inference.gguf.v1", "training.patch.v1"))
        assertIs<BundleFormat.Result.Ok>(verify(needsEngine, v2Host))
        // Same APK on a CPU where the engine could not initialise: capability not advertised -> refused, not crashed.
        val r = rejects(verify(needsEngine, v2NoEngineHost))
        assertEquals(listOf(RejectCode.MISSING_CAPABILITY, RejectCode.MISSING_CAPABILITY), r)
        assertTrue(Compatibility.needsNewHost(Compatibility.evaluate(needsEngine.requires, v2NoEngineHost)))
    }

    @Test fun hostApiLevel1BundlesStillWorkOnLevel2AndLevel2BundlesNotOnLevel1() {
        assertTrue(Compatibility.evaluate(m(4, 2, api = IntRangeSpec(1, 99)).requires, v2Host).isEmpty())
        val needs2 = m(4, 1, api = IntRangeSpec(2, 99)).requires
        assertEquals(listOf(RejectCode.HOST_API_TOO_OLD), Compatibility.evaluate(needs2, v1Host.copy(nativeAbi = 1)).map { it.code })
    }

    @Test fun builtinV2BundleSupersedesEveryV1EraOtaSlot() {
        val store = UpdateStore(TestKit.tmp())
        // v1.1 (OTA sequence #2) was staged and promoted on a native v1 APK; the user then installed the v2 APK in place.
        val v1Ota = TestKit.verified(2)
        val slot = store.stage(v1Ota, Hashing.sha256Hex(TestKit.bundle(2)), builtinVersion = 1).getOrThrow()
        store.planBoot(); store.markHealthy(slot.id)
        assertEquals(slot.id, store.load().active)

        store.reconcileBuiltin(builtinVersion = 3)

        val st = store.load()
        assertTrue(st.slots.isEmpty()); assertEquals(null, st.active); assertEquals(null, st.lastKnownGood)
        assertIs<BootPlan.Builtin>(store.planBoot())
        assertFalse(store.slotDir(slot.id).exists(), "stale slot files must be deleted")
    }

    @Test fun v1EraSequenceNumbersCanNeverBeStagedOverTheV2Builtin() {
        val store = UpdateStore(TestKit.tmp())
        for (v in 1..3) {
            val r = store.stage(TestKit.verified(v), Hashing.sha256Hex(TestKit.bundle(v)), builtinVersion = 3)
            assertEquals(RejectCode.NOT_NEWER, (r.exceptionOrNull() as UpdateStore.RejectedException).reject.code)
        }
        assertTrue(store.stage(TestKit.verified(4), Hashing.sha256Hex(TestKit.bundle(4)), builtinVersion = 3).isSuccess)
    }

    @Test fun coordinatorOnAV2HostNeverDownloadsAV1EraBundle() {
        val v1Bundle = TestKit.bundle(manifest = m(version = 5, abi = 1)) // newer sequence number, but old runtime
        val (idx, files) = TestKit.channelFor(mapOf(5 to v1Bundle)) { Requires(IntRangeSpec(1, 99), 1, listOf("core.v1")) }
        val f = TestKit.FakeFetcher(idx, files)
        val store = UpdateStore(TestKit.tmp())
        val r = UpdateCoordinator(v2Host, TestKit.trusted, store, f, TestKit.tmp(), "https://x/channel.json").check()
        assertTrue(r !is CheckResult.Staged)
        assertTrue(f.downloads.isEmpty())
        assertEquals(null, store.load().pending)
    }

    @Test fun coordinatorOnAV1HostNeverDownloadsAV2Bundle() {
        val v2Bundle = TestKit.bundle(manifest = m(version = 4, abi = 2, caps = listOf("core.v1", "inference.gguf.v1")))
        val (idx, files) = TestKit.channelFor(mapOf(4 to v2Bundle)) { Requires(IntRangeSpec(1, 99), 2, listOf("core.v1", "inference.gguf.v1")) }
        val f = TestKit.FakeFetcher(idx, files)
        val r = UpdateCoordinator(v1Host, TestKit.trusted, UpdateStore(TestKit.tmp()), f, TestKit.tmp(), "https://x/channel.json").check()
        assertIs<CheckResult.NeedsNewApk>(r)
        assertTrue(f.downloads.isEmpty(), "the old native v1 host must not even download an OTA that needs the new runtime")
    }

    @Test fun coordinatorStagesAMatchingAbi2BundleOnAV2Host() {
        val b = TestKit.bundle(manifest = m(version = 4, abi = 2, caps = listOf("core.v1", "inference.gguf.v1")))
        val (idx, files) = TestKit.channelFor(mapOf(4 to b)) { Requires(IntRangeSpec(1, 99), 2, listOf("core.v1", "inference.gguf.v1")) }
        val r = UpdateCoordinator(v2Host, TestKit.trusted, UpdateStore(TestKit.tmp()), TestKit.FakeFetcher(idx, files), TestKit.tmp(), "https://x/channel.json").check()
        assertEquals(4, assertIs<CheckResult.Staged>(r).slot.bundleVersion)
    }

    @Test fun diagnosticsCarryBuildShaAndEngineStatus() {
        val json = Diagnostics.toJson(
            Diagnostics.build(v2Host.copy(sourceSha = "abc1234"), UpdateStore(TestKit.tmp()).load(), "builtin", 3, "2.0", null, engineStatus = "ready"),
        )
        assertTrue(json.contains("\"buildSha\": \"abc1234\"") && json.contains("\"engineStatus\": \"ready\"") && json.contains("\"nativeAbi\": 2"))
    }
}
