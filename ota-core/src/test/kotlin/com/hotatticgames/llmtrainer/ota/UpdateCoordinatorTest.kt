package com.hotatticgames.llmtrainer.ota

import com.hotatticgames.llmtrainer.ota.TestKit.bundle
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class UpdateCoordinatorTest {
    private fun run(
        fetcher: Fetcher, host: HostInfo = TestKit.host(), store: UpdateStore = UpdateStore(TestKit.tmp()),
        trusted: List<TrustedKey> = TestKit.trusted,
    ) = UpdateCoordinator(host, trusted, store, fetcher, TestKit.tmp(), "https://x/channel.json").check()

    @Test fun stagesNewestCompatibleBundle() {
        val (idx, files) = TestKit.channelFor(mapOf(2 to bundle(2), 3 to bundle(3)))
        val r = run(TestKit.FakeFetcher(idx, files))
        assertEquals(3, assertIs<CheckResult.Staged>(r).slot.bundleVersion)
    }

    @Test fun upToDateWhenNothingNewer() {
        val (idx, files) = TestKit.channelFor(mapOf(1 to bundle(1)))
        assertIs<CheckResult.UpToDate>(run(TestKit.FakeFetcher(idx, files)))
    }

    @Test fun incompatibleNativeBundleIsNeverDownloadedOrStaged() {
        val abi2 = TestKit.manifest(version = 5, abi = 2)
        val b = bundle(manifest = abi2)
        val (idx, files) = TestKit.channelFor(mapOf(5 to b)) { Requires(IntRangeSpec(1, 1), 2, listOf("core.v1")) }
        val f = TestKit.FakeFetcher(idx, files)
        val store = UpdateStore(TestKit.tmp())
        val r = run(f, store = store)
        assertIs<CheckResult.NeedsNewApk>(r)
        assertTrue(f.downloads.isEmpty(), "must not even download an incompatible bundle")
        assertTrue(store.load().pending == null)
    }

    @Test fun picksOlderCompatibleBundleWhenNewestNeedsNewApk() {
        val newest = bundle(manifest = TestKit.manifest(version = 6, abi = 2))
        val (idx, files) = TestKit.channelFor(mapOf(4 to bundle(4), 6 to newest)) { v ->
            Requires(IntRangeSpec(1, 1), if (v == 6) 2 else 1, listOf("core.v1"))
        }
        assertEquals(4, assertIs<CheckResult.Staged>(run(TestKit.FakeFetcher(idx, files))).slot.bundleVersion)
    }

    @Test fun lyingIndexCannotSmuggleIncompatibleBundle() {
        // Index claims compatibility, signed manifest says native ABI 2 -> device must still refuse.
        val b = bundle(manifest = TestKit.manifest(version = 5, abi = 2))
        val (idx, files) = TestKit.channelFor(mapOf(5 to b)) // index lies: abi 1
        val store = UpdateStore(TestKit.tmp())
        val r = run(TestKit.FakeFetcher(idx, files), store = store)
        assertIs<CheckResult.Failed>(r)
        assertTrue(r.reasons.any { it.code == RejectCode.NATIVE_ABI_MISMATCH })
        assertTrue(store.load().pending == null)
    }

    @Test fun tamperedDownloadRejectedByHash() {
        val (idx, files) = TestKit.channelFor(mapOf(2 to bundle(2)))
        val bad = files.mapValues { it.value.copyOf().also { b -> b[b.size / 2] = (b[b.size / 2] + 1).toByte() } }
        assertIs<CheckResult.Failed>(run(TestKit.FakeFetcher(idx, bad)))
    }

    @Test fun bundleSignedByUntrustedKeyRejectedEvenWithMatchingIndexHash() {
        val evil = bundle(2, privateKey = TestKit.otherKeys.privateDerBase64)
        val (idx, files) = TestKit.channelFor(mapOf(2 to evil))
        val r = assertIs<CheckResult.Failed>(run(TestKit.FakeFetcher(idx, files)))
        assertTrue(r.reasons.any { it.code == RejectCode.BAD_SIGNATURE })
    }

    @Test fun offlineIsAFailureNotACrash() { assertIs<CheckResult.Failed>(run(TestKit.FakeFetcher(null))) }

    @Test fun garbageIndexIsAFailure() { assertIs<CheckResult.Failed>(run(TestKit.FakeFetcher("nope".toByteArray()))) }

    @Test fun quarantinedBundleIsNotRetried() {
        val b = bundle(2)
        val (idx, files) = TestKit.channelFor(mapOf(2 to b))
        val store = UpdateStore(TestKit.tmp())
        val first = assertIs<CheckResult.Staged>(run(TestKit.FakeFetcher(idx, files), store = store))
        store.reportFailure(first.slot.id, "crashed")
        val f = TestKit.FakeFetcher(idx, files)
        assertIs<CheckResult.UpToDate>(run(f, store = store))
        assertTrue(f.downloads.isEmpty())
    }

    @Test fun diagnosticsDescribeInstalledState() {
        val store = UpdateStore(TestKit.tmp())
        val (idx, files) = TestKit.channelFor(mapOf(2 to bundle(2)))
        val slot = (run(TestKit.FakeFetcher(idx, files), store = store) as CheckResult.Staged).slot
        val json = Diagnostics.toJson(Diagnostics.build(TestKit.host(), store.load(), "ota", 2, "1.0.2", slot.id))
        assertTrue(json.contains(slot.id) && json.contains("nativeAbi"))
    }

    @Test fun missingChannelIs404IsACleanUpToDateNotAFailure() {
        val f404 = object : Fetcher {
            override fun getBytes(url: String, maxBytes: Long): ByteArray = throw HttpStatusException(404, "HTTP 404")
            override fun download(url: String, dest: java.io.File, maxBytes: Long) = throw HttpStatusException(404, "HTTP 404")
        }
        val store = UpdateStore(TestKit.tmp())
        assertIs<CheckResult.UpToDate>(run(f404, store = store))
        assertTrue(store.load().history.any { it.event == "CHECK_NO_CHANNEL" })
    }

    @Test fun serverErrorsAreStillFailures() {
        val f500 = object : Fetcher {
            override fun getBytes(url: String, maxBytes: Long): ByteArray = throw HttpStatusException(500, "HTTP 500")
            override fun download(url: String, dest: java.io.File, maxBytes: Long) = throw HttpStatusException(500, "HTTP 500")
        }
        assertIs<CheckResult.Failed>(run(f500))
    }
}
