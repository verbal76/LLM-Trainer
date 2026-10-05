package com.hotatticgames.llmtrainer.ota

import com.hotatticgames.llmtrainer.ota.TestKit.bundle
import com.hotatticgames.llmtrainer.ota.TestKit.host
import com.hotatticgames.llmtrainer.ota.TestKit.trusted
import com.hotatticgames.llmtrainer.ota.TestKit.write
import java.util.zip.ZipFile
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BundleFormatTest {
    private fun codes(bytes: ByteArray, h: HostInfo = host(), keys: List<TrustedKey> = trusted): List<RejectCode> =
        (BundleFormat.verify(write(bytes), keys, h) as? BundleFormat.Result.Rejected)?.reasons?.map { it.code } ?: emptyList()

    private fun entriesOf(bytes: ByteArray): MutableMap<String, ByteArray> =
        ZipFile(write(bytes)).use { z -> z.entries().asSequence().associate { it.name to z.getInputStream(it).readBytes() } }.toMutableMap()

    @Test fun validBundleVerifies() {
        val r = BundleFormat.verify(write(bundle()), trusted, host())
        assertTrue(r is BundleFormat.Result.Ok, r.toString())
        assertEquals(2, (r as BundleFormat.Result.Ok).bundle.manifest.bundleVersion)
    }

    @Test fun assetsAreAllowed() {
        val b = bundle(extra = mapOf("assets/strings.json" to "{}".toByteArray()))
        assertTrue(BundleFormat.verify(write(b), trusted, host()) is BundleFormat.Result.Ok)
    }

    @Test fun wrongSigningKeyRejected() =
        assertEquals(listOf(RejectCode.BAD_SIGNATURE), codes(bundle(privateKey = TestKit.otherKeys.privateDerBase64)))

    @Test fun unknownKeyIdRejected() =
        assertEquals(listOf(RejectCode.UNKNOWN_KEY), codes(bundle(manifest = TestKit.manifest(keyId = "nope"))))

    @Test fun tamperedManifestRejected() {
        val e = entriesOf(bundle())
        e["manifest.json"] = e["manifest.json"]!!.toString(Charsets.UTF_8).replace("\"bundleVersion\": 2", "\"bundleVersion\": 99").toByteArray()
        assertEquals(listOf(RejectCode.BAD_SIGNATURE), codes(BundleFormat.zip(e)))
    }

    @Test fun tamperedDexRejected() {
        val e = entriesOf(bundle())
        e["classes.dex"] = "evil".toByteArray()
        assertTrue(RejectCode.HASH_MISMATCH in codes(BundleFormat.zip(e)))
    }

    @Test fun unlistedExtraFileRejected() {
        val e = entriesOf(bundle())
        e["assets/sneaky.bin"] = byteArrayOf(1)
        assertTrue(RejectCode.UNLISTED_FILE in codes(BundleFormat.zip(e)))
    }

    @Test fun missingListedFileRejected() {
        val e = entriesOf(bundle(extra = mapOf("assets/x.txt" to "x".toByteArray())))
        e.remove("assets/x.txt")
        assertTrue(RejectCode.MISSING_FILE in codes(BundleFormat.zip(e)))
    }

    @Test fun nativeLibrariesAreForbiddenEvenWhenSigned() {
        for (name in listOf("lib/arm64-v8a/libevil.so", "assets/libx.so", "assets/payload.jar")) {
            // Sign a manifest that lists the forbidden file: the *format* must still refuse it.
            val c = codes(bundle(extra = mapOf(name to byteArrayOf(1, 2, 3))))
            assertTrue(RejectCode.FORBIDDEN_FILE in c, "$name -> $c")
        }
    }

    @Test fun pathTraversalRejected() {
        for (name in listOf("assets/../../evil.dex", "/abs.dex", "assets//x")) {
            val c = codes(bundle(extra = mapOf(name to byteArrayOf(1))))
            assertTrue(RejectCode.UNSAFE_PATH in c, "$name -> $c")
        }
    }

    @Test fun wrongBundleIdAndChannelRejected() {
        assertEquals(listOf(RejectCode.WRONG_BUNDLE_ID), codes(bundle(manifest = TestKit.manifest(bundleId = "other"))))
        assertEquals(listOf(RejectCode.WRONG_CHANNEL), codes(bundle(manifest = TestKit.manifest(channel = "nightly"))))
    }

    @Test fun incompatibleRuntimeRejectedAfterSignatureCheck() {
        assertEquals(listOf(RejectCode.NATIVE_ABI_MISMATCH), codes(bundle(manifest = TestKit.manifest(abi = 2))))
        assertEquals(listOf(RejectCode.MISSING_CAPABILITY), codes(bundle(manifest = TestKit.manifest(caps = listOf("gguf.v1")))))
        assertEquals(listOf(RejectCode.HOST_API_TOO_NEW), codes(bundle(), host(api = 2)))
    }

    @Test fun garbageIsNotAZip() = assertEquals(listOf(RejectCode.BAD_ARCHIVE), codes("not a zip".toByteArray()))

    @Test fun missingSignatureRejected() {
        val e = entriesOf(bundle()); e.remove("manifest.sig")
        assertEquals(listOf(RejectCode.BAD_SIGNATURE), codes(BundleFormat.zip(e)))
    }

    @Test fun bundleWithoutDexRejected() {
        val b = BundleFormat.pack(mapOf("assets/a.txt" to "a".toByteArray()), TestKit.manifest(), TestKit.keys.privateDerBase64)
        assertTrue(RejectCode.NO_DEX in codes(b))
    }
}
