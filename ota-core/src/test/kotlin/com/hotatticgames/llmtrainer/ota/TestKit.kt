package com.hotatticgames.llmtrainer.ota

import java.io.File
import java.io.IOException
import java.nio.file.Files

object TestKit {
    val keys = Signing.generate()
    val otherKeys = Signing.generate()
    val trusted = listOf(TrustedKey("k1", keys.publicDerBase64))

    fun host(
        api: Int = 1, abi: Int = 1, caps: Set<String> = setOf("core.v1"), sdk: Int = 34, builtin: Int = 1,
    ) = HostInfo(
        hostVersionCode = 1, hostVersionName = "1", hostApiLevel = api, nativeAbi = abi, nativeRuntimeId = "test-rt",
        capabilities = caps, sdkInt = sdk, bundleId = "llmtrainer-main", channel = "stable", builtinBundleVersion = builtin,
    )

    fun manifest(
        version: Int = 2, api: IntRangeSpec = IntRangeSpec(1, 1), abi: Int = 1, caps: List<String> = listOf("core.v1"),
        minSdk: Int = 26, keyId: String = "k1", bundleId: String = "llmtrainer-main", channel: String = "stable",
    ) = BundleManifest(
        BUNDLE_SCHEMA, bundleId, version, "1.0.$version", channel, "2026-01-01T00:00:00Z", keyId,
        "com.example.Entry", Requires(api, abi, caps, minSdk), emptyList(),
    )

    fun dex(tag: String = "a") = "dex-bytes-$tag".toByteArray()

    fun bundle(
        version: Int = 2, privateKey: String = keys.privateDerBase64, extra: Map<String, ByteArray> = emptyMap(),
        manifest: BundleManifest = manifest(version),
    ): ByteArray = BundleFormat.pack(mapOf("classes.dex" to dex("v$version")) + extra, manifest, privateKey)

    fun tmp(): File = Files.createTempDirectory("ota-test").toFile()

    fun write(bytes: ByteArray): File = File.createTempFile("bnd", ".hagb").apply { writeBytes(bytes); deleteOnExit() }

    fun verified(version: Int = 2, host: HostInfo = host()): BundleFormat.Verified {
        val r = BundleFormat.verify(write(bundle(version)), trusted, host)
        return (r as BundleFormat.Result.Ok).bundle
    }

    class FakeFetcher(
        val index: ByteArray?, val files: Map<String, ByteArray> = emptyMap(),
    ) : Fetcher {
        val downloads = mutableListOf<String>()
        override fun getBytes(url: String, maxBytes: Long): ByteArray = index ?: throw IOException("offline")
        override fun download(url: String, dest: File, maxBytes: Long) {
            downloads += url
            dest.writeBytes(files[url] ?: throw IOException("404 $url"))
        }
    }

    fun channelFor(bundles: Map<Int, ByteArray>, requires: (Int) -> Requires = { Requires(IntRangeSpec(1, 1), 1, listOf("core.v1")) }): Pair<ByteArray, Map<String, ByteArray>> {
        val entries = bundles.map { (v, b) ->
            ChannelEntry(v, "1.0.$v", "https://x/b$v.hagb", Hashing.sha256Hex(b), b.size.toLong(), requires(v))
        }
        val idx = ChannelIndex(CHANNEL_SCHEMA, "llmtrainer-main", "stable", "now", entries, HostReleaseInfo(9, "https://x/apk"))
        return OtaJson.encodeToString(ChannelIndex.serializer(), idx).toByteArray() to
            bundles.map { (v, b) -> "https://x/b$v.hagb" to b }.toMap()
    }
}
