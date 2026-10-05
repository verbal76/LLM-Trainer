package com.hotatticgames.llmtrainer.ota.tool

import com.hotatticgames.llmtrainer.ota.*
import java.io.File
import java.time.Instant
import kotlin.system.exitProcess

/**
 * CI-facing tool (run via `./gradlew :ota-core:run --args="..."`):
 *   keygen <outPrefix>                       -> <outPrefix>.pub (X.509 DER b64) and <outPrefix>.key (PKCS8 DER b64)
 *   pack   --dex <dir> [--assets <dir>] --out <file.hagb> --key <privfile> --key-id <id> --version <int>
 *          --version-name <s> --bundle-id <s> --channel <s> --entry <class> --host-api-min <n> --host-api-max <n>
 *          --native-abi <n> [--cap <c>]... [--min-sdk <n>] [--notes <s>]
 *   verify --bundle <file> --pub <pubfile> --key-id <id> --host-api <n> --native-abi <n> [--cap <c>]... --sdk <n>
 *          --bundle-id <s> --channel <s>
 *   index  --out <channel.json> --bundle-id <s> --channel <s> --base-url <url> [--host-version-code <n> --host-url <u>] <bundle.hagb>...
 */
fun main(args: Array<String>) {
    if (args.isEmpty()) usage()
    val rest = args.drop(1)
    when (args[0]) {
        "keygen" -> keygen(rest)
        "pack" -> pack(Opts(rest))
        "verify" -> verify(Opts(rest))
        "index" -> index(Opts(rest))
        else -> usage()
    }
}

private fun usage(): Nothing {
    System.err.println("usage: keygen | pack | verify | index (see source header)")
    exitProcess(2)
}

private class Opts(args: List<String>) {
    val kv = mutableMapOf<String, MutableList<String>>()
    val positional = mutableListOf<String>()

    init {
        var i = 0
        while (i < args.size) {
            val a = args[i]
            if (a.startsWith("--")) {
                kv.getOrPut(a.removePrefix("--")) { mutableListOf() }.add(args.getOrNull(i + 1) ?: error("missing value for $a"))
                i += 2
            } else {
                positional += a
                i++
            }
        }
    }

    fun req(k: String) = kv[k]?.firstOrNull() ?: error("missing --$k")
    fun opt(k: String) = kv[k]?.firstOrNull()
    fun all(k: String) = kv[k] ?: emptyList()
}

private fun keygen(a: List<String>) {
    val prefix = a.firstOrNull() ?: usage()
    val g = Signing.generate()
    File("$prefix.pub").writeText(g.publicDerBase64)
    File("$prefix.key").writeText(g.privateDerBase64)
    println("wrote $prefix.pub and $prefix.key (keep .key secret)")
}

private fun pack(o: Opts) {
    val files = sortedMapOf<String, ByteArray>()
    val dexDir = File(o.req("dex"))
    dexDir.listFiles { f -> f.name.matches(Regex("classes\\d*\\.dex")) }?.forEach { files[it.name] = it.readBytes() }
    o.opt("assets")?.let { dir ->
        val base = File(dir)
        base.walkTopDown().filter { it.isFile }.forEach {
            files["assets/" + it.relativeTo(base).invariantSeparatorsPath] = it.readBytes()
        }
    }
    val manifest = BundleManifest(
        schema = BUNDLE_SCHEMA, bundleId = o.req("bundle-id"), bundleVersion = o.req("version").toInt(),
        bundleVersionName = o.req("version-name"), channel = o.req("channel"),
        createdAt = o.opt("created-at") ?: Instant.now().toString(), keyId = o.req("key-id"),
        entryClass = o.req("entry"),
        requires = Requires(
            IntRangeSpec(o.req("host-api-min").toInt(), o.req("host-api-max").toInt()),
            o.req("native-abi").toInt(), o.all("cap"), o.opt("min-sdk")?.toInt() ?: 26,
        ),
        files = emptyList(), notes = o.opt("notes"),
    )
    val bytes = BundleFormat.pack(files, manifest, File(o.req("key")).readText())
    File(o.req("out")).apply { parentFile?.mkdirs() }.writeBytes(bytes)
    println("packed ${o.req("out")} sha256=${Hashing.sha256Hex(bytes)} size=${bytes.size}")
}

private fun verify(o: Opts) {
    val host = HostInfo(
        hostVersionCode = 0, hostVersionName = "verify-cli", hostApiLevel = o.req("host-api").toInt(),
        nativeAbi = o.req("native-abi").toInt(), nativeRuntimeId = "verify-cli", capabilities = o.all("cap").toSet(),
        sdkInt = o.req("sdk").toInt(), bundleId = o.req("bundle-id"), channel = o.req("channel"), builtinBundleVersion = 0,
    )
    val key = TrustedKey(o.req("key-id"), File(o.req("pub")).readText())
    when (val r = BundleFormat.verify(File(o.req("bundle")), listOf(key), host)) {
        is BundleFormat.Result.Ok -> println("OK v${r.bundle.manifest.bundleVersion} ${r.bundle.manifest.bundleVersionName}")
        is BundleFormat.Result.Rejected -> {
            System.err.println("REJECTED: ${r.reasons.joinToString()}")
            exitProcess(1)
        }
    }
}

private fun index(o: Opts) {
    val entries = o.positional.map { path ->
        val f = File(path)
        // Reads the (unverified) manifest only to describe the file; devices re-verify everything.
        val zip = java.util.zip.ZipFile(f)
        val m = zip.use {
            OtaJson.decodeFromString(
                BundleManifest.serializer(),
                it.getInputStream(it.getEntry(BundleFormat.MANIFEST)).readBytes().toString(Charsets.UTF_8),
            )
        }
        ChannelEntry(
            m.bundleVersion, m.bundleVersionName, o.req("base-url").trimEnd('/') + "/" + f.name,
            Hashing.sha256Hex(f), f.length(), m.requires, m.notes,
        )
    }.sortedBy { it.bundleVersion }
    val host = o.opt("host-version-code")?.let { HostReleaseInfo(it.toInt(), o.req("host-url")) }
    val idx = ChannelIndex(
        CHANNEL_SCHEMA, o.req("bundle-id"), o.req("channel"), Instant.now().toString(), entries, host,
    )
    File(o.req("out")).apply { parentFile?.mkdirs() }
        .writeText(OtaJson.encodeToString(ChannelIndex.serializer(), idx))
    println("index with ${entries.size} bundle(s)")
}
