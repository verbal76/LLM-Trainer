package com.hotatticgames.llmtrainer.ota

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** Schema versions start honestly at 1. */
const val BUNDLE_SCHEMA = 1
const val CHANNEL_SCHEMA = 1
const val STATE_SCHEMA = 1

val OtaJson = Json {
    ignoreUnknownKeys = true
    prettyPrint = true
    encodeDefaults = true
}

@Serializable
data class IntRangeSpec(val min: Int, val max: Int)

/**
 * What a bundle needs from the installed native host. Every field is checked; any mismatch
 * rejects the bundle (see [Compatibility]). Native code is NEVER shipped in a bundle.
 */
@Serializable
data class Requires(
    /** Inclusive range of host API levels the bundle was built and tested against. */
    val hostApi: IntRangeSpec,
    /** Exact native runtime ABI (JNI surface + bundled .so set). Bumped on any native change. */
    val nativeAbi: Int,
    /** Host capabilities that must all be present, e.g. "inference.gguf.v1". */
    val capabilities: List<String> = emptyList(),
    val minSdk: Int = 26,
)

@Serializable
data class FileEntry(val path: String, val sha256: String, val size: Long)

/** manifest.json inside a .hagb bundle. Signed byte-for-byte via manifest.sig. */
@Serializable
data class BundleManifest(
    val schema: Int,
    val bundleId: String,
    /** Strictly monotonic across all releases of the bundle. */
    val bundleVersion: Int,
    val bundleVersionName: String,
    val channel: String,
    val createdAt: String,
    val keyId: String,
    /** Class implementing com.hotatticgames.llmtrainer.hostapi.BundleEntry. */
    val entryClass: String,
    val requires: Requires,
    val files: List<FileEntry>,
    val notes: String? = null,
)

/** The facts about the installed APK that compatibility is decided against. */
data class HostInfo(
    val hostVersionCode: Int,
    val hostVersionName: String,
    val hostApiLevel: Int,
    val nativeAbi: Int,
    val nativeRuntimeId: String,
    val capabilities: Set<String>,
    val sdkInt: Int,
    val bundleId: String,
    val channel: String,
    /** Bundle shipped inside the APK; also the floor for "never run older than the APK". */
    val builtinBundleVersion: Int,
)

@Serializable
data class ChannelEntry(
    val bundleVersion: Int,
    val bundleVersionName: String,
    val url: String,
    val sha256: String,
    val size: Long,
    val requires: Requires,
    val notes: String? = null,
)

/** Informational only: the app never installs an APK by itself. */
@Serializable
data class HostReleaseInfo(val latestHostVersionCode: Int, val url: String, val notes: String? = null)

@Serializable
data class ChannelIndex(
    val schema: Int,
    val bundleId: String,
    val channel: String,
    val generatedAt: String,
    val bundles: List<ChannelEntry>,
    val hostRelease: HostReleaseInfo? = null,
)

enum class RejectCode {
    BAD_ARCHIVE, BAD_MANIFEST, BAD_SCHEMA, WRONG_BUNDLE_ID, WRONG_CHANNEL, UNKNOWN_KEY, BAD_SIGNATURE,
    HASH_MISMATCH, UNLISTED_FILE, MISSING_FILE, FORBIDDEN_FILE, UNSAFE_PATH, TOO_LARGE, NO_DEX,
    HOST_API_TOO_OLD, HOST_API_TOO_NEW, NATIVE_ABI_MISMATCH, MISSING_CAPABILITY, SDK_TOO_LOW,
    NOT_NEWER, QUARANTINED, DOWNLOAD_FAILED, INDEX_INVALID, TAMPERED_ON_DISK,
}

@Serializable
data class Reject(val code: RejectCode, val detail: String = "") {
    override fun toString() = if (detail.isEmpty()) code.name else "${code.name}: $detail"
}
