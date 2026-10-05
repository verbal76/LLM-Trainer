package com.hotatticgames.llmtrainer.ota

import kotlinx.serialization.Serializable

/**
 * The five identities of docs/VERSIONING.md, kept apart so none can be presented as another.
 * An OTA never changes [nativeVersion]; [appVersion] is `<native>.<minor>` from the bundle manifest.
 */
@Serializable
data class VersionIdentity(
    /** APK versionName, e.g. "2". */
    val nativeVersion: String,
    /** Active bundle manifest bundleVersionName, e.g. "2.0". */
    val appVersion: String,
    /** Active bundle manifest bundleVersion (engineering only). */
    val otaSequence: Int,
    val nativeAbi: Int,
    val nativeRuntimeId: String,
    /** Git short SHA the APK was built from ("dev" locally). */
    val sourceSha: String,
    /** "builtin" | "ota" | "starting" (bundle being brought up) | "none" (no bundle: host safe mode). */
    val runningSource: String,
) {
    /** Owner-facing identity block (About / Diagnostics). Always the same five labelled lines. */
    fun block(): String = buildString {
        val none = runningSource == "none"
        appendLine("Native version: $nativeVersion")
        appendLine("Application version: ${if (none) "none (safe mode)" else appVersion}")
        appendLine("OTA sequence: ${if (none) "-" else "#$otaSequence"} ($runningSource)")
        appendLine("Runtime: ABI $nativeAbi, $nativeRuntimeId")
        append("Source: $sourceSha")
    }

    companion object {
        /** Convention: the application version's major equals the native version. */
        fun majorOf(versionName: String): String = versionName.substringBefore('.')
    }
}
