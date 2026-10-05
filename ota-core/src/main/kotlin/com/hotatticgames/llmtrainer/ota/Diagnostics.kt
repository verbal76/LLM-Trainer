package com.hotatticgames.llmtrainer.ota

import kotlinx.serialization.Serializable

/** Everything needed to identify exactly what is installed and running. Shown in-app and exportable. */
@Serializable
data class DiagnosticsReport(
    val hostVersionName: String,
    val hostVersionCode: Int,
    val hostApiLevel: Int,
    val nativeAbi: Int,
    val nativeRuntimeId: String,
    val capabilities: List<String>,
    val sdkInt: Int,
    val channel: String,
    val builtinBundleVersion: Int,
    val runningSource: String,
    val runningBundleVersion: Int,
    val runningBundleName: String,
    val runningSlot: String?,
    val activeSlot: String?,
    val lastKnownGoodSlot: String?,
    val pendingSlot: String?,
    val quarantined: Map<String, String>,
    val recentHistory: List<HistoryEvent>,
    /** Source commit of the APK (BuildConfig.GIT_SHA). Defaults keep older callers/readers working. */
    val buildSha: String = "",
    /** "ready" when the native engine initialised, else "unavailable: <stage>: <reason>". */
    val engineStatus: String = "",
)

object Diagnostics {
    fun build(
        host: HostInfo,
        state: StoreState,
        runningSource: String,
        runningVersion: Int,
        runningName: String,
        runningSlot: String?,
        buildSha: String = "",
        engineStatus: String = "",
    ) = DiagnosticsReport(
        host.hostVersionName, host.hostVersionCode, host.hostApiLevel, host.nativeAbi, host.nativeRuntimeId,
        host.capabilities.sorted(), host.sdkInt, host.channel, host.builtinBundleVersion,
        runningSource, runningVersion, runningName, runningSlot,
        state.active, state.lastKnownGood, state.pending, state.quarantined, state.history.takeLast(20),
        buildSha, engineStatus,
    )

    fun toJson(r: DiagnosticsReport): String = OtaJson.encodeToString(DiagnosticsReport.serializer(), r)
}
