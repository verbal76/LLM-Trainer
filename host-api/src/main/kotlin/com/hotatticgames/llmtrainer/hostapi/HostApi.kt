package com.hotatticgames.llmtrainer.hostapi

import android.content.Context
import android.view.View

/**
 * The stable contract between the native host (APK) and OTA-updatable bundles.
 *
 * Rules (see docs/OTA.md):
 *  - Append-only inside an API major (level / 100). Adding a method to [HostServices] or a new
 *    interface bumps [HOST_API_LEVEL] by one; bundles declare the minimum level they use.
 *  - A breaking change (removal/signature change/behaviour change) jumps to the next hundred, which
 *    makes every older bundle incompatible BY DECLARATION, so it is rejected instead of crashing.
 *  - The surface is deliberately primitive-only (String/Int/Boolean/Long/lambdas + platform types) so
 *    host and bundle never need to share library classes beyond Kotlin stdlib, which the host provides.
 *  - Bundles contain no native code; anything native is reached only through this API and a declared capability.
 */
const val HOST_API_LEVEL = 1

/** Capability names a host may advertise. Bundles list the ones they need in their manifest. */
object Capabilities {
    const val CORE_V1 = "core.v1"
    const val DEVICE_SNAPSHOT_V1 = "device.snapshot.v1"
    const val UPDATE_CHECK_V1 = "update.check.v1"
}

/** Implemented by the bundle's entry class (named in the manifest). Public no-arg constructor required. */
interface BundleEntry {
    fun create(host: HostServices): BundleApp
}

interface BundleApp {
    /** Build the whole UI. Called on the main thread. Use platform Views only (no shared resources). */
    fun createContentView(context: Context): View

    fun onResume()
    fun onPause()

    /**
     * Fast, side-effect-free sanity check run before the bundle is shown. Return null if healthy, else a message.
     * The host promotes a trial update to "good" only after selfTest passes AND the UI has drawn a frame.
     */
    fun selfTest(): String?

    /** Return true if the bundle consumed the back press. */
    fun onBackPressed(): Boolean = false
}

interface HostServices {
    val hostVersionName: String
    val hostApiLevel: Int
    val nativeAbi: Int
    val nativeRuntimeId: String

    /** JSON describing installed APK, native runtime, running bundle and update slots. */
    fun diagnosticsJson(): String

    /** JSON: RAM/storage/thermal/SoC facts used for device qualification. Cheap; safe on main thread. */
    fun deviceSnapshotJson(): String

    /** Check the update channel now (background thread); callback is invoked on the main thread. */
    fun checkForUpdates(callback: (UpdateStatus) -> Unit)

    /** Restart the app process so a staged update takes effect. */
    fun restartApp()
}

/** kind: UP_TO_DATE | STAGED | NEEDS_NEW_APK | FAILED | CHECKING */
data class UpdateStatus(val kind: String, val message: String)
