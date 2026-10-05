package com.hotatticgames.llmtrainer.app

import android.content.Context
import android.view.View
import com.hotatticgames.llmtrainer.hostapi.BundleApp
import com.hotatticgames.llmtrainer.hostapi.BundleEntry
import com.hotatticgames.llmtrainer.hostapi.HostServices
import org.json.JSONObject

/**
 * The OTA-updatable "LLM Trainer product experience": the owner workflow (specialist projects, sources, dataset
 * review, base-model license evidence, method, training-job / reference packages, evaluation import, specialist
 * package, updates). Everything here uses only host API level 1 (core.v1, device.snapshot.v1, update.check.v1)
 * plus platform APIs reached through the host Activity. See docs/studio/PRODUCT.md "OTA boundary".
 */
class MainBundle : BundleEntry {
    override fun create(host: HostServices): BundleApp {
        if (BuildInfo.FAULT_MODE == "entry_throws") error("fault injection: entry_throws")
        return MainApp(host)
    }
}

private class MainApp(private val host: HostServices) : BundleApp {
    private var ctl: Ctl? = null

    /** Fast and side-effect free: no Studio, no file IO, no diagnostics (read lazily after attach). */
    override fun selfTest(): String? {
        if (BuildInfo.FAULT_MODE == "selftest_fails") return "fault injection: selftest_fails"
        if (host.hostApiLevel < 1) return "host api level missing"
        return try {
            JSONObject(host.deviceSnapshotJson()); null
        } catch (e: Exception) {
            "host services unusable: ${e.message}"
        }
    }

    override fun createContentView(context: Context): View {
        val c = Ctl(context, host)
        ctl = c
        c.start()
        return c.root
    }

    override fun onResume() { ctl?.onResume() }
    override fun onPause() {}

    override fun onBackPressed(): Boolean = ctl?.back() ?: false
}
