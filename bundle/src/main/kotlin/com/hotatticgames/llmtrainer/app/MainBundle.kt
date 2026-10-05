package com.hotatticgames.llmtrainer.app

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.hotatticgames.llmtrainer.hostapi.BundleApp
import com.hotatticgames.llmtrainer.hostapi.BundleEntry
import com.hotatticgames.llmtrainer.hostapi.Capabilities
import com.hotatticgames.llmtrainer.hostapi.HostServices
import org.json.JSONObject

/**
 * The OTA-updatable "LLM Trainer product experience". Everything here can ship without a new APK as
 * long as it only uses the host API (see docs/OTA.md). This first bundle is the product shell:
 * status, device snapshot, update controls and diagnostics. Specialist-project workflows land here next.
 */
class MainBundle : BundleEntry {
    override fun create(host: HostServices): BundleApp {
        if (BuildInfo.FAULT_MODE == "entry_throws") error("fault injection: entry_throws")
        return MainApp(host)
    }
}

private class MainApp(private val host: HostServices) : BundleApp {
    private val bg = Color.parseColor("#0E0E12")
    private val card = Color.parseColor("#1A1A22")
    private val accent = Color.parseColor("#FF8A1F")
    private val text = Color.parseColor("#F2F2F5")
    private val muted = Color.parseColor("#9A9AA8")

    private var updateStatus: TextView? = null
    private var root: View? = null

    override fun selfTest(): String? {
        if (BuildInfo.FAULT_MODE == "selftest_fails") return "fault injection: selftest_fails"
        if (host.hostApiLevel < 1) return "host api level missing"
        return try {
            JSONObject(host.deviceSnapshotJson()); JSONObject(host.diagnosticsJson()); null
        } catch (e: Exception) {
            "host services unusable: ${e.message}"
        }
    }

    override fun createContentView(context: Context): View {
        val dp = context.resources.displayMetrics.density
        fun px(v: Int) = (v * dp).toInt()

        val col = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(px(16), px(24), px(16), px(24))
        }
        fun label(s: String, size: Float, color: Int, bold: Boolean = false) = TextView(context).apply {
            this.text = s; textSize = size; setTextColor(color)
            if (bold) setTypeface(typeface, Typeface.BOLD)
        }
        fun cardView(title: String, body: View): View = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(px(16), px(14), px(16), px(14))
            background = GradientDrawable().apply { setColor(card); cornerRadius = px(14).toFloat() }
            addView(label(title, 13f, accent, true))
            addView(body, LinearLayout.LayoutParams(-1, -2).apply { topMargin = px(6) })
            layoutParams = LinearLayout.LayoutParams(-1, -2).apply { topMargin = px(14) }
        }
        fun button(s: String, onClick: () -> Unit) = Button(context).apply {
            this.text = s; isAllCaps = false; setTextColor(Color.BLACK)
            background = GradientDrawable().apply { setColor(accent); cornerRadius = px(10).toFloat() }
            setOnClickListener { onClick() }
        }

        col.addView(label("LLM Trainer", 28f, text, true))
        col.addView(label("Hot Attic Games · specialist-model factory", 13f, muted))

        // Status
        val diag = JSONObject(host.diagnosticsJson())
        val status = "Running ${diag.optString("runningSource")} bundle ${diag.optString("runningBundleName")} " +
            "(v${diag.optInt("runningBundleVersion")})\n" +
            "APK ${diag.optString("hostVersionName")} · host API ${diag.optInt("hostApiLevel")} · " +
            "native ABI ${diag.optInt("nativeAbi")} (${diag.optString("nativeRuntimeId")})"
        col.addView(cardView("STATUS", label(status, 14f, text)))

        // Device
        col.addView(cardView("THIS DEVICE", label(deviceSummary(), 14f, text)))

        // Projects placeholder (honest about what exists)
        col.addView(
            cardView(
                "SPECIALIST PROJECTS",
                label(
                    "No projects yet. Project creation, package import and on-device qualification arrive in " +
                        "upcoming bundle updates; the update system delivers them without reinstalling.",
                    14f, muted,
                ),
            ),
        )

        // Updates
        updateStatus = label("Not checked yet.", 14f, text)
        val updBody = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            addView(updateStatus)
            addView(button("Check for updates") { runCheck() }, LinearLayout.LayoutParams(-2, -2).apply { topMargin = px(10) })
        }
        col.addView(cardView("UPDATES", updBody))

        // Diagnostics
        val diagText = label(diag.toString(2), 11f, muted).apply { typeface = Typeface.MONOSPACE; setTextIsSelectable(true) }
        col.addView(cardView("DIAGNOSTICS (long-press to copy)", diagText))

        val scroll = ScrollView(context).apply {
            setBackgroundColor(bg); isFillViewport = true; addView(col)
            fitsSystemWindows = true
        }
        root = scroll
        return scroll
    }

    private fun deviceSummary(): String = try {
        val d = JSONObject(host.deviceSnapshotJson())
        fun gb(k: String) = "%.1f GB".format(d.optLong(k) / 1_073_741_824.0)
        "${d.optString("manufacturer")} ${d.optString("model")} · Android ${d.optInt("sdkInt")}\n" +
            "RAM ${gb("totalRamBytes")} total, ${gb("availRamBytes")} available" +
            (if (d.optBoolean("lowMemory")) " (LOW MEMORY)" else "") + "\n" +
            "Storage ${gb("freeStorageBytes")} free of ${gb("totalStorageBytes")}\n" +
            "Thermal status ${d.optString("thermalStatus", "n/a")}"
    } catch (e: Exception) {
        "Device info unavailable: ${e.message}"
    }

    private fun runCheck() {
        updateStatus?.text = "Checking…"
        host.checkForUpdates { s ->
            updateStatus?.text = when (s.kind) {
                "UP_TO_DATE" -> "Up to date. ${s.message}"
                "STAGED" -> "Update downloaded and verified: ${s.message}. Restart the app to apply."
                "NEEDS_NEW_APK" -> "A newer version needs a new app install (native/runtime change): ${s.message}"
                else -> "Update check failed: ${s.message}"
            }
        }
    }

    override fun onResume() {}
    override fun onPause() {}
}
