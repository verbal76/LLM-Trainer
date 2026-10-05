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
import com.hotatticgames.llmtrainer.qualify.DeviceSnapshot
import com.hotatticgames.llmtrainer.qualify.Explain
import com.hotatticgames.llmtrainer.qualify.ProfilePick
import com.hotatticgames.llmtrainer.qualify.QualificationResult
import com.hotatticgames.llmtrainer.qualify.ReferenceCatalog
import com.hotatticgames.llmtrainer.qualify.Tier
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

        // Model recommendations (qualification only; see recommendationsBody)
        col.addView(cardView("MODEL RECOMMENDATIONS", recommendationsBody(context, dp)))

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

    /**
     * Runs the on-device qualifier (the :qualify module, dexed into this bundle) on the live device snapshot.
     * Platform Views only. Honest by construction: no inference runtime ships in this APK, and every figure
     * here is a conservative estimate, so no profile can be labelled "Recommended" until it is benchmarked.
     */
    private fun recommendationsBody(context: Context, dp: Float): View {
        val box = LinearLayout(context).apply { orientation = LinearLayout.VERTICAL }
        fun line(s: String, size: Float, color: Int, bold: Boolean = false, topDp: Int = 0): TextView {
            val t = TextView(context)
            t.text = s; t.textSize = size; t.setTextColor(color)
            if (bold) t.setTypeface(t.typeface, Typeface.BOLD)
            box.addView(t, LinearLayout.LayoutParams(-1, -2).apply { topMargin = (topDp * dp).toInt() })
            return t
        }
        line(
            "No on-device inference runtime ships in this app yet (native ABI ${host.nativeAbi}, runtime " +
                "'${host.nativeRuntimeId}'). This is a qualification view of what this phone could sustain, not a running model.",
            12f, muted,
        )
        fun addPick(result: QualificationResult, pick: ProfilePick) {
            line(pick.label, 14f, text, true, 14)
            val c = pick.config
            if (c == null || pick.tier == Tier.NONE) {
                line("No safe configuration", 13f, accent, true, 2)
                line(pick.reason, 12f, muted, false, 2)
                return
            }
            val status = if (pick.recommended) "Recommended (measured on this device)" else "Unverified estimate"
            line(status, 12f, accent, true, 2)
            line(
                "${c.model.modelId} · ${c.quant.name} · ${c.contextTokens} token context · retrieval ${c.retrieval.indexRamMb.toInt()} MB",
                13f, text, false, 2,
            )
            line(pick.reason, 12f, muted, false, 2)
            val a = result.assessed.firstOrNull { it.config.configId == c.configId }
            if (a != null) line(Explain.pickLines(a).joinToString("\n"), 11f, muted, false, 2)
            val ex = Explain.exclusionLines(result.assessed, pick, result.policy)
            if (ex.isNotEmpty()) line("Larger models not chosen:\n" + ex.joinToString("\n"), 11f, muted, false, 2)
            for (w in pick.warnings) line(w, 11f, accent, false, 2)

        }
        val result: QualificationResult = try {
            DeviceSnapshot.qualify(host.deviceSnapshotJson(), ReferenceCatalog.load().candidates())
        } catch (e: Throwable) {
            line("Qualification unavailable: ${e.message}", 14f, text, false, 8)
            return box
        }
        val profile = result.profile
        if (profile != null) {
            line("RAM budget for a model on this device", 12f, accent, true, 10)
            line(Explain.budgetLines(profile, result.policy).joinToString("\n"), 12f, text)
        }
        if (result.notes.isNotEmpty()) line("Signals: " + result.notes.joinToString(", "), 12f, muted, false, 4)
        for (pick in result.picks) addPick(result, pick)
        line(
            "Sizes are generic classes (not specific models) with conservative estimates. A profile is only " +
                "'Recommended' after a sustained on-device benchmark; until then it is shown as an unverified estimate.",
            11f, muted, false, 12,
        )
        return box
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
