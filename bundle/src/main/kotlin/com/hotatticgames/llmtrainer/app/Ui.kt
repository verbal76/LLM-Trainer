package com.hotatticgames.llmtrainer.app

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.text.InputType
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import com.hotatticgames.llmtrainer.studio.api.LicenseState
import com.hotatticgames.llmtrainer.studio.api.StageStatus
import com.hotatticgames.llmtrainer.studio.api.Verdict
import java.util.Locale

/** Minimal platform-View toolkit (dark theme, large tap targets). No resources, no libraries. */
internal class Ui(val ctx: Context) {
    private val dens = ctx.resources.displayMetrics.density
    fun px(v: Int) = (v * dens).toInt()

    val bg = Color.parseColor("#0E0E12")
    val cardBg = Color.parseColor("#1A1A22")
    val accent = Color.parseColor("#FF8A1F")
    val ink = Color.parseColor("#F2F2F5")
    val muted = Color.parseColor("#9A9AA8")
    val ok = Color.parseColor("#4CAF50")
    val warn = Color.parseColor("#FFB300")
    val bad = Color.parseColor("#EF5350")
    val info = Color.parseColor("#64B5F6")

    fun tv(s: CharSequence, size: Float = 14f, color: Int = ink, bold: Boolean = false, topDp: Int = 0): TextView {
        val t = TextView(ctx)
        t.text = s; t.textSize = size; t.setTextColor(color)
        if (bold) t.setTypeface(t.typeface, Typeface.BOLD)
        t.layoutParams = LinearLayout.LayoutParams(-1, -2).apply { topMargin = px(topDp) }
        return t
    }

    fun mono(s: CharSequence, size: Float = 11f, color: Int = muted): TextView {
        val t = tv(s, size, color)
        t.typeface = Typeface.MONOSPACE; t.setTextIsSelectable(true)
        return t
    }

    fun col(): LinearLayout = LinearLayout(ctx).apply { orientation = LinearLayout.VERTICAL }

    fun row(): LinearLayout = LinearLayout(ctx).apply {
        orientation = LinearLayout.HORIZONTAL
        layoutParams = LinearLayout.LayoutParams(-1, -2)
    }

    fun card(topDp: Int = 12): LinearLayout = LinearLayout(ctx).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(px(14), px(12), px(14), px(12))
        background = GradientDrawable().apply { setColor(cardBg); cornerRadius = px(12).toFloat() }
        layoutParams = LinearLayout.LayoutParams(-1, -2).apply { topMargin = px(topDp) }
    }

    fun button(s: String, tagName: String, primary: Boolean = true, onClick: () -> Unit): Button {
        val b = Button(ctx)
        b.text = s; b.isAllCaps = false; b.minimumHeight = px(52)
        setPrimary(b, primary)
        b.layoutParams = LinearLayout.LayoutParams(-1, -2).apply { topMargin = px(8) }
        b.tag = tagName
        b.setOnClickListener { onClick() }
        return b
    }

    /** (Re)style a button as primary (accent) or secondary. */
    fun setPrimary(b: Button, primary: Boolean) {
        b.setTextColor(if (primary) Color.BLACK else ink)
        b.background = GradientDrawable().apply {
            setColor(if (primary) accent else Color.parseColor("#2A2A36")); cornerRadius = px(10).toFloat()
        }
    }

    fun edit(hint: String, tagName: String, number: Boolean = false, decimal: Boolean = false, multiline: Boolean = false): EditText {
        val e = EditText(ctx)
        e.hint = hint; e.setHintTextColor(muted); e.setTextColor(ink); e.textSize = 15f
        e.minimumHeight = px(52)
        e.setPadding(px(12), px(10), px(12), px(10))
        e.background = GradientDrawable().apply { setColor(Color.parseColor("#24242E")); cornerRadius = px(8).toFloat() }
        e.inputType = when {
            number && decimal -> InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            number -> InputType.TYPE_CLASS_NUMBER
            multiline -> InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            else -> InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
        }
        e.layoutParams = LinearLayout.LayoutParams(-1, -2).apply { topMargin = px(8) }
        e.tag = tagName
        return e
    }

    fun check(s: String, tagName: String, checked: Boolean = false, onChange: (Boolean) -> Unit = {}): CheckBox {
        val c = CheckBox(ctx)
        c.text = s; c.setTextColor(ink); c.textSize = 14f; c.minimumHeight = px(48); c.isChecked = checked
        c.layoutParams = LinearLayout.LayoutParams(-1, -2).apply { topMargin = px(2) }
        c.tag = tagName
        c.setOnCheckedChangeListener { _, v -> onChange(v) }
        return c
    }

    fun badge(s: String, color: Int): TextView {
        val t = TextView(ctx)
        t.text = s; t.textSize = 12f; t.setTextColor(Color.BLACK); t.setTypeface(t.typeface, Typeface.BOLD)
        t.setPadding(px(8), px(3), px(8), px(3))
        t.background = GradientDrawable().apply { setColor(color); cornerRadius = px(6).toFloat() }
        t.layoutParams = LinearLayout.LayoutParams(-2, -2).apply { topMargin = px(4) }
        return t
    }

    fun bar(fraction: Double, tagName: String): ProgressBar {
        val p = ProgressBar(ctx, null, android.R.attr.progressBarStyleHorizontal)
        p.max = 1000
        if (fraction < 0) p.isIndeterminate = true else p.progress = (fraction * 1000).toInt()
        p.layoutParams = LinearLayout.LayoutParams(-1, px(10)).apply { topMargin = px(6) }
        p.tag = tagName; p.contentDescription = tagName
        return p
    }

    fun section(s: String): TextView = tv(s, 12f, accent, true, 14)
    fun line(s: String, color: Int = muted, size: Float = 12f): TextView = tv(s, size, color)

    fun conditionColor(status: String) = when (status) { "met" -> ok; "unmet" -> bad; else -> muted }
    fun conditionText(status: String) = when (status) { "met" -> "MET"; "unmet" -> "NOT MET"; else -> "UNKNOWN" }

    fun spacer(dp: Int): View = View(ctx).apply { layoutParams = ViewGroup.LayoutParams(-1, px(dp)) }

    // ---- colour+text semantics (never colour alone) -----------------------------------------------------
    fun licenseColor(s: LicenseState) = when (s) { LicenseState.VERIFIED -> ok; LicenseState.UNVERIFIED -> warn; LicenseState.DISALLOWED -> bad }
    fun licenseText(s: LicenseState) = when (s) {
        LicenseState.VERIFIED -> "LICENSE VERIFIED (owner-reviewed text)"
        LicenseState.UNVERIFIED -> "LICENSE UNVERIFIED (read the text and attest before use)"
        LicenseState.DISALLOWED -> "LICENSE DISALLOWS THIS USE"
    }
    fun verdictColor(v: Verdict) = when (v) { Verdict.FITS_SAFELY -> ok; Verdict.TIGHT -> warn; Verdict.DOES_NOT_FIT -> bad; Verdict.UNKNOWN -> muted }
    fun verdictText(v: Verdict) = when (v) {
        Verdict.FITS_SAFELY -> "Fits safely (estimate)"; Verdict.TIGHT -> "Tight (estimate)"
        Verdict.DOES_NOT_FIT -> "Does not fit"; Verdict.UNKNOWN -> "Unknown"
    }
    fun stageColor(s: StageStatus) = when (s) {
        StageStatus.DONE -> ok; StageStatus.IN_PROGRESS -> info; StageStatus.NEEDS_ATTENTION -> warn
        StageStatus.BLOCKED -> bad; StageStatus.NOT_STARTED -> muted
    }
    fun stageText(s: StageStatus) = when (s) {
        StageStatus.DONE -> "DONE"; StageStatus.IN_PROGRESS -> "IN PROGRESS"; StageStatus.NEEDS_ATTENTION -> "NEEDS ATTENTION"
        StageStatus.BLOCKED -> "BLOCKED"; StageStatus.NOT_STARTED -> "NOT STARTED"
    }
}

internal fun fmtBytes(b: Long): String = when {
    b < 0 -> "unknown"
    b >= 1_073_741_824L -> String.format(Locale.US, "%.2f GB", b / 1_073_741_824.0)
    b >= 1_048_576L -> String.format(Locale.US, "%.1f MB", b / 1_048_576.0)
    b >= 1024L -> String.format(Locale.US, "%.0f KB", b / 1024.0)
    else -> "$b B"
}

internal fun fmtMb(mb: Long): String = fmtBytes(mb * 1_048_576L)
internal fun fmtNum(d: Double): String = String.format(Locale.US, "%.3f", d)
internal fun shortHash(h: String?): String = if (h == null) "-" else if (h.length > 16) h.substring(0, 16) + "..." else h
internal fun slug(s: String): String = s.lowercase(Locale.US).replace(Regex("[^a-z0-9]+"), "-").trim('-').ifEmpty { "specialist" }

internal fun fmtSecs(ms: Double): String = String.format(Locale.US, "%.1f s", ms / 1000.0)

/** One honest line of generation measurements; absent numbers are left out, never invented. */
internal fun fmtGenStats(g: com.hotatticgames.llmtrainer.studio.api.GenerationStats): String {
    val parts = ArrayList<String>()
    g.modelLoadMs?.let { parts.add("model load " + fmtSecs(it.toDouble())) }
    g.timeToFirstTokenMs?.let { parts.add("first token " + fmtSecs(it)) }
    parts.add(String.format(Locale.US, "%.1f tok/s", g.tokensPerSecond))
    parts.add("${g.generatedTokens} tokens (prompt ${g.promptTokens})")
    g.peakRssMb?.let { parts.add("process RAM peak $it MB") }
    parts.add("stopped: " + g.stopReason.name.lowercase().replace('_', ' '))
    return parts.joinToString(" - ")
}
