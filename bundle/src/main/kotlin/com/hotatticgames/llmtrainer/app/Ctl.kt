package com.hotatticgames.llmtrainer.app

import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import com.hotatticgames.llmtrainer.hostapi.HostServices
import com.hotatticgames.llmtrainer.studio.api.ExportedPackage
import com.hotatticgames.llmtrainer.studio.api.ProjectSummary
import com.hotatticgames.llmtrainer.studio.api.Screen
import com.hotatticgames.llmtrainer.studio.api.Studio
import com.hotatticgames.llmtrainer.studio.api.StudioError
import com.hotatticgames.llmtrainer.studio.api.StudioResult
import com.hotatticgames.llmtrainer.studio.core.StudioFactory
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.io.OutputStream
import java.util.concurrent.Executors

internal enum class Kind(val title: String) {
    FIRSTRUN("What this app does"), DASHBOARD("Specialists"), CREATE("Create specialist"), HUB("Project"),
    DEVICE("Device & recommendations"), CATALOG("Model catalog"), MODEL("Model"), LICENSE("License evidence"),
    ACQUIRE("Acquire model"), SOURCES("Sources"), INGEST("Ingestion report"), DATASET("Dataset"),
    REVIEW("Dataset review"), METHOD("Method"), TRAINING("Training job / reference package"),
    EVAL("Evaluation"), SPECIALIST("Specialist package"), UPDATES("Updates & diagnostics"),
}

internal data class Route(val kind: Kind, val projectId: String? = null, val modelId: String? = null, val variantId: String? = null) {
    fun toJson(): JSONObject = JSONObject().put("k", kind.name).put("p", projectId ?: JSONObject.NULL)
        .put("m", modelId ?: JSONObject.NULL).put("v", variantId ?: JSONObject.NULL)

    companion object {
        fun fromJson(o: JSONObject): Route? = try {
            Route(
                Kind.valueOf(o.getString("k")),
                if (o.isNull("p")) null else o.getString("p"),
                if (o.isNull("m")) null else o.getString("m"),
                if (o.isNull("v")) null else o.getString("v"),
            )
        } catch (_: Exception) { null }
    }
}

/**
 * UI controller: simple back-stack navigator, single-flight worker for blocking Studio calls, error banner,
 * SAF pick/create via [PickerFragment]. All Studio access happens on the worker thread, never on the UI thread.
 */
internal class Ctl(val context: Context, val host: HostServices) {
    val ui = Ui(context)
    val activity: Activity? = context as? Activity
    private val appCtx: Context = context.applicationContext ?: context
    private val main = Handler(Looper.getMainLooper())
    private val exec = Executors.newSingleThreadExecutor { r -> Thread(r, "studio-worker").apply { isDaemon = true } }
    private val uiDir = File(appCtx.filesDir, "studio-ui")

    /** Created on first use, on the worker thread only. */
    val studio: Studio by lazy { StudioFactory.create(File(appCtx.filesDir, "studio")) { host.deviceSnapshotJson() } }

    @Volatile var alive = true
    private var epoch = 0
    private var busy = 0
    private val stack = ArrayList<Route>()
    var resumeHook: (() -> Unit)? = null

    // ---- views -----------------------------------------------------------------------------------------
    val root = StudioRoot(context)
    private val busyBar = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
        isIndeterminate = true; visibility = View.GONE; tag = "busy"; contentDescription = "busy"
    }
    private val banner = TextView(context).apply {
        visibility = View.GONE; tag = "banner"; setTextColor(android.graphics.Color.BLACK); textSize = 13f
        setPadding(ui.px(14), ui.px(10), ui.px(14), ui.px(10))
        setOnClickListener { visibility = View.GONE }
    }
    private val scroll = ScrollView(context).apply { setBackgroundColor(ui.bg); isFillViewport = true }
    private var content = ui.col()

    init {
        root.orientation = LinearLayout.VERTICAL
        root.setBackgroundColor(ui.bg)
        root.fitsSystemWindows = true
        root.addView(busyBar, LinearLayout.LayoutParams(-1, ui.px(4)))
        root.addView(banner, LinearLayout.LayoutParams(-1, -2))
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
    }

    // ---- navigation ------------------------------------------------------------------------------------
    fun start() {
        restore()
        if (stack.isEmpty()) stack.add(if (!File(uiDir, "firstrun.done").exists()) Route(Kind.FIRSTRUN) else Route(Kind.DASHBOARD))
        render()
    }

    fun go(r: Route) { stack.add(r); persist(); render() }
    fun replaceTop(r: Route) { if (stack.isNotEmpty()) stack.removeAt(stack.size - 1); stack.add(r); persist(); render() }
    fun resetTo(r: Route) { stack.clear(); stack.add(r); persist(); render() }
    fun back(): Boolean {
        if (stack.size <= 1) return false
        stack.removeAt(stack.size - 1); persist(); render(); return true
    }
    fun canGoBack() = stack.size > 1
    fun current(): Route = stack.last()

    fun markFirstRunDone() { try { uiDir.mkdirs(); File(uiDir, "firstrun.done").writeText("1") } catch (_: Throwable) {} }

    private fun persist() {
        try {
            uiDir.mkdirs()
            val a = JSONArray(); for (r in stack) a.put(r.toJson())
            File(uiDir, "nav.json").writeText(a.toString())
        } catch (_: Throwable) {}
    }

    private fun restore() {
        try {
            val f = File(uiDir, "nav.json"); if (!f.exists()) return
            val a = JSONArray(f.readText())
            for (i in 0 until a.length()) Route.fromJson(a.getJSONObject(i))?.let { stack.add(it) }
        } catch (_: Throwable) { stack.clear() }
    }

    /** Rebuilds the visible screen. Screens fetch their data from the worker and fill views when it returns. */
    fun render() {
        epoch++
        resumeHook = null
        content = ui.col().apply { setPadding(ui.px(16), ui.px(16), ui.px(16), ui.px(32)) }
        scroll.removeAllViews(); scroll.addView(content, android.view.ViewGroup.LayoutParams(-1, -2))
        scroll.scrollTo(0, 0)
        val r = current()
        header(r.kind)
        try { buildScreen(this, r, content) } catch (t: Throwable) { error("Screen failed: ${t.message}") }
    }

    private fun header(kind: Kind) {
        val row = ui.row().apply { gravity = android.view.Gravity.CENTER_VERTICAL }
        if (canGoBack()) {
            val b = ui.button("< Back", "btn:back", false) { back() }
            b.layoutParams = LinearLayout.LayoutParams(-2, -2).apply { rightMargin = ui.px(12) }
            row.addView(b)
        }
        val t = ui.tv(kind.title, 22f, ui.ink, true)
        t.tag = "screen:" + kind.name; t.contentDescription = "screen:" + kind.name
        row.addView(t, LinearLayout.LayoutParams(0, -2, 1f))
        content.addView(row)
    }

    // ---- worker ----------------------------------------------------------------------------------------
    /**
     * Run blocking [work] off the UI thread; deliver to [done] on the UI thread if the app is alive and (unless
     * [sticky]) the user is still on the same screen. Exceptions become an error banner, never a crash.
     */
    fun <T> bg(work: () -> T, sticky: Boolean = false, done: (T) -> Unit) {
        val e = epoch
        setBusy(+1)
        exec.execute {
            var result: T? = null
            var failure: Throwable? = null
            try { result = work() } catch (t: Throwable) { failure = t }
            main.post {
                setBusy(-1)
                if (!alive) return@post
                if (failure != null) { error("Unexpected error: ${failure.javaClass.simpleName}: ${failure.message}"); return@post }
                if (!sticky && e != epoch) return@post
                try {
                    @Suppress("UNCHECKED_CAST")
                    done(result as T)
                } catch (t: Throwable) { error("Display error: ${t.javaClass.simpleName}: ${t.message}") }
            }
        }
    }

    /** Studio call returning a [StudioResult]: errors go to the banner, success to [ok]. */
    fun <T> call(work: () -> StudioResult<T>, sticky: Boolean = false, ok: (T) -> Unit) {
        bg(work, sticky) { r -> handle(r, ok) }
    }

    fun <T> handle(r: StudioResult<T>, ok: (T) -> Unit) {
        when (r) {
            is StudioResult.Ok -> ok(r.value)
            is StudioResult.Err -> error(describe(r.error))
        }
    }

    fun describe(e: StudioError): String = when (e) {
        is StudioError.Blocked -> "Blocked: " + e.message + (if (e.reasons.isEmpty()) "" else "\n" + e.reasons.joinToString("\n") { "- ${it.message}" })
        is StudioError.Cancelled -> "Cancelled."
        else -> e.message
    }

    private fun setBusy(d: Int) {
        busy += d
        busyBar.visibility = if (busy > 0) View.VISIBLE else View.GONE
    }

    fun isBusy() = busy > 0

    fun post(r: () -> Unit) { main.post { if (alive) r() } }
    fun postDelayed(ms: Long, ep: Int = epoch, r: () -> Unit) { main.postDelayed({ if (alive && ep == epoch) r() }, ms) }
    fun epochNow() = epoch

    // ---- banner ----------------------------------------------------------------------------------------
    fun error(msg: String) = showBanner(msg, ui.bad)
    fun notice(msg: String) = showBanner(msg, ui.warn)
    fun success(msg: String) = showBanner(msg, ui.ok)
    private fun showBanner(msg: String, color: Int) {
        banner.text = msg + "  (tap to dismiss)"
        banner.setBackgroundColor(color)
        banner.visibility = View.VISIBLE
    }
    fun clearBanner() { banner.visibility = View.GONE }

    fun confirm(title: String, message: String, yes: String, action: () -> Unit) {
        val a = activity
        if (a == null) { error("Cannot show a confirmation dialog without an Activity."); return }
        AlertDialog.Builder(a).setTitle(title).setMessage(message)
            .setPositiveButton(yes) { _, _ -> action() }.setNegativeButton("Cancel", null).show()
    }

    // ---- SAF -------------------------------------------------------------------------------------------
    /** Starts a system picker through the bundle-owned fragment; [cb] runs on the UI thread. */
    fun pick(intent: Intent, cb: (Intent?) -> Unit) {
        val a = activity
        if (a == null) { error("The system file picker needs the host Activity, which is unavailable."); return }
        try {
            PickerFragment.launch(a, intent) { code, data ->
                if (alive) { if (code == Activity.RESULT_OK) cb(data) else notice("Nothing selected.") }
            }
        } catch (t: Throwable) {
            error("Could not open the system picker: ${t.message}")
        }
    }

    /** Write an export through the Studio's OutputStream API into a user-chosen document (or the test sink). */
    fun runExport(
        suggestedName: String, mime: String,
        write: (Studio, OutputStream) -> StudioResult<ExportedPackage>,
        done: (ExportedPackage, Uri?) -> Unit,
    ) {
        val sink = StudioTestHooks.exportSink()
        if (sink != null) {
            call({ write(studio, sink) }, sticky = true) { done(it, null) }
            return
        }
        pick(Pickers.create(suggestedName, mime)) { data ->
            val uri = data?.data
            if (uri == null) { error("No destination was chosen."); return@pick }
            val cr = appCtx.contentResolver
            bg({
                val out = cr.openOutputStream(uri, "wt") ?: throw IOException("Cannot write to the chosen document")
                out.use { write(studio, it) }
            }, sticky = true) { r -> handle(r) { p -> done(p, uri) } }
        }
    }

    fun share(uri: Uri, mime: String) {
        val a = activity ?: return
        try {
            val i = Intent(Intent.ACTION_SEND).setType(mime).putExtra(Intent.EXTRA_STREAM, uri)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            a.startActivity(Intent.createChooser(i, "Share package"))
        } catch (t: Throwable) { error("Could not share: ${t.message}") }
    }

    fun contentResolver() = appCtx.contentResolver

    // ---- host lifecycle --------------------------------------------------------------------------------
    fun onResume() { resumeHook?.invoke() }
    fun destroy() { alive = false; exec.shutdown() }

    fun routeFor(s: Screen?, p: ProjectSummary): Route = when (s) {
        null, Screen.DASHBOARD -> Route(Kind.HUB, p.id.value)
        Screen.CREATE_PROJECT -> Route(Kind.CREATE)
        Screen.DEVICE_PROFILE, Screen.RECOMMENDATIONS -> Route(Kind.DEVICE, p.id.value)
        Screen.MODEL_DETAIL -> if (p.baseModelId != null) Route(Kind.MODEL, p.id.value, p.baseModelId) else Route(Kind.DEVICE, p.id.value)
        Screen.ACQUIRE_MODEL -> Route(Kind.ACQUIRE, p.id.value, p.baseModelId)
        Screen.ADD_SOURCES -> Route(Kind.SOURCES, p.id.value)
        Screen.INGESTION_REPORT -> Route(Kind.INGEST, p.id.value)
        Screen.DATASET_BUILD -> Route(Kind.DATASET, p.id.value)
        Screen.DATASET_REVIEW -> Route(Kind.REVIEW, p.id.value)
        Screen.METHOD -> Route(Kind.METHOD, p.id.value)
        Screen.TRAINING_PACKAGE -> Route(Kind.TRAINING, p.id.value)
        Screen.EVALUATION -> Route(Kind.EVAL, p.id.value)
        Screen.EXPORT, Screen.IMPORT_PACKAGE -> Route(Kind.SPECIALIST, p.id.value)
    }
}

/** Named root so tests can reach the bundle's class loader (platform Views would resolve to the framework loader). */
class StudioRoot(context: Context) : LinearLayout(context)
