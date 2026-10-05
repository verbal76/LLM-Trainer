package com.hotatticgames.llmtrainer.app

import android.app.Activity
import android.app.Fragment
import android.content.ContentResolver
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import com.hotatticgames.llmtrainer.studio.api.FakeStudio
import com.hotatticgames.llmtrainer.studio.api.SourceInput
import com.hotatticgames.llmtrainer.studio.api.Studio
import java.io.ByteArrayInputStream
import java.io.IOException
import java.io.OutputStream

/**
 * Headless fragment owned by the bundle. The v1 host does not forward activity results to bundles, so the picker
 * intents are started from this fragment (added to the host Activity's FragmentManager only while a pick is
 * pending) and its onActivityResult receives the answer. No manifest entries, no storage permission (SAF only).
 */
@Suppress("DEPRECATION")
internal class PickerFragment : Fragment() {
    private val pending = HashMap<Int, (Int, Intent?) -> Unit>()

    fun start(intent: Intent, code: Int, cb: (Int, Intent?) -> Unit) {
        pending[code] = cb
        startActivityForResult(intent, code)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        val cb = pending.remove(requestCode)
        if (pending.isEmpty()) {
            try { fragmentManager?.beginTransaction()?.remove(this)?.commitAllowingStateLoss() } catch (_: Throwable) {}
        }
        cb?.invoke(resultCode, data)
    }

    companion object {
        private const val TAG = "hag-llmtrainer-picker"
        private var nextCode = 0x4C00

        /** Main thread only. */
        fun launch(activity: Activity, intent: Intent, cb: (Int, Intent?) -> Unit) {
            val fm = activity.fragmentManager
            var f = fm.findFragmentByTag(TAG) as? PickerFragment
            if (f == null) {
                f = PickerFragment()
                fm.beginTransaction().add(f, TAG).commitAllowingStateLoss()
                fm.executePendingTransactions()
            }
            nextCode = if (nextCode >= 0x4CFF) 0x4C00 else nextCode + 1
            f.start(intent, nextCode, cb)
        }
    }
}

internal object Pickers {
    private val DOC_TYPES = arrayOf(
        "application/pdf", "text/plain", "text/markdown", "text/x-markdown", "text/csv", "application/json",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    )

    fun openMany(): Intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
        addCategory(Intent.CATEGORY_OPENABLE); type = "*/*"
        putExtra(Intent.EXTRA_MIME_TYPES, DOC_TYPES)
        putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
    }

    fun openOne(): Intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
        addCategory(Intent.CATEGORY_OPENABLE); type = "*/*"
    }

    fun create(name: String, mime: String): Intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
        addCategory(Intent.CATEGORY_OPENABLE); type = mime; putExtra(Intent.EXTRA_TITLE, name)
    }

    /** Uris from a pick result (single or multi). */
    fun uris(data: Intent?): List<Uri> {
        if (data == null) return emptyList()
        val out = ArrayList<Uri>()
        val clip = data.clipData
        if (clip != null) for (i in 0 until clip.itemCount) clip.getItemAt(i).uri?.let { out.add(it) }
        else data.data?.let { out.add(it) }
        return out
    }

    /** May block (content provider query): call from the worker thread. */
    fun inputFrom(cr: ContentResolver, uri: Uri): SourceInput {
        var name = uri.lastPathSegment ?: "file"
        var size = -1L
        try {
            cr.query(uri, null, null, null, null)?.use { c ->
                if (c.moveToFirst()) {
                    val ni = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    val si = c.getColumnIndex(OpenableColumns.SIZE)
                    if (ni >= 0 && !c.isNull(ni)) name = c.getString(ni)
                    if (si >= 0 && !c.isNull(si)) size = c.getLong(si)
                }
            }
        } catch (_: Throwable) {
        }
        val mime = cr.getType(uri) ?: "application/octet-stream"
        return SourceInput(name, mime, size) { cr.openInputStream(uri) ?: throw IOException("Cannot open $name") }
    }
}

/**
 * Bundle-internal hooks so instrumented tests can drive the real UI without the system picker. The real picker
 * path remains the default; hooks are inert until a test sets them. Test code reaches this object reflectively
 * through the bundle's class loader (see StudioUiFlowTest).
 */
object StudioTestHooks {
    @Volatile private var names: Array<String>? = null
    @Volatile private var contents: Array<String>? = null
    @Volatile private var sink: OutputStream? = null
    @Volatile private var licName: String? = null
    @Volatile private var licBytes: ByteArray? = null

    /** Replace the next document pick(s) with these in-memory text files. */
    @JvmStatic fun supplySources(fileNames: Array<String>, fileContents: Array<String>) {
        names = fileNames; contents = fileContents
    }

    /** Replace the next export destination (ACTION_CREATE_DOCUMENT) with this stream. */
    @JvmStatic fun supplyExportSink(out: OutputStream?) { sink = out }

    /** Replace the next license-file pick (license evidence screen, "import a license file"). */
    @JvmStatic fun supplyLicenseFile(fileName: String, text: String) { licName = fileName; licBytes = text.toByteArray(Charsets.UTF_8) }

    /** Replace the next model-file pick (model manager import / acquire import) with this in-memory file. */
    @JvmStatic fun supplyModelFile(fileName: String, content: String) { modelName = fileName; modelBytes = content.toByteArray(Charsets.UTF_8); modelPath = null }

    /** Replace the next model-file pick with a file on disk (a real tiny GGUF in the real-engine instrumented test); streamed, never read into memory. */
    @JvmStatic fun supplyModelPath(path: String) { modelPath = path; modelName = java.io.File(path).name; modelBytes = null }

    /** Run every confirmation (download, delete, cancel training) without showing the dialog; dialogs are not in the view tree. */
    @JvmStatic fun setAutoConfirm(on: Boolean) { confirmAll = on }

    /** Replace the whole Studio with the scripted [FakeStudio] (sample v2 data when [sampleV2]) and redraw the current screen. */
    @JvmStatic fun useFakeStudio(sampleV2: Boolean) {
        fake = if (sampleV2) FakeStudio.sampleV2() else FakeStudio(seedSampleData = true, preinstallBaseModels = false)
        rerender?.invoke()
    }

    /**
     * FakeStudio v2 knobs, applied on the studio worker thread: engineAvailable, charging ("true"|"false"|"null"), batteryPercent, thermal,
     * freeStorageMb, availableRamMb; "tick" advances scripted time (training steps, evaluation, downloads); "death" simulates process death.
     */
    @JvmStatic fun fakeKnob(name: String, value: String) {
        val f = fake as? FakeStudio ?: return
        runOnWorker?.invoke(Runnable {
            when (name) {
                "engineAvailable" -> f.v2.engineAvailable = value == "true"
                "charging" -> f.v2.charging = if (value == "null") null else value == "true"
                "batteryPercent" -> f.v2.batteryPercent = value.toIntOrNull()
                "thermal" -> f.v2.thermal = if (value == "null") null else value
                "freeStorageMb" -> f.v2.freeStorageMb = value.toLongOrNull() ?: f.v2.freeStorageMb
                "availableRamMb" -> f.v2.availableRamMb = value.toIntOrNull() ?: f.v2.availableRamMb
                "tick" -> repeat(value.toIntOrNull() ?: 1) { f.tick() }
                "death" -> f.simulateProcessDeath()
            }
        })
    }

    @JvmStatic fun clear() {
        names = null; contents = null; sink = null; licName = null; licBytes = null
        modelName = null; modelBytes = null; modelPath = null; confirmAll = false; fake = null
    }

    @Volatile private var fake: Studio? = null
    @Volatile private var confirmAll = false
    @Volatile private var modelName: String? = null
    @Volatile private var modelBytes: ByteArray? = null
    @Volatile private var modelPath: String? = null
    @Volatile private var rerender: (() -> Unit)? = null
    @Volatile private var runOnWorker: ((Runnable) -> Unit)? = null

    internal fun attach(redraw: () -> Unit, worker: (Runnable) -> Unit) { rerender = redraw; runOnWorker = worker }
    internal fun detach() { rerender = null; runOnWorker = null }
    internal fun studioOverride(): Studio? = fake
    internal fun autoConfirm(): Boolean = confirmAll

    /** The injected model file, if a test supplied one. */
    internal fun modelFile(): SourceInput? {
        val n = modelName ?: return null
        val path = modelPath
        if (path != null) { val f = java.io.File(path); return SourceInput(n, "application/octet-stream", f.length()) { f.inputStream() } }
        val b = modelBytes ?: return null
        return SourceInput(n, "application/octet-stream", b.size.toLong()) { ByteArrayInputStream(b) }
    }

    internal fun licenseFile(): Pair<String, ByteArray>? {
        val n = licName; val b = licBytes
        return if (n != null && b != null) Pair(n, b) else null
    }

    internal fun sources(): List<SourceInput>? {
        val n = names ?: return null
        val c = contents ?: return null
        return n.indices.map { i ->
            val bytes = c[i].toByteArray(Charsets.UTF_8)
            val mime = when {
                n[i].endsWith(".pdf") -> "application/pdf"
                n[i].endsWith(".md") -> "text/markdown"
                n[i].endsWith(".csv") -> "text/csv"
                n[i].endsWith(".json") -> "application/json"
                else -> "text/plain"
            }
            SourceInput(n[i], mime, bytes.size.toLong()) { ByteArrayInputStream(bytes) }
        }
    }

    internal fun exportSink(): OutputStream? = sink
}
