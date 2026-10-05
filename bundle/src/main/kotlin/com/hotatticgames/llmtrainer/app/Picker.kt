package com.hotatticgames.llmtrainer.app

import android.app.Activity
import android.app.Fragment
import android.content.ContentResolver
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import com.hotatticgames.llmtrainer.studio.api.SourceInput
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

    @JvmStatic fun clear() { names = null; contents = null; sink = null; licName = null; licBytes = null }

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
