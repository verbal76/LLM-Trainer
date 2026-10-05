package com.hotatticgames.llmtrainer.host

import android.os.ParcelFileDescriptor
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assume
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * The tiny REAL GGUF the runtime's engine tests already use (SmolLM2-135M-Instruct Q8_0; CI downloads it, verifies its sha256 and
 * `adb push`es it to /data/local/tmp/hag/). Same convention as runtime/src/androidTest/.../Fixtures.kt, which is not visible to the app's
 * test APK: instrumentation args hagModel Q8 (path) and hagRequireModels=true (CI: a missing fixture is a FAILURE, never a silent skip).
 */
object TinyModelFixture {
    private const val TAG = "HagTest"
    private val args get() = InstrumentationRegistry.getArguments()
    val requireModels: Boolean get() = args.getString("hagRequireModels") == "true"

    /** A readable app-private copy of the Q8_0 model, or skips (local runs) / fails (CI) when absent. */
    fun q8(): File {
        val src = args.getString("hagModelQ8") ?: "/data/local/tmp/hag/model-q8_0.gguf"
        val srcFile = File(src)
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dst = File(ctx.filesDir, "fixtures/${srcFile.name}")
        if (dst.isFile && dst.length() > 0 && sizeViaShell(src) == dst.length()) return dst
        dst.parentFile!!.mkdirs()
        if (srcFile.canRead()) {
            FileInputStream(srcFile).use { i -> FileOutputStream(dst).use { o -> i.copyTo(o, 1 shl 20) } }
        } else {
            // /data/local/tmp is owned by the shell user and not always readable by the app: stream it through the shell.
            val pfd = InstrumentationRegistry.getInstrumentation().uiAutomation.executeShellCommand("cat $src")
            ParcelFileDescriptor.AutoCloseInputStream(pfd).use { i -> FileOutputStream(dst).use { o -> i.copyTo(o, 1 shl 20) } }
        }
        if (dst.length() == 0L) {
            dst.delete()
            if (requireModels) error("fixture $src is missing or empty (hagRequireModels=true)")
            Assume.assumeTrue("fixture $src not provided", false)
        }
        Log.i(TAG, "tiny model fixture ${dst.name}: ${dst.length()} bytes (from $src)")
        return dst
    }

    private fun sizeViaShell(path: String): Long {
        val f = File(path)
        if (f.canRead()) return f.length()
        return try {
            val pfd = InstrumentationRegistry.getInstrumentation().uiAutomation.executeShellCommand("stat -c %s $path")
            ParcelFileDescriptor.AutoCloseInputStream(pfd).use { it.readBytes().toString(Charsets.UTF_8).trim().toLong() }
        } catch (_: Throwable) {
            -1L
        }
    }
}
