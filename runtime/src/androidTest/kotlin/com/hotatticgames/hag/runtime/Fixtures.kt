package com.hotatticgames.hag.runtime

import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assume
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.MessageDigest

/**
 * Real-model fixtures for the engine tests. CI downloads real GGUF files, verifies their sha256 and `adb push`es them to
 * /data/local/tmp/hag/. Instrumentation args: hagModelQ8, hagModelF16 (paths), hagRequireModels=true (CI: a missing
 * fixture is a FAILURE, never a silent skip).
 */
object Fixtures {
    const val TAG = "HagTest"
    private val args get() = InstrumentationRegistry.getArguments()

    val requireModels: Boolean get() = args.getString("hagRequireModels") == "true"

    /** Returns a readable app-private copy of the fixture, or skips (local runs) / fails (CI) when absent. */
    fun model(argName: String, defaultPath: String): File {
        val src = args.getString(argName) ?: defaultPath
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
            android.os.ParcelFileDescriptor.AutoCloseInputStream(pfd).use { i -> FileOutputStream(dst).use { o -> i.copyTo(o, 1 shl 20) } }
        }
        if (dst.length() == 0L) {
            dst.delete()
            if (requireModels) error("fixture $src is missing or empty (hagRequireModels=true)")
            Assume.assumeTrue("fixture $src not provided", false)
        }
        Log.i(TAG, "fixture ${dst.name}: ${dst.length()} bytes (from $src)")
        return dst
    }

    private fun sizeViaShell(path: String): Long {
        val f = File(path)
        if (f.canRead()) return f.length()
        return try {
            val pfd = InstrumentationRegistry.getInstrumentation().uiAutomation.executeShellCommand("stat -c %s $path")
            android.os.ParcelFileDescriptor.AutoCloseInputStream(pfd).use { it.readBytes().toString(Charsets.UTF_8).trim().toLong() }
        } catch (_: Throwable) {
            -1L
        }
    }

    val q8: File get() = model("hagModelQ8", "/data/local/tmp/hag/model-q8_0.gguf")
    val f16: File get() = model("hagModelF16", "/data/local/tmp/hag/model-f16.gguf")

    fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { i -> val buf = ByteArray(1 shl 20); while (true) { val n = i.read(buf); if (n < 0) break; md.update(buf, 0, n) } }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    fun intArg(name: String, default: Int) = args.getString(name)?.toIntOrNull() ?: default
    fun floatArg(name: String, default: Float) = args.getString(name)?.toFloatOrNull() ?: default
}

/** Brings the real engine up once per process; a failure here fails every test with the engine's own reason. */
object EngineHolder {
    val status: EngineStatus by lazy { HagRuntime.load() }
    val engine: HagEngine
        get() = (status as? EngineStatus.Ready)?.engine
            ?: error("engine unavailable: " + (status as EngineStatus.Unavailable).let { "${it.stage}: ${it.reason}" })
}
