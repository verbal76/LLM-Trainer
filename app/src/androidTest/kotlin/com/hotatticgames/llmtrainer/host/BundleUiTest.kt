package com.hotatticgames.llmtrainer.host

import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import com.hotatticgames.llmtrainer.hostapi.UpdateStatus
import com.hotatticgames.llmtrainer.ota.BundleManifest
import com.hotatticgames.llmtrainer.ota.CHANNEL_SCHEMA
import com.hotatticgames.llmtrainer.ota.ChannelEntry
import com.hotatticgames.llmtrainer.ota.ChannelIndex
import com.hotatticgames.llmtrainer.ota.Fetcher
import com.hotatticgames.llmtrainer.ota.Hashing
import com.hotatticgames.llmtrainer.ota.OtaJson
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import java.io.File
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.zip.ZipInputStream

/**
 * Base for tests that drive the REAL host Activity running the CI fixture OTA bundle (built from the current bundle source and signed with
 * the CI key), staged exactly like [OtaQualificationTest] does. Views are found by their stable tags; the bundle's test hooks
 * (StudioTestHooks) are reached through the bundle's own class loader.
 */
abstract class BundleUiTest {
    protected val instr = InstrumentationRegistry.getInstrumentation()
    protected val target = instr.targetContext
    protected var scenario: ActivityScenario<HostActivity>? = null

    private fun fixture(name: String) = instr.context.assets.open("ota-fixtures/$name.hagb").readBytes()

    private class ChannelFetcher(bundles: Map<String, ByteArray>) : Fetcher {
        private val files = bundles.mapKeys { "https://test.invalid/${it.key}.hagb" }
        private val index: ByteArray

        init {
            val entries = bundles.map { (name, bytes) ->
                val m = manifestOf(bytes)
                ChannelEntry(m.bundleVersion, m.bundleVersionName, "https://test.invalid/$name.hagb", Hashing.sha256Hex(bytes), bytes.size.toLong(), m.requires)
            }
            index = OtaJson.encodeToString(ChannelIndex.serializer(), ChannelIndex(CHANNEL_SCHEMA, "llmtrainer-main", "stable", "now", entries)).toByteArray()
        }

        override fun getBytes(url: String, maxBytes: Long) = index
        override fun download(url: String, dest: File, maxBytes: Long) { dest.writeBytes(files[url] ?: throw IOException("404 $url")) }

        companion object {
            fun manifestOf(bytes: ByteArray): BundleManifest {
                ZipInputStream(bytes.inputStream()).use { z ->
                    while (true) {
                        val e = z.nextEntry ?: break
                        if (e.name == "manifest.json") return OtaJson.decodeFromString(BundleManifest.serializer(), z.readBytes().toString(Charsets.UTF_8))
                    }
                }
                error("no manifest")
            }
        }
    }

    private fun wipe() {
        for (n in listOf("ota", "studio", "studio-ui")) File(target.filesDir, n).deleteRecursively()
    }

    @Before fun stageFixtureBundle() {
        wipe()
        val rt = HostRuntime(target, File(target.filesDir, "ota"), ChannelFetcher(mapOf("good" to fixture("bundle-4-good"))))
        var status: UpdateStatus? = null
        val latch = CountDownLatch(1)
        instr.runOnMainSync { rt.checkForUpdates { status = it; latch.countDown() } }
        assertTrue("staging timed out", latch.await(60, TimeUnit.SECONDS))
        assertEquals("STAGED", status!!.kind)
    }

    @After fun cleanupBundleUi() {
        try { hooks("clear") } catch (_: Throwable) {}
        scenario?.close()
        wipe()
    }

    // ---- view helpers (all view access on the main thread) --------------------------------------------------
    private var decorView: View? = null

    protected fun decor(): View = decorView!!

    protected fun walk(v: View, f: (View) -> Boolean): View? {
        if (f(v)) return v
        if (v is ViewGroup) for (i in 0 until v.childCount) walk(v.getChildAt(i), f)?.let { return it }
        return null
    }

    protected fun findTag(tag: String): View? {
        var r: View? = null
        instr.runOnMainSync { r = walk(decor()) { it.tag == tag } }
        return r
    }

    protected fun findTagPrefix(prefix: String): View? {
        var r: View? = null
        instr.runOnMainSync { r = walk(decor()) { (it.tag as? String)?.startsWith(prefix) == true } }
        return r
    }

    /** The full tag of the first view whose tag starts with [prefix] (to learn ids such as a variant id). */
    protected fun tagOfPrefix(prefix: String): String? = findTagPrefix(prefix)?.tag as? String

    protected fun <T> waitUntil(what: String, timeoutMs: Long = 90_000, probe: () -> T?): T {
        val end = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < end) {
            val r = probe()
            if (r != null) return r
            Thread.sleep(250)
        }
        throw AssertionError("Timed out waiting for $what.\nVisible text:\n" + visibleText())
    }

    protected fun waitTag(tag: String, timeoutMs: Long = 90_000) = waitUntil("view tagged '$tag'", timeoutMs) { findTag(tag) }
    protected fun waitTagPrefix(prefix: String) = waitUntil("view tagged '$prefix*'") { findTagPrefix(prefix) }
    protected fun waitScreen(name: String) = waitTag("screen:$name")
    protected fun waitGone(tag: String) = waitUntil("view '$tag' to disappear") { if (findTag(tag) == null) true else null }

    protected fun visibleText(): String {
        var s = ""
        instr.runOnMainSync {
            val sb = StringBuilder()
            walk(decor()) { v -> if (v is TextView && v.visibility == View.VISIBLE) sb.append(v.text).append('\n'); false }
            s = sb.toString()
        }
        return s
    }

    protected fun waitText(fragment: String, timeoutMs: Long = 90_000) =
        waitUntil("text containing '$fragment'", timeoutMs) { if (visibleText().contains(fragment, ignoreCase = true)) true else null }

    protected fun textOf(tag: String): String? {
        var t: String? = null
        instr.runOnMainSync { t = (walk(decor()) { it.tag == tag } as? TextView)?.text?.toString() }
        return t
    }

    protected fun waitTextOf(tag: String, what: String, timeoutMs: Long = 90_000, ok: (String) -> Boolean) =
        waitUntil("text of '$tag' $what", timeoutMs) { textOf(tag)?.takeIf(ok) }

    protected fun click(tag: String) {
        // (views with GONE visibility are still clickable programmatically, e.g. the banner)
        waitUntil("clickable '$tag'") {
            var done = false
            instr.runOnMainSync { val v = walk(decor()) { it.tag == tag }; if (v != null) { v.performClick(); done = true } }
            if (done) true else null
        }
    }

    protected fun clickPrefix(prefix: String) {
        waitUntil("clickable '$prefix*'") {
            var done = false
            instr.runOnMainSync {
                val v = walk(decor()) { (it.tag as? String)?.startsWith(prefix) == true }
                if (v != null) { v.performClick(); done = true }
            }
            if (done) true else null
        }
    }

    protected fun setText(tag: String, s: String) {
        waitTag(tag)
        instr.runOnMainSync { (walk(decor()) { it.tag == tag } as EditText).setText(s) }
    }

    protected fun bannerText(): String? {
        var t: String? = null
        instr.runOnMainSync {
            val v = walk(decor()) { it.tag == "banner" } as? TextView
            if (v != null && v.visibility == View.VISIBLE) t = v.text.toString()
        }
        return t
    }

    /** Tag of the first view whose tag starts with [prefix] and whose descendant text contains [needle]. */
    protected fun cardTagWithText(prefix: String, needle: String): String? {
        var found: String? = null
        instr.runOnMainSync {
            walk(decor()) { v ->
                val t = v.tag as? String
                if (found == null && t != null && t.startsWith(prefix)) {
                    var text = ""
                    walk(v) { c -> if (c is TextView) text += c.text.toString() + "\n"; false }
                    if (text.contains(needle)) found = t
                }
                false
            }
        }
        return found
    }

    /** Reaches StudioTestHooks through the bundle's class loader (a DexClassLoader, not the test's). */
    protected fun hooks(method: String, vararg args: Any?) {
        var loader: ClassLoader? = null
        instr.runOnMainSync {
            loader = walk(decor()) { it.javaClass.name == "com.hotatticgames.llmtrainer.app.StudioRoot" }?.javaClass?.classLoader
        }
        val cl = loader ?: throw AssertionError("bundle root view not found")
        val cls = Class.forName("com.hotatticgames.llmtrainer.app.StudioTestHooks", true, cl)
        val m = cls.methods.first { it.name == method && it.parameterTypes.size == args.size }
        m.invoke(null, *args)
    }

    /** Launches the host Activity on the staged bundle and gets past the first-run explanation to the dashboard. */
    protected fun launchToDashboard() {
        scenario = ActivityScenario.launch(HostActivity::class.java)
        scenario!!.onActivity { decorView = it.window.decorView }
        waitUntil("first-run or dashboard") { findTag("screen:FIRSTRUN") ?: findTag("screen:DASHBOARD") }
        if (findTag("screen:FIRSTRUN") != null) {
            waitText("NOT training")
            click("btn:firstrun-ok")
        }
        waitScreen("DASHBOARD")
    }
}
