package com.hotatticgames.llmtrainer.host

import android.view.View
import android.view.ViewGroup
import android.widget.CheckBox
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
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.zip.ZipInputStream

/**
 * Drives the REAL host Activity with the CI fixture OTA bundle (built from the current bundle source and signed with
 * the CI key), staged exactly like [OtaQualificationTest] does, through the owner workflow. Views are found by their
 * stable tags; assertions use screen identities and flow outcomes, not specific numbers, so the test holds against
 * whatever StudioFactory returns (fake today, real implementation later).
 */
class StudioUiFlowTest {
    private val instr = InstrumentationRegistry.getInstrumentation()
    private val target = instr.targetContext
    private var scenario: ActivityScenario<HostActivity>? = null

    private fun fixture(name: String) = instr.context.assets.open("ota-fixtures/$name.hagb").readBytes()

    private class ChannelFetcher(bundles: Map<String, ByteArray>) : Fetcher {
        private val files = bundles.mapKeys { "https://test.invalid/${it.key}.hagb" }
        private val index: ByteArray

        init {
            val entries = bundles.map { (name, bytes) ->
                val m = manifestOf(bytes)
                ChannelEntry(m.bundleVersion, m.bundleVersionName, "https://test.invalid/$name.hagb", Hashing.sha256Hex(bytes), bytes.size.toLong(), m.requires)
            }
            index = OtaJson.encodeToString(
                ChannelIndex.serializer(),
                ChannelIndex(CHANNEL_SCHEMA, "llmtrainer-main", "stable", "now", entries),
            ).toByteArray()
        }

        override fun getBytes(url: String, maxBytes: Long) = index
        override fun download(url: String, dest: File, maxBytes: Long) {
            dest.writeBytes(files[url] ?: throw IOException("404 $url"))
        }

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
        // Stage v2 into the SAME root the real HostActivity uses (filesDir/ota); the Activity then boots it as a trial.
        val rt = HostRuntime(target, File(target.filesDir, "ota"), ChannelFetcher(mapOf("good" to fixture("bundle-2-good"))))
        var status: UpdateStatus? = null
        val latch = CountDownLatch(1)
        instr.runOnMainSync { rt.checkForUpdates { status = it; latch.countDown() } }
        assertTrue("staging timed out", latch.await(60, TimeUnit.SECONDS))
        assertEquals("STAGED", status!!.kind)
    }

    @After fun cleanup() {
        try { hooks("clear") } catch (_: Throwable) {}
        scenario?.close()
        wipe()
    }

    // ---- view helpers (all view access on the main thread) --------------------------------------------------
    private var decorView: View? = null

    /** Captured once after launch: ActivityScenario.onActivity must not be called from the main thread. */
    private fun decor(): View = decorView!!

    private fun walk(v: View, f: (View) -> Boolean): View? {
        if (f(v)) return v
        if (v is ViewGroup) for (i in 0 until v.childCount) walk(v.getChildAt(i), f)?.let { return it }
        return null
    }

    private fun findTag(tag: String): View? {
        var r: View? = null
        instr.runOnMainSync { r = walk(decor()) { it.tag == tag } }
        return r
    }

    private fun findTagPrefix(prefix: String): View? {
        var r: View? = null
        instr.runOnMainSync { r = walk(decor()) { (it.tag as? String)?.startsWith(prefix) == true } }
        return r
    }

    private fun <T> waitUntil(what: String, timeoutMs: Long = 90_000, probe: () -> T?): T {
        val end = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < end) {
            val r = probe()
            if (r != null) return r
            Thread.sleep(250)
        }
        throw AssertionError("Timed out waiting for $what.\nVisible text:\n" + visibleText())
    }

    private fun waitTag(tag: String) = waitUntil("view tagged '$tag'") { findTag(tag) }
    private fun waitTagPrefix(prefix: String) = waitUntil("view tagged '$prefix*'") { findTagPrefix(prefix) }
    private fun waitScreen(name: String) = waitTag("screen:$name")

    private fun visibleText(): String {
        var s = ""
        instr.runOnMainSync {
            val sb = StringBuilder()
            walk(decor()) { v -> if (v is TextView && v.visibility == View.VISIBLE) sb.append(v.text).append('\n'); false }
            s = sb.toString()
        }
        return s
    }

    private fun waitText(fragment: String) = waitUntil("text containing '$fragment'") { if (visibleText().contains(fragment, ignoreCase = true)) true else null }

    private fun click(tag: String) {
        waitUntil("clickable '$tag'") {
            var done = false
            instr.runOnMainSync { val v = walk(decor()) { it.tag == tag }; if (v != null) { v.performClick(); done = true } }
            if (done) true else null
        }
    }

    private fun clickPrefix(prefix: String) {
        waitUntil("clickable '$prefix*'") {
            var done = false
            instr.runOnMainSync {
                val v = walk(decor()) { (it.tag as? String)?.startsWith(prefix) == true }
                if (v != null) { v.performClick(); done = true }
            }
            if (done) true else null
        }
    }

    private fun setText(tag: String, s: String) {
        waitTag(tag)
        instr.runOnMainSync { (walk(decor()) { it.tag == tag } as EditText).setText(s) }
    }

    /** Reaches StudioTestHooks through the bundle's class loader (a DexClassLoader, not the test's). */
    private fun hooks(method: String, vararg args: Any?) {
        var loader: ClassLoader? = null
        instr.runOnMainSync {
            loader = walk(decor()) { it.javaClass.name == "com.hotatticgames.llmtrainer.app.StudioRoot" }?.javaClass?.classLoader
        }
        val cl = loader ?: throw AssertionError("bundle root view not found")
        val cls = Class.forName("com.hotatticgames.llmtrainer.app.StudioTestHooks", true, cl)
        val m = cls.methods.first { it.name == method && it.parameterTypes.size == args.size }
        m.invoke(null, *args)
    }

    // ---- the flow -------------------------------------------------------------------------------------------
    @Test fun ownerWorkflowThroughRealHostActivity() {
        scenario = ActivityScenario.launch(HostActivity::class.java)
        scenario!!.onActivity { decorView = it.window.decorView }

        // First run explanation (fresh install) -> dashboard.
        waitUntil("first-run or dashboard") { findTag("screen:FIRSTRUN") ?: findTag("screen:DASHBOARD") }
        if (findTag("screen:FIRSTRUN") != null) {
            waitText("does not train")
            click("btn:firstrun-ok")
        }
        waitScreen("DASHBOARD")

        // Updates & diagnostics: diagnostics are read lazily after attach.
        click("btn:updates")
        waitScreen("UPDATES")
        waitUntil("diagnostics JSON") {
            val v = findTag("diagnostics-json") as? TextView
            var t: String? = null
            instr.runOnMainSync { t = v?.text?.toString() }
            if (t != null && t!!.contains("hostApiLevel")) t else null
        }
        click("btn:back")
        waitScreen("DASHBOARD")

        // Create specialist.
        click("btn:create")
        waitScreen("CREATE")
        setText("field:name", "Test Mechanic")
        setText("field:domain", "motorcycle repair")
        setText("field:purpose", "Diagnose and repair carburetted motorcycles")
        click("btn:create-submit")
        waitScreen("HUB")
        waitText("Test Mechanic")

        // Device & recommendations -> catalog -> model -> select base model.
        click("btn:hub-models")
        waitScreen("DEVICE")
        waitTagPrefix("profile:")
        waitText("estimate")
        click("btn:open-catalog")
        waitScreen("CATALOG")
        waitTagPrefix("model:")
        clickPrefix("model:")
        waitScreen("MODEL")
        waitTag("license-card")
        waitTag("select-card")
        click("btn:select-base-model")
        waitScreen("HUB")

        // Add a source through the injectable test hook (the real picker stays the default path).
        click("btn:hub-sources")
        waitScreen("SOURCES")
        click("rights:OWNER_AUTHORED")
        val body = (1..40).joinToString("\n\n") { i ->
            "Section $i. The carburettor float level must be checked whenever the bowl is removed. " +
                "Step $i: inspect the needle valve seat for wear, clean the jets with compressed air, and verify idle mixture " +
                "after reassembly. Record the observation number $i in the service log before continuing."
        }
        hooks("supplySources", arrayOf("workshop-manual.txt", "field-notes.md"), arrayOf(body, "# Notes\n\n" + body.reversed().take(1500)))
        click("btn:add-sources")
        waitTag("ingest-report")
        waitText("ingested")

        // Dataset build and review.
        click("btn:to-dataset")
        waitScreen("DATASET")
        click("btn:build-dataset")
        waitTag("dataset-stats")
        click("btn:open-review")
        waitScreen("REVIEW")
        click("filter:flagged")
        Thread.sleep(800)
        repeat(3) {
            instr.runOnMainSync {
                walk(decor()) { v ->
                    if (v is CheckBox && (v.tag as? String)?.startsWith("item:") == true && v.isChecked) {
                        var p: View? = v.parent as? View
                        while (p != null && (p.tag as? String)?.startsWith("review:") != true) p = p.parent as? View
                        var text = ""
                        if (p != null) walk(p) { c -> if (c is TextView && c !is CheckBox) text += c.text; false }
                        if (text.contains("possible_leakage", ignoreCase = true)) v.performClick()
                    }
                    false
                }
            }
            click("btn:review-next")
            Thread.sleep(600)
        }
        click("btn:review-approve")
        waitScreen("DATASET")
        waitText("APPROVED")
        click("btn:back"); waitScreen("SOURCES")
        click("btn:back"); waitScreen("HUB")

        // Method screen: honest options, then choose a training method (desktop).
        click("btn:stage:METHOD")
        waitScreen("METHOD")
        waitText("does not train models")
        waitTagPrefix("method:")
        waitText("NOT TRAINING")
        click("btn:method:adapter-training-desktop")
        waitScreen("HUB")

        // Training job export to a temp stream (stands in for the SAF document).
        val sink = ByteArrayOutputStream()
        hooks("supplyExportSink", sink)
        click("btn:stage:TRAINING_PACKAGE")
        waitScreen("TRAINING")
        click("btn:export-job")
        waitTag("export-result")
        assertTrue("training job package bytes were written", sink.size() > 0)
    }
}
