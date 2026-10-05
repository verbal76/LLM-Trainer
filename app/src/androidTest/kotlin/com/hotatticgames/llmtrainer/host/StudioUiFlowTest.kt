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
        val rt = HostRuntime(target, File(target.filesDir, "ota"), ChannelFetcher(mapOf("good" to fixture("bundle-4-good"))))
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

    private fun waitTextOf(tag: String, what: String, ok: (String) -> Boolean) = waitUntil("text of '$tag' $what") {
        var t: String? = null
        instr.runOnMainSync { t = (walk(decor()) { it.tag == tag } as? TextView)?.text?.toString() }
        t?.takeIf(ok)
    }

    private fun waitText(fragment: String) = waitUntil("text containing '$fragment'") { if (visibleText().contains(fragment, ignoreCase = true)) true else null }

    private fun click(tag: String) {
        // (views with GONE visibility are still clickable programmatically, e.g. the banner)
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
            waitText("NOT training")
            click("btn:firstrun-ok")
        }
        waitScreen("DASHBOARD")

        // Model manager: never downloads by itself, shows storage and the owner-facing picks (real catalog; files may be unrefreshed previews).
        click("btn:models")
        waitScreen("MODELS")
        waitTag("models-storage")
        waitText("Nothing downloads automatically")
        waitTagPrefix("pick:")
        waitTag("models-import")
        click("btn:back")
        waitScreen("DASHBOARD")

        // About: the five identities in one block, plus an honest phone summary.
        click("btn:about-phone")
        waitScreen("ABOUT")
        waitTag("identity-block")
        waitTextOf("identity-text", "identity lines") { it.contains("Native version:") && it.contains("Application version:") && it.contains("Source:") }
        waitTag("phone-summary")
        click("btn:back")
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

        // Phone-first screens are honest before any model is installed: chat is blocked with a reason, training shows its gates.
        waitTag("hub-phone")
        click("btn:hub-chat")
        waitScreen("CHAT")
        waitTag("blocked-card")
        click("btn:back"); waitScreen("HUB")
        click("btn:hub-train")
        waitScreen("TRAIN_LOCAL")
        waitTag("train-conditions")
        waitTagPrefix("option:")
        waitText("NOT training")
        click("btn:back"); waitScreen("HUB")

        // Device & recommendations -> catalog -> model -> select base model (real registry: all UNVERIFIED or DISALLOWED).
        click("btn:hub-models")
        waitScreen("DEVICE")
        waitTagPrefix("profile:")
        waitText("estimate")
        click("btn:open-catalog")
        waitScreen("CATALOG")
        waitTagPrefix("model:")
        // A DISALLOWED model must be refused cleanly with an error banner (if the registry has one).
        val disallowed = cardTagWithText("model:", "DISALLOWS")
        if (disallowed != null) {
            click(disallowed)
            waitScreen("MODEL")
            click("banner") // dismiss any stale banner so the next one is the answer to this tap
            click("btn:select-base-model")
            waitUntil("blocked banner") { bannerText()?.takeIf { it.contains("Blocked") || it.contains("not allow", true) || it.contains("disallow", true) } }
            click("btn:back")
            waitScreen("CATALOG")
        }
        click(waitUntil("an UNVERIFIED model card") { cardTagWithText("model:", "UNVERIFIED") })
        waitScreen("MODEL")
        waitTag("license-card")
        waitTag("select-card")
        click("btn:select-base-model")
        waitScreen("HUB")

        // Add sources through the injectable test hook (the real picker stays the default path).
        click("btn:hub-sources")
        waitScreen("SOURCES")
        click("rights:OWNER_AUTHORED")
        val names = arrayOf("carburettor.txt", "ignition.md", "brakes.txt", "suspension.txt")
        hooks("supplySources", names, Array(names.size) { d -> manual(d) })
        click("btn:add-sources")
        waitTag("ingest-report")
        waitText("ingested")

        // Dataset build, review, approve.
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

        // Method screen: honest options. The desktop training method is unavailable while the license is UNVERIFIED.
        click("btn:stage:METHOD")
        waitScreen("METHOD")
        waitTag("method-warning")
        waitText("NOT training")
        waitTag("method:adapter-training-desktop")
        waitText("NOT TRAINING")
        waitText("NOT AVAILABLE")
        waitText("UNVERIFIED")
        click("btn:back"); waitScreen("HUB")

        // NEGATIVE: the training job export is refused (fail-closed) and writes nothing.
        val blockedSink = ByteArrayOutputStream()
        hooks("supplyExportSink", blockedSink)
        click("btn:hub-packages")
        waitScreen("TRAINING")
        click("banner")
        click("btn:export-job")
        waitUntil("blocked export banner") { bannerText()?.takeIf { it.contains("Blocked") } }
        assertEquals("blocked export must write 0 bytes", 0, blockedSink.size())
        click("btn:back"); waitScreen("HUB")

        // License evidence flow: import the license text (hook), owner attests the uses -> VERIFIED.
        click("btn:hub-license")
        waitScreen("LICENSE")
        hooks("supplyLicenseFile", "LICENSE.txt", "Test license text for the instrumented flow.\nYou may use, modify and fine-tune this model, and train adapters.\n")
        click("btn:license-import")
        waitTag("license-text")
        waitTag("btn:attest")
        click("attest:FINE_TUNE")
        click("attest:ADAPTER")
        click("attest:read")
        click("btn:attest")
        waitText("LICENSE VERIFIED")
        click("btn:back"); waitScreen("HUB")

        // Now the desktop training method is available; choose it and export the job package.
        click("btn:stage:METHOD")
        waitScreen("METHOD")
        click("btn:method:adapter-training-desktop")
        waitScreen("HUB")
        val sink = ByteArrayOutputStream()
        hooks("supplyExportSink", sink)
        click("btn:hub-packages")
        waitScreen("TRAINING")
        click("btn:export-job")
        waitTag("export-result")
        assertTrue("training job package bytes were written", sink.size() > 0)
    }

    private fun bannerText(): String? {
        var t: String? = null
        instr.runOnMainSync {
            val v = walk(decor()) { it.tag == "banner" } as? TextView
            if (v != null && v.visibility == View.VISIBLE) t = v.text.toString()
        }
        return t
    }

    /** Tag of the first view whose tag starts with [prefix] and whose descendant text contains [needle]. */
    private fun cardTagWithText(prefix: String, needle: String): String? {
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

    /** Distinct pseudo-random prose per document so near-duplicate detection does not collapse the corpus. */
    private fun manual(doc: Int): String {
        val vocab = ("float needle valve jet bowl choke piston spring gasket bearing sprocket chain caliper rotor pad fluid hose clamp bolt " +
            "torque socket gauge shim cam lobe tappet rocker coil plug lead battery stator regulator fuse relay switch harness fork seal " +
            "damper preload sag swingarm linkage bushing axle spoke rim tyre tube bead valve stem inspect measure replace tighten loosen " +
            "clean lubricate adjust verify record remove install align bleed flush torquing seating sealing wear crack leak noise heat").split(" ")
        var seed = 1234567L + doc * 7919L
        fun next(): Int { seed = (seed * 6364136223846793005L + 1442695040888963407L); return ((seed ushr 33) % vocab.size).toInt() }
        val sb = StringBuilder("# Workshop manual part $doc\n\n")
        for (sec in 1..8) {
            sb.append("## Section $doc.$sec\n\n")
            repeat(3) {
                sb.append((1..60).joinToString(" ") { vocab[next()] }.replaceFirstChar { it.uppercase() }).append(".\n\n")
            }
        }
        return sb.toString()
    }
}
