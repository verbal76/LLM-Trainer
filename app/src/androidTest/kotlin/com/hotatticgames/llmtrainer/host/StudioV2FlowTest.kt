package com.hotatticgames.llmtrainer.host

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream

/**
 * The phone-first (native v2) screens driven through the REAL host Activity and the OTA bundle, with the scripted FakeStudio v2
 * substituted by a bundle test hook. The fake does not run a model: this test proves the screens' structure, honesty labels,
 * gating and flow (model manager -> chat -> train -> evaluate -> A/B -> specialists -> about). The real engine is covered by
 * [RealEngineChatFlowTest].
 */
class StudioV2FlowTest : BundleUiTest() {
    @Test fun modelManagerChatTrainEvaluateCompareSpecialistsAbout() {
        launchToDashboard()
        hooks("setAutoConfirm", true)
        hooks("useFakeStudio", true)                         // base model installed, one trained + selected specialist, engine available

        // ---- model manager ---------------------------------------------------------------------------
        click("btn:models")
        waitScreen("MODELS")
        waitTag("models-storage")
        waitText("Nothing downloads automatically")
        waitTagPrefix("pick:")
        waitTag("pick:best_quality")
        waitTag("pick:best_specialize")
        waitText("PROVISIONAL")                              // estimates are never presented as recommended/measured
        waitText("INSTALLED ON THIS PHONE")
        waitText("CHAT / RUN ONLY")                          // capability class of a quantized file
        waitText("LICENSE VERIFIED")
        // a download that does not fit the phone is refused with the reason, before anything is fetched
        click("banner")
        click("btn:download:qwen3-8b:q4_k_m")
        waitUntil("blocked download banner") { bannerText()?.takeIf { it.contains("blocked", ignoreCase = true) } }
        // manual GGUF import through the bundle-owned picker hook
        hooks("supplyModelFile", "my-model.gguf", "GGUF" + "x".repeat(2000))
        click("btn:models-import")
        waitUntil("import banner") { bannerText()?.takeIf { it.contains("Import started") } }
        click("btn:back"); waitScreen("DASHBOARD")

        // ---- open the project that has a trained specialist --------------------------------------------
        waitTagPrefix("project:")
        click(waitUntil("Town Design card") { cardTagWithText("project:", "Town Design") })
        waitScreen("HUB")
        waitTag("hub-phone")
        waitTextOf("hub-phone-status", "to describe chat") { it.contains("Chat with the base model: ready") }

        // ---- chat: base, streaming, speed, retrieval labelled "not training" -------------------------------
        click("btn:hub-chat")
        waitScreen("CHAT")
        waitTag("chat-model-label")
        waitTextOf("chat-model-label", "base label") { it.contains("Base:") }
        setText("field:chat-input", "How tight is the rear axle nut?")
        click("btn:chat-send")
        waitTextOf("chat-answer", "scripted base answer") { it.contains("Base model (scripted)") }
        waitTextOf("chat-stats", "measured stats") { it.contains("tok/s") && it.contains("model load") }
        click("chat:use-context")
        waitText("Retrieval is ON")
        setText("field:chat-input", "And the front one?")
        click("btn:chat-send")
        waitTextOf("chat-context-label", "retrieval label") { it.contains("not training", ignoreCase = true) }
        click("btn:chat-target-specialist")
        waitTextOf("chat-model-label", "specialist label") { it.contains("parameters changed on this phone") }
        // engine unavailable: honest, with the reason, and no input box
        hooks("fakeKnob", "engineAvailable", "false")
        click("btn:back"); waitScreen("HUB")
        click("btn:hub-chat"); waitScreen("CHAT")
        waitTag("blocked-card")
        waitText("scripted state")
        assertTrue("no input while blocked", findTag("field:chat-input") == null)
        hooks("fakeKnob", "engineAvailable", "true")
        click("btn:back"); waitScreen("HUB")

        // ---- train: gates are displayed, LoRA is offered, desktop is an optional fallback ------------------
        hooks("fakeKnob", "charging", "false")
        click("btn:hub-train"); waitScreen("TRAIN_LOCAL")
        waitTag("train-conditions")
        waitTag("option:LOCAL_LORA"); waitTag("option:LOCAL_PARTIAL"); waitTag("option:LOCAL_FULL"); waitTag("option:EXTERNAL_COMPUTE")
        waitText("NOT training")                             // retrieval / prompt options say so
        waitTag("blocker:NOT_CHARGING")
        waitText("No on-phone configuration is available")
        assertTrue("start is not offered while a gate fails", findTag("btn:train-start") == null)
        waitText("forget")                                   // the retention caveat
        hooks("fakeKnob", "charging", "true")
        click("btn:back"); waitScreen("HUB")
        click("btn:hub-train"); waitScreen("TRAIN_LOCAL")
        waitTag("btn:train-select:LOCAL_LORA")
        click("btn:train-select:LOCAL_LORA")
        waitTag("train-settings")
        click("btn:train-select:LOCAL_PARTIAL")
        waitTag("train-settings")
        click("btn:train-start")                             // auto-confirmed by the hook
        waitTextOf("train-state", "RUNNING") { it == "RUNNING" }
        hooks("fakeKnob", "tick", "1")
        waitText("Step 3 of 12")
        waitTextOf("train-checkpoint", "checkpoint after the first steps") { it.contains("Checkpoint: saved") }
        click("btn:train-pause")
        waitTextOf("train-state", "PAUSED") { it == "PAUSED" }
        waitTag("btn:train-resume")
        click("btn:train-resume")
        waitTextOf("train-state", "RUNNING again") { it == "RUNNING" }
        hooks("fakeKnob", "tick", "3")
        waitTag("train-done")
        waitText("Specialist ready")
        waitText("does not prove")                           // a falling loss is not evidence

        // ---- evaluate: held-out TEST split, claim gating shown honestly -------------------------------------
        click("btn:train-to-eval"); waitScreen("EVAL")
        waitTag("local-eval-card")
        click("btn:eval-local-start")
        waitTextOf("eval-state", "RUNNING") { it == "RUNNING" }
        hooks("fakeKnob", "tick", "1")
        waitTag("eval-performance")
        waitText("NO IMPROVEMENT CLAIM")                     // 60 questions, intervals include zero
        waitText("Intervals include zero")
        waitText("General-capability retention")
        waitText("TEST")
        setText("field:eval-items", "120")
        click("btn:eval-local-start")
        waitTextOf("eval-state", "second run RUNNING") { it == "RUNNING" }
        hooks("fakeKnob", "tick", "1")
        waitText("IMPROVEMENT CLAIM SUPPORTED")
        click("btn:back"); waitScreen("TRAIN_LOCAL")
        click("btn:back")

        // ---- A/B: the same question to both, side by side ---------------------------------------------------
        waitScreen("HUB")
        click("btn:hub-ab"); waitScreen("AB")
        waitTag("ab-columns")
        setText("field:ab-prompt", "Rear axle nut torque?")
        click("btn:ab-run")
        waitTag("ab-result")
        waitTextOf("ab:base", "base answer") { it.contains("Base model (scripted)") }
        waitTextOf("ab:specialist", "specialist answer") { it.contains("Specialist (scripted)") }
        setText("field:ab-note", "specialist knows the vocabulary")
        click("btn:ab-save-note")
        waitUntil("note saved banner") { bannerText()?.takeIf { it.contains("Note saved") } }
        click("btn:back"); waitScreen("HUB")

        // ---- specialists: verify, export the patch, deselect -----------------------------------------------
        click("btn:hub-specialists-local"); waitScreen("SPECIALISTS")
        waitTagPrefix("specialist:")
        waitText("VERIFIED")
        clickPrefix("btn:sp-verify:")
        waitUntil("verified banner") { bannerText()?.takeIf { it.contains("Verified") } }
        val sink = ByteArrayOutputStream()
        hooks("supplyExportSink", sink)
        clickPrefix("btn:sp-export:")
        waitTag("export-result")
        assertTrue("patch package bytes were written", sink.size() > 0)
        clickPrefix("btn:sp-deselect:")
        waitTagPrefix("btn:sp-select:")
        click("btn:back"); waitScreen("HUB")

        // ---- about: the five identities and an honest summary -----------------------------------------------
        click("btn:hub-about"); waitScreen("ABOUT")
        waitTag("identity-block")
        waitTextOf("identity-text", "five identity lines") {
            it.contains("Native version:") && it.contains("Application version:") && it.contains("OTA sequence:") && it.contains("Runtime:") && it.contains("Source:")
        }
        waitTextOf("identity-fields", "fields") { it.contains("Git sha:") && it.contains("Engine status:") && it.contains("ABI:") }
        waitTag("phone-summary")
        waitText("ESTIMATES")
    }
}
