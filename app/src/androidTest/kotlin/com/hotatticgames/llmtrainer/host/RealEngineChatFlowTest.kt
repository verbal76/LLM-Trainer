package com.hotatticgames.llmtrainer.host

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The bundle's engine adapter against the REAL native engine and a REAL tiny GGUF (the runtime tests' SmolLM2-135M-Instruct Q8_0):
 * create a specialist, pick a catalog base model, import the GGUF into that variant through the model manager's import path,
 * open Chat and get a real streamed answer with measured speed. Skipped locally when the model fixture is not provided; a CI run with
 * hagRequireModels=true fails instead of skipping.
 */
class RealEngineChatFlowTest : BundleUiTest() {
    @Test fun importTinyGgufThenChatWithTheRealEngine() {
        val model = TinyModelFixture.q8()
        launchToDashboard()
        hooks("setAutoConfirm", true)

        // The engine really came up on this device (the host advertises it only then).
        click("btn:about-phone"); waitScreen("ABOUT")
        waitTextOf("identity-fields", "engine ready") { it.contains("Engine status: ready") }
        waitText("Run a language model here")
        click("btn:back"); waitScreen("DASHBOARD")

        // Create a specialist and choose a catalog base model (the first one that is not disallowed).
        click("btn:create"); waitScreen("CREATE")
        setText("field:name", "Engine Test")
        setText("field:domain", "workshop")
        setText("field:purpose", "Answer simple questions")
        click("btn:create-submit"); waitScreen("HUB")
        click("btn:hub-models"); waitScreen("DEVICE")
        click("btn:open-catalog"); waitScreen("CATALOG")
        click(waitUntil("a usable catalog model") { cardTagWithText("model:", "135M") ?: cardTagWithText("model:", "UNVERIFIED") })
        waitScreen("MODEL")

        // Import the real tiny GGUF into the first listed variant (the bundle-owned picker is replaced by the test hook).
        val acquireTag = waitUntil("a variant to import into") { tagOfPrefix("btn:acquire:") }
        val variantId = acquireTag.removePrefix("btn:acquire:")
        click(acquireTag); waitScreen("ACQUIRE")
        hooks("supplyModelPath", model.absolutePath)
        click("btn:import-model")
        waitUntil("import to finish", 180_000) { if (visibleText().contains("SUCCEEDED")) true else null }
        click("btn:back"); waitScreen("MODEL")
        click("btn:select-base:$variantId"); waitScreen("HUB")

        // Chat with the base model: a real answer streams in, with measured load time and speed.
        waitTextOf("hub-phone-status", "chat ready", 60_000) { it.contains("Chat with the base model: ready") }
        click("btn:hub-chat"); waitScreen("CHAT")
        waitTextOf("chat-model-label", "base label") { it.contains("Base:") }
        setText("field:chat-input", "Say hello in one short sentence.")
        click("btn:chat-send")
        val answer = waitTextOf("chat-answer", "a real answer", 240_000) { it.isNotBlank() }
        assertFalse("the answer is model output, not a placeholder: $answer", answer.contains("scripted", ignoreCase = true))
        val stats = waitTextOf("chat-stats", "measured stats") { it.contains("tok/s") }
        assertTrue(stats, stats.contains("model load"))
        assertTrue(stats, stats.contains("stopped:"))
        val banner = bannerText() // informational banners ("Base model selected.") are fine; anything else would be an error
        assertTrue("no error banner expected, was: $banner", banner == null || banner.startsWith("Base model selected"))
    }
}
