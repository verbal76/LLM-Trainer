package com.hotatticgames.llmtrainer.engine

import com.hotatticgames.llmtrainer.studio.api.ChatOptions
import com.hotatticgames.llmtrainer.studio.api.ChatTarget
import com.hotatticgames.llmtrainer.studio.api.ProjectId
import com.hotatticgames.llmtrainer.studio.api.StudioError
import com.hotatticgames.llmtrainer.studio.api.NewProject
import com.hotatticgames.llmtrainer.studio.core.StudioFactory
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/** The real StudioCore wired to the adapter over a fake engine: the same call the bundle makes (StudioFactory.createLocal). */
class StudioWiringTest {
    private val snapshot = """{"manufacturer":"Google","model":"Pixel","sdkInt":36,"totalRamBytes":16000000000,"availRamBytes":9000000000,"lowMemoryThresholdBytes":400000000,"lowMemory":false,"freeStorageBytes":60000000000,"charging":true,"batteryPct":90,"thermalStatus":0,"powerSaveMode":false}"""

    @Test fun studioReportsTheEngineAsAvailableWhenTheHostAdvertisedIt() {
        val dir = Files.createTempDirectory("wiring").toFile()
        val b = EngineWiring.create(FakeEngine(), null, setOf(EngineWiring.INFERENCE_CAP, EngineWiring.TRAINING_CAP))
        val s = StudioFactory.createLocal(dir, { snapshot }, null, b.inference, b.trainer)
        val st = s.engineStatus()
        assertTrue(st.inferenceAvailable && st.trainingAvailable, st.toString())
        assertEquals("hag-engine 1; llama.cpp 0c1e570", st.runtimeId)
    }

    @Test fun studioReportsWhyTheEngineIsUnavailableAndRefusesChatInsteadOfPretending() {
        val dir = Files.createTempDirectory("wiring").toFile()
        val b = EngineWiring.create(null, "cpuGate: CPU lacks dotprod", setOf("core.v1"))
        val s = StudioFactory.createLocal(dir, { snapshot }, null, b.inference, b.trainer)
        val st = s.engineStatus()
        assertFalse(st.inferenceAvailable); assertFalse(st.trainingAvailable)
        assertEquals("cpuGate: CPU lacks dotprod", st.inferenceReason); assertEquals("cpuGate: CPU lacks dotprod", st.trainingReason)
        val p = (s.createProject(NewProject("Town", "town design", "p")) as com.hotatticgames.llmtrainer.studio.api.StudioResult.Ok).value.id
        val ms = (s.projectModelState(p) as com.hotatticgames.llmtrainer.studio.api.StudioResult.Ok).value
        assertFalse(ms.canChatBase.ok); assertNotNull(ms.canChatBase.reason)
        assertTrue(ms.canChatBase.reason!!.contains("dotprod"), ms.canChatBase.reason)
        val e = (s.createChat(p, ChatTarget.BASE, null, ChatOptions()) as com.hotatticgames.llmtrainer.studio.api.StudioResult.Err).error
        assertTrue(e is StudioError.Blocked && e.code == "CHAT_UNAVAILABLE", e.toString())
    }

    @Test fun unknownProjectsStillGetNotFound() {
        val dir = Files.createTempDirectory("wiring").toFile()
        val b = EngineWiring.create(FakeEngine(), null, null)
        val s = StudioFactory.createLocal(dir, { snapshot }, null, b.inference, b.trainer)
        val e = (s.projectModelState(ProjectId("nope")) as com.hotatticgames.llmtrainer.studio.api.StudioResult.Err).error
        assertEquals("NOT_FOUND", e.code)
    }
}
