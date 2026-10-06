package com.hotatticgames.llmtrainer.studio.api

import java.io.ByteArrayOutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.fail

class FakeStudioV2Test {
    private fun <T> StudioResult<T>.ok(): T = (this as? StudioResult.Ok)?.value ?: fail("expected Ok but got ${(this as StudioResult.Err).error}")
    private fun <T> StudioResult<T>.err(): StudioError = (this as? StudioResult.Err)?.error ?: fail("expected Err")
    private fun town(s: FakeStudio) = s.listProjects().first { it.name == "Town Design" }.id

    @Test fun sampleFixtureHasSelectedSpecialistAndBaseInstalled() {
        val s = FakeStudio.sampleV2()
        val p = town(s)
        val st = s.projectModelState(p).ok()
        assertTrue(st.baseInstalled); assertTrue(st.canChatBase.ok); assertTrue(st.canChatSpecialist.ok)
        assertEquals(1, s.specialists(p).size)
        assertTrue(s.specialists(p).single().selected)
        assertTrue(s.installedModels().any { it.variantId == "qwen3-4b:q4_k_m" && p in it.activeForProjects })
    }

    @Test fun chatStreamsAndLabelsRetrievalContext() {
        val s = FakeStudio.sampleV2()
        val p = town(s)
        val chat = s.createChat(p, ChatTarget.BASE, null, ChatOptions(useSourceContext = true)).ok()
        val pieces = ArrayList<String>()
        val reply = s.sendMessage(chat.id, "What is the minimum lane width?", onToken = { pieces.add(it) }).ok()
        assertEquals(reply.text, pieces.joinToString(""))
        assertNotNull(reply.contextLabel)
        assertTrue(reply.contextLabel!!.contains("not training"))
        assertEquals(2, s.chatHistory(chat.id).ok().size)
        val sp = s.createChat(p, ChatTarget.SPECIALIST).ok()
        assertTrue(sp.modelLabel.contains("parameters changed"))
        val tok = CancelToken()
        val partial = s.sendMessage(sp.id, "hello there", tok) { tok.cancel() }.ok()
        assertTrue(partial.interrupted)
    }

    @Test fun chatRefusedWithoutEngineOrModel() {
        val s = FakeStudio.sampleV2()
        val p = town(s)
        s.v2.engineAvailable = false
        assertEquals("CHAT_UNAVAILABLE", s.createChat(p, ChatTarget.BASE).err().code)
        val plain = FakeStudio()                         // base model not installed
        assertEquals("CHAT_UNAVAILABLE", plain.createChat(town(plain), ChatTarget.BASE).err().code)
    }

    @Test fun trainingPlanIsHonestAboutWhatIsTraining() {
        val s = FakeStudio.sampleV2()
        val moto = s.listProjects().first { it.name == "Motorcycle Mechanic" }.id
        val plan = s.localTrainingPlan(town(s)).ok()
        val byKind = plan.options.associateBy { it.kind }
        assertFalse(byKind.getValue(TrainingMethodKind.RAG_ONLY).isTraining)
        assertFalse(byKind.getValue(TrainingMethodKind.PROMPT_ONLY).changesParameters)
        assertTrue(byKind.getValue(TrainingMethodKind.LOCAL_PARTIAL).changesParameters)
        assertEquals(RunLocation.DESKTOP, byKind.getValue(TrainingMethodKind.EXTERNAL_COMPUTE).whereItRuns)
        assertFalse(byKind.getValue(TrainingMethodKind.LOCAL_FULL).available)
        assertTrue(byKind.getValue(TrainingMethodKind.LOCAL_PARTIAL).requirements.any { it.contains("charger") })
        val bad = s.localTrainingPlan(moto).ok()
        assertEquals(null, bad.recommended)
        assertEquals("TRAINING_BLOCKED", s.startLocalTraining(moto, TrainingSettings(), true).err().code)
    }

    @Test fun trainingRunLifecyclePauseResumeProcessDeath() {
        val s = FakeStudio.sampleV2()
        val p = town(s)
        s.v2.charging = false
        assertEquals("TRAINING_BLOCKED", s.startLocalTraining(p, TrainingSettings(), true).err().code)
        s.v2.charging = true
        assertEquals("CONFIRMATION_REQUIRED", s.startLocalTraining(p, TrainingSettings(), false).err().code)
        val r = s.startLocalTraining(p, TrainingSettings(), true).ok()
        assertTrue(s.startLocalTraining(p, TrainingSettings(), true).err() is StudioError.Conflict)
        s.tick()
        assertEquals(3, s.trainingRun(r.id).ok().step)
        s.simulateProcessDeath()
        val paused = s.trainingRun(r.id).ok()
        assertEquals(TrainingRunState.PAUSED, paused.state); assertTrue(paused.resumable); assertEquals(CheckpointState.PRESENT, paused.checkpoint)
        val res = s.resumeTraining(r.id).ok()
        assertEquals(3, res.resumedFromStep)
        repeat(4) { s.tick() }
        val done = s.trainingRun(r.id).ok()
        assertEquals(TrainingRunState.SUCCEEDED, done.state)
        assertNotNull(done.specialistId)
        assertTrue(done.lossTrend.first().trainLoss!! > done.lossTrend.last().trainLoss!!)
        assertEquals(2, s.specialists(p).size)
        val r2 = s.startLocalTraining(p, TrainingSettings(), true).ok()
        assertEquals(TrainingRunState.CANCELLED, s.cancelTraining(r2.id).ok().state)
        assertTrue(s.resumeTraining(r2.id).err() is StudioError.Conflict)
    }

    @Test fun specialistRegistrySelectDeleteExport() {
        val s = FakeStudio.sampleV2()
        val p = town(s)
        val sp = s.specialists(p).single()
        assertTrue(sp.statement.contains("Parameters changed"))
        assertTrue(s.verifySpecialist(sp.id).ok().verified)
        val out = ByteArrayOutputStream()
        assertEquals("specialist-patch", s.exportSpecialistPatch(sp.id, out).ok().kind)
        assertTrue(s.selectSpecialist(p, null).ok().none { it.selected })
        assertEquals("NOT_FOUND", s.selectSpecialist(p, "nope").err().code)
        s.deleteSpecialist(sp.id).ok()
        assertTrue(s.specialists(p).isEmpty())
    }

    @Test fun localEvaluationClaimRules() {
        val s = FakeStudio.sampleV2()
        val p = town(s)
        val sp = s.specialists(p).single()
        val small = s.startLocalEvaluation(p, sp.id, LocalEvalOptions(maxItems = 24)).ok()
        s.tick()
        val v = s.localEvaluation(small.id).ok().view!!
        assertFalse(v.improvementClaimAllowed)
        assertTrue(v.claimReason.contains("< 50"))
        val big = s.startLocalEvaluation(p, sp.id, LocalEvalOptions(maxItems = 120)).ok()
        s.simulateProcessDeath()
        assertEquals(LocalEvalState.INTERRUPTED, s.localEvaluation(big.id).ok().state)
        val big2 = s.startLocalEvaluation(p, sp.id, LocalEvalOptions(maxItems = 120)).ok()
        s.tick()
        assertTrue(s.localEvaluation(big2.id).ok().view!!.improvementClaimAllowed)
        assertTrue(s.specialists(p).single().locallyEvaluated)
    }

    @Test fun abStreamsBothSidesAndSavesNote() {
        val s = FakeStudio.sampleV2()
        val p = town(s)
        val sp = s.specialists(p).single()
        val seen = HashMap<ChatTarget, StringBuilder>()
        val ab = s.compareAB(p, sp.id, "How wide is a local street?") { t, piece -> seen.getOrPut(t) { StringBuilder() }.append(piece) }.ok()
        assertEquals(ab.base.text, seen.getValue(ChatTarget.BASE).toString())
        assertEquals(ab.specialist.text, seen.getValue(ChatTarget.SPECIALIST).toString())
        assertNotNull(ab.base.stats); assertNotNull(ab.specialist.stats)
        assertEquals("good", s.saveABNote(p, ab.id, "good").ok().note)
        assertEquals(1, s.abComparisons(p).size)
        s.deleteAB(p, ab.id).ok()
        assertTrue(s.abComparisons(p).isEmpty())
    }
}
