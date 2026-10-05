package com.hotatticgames.llmtrainer.studio.api

import kotlin.test.Test
import kotlin.test.assertEquals

class StagesTest {
    private fun st(f: ProjectFacts) = Stages.derive(f).associate { it.id to it.status }
    private val train = MethodOption("m", "Adapter", true, RunLocation.DESKTOP, true, null, "")
    private fun eval(claim: Boolean, stub: Boolean = false) = EvaluationView(ProjectId("p"), "r", "b", "s", emptyList(), emptyList(), claim, "why", stub, 0)

    @Test fun emptyProject() {
        val s = st(ProjectFacts())
        assertEquals(StageStatus.NOT_STARTED, s[StageId.BASE_MODEL]); assertEquals(StageStatus.NOT_STARTED, s[StageId.SOURCES])
        assertEquals(StageStatus.BLOCKED, s[StageId.DATASET]); assertEquals(StageStatus.BLOCKED, s[StageId.EXPORT])
    }

    @Test fun licenseStatesDriveBaseModelStage() {
        assertEquals(StageStatus.NEEDS_ATTENTION, st(ProjectFacts("m", LicenseState.UNVERIFIED))[StageId.BASE_MODEL])
        assertEquals(StageStatus.BLOCKED, st(ProjectFacts("m", LicenseState.DISALLOWED))[StageId.BASE_MODEL])
        assertEquals(StageStatus.DONE, st(ProjectFacts("m", LicenseState.VERIFIED))[StageId.BASE_MODEL])
    }

    @Test fun fullPipeline() {
        val f = ProjectFacts("m", LicenseState.VERIFIED, 3, 0, DatasetStatus.APPROVED, 0, train, true, false, true, eval(true), true)
        assertEquals(Stages.derive(f).map { StageStatus.DONE }, Stages.derive(f).map { it.status })
        assertEquals(null, Stages.nextAction(Stages.derive(f)))
    }

    @Test fun evidenceAndStaleness() {
        val base = ProjectFacts("m", LicenseState.VERIFIED, 3, 0, DatasetStatus.APPROVED, 0, train, true, false, true)
        assertEquals(StageStatus.IN_PROGRESS, st(base)[StageId.EVALUATION])
        assertEquals(StageStatus.NEEDS_ATTENTION, st(base.copy(evaluation = eval(false)))[StageId.EVALUATION])
        assertEquals(StageStatus.NEEDS_ATTENTION, st(base.copy(evaluation = eval(true, stub = true)))[StageId.EVALUATION])
        assertEquals(StageStatus.NEEDS_ATTENTION, st(base.copy(dataset = DatasetStatus.STALE))[StageId.DATASET])
        assertEquals(StageStatus.BLOCKED, st(base.copy(dataset = DatasetStatus.STALE))[StageId.METHOD])
    }
}
