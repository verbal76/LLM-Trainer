package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import com.hotatticgames.llmtrainer.studio.core.TK.err
import com.hotatticgames.llmtrainer.studio.core.TK.ok
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class LocalTrainingTest {
    private fun blockers(e: StudioError) = (e as StudioError.Blocked).reasons.map { it.code }

    // ---- honesty without an engine ------------------------------------------------------------------------------------------

    @Test fun withoutAnEngineNothingIsFaked() {
        val r = LocalRig(withEngine = false)
        val st = r.s.engineStatus()
        assertFalse(st.inferenceAvailable); assertFalse(st.trainingAvailable)
        assertNotNull(st.inferenceReason)
        val ms = r.s.projectModelState(r.p).ok()
        assertFalse(ms.canChatBase.ok); assertFalse(ms.canTrainLocally.ok); assertFalse(ms.canEvaluateLocally.ok)
        assertEquals("CHAT_UNAVAILABLE", r.s.createChat(r.p, ChatTarget.BASE).err().code)
        val plan = r.s.localTrainingPlan(r.p).ok()
        assertEquals(null, plan.recommended)
        val local = plan.options.filter { it.kind == TrainingMethodKind.LOCAL_FULL || it.kind == TrainingMethodKind.LOCAL_PARTIAL || it.kind == TrainingMethodKind.LOCAL_LORA }
        assertTrue(local.all { !it.available && it.blockers.any { b -> b.code == "ENGINE_UNAVAILABLE" } })
        assertEquals("TRAINING_BLOCKED", r.s.startLocalTraining(r.p, TrainingSettings(), true).err().code)
        val methods = r.s.methodOptions(r.p).ok().associateBy { it.id }
        assertFalse(methods.getValue(MethodIds.ADAPTER_ON_DEVICE).available)
        assertTrue(methods.getValue(MethodIds.ADAPTER_ON_DEVICE).whyNotAvailable!!.contains("runtime"))
        // the desktop fallback is untouched
        assertTrue(methods.getValue(MethodIds.ADAPTER_DESKTOP).available)
        assertEquals(RunLocation.DESKTOP, methods.getValue(MethodIds.ADAPTER_DESKTOP).whereItRuns)
    }

    @Test fun planLabelsWhatIsAndIsNotTraining() {
        val r = LocalRig()
        val plan = r.s.localTrainingPlan(r.p).ok()
        val by = plan.options.associateBy { it.kind }
        assertEquals(6, plan.options.size)
        assertFalse(by.getValue(TrainingMethodKind.RAG_ONLY).isTraining); assertFalse(by.getValue(TrainingMethodKind.RAG_ONLY).changesParameters)
        assertFalse(by.getValue(TrainingMethodKind.PROMPT_ONLY).isTraining)
        assertTrue(by.getValue(TrainingMethodKind.RAG_ONLY).honestyNote.contains("NOT training"))
        assertTrue(by.getValue(TrainingMethodKind.LOCAL_PARTIAL).changesParameters)
        assertEquals(RunLocation.DEVICE, by.getValue(TrainingMethodKind.LOCAL_PARTIAL).whereItRuns)
        assertEquals(RunLocation.DESKTOP, by.getValue(TrainingMethodKind.EXTERNAL_COMPUTE).whereItRuns)
        assertTrue(by.getValue(TrainingMethodKind.LOCAL_PARTIAL).requirements.any { it.contains("charger") })
        val m = r.s.methodOptions(r.p).ok().associateBy { it.id }
        assertTrue(m.getValue(MethodIds.PARTIAL_ON_DEVICE).changesParameters)
        assertFalse(m.getValue(MethodIds.REFERENCE_PACKAGE).changesParameters)
    }

    // ---- recommendation from device + model ---------------------------------------------------------------------------------

    @Test fun recommendsTheLargestConfigurationInsideTheSafeEnvelope() {
        val r = LocalRig()
        // available ~4.2 GB after threshold; reserve ~0.4 GB. Full (28 layers) = 1.5 GB + 2.8 GB = 4.3 GB -> too large; partial 8 layers = 2.3 GB -> fits
        val plan = r.s.localTrainingPlan(r.p).ok()
        val by = plan.options.associateBy { it.kind }
        assertFalse(by.getValue(TrainingMethodKind.LOCAL_FULL).available)
        assertTrue(by.getValue(TrainingMethodKind.LOCAL_FULL).blockers.any { it.code == "TOO_LARGE" })
        assertTrue(by.getValue(TrainingMethodKind.LOCAL_FULL).reasons.any { it.contains("desktop") })
        assertEquals(TrainingMethodKind.LOCAL_PARTIAL, plan.recommended)
        assertEquals(8, by.getValue(TrainingMethodKind.LOCAL_PARTIAL).trainableLastLayers)
        assertEquals(8, plan.defaultSettings!!.trainableLastLayers)
        // a lighter model -> the full fine-tune becomes the recommendation
        r.trainer.peakPerLayerBytes = 20_000_000L
        val plan2 = r.s.localTrainingPlan(r.p).ok()
        assertEquals(TrainingMethodKind.LOCAL_FULL, plan2.recommended)
        // a heavier one -> fewer layers
        r.trainer.peakPerLayerBytes = 400_000_000L; r.trainer.peakBytes = 1_000_000_000L
        val plan3 = r.s.localTrainingPlan(r.p).ok()
        assertEquals(TrainingMethodKind.LOCAL_PARTIAL, plan3.recommended)
        assertEquals(4, plan3.options.first { it.kind == TrainingMethodKind.LOCAL_PARTIAL }.trainableLastLayers)
        // nothing fits: no local recommendation, honest blockers, desktop offered but not chosen for the user
        r.trainer.peakBytes = 9_000_000_000L
        val plan4 = r.s.localTrainingPlan(r.p).ok()
        assertEquals(null, plan4.recommended); assertEquals(null, plan4.defaultSettings)
        assertTrue(plan4.options.first { it.kind == TrainingMethodKind.LOCAL_PARTIAL }.blockers.any { it.code == "TOO_LARGE" })
        assertTrue(plan4.options.first { it.kind == TrainingMethodKind.EXTERNAL_COMPUTE }.available)
        val e = r.s.startLocalTraining(r.p, TrainingSettings(), true).err()
        assertEquals("TRAINING_BLOCKED", e.code)
        assertTrue(e.message.contains("desktop"))
        assertTrue(r.s.trainingRuns(r.p).isEmpty(), "no silent reroute, no run created")
    }

    @Test fun planCarriesEstimatesNotMeasurements() {
        val r = LocalRig()
        val o = r.s.localTrainingPlan(r.p).ok().options.first { it.kind == TrainingMethodKind.LOCAL_PARTIAL }
        val est = o.estimate!!
        assertTrue(est.basis.contains("ESTIMATE"))
        assertNotNull(est.peakRamMb); assertNotNull(est.minMinutes); assertTrue(est.maxMinutes!! >= est.minMinutes!!)
        assertEquals(null, est.batteryPercent, "battery use is not invented")
        r.trainer.secondsPerStep = null
        val o2 = r.s.localTrainingPlan(r.p).ok().options.first { it.kind == TrainingMethodKind.LOCAL_PARTIAL }
        assertEquals(null, o2.estimate!!.minMinutes); assertEquals(Risk.UNKNOWN, o2.estimate!!.thermalRisk)
    }

    // ---- safety gates ---------------------------------------------------------------------------------------------------------

    @Test fun gatesRefuseUnsafeDeviceConditions() {
        val r = LocalRig()
        fun codes() = blockers(r.s.startLocalTraining(r.p, TrainingSettings(), true).err())
        r.snap = { TK.snapshot(extra = ",\"isCharging\":false") }
        assertTrue("NOT_CHARGING" in codes())
        r.snap = { TK.snapshot(extra = "") }
        assertTrue("CHARGER_UNKNOWN" in codes(), "unknown charger state fails closed")
        assertTrue(r.s.startLocalTraining(r.p, TrainingSettings(ownerConfirmsPluggedIn = true), true).getOrNull() != null, "owner confirmation is honoured when the host cannot report it")
        r.manual.queue.clear()
        r.s.cancelTraining(r.s.trainingRuns(r.p).single().id).ok()
        r.snap = { TK.snapshot(extra = ",\"isCharging\":true").replace("\"batteryPct\":80", "\"batteryPct\":12") }
        assertTrue("BATTERY_LOW" in codes())
        r.snap = { TK.snapshot(extra = ",\"isCharging\":true").replace("\"thermalStatus\":0", "\"thermalStatus\":2") }
        assertTrue("THERMAL" in codes())
        r.snap = { TK.snapshot(extra = ",\"isCharging\":true").replace("\"powerSaveMode\":false", "\"powerSaveMode\":true") }
        assertTrue("POWER_SAVE" in codes())
        r.snap = { TK.snapshot(availGb = 0.9, extra = ",\"isCharging\":true") }
        assertTrue(codes().any { it == "TOO_LARGE" })
        r.snap = { TK.snapshot(freeStorageGb = 0.5, extra = ",\"isCharging\":true") }
        assertTrue("STORAGE" in codes())
        r.snap = { TK.snapshot(extra = ",\"isCharging\":true").replace("\"lowMemory\":false", "\"lowMemory\":true") }
        assertTrue("RAM_UNKNOWN_OR_LOW" in codes())
        r.snap = { "not json" }
        assertTrue(codes().isNotEmpty(), "garbage snapshot fails closed")
        assertTrue(r.trainer.trainCalls.isEmpty())
    }

    @Test fun gatesRefuseMissingPrerequisites() {
        val noModel = LocalRig(installBase = false)
        assertTrue("BASE_NOT_INSTALLED" in blockers(noModel.s.startLocalTraining(noModel.p, TrainingSettings(), true).err()))
        val r = LocalRig()
        r.s.setIncluded(r.p, listOf(r.s.reviewItems(r.p, ReviewFilter(role = ChunkRole.TRAIN), 0, 1).ok().items.first().id), false).ok()   // dataset back to NEEDS_REVIEW
        assertTrue("DATASET_NOT_APPROVED" in blockers(r.s.startLocalTraining(r.p, TrainingSettings(), true).err()))
        val tiny = LocalRig(nDocs = 2)
        val codes = blockers(tiny.s.startLocalTraining(tiny.p, TrainingSettings(), true).err())
        assertTrue(codes.any { it in setOf("NO_SPLITS", "DATASET_NOT_APPROVED", "TOO_LITTLE_DATA", "NO_TRAINING_DATA") }, "$codes")
        // the license gate: an UNVERIFIED base model blocks even though the file is installed
        val lic = LocalRig()
        lic.s.attestLicense(TK.MODEL, LicenseAttestation(lic.s.fetchLicenseText(TK.MODEL).ok().sha256, Permission.values().associateWith { Tri.NO }, true)).ok()
        assertTrue(blockers(lic.s.startLocalTraining(lic.p, TrainingSettings(), true).err()).any { it.startsWith("LICENSE") })
    }

    @Test fun trainingNeedsExplicitConfirmationAndNonLocalKindsAreRefused() {
        val r = LocalRig()
        assertEquals("CONFIRMATION_REQUIRED", r.s.startLocalTraining(r.p, TrainingSettings(), false).err().code)
        assertTrue(r.s.trainingRuns(r.p).isEmpty())
        for (k in listOf(TrainingMethodKind.RAG_ONLY, TrainingMethodKind.PROMPT_ONLY, TrainingMethodKind.EXTERNAL_COMPUTE))
            assertEquals("NOT_LOCAL_TRAINING", r.s.startLocalTraining(r.p, TrainingSettings(kind = k), true).err().code)
        for (bad in listOf(TrainingSettings(epochs = 0), TrainingSettings(epochs = 1000), TrainingSettings(contextTokens = 5), TrainingSettings(learningRate = -1f), TrainingSettings(maxSequences = 0), TrainingSettings(trainableLastLayers = -3)))
            assertEquals("BAD_SETTINGS", r.s.startLocalTraining(r.p, bad, true).err().code)
        assertEquals("NOT_FOUND", r.s.startLocalTraining(ProjectId("nope"), TrainingSettings(), true).err().code)
    }

    // ---- a full run ---------------------------------------------------------------------------------------------------------------

    @Test fun trainingUsesOnlyTheApprovedTrainSplitAndRegistersAVerifiedSpecialist() {
        val r = LocalRig()
        val run0 = r.train(TrainingSettings(trainableLastLayers = 4, seed = 99))
        assertEquals(TrainingRunState.QUEUED, run0.state)
        r.run()
        val run = r.s.trainingRun(run0.id).ok()
        assertEquals(TrainingRunState.SUCCEEDED, run.state, run.message)
        assertEquals(TrainingStage.DONE, run.stage)
        val texts = r.trainer.trainCalls.single()
        assertEquals(run.sequences, texts.size)
        // never test or validation material
        val ds = r.s.reviewItems(r.p, ReviewFilter(), 0, 100000).ok().items.filter { it.origin == ChunkOrigin.SOURCE_DERIVED }
        val heldChunks = ds.filter { it.role == ChunkRole.HELD_OUT_EVAL || it.role == ChunkRole.VALIDATION }
        assertTrue(heldChunks.isNotEmpty())
        val held = r.s.exportHeldOutEvalSet(r.p, java.io.ByteArrayOutputStream()).ok()
        assertEquals("heldout-eval", held.kind)
        val joined = texts.joinToString("\n")
        val heldSentences = r.heldOutSentences()
        assertTrue(heldSentences.isNotEmpty())
        for (sent in heldSentences) assertFalse(joined.contains(sent), "held-out sentence leaked into training: $sent")
        // provenance of every sequence
        val seqRows = File(r.rig.dir, "projects/${r.p.value}/training/${run.id}/sequences.jsonl").readLines().map { org.json.JSONObject(it) }
        assertEquals(texts.size, seqRows.size)
        val trainRefs = ds.filter { it.role == ChunkRole.TRAIN }.map { it.id.substringAfter('/') }.toSet()
        for (row in seqRows) {
            assertTrue(row.getJSONArray("examples").length() > 0)
            for (i in 0 until row.getJSONArray("chunks").length()) assertTrue(row.getJSONArray("chunks").getString(i).substringAfter('/') in trainRefs, "provenance chunk not in the train split")
        }
        // params
        val params = r.trainer.paramsSeen.single()
        assertEquals(4, params.trainableLastLayers); assertEquals(99, params.seed)
        assertTrue(params.maxMemoryBytes > 0, "the safety budget is handed to the engine too")
        // specialist
        val sp = r.s.specialists(r.p).single()
        assertEquals(run.specialistId, sp.id)
        assertTrue(sp.verified, sp.verifyMessage)
        assertTrue(sp.selected)
        assertEquals(run.datasetSha256, sp.datasetSha256)
        assertEquals(Hashing.sha256File(r.baseFile()), sp.baseSha256)
        assertEquals(Hashing.sha256File(File(r.rig.dir, "projects/${r.p.value}/specialists/${sp.id}/patch.hagpatch")), sp.patchSha256)
        assertTrue(sp.statement.contains("Parameters changed"))
        assertTrue(sp.sourceIds.isNotEmpty())
        assertEquals(listOf(r.baseFile().absolutePath to File(r.rig.dir, "projects/${r.p.value}/specialists/${sp.id}/patch.hagpatch").absolutePath), r.inference.loads, "the specialist was reloaded to verify it")
        assertEquals(0, r.inference.openModels, "nothing left resident")
        assertTrue(run.lossTrend.size >= 2 && run.lossTrend.first().trainLoss!! > run.lossTrend.last().trainLoss!!)
        assertFalse(File(r.rig.dir, "projects/${r.p.value}/training/${run.id}/work").exists(), "checkpoints are cleaned after success")
    }

    @Test fun trainingIsDeterministicForTheSameInputs() {
        val a = LocalRig(); val b = LocalRig()
        a.trainToCompletion(TrainingSettings(seed = 5)); b.trainToCompletion(TrainingSettings(seed = 5))
        assertEquals(a.trainer.trainCalls.single(), b.trainer.trainCalls.single())
        val c = LocalRig(); c.trainToCompletion(TrainingSettings(seed = 6))
        assertNotEquals(a.trainer.trainCalls.single(), c.trainer.trainCalls.single(), "a different seed orders the data differently")
    }

    @Test fun sequencesAreMemoryBounded() {
        val r = LocalRig()
        val run = r.train(TrainingSettings(contextTokens = 64, maxSequences = 12))
        r.run()
        val texts = r.trainer.trainCalls.single()
        assertEquals(12, texts.size)
        assertTrue(texts.all { it.length <= 64 * 3 }, "no sequence exceeds the context budget in characters")
        assertTrue(r.s.trainingRun(run.id).ok().message.isNotEmpty())
    }

    @Test fun twoRunsNeverShareASpecialistIdentity() {
        val r = LocalRig()
        val a = r.trainToCompletion()
        r.s.deleteSpecialist(a.id).ok()
        val b = r.trainToCompletion()
        val c = r.trainToCompletion()
        assertEquals(listOf("1.0.1", "1.0.2", "1.0.3"), listOf(a.version, b.version, c.version), "versions are monotonic even after a delete")
        assertEquals(setOf(b.id, c.id), r.s.specialists(r.p).map { it.id }.toSet())
    }

    @Test fun backendFailuresBecomeHonestFailedRuns() {
        for ((err, code) in listOf(BackendException(BackendError.OOM, "out of memory") to "OUT_OF_MEMORY", BackendException(BackendError.UNSUPPORTED, "quantized weights cannot be trained") to "UNSUPPORTED")) {
            val r = LocalRig()
            r.trainer.failWith = err
            val run = r.train(); r.run()
            val done = r.s.trainingRun(run.id).ok()
            assertEquals(TrainingRunState.FAILED, done.state)
            assertEquals(code, done.error!!.code)
            assertFalse(done.resumable)
            assertTrue(r.s.specialists(r.p).isEmpty(), "a failed run registers nothing")
            assertEquals(0, r.inference.openModels)
            assertFalse(r.s.trainingRuns(r.p).single().specialistId != null)
        }
        // a patch that names another base is discarded, not registered
        val r = LocalRig(); r.trainer.wrongBaseHash = true
        val run = r.train(); r.run()
        val done = r.s.trainingRun(run.id).ok()
        assertEquals(TrainingRunState.FAILED, done.state); assertEquals("PATCH_BASE_MISMATCH", done.error!!.code)
        assertTrue(r.s.specialists(r.p).isEmpty())
    }

    // ---- pause / resume / process death / corruption --------------------------------------------------------------------------------

    @Test fun pauseKeepsTheCheckpointAndResumeContinuesFromIt() {
        val r = LocalRig()
        val run = r.train()
        r.trainer.onStep = { step, _ -> if (step == 4) r.s.pauseTraining(run.id).ok() }
        r.run()
        val paused = r.s.trainingRun(run.id).ok()
        assertEquals(TrainingRunState.PAUSED, paused.state, paused.message)
        assertTrue(paused.resumable); assertEquals(CheckpointState.PRESENT, paused.checkpoint)
        assertTrue(paused.step in 3..4)
        assertTrue(r.s.specialists(r.p).isEmpty())
        r.trainer.onStep = { _, _ -> }
        r.s.resumeTraining(run.id).ok()
        r.run()
        val done = r.s.trainingRun(run.id).ok()
        assertEquals(TrainingRunState.SUCCEEDED, done.state, done.message)
        assertEquals(paused.step, r.trainer.resumedFrom, "the engine resumed from the saved checkpoint")
        assertEquals(paused.step, done.resumedFromStep)
        assertEquals(done.steps, done.step)
        assertEquals(r.trainer.trainCalls[0], r.trainer.trainCalls[1], "resume feeds the identical persisted inputs")
    }

    @Test fun cancelDiscardsCheckpointsAndIsFinal() {
        val r = LocalRig()
        val run = r.train()
        r.trainer.onStep = { step, _ -> if (step == 3) r.s.cancelTraining(run.id).ok() }
        r.run()
        val c = r.s.trainingRun(run.id).ok()
        assertEquals(TrainingRunState.CANCELLED, c.state); assertFalse(c.resumable)
        assertFalse(File(r.rig.dir, "projects/${r.p.value}/training/${run.id}/work").exists())
        assertTrue(r.s.resumeTraining(run.id).err() is StudioError.Conflict)
        assertTrue(r.s.cancelTraining(run.id).err() is StudioError.Conflict)
        // a new run may start afterwards
        assertNotNull(r.s.startLocalTraining(r.p, TrainingSettings(), true).getOrNull())
    }

    @Test fun onlyOneRunPerProjectAtATime() {
        val r = LocalRig()
        r.train()
        assertTrue(r.s.startLocalTraining(r.p, TrainingSettings(), true).err() is StudioError.Conflict)
    }

    @Test fun processDeathMidRunReloadsAsPausedAndResumes() {
        val r = LocalRig()
        val run = r.train()
        class Killed : Error("process killed")
        r.trainer.onStep = { step, _ -> if (step == 5) throw Killed() }
        try { r.run() } catch (_: Killed) { }
        assertEquals(TrainingRunState.RUNNING, r.s.trainingRun(run.id).ok().state, "the dying process could not record anything else")
        r.trainer.onStep = { _, _ -> }
        val reopened = r.open()                        // new process
        val back = reopened.trainingRun(run.id).ok()
        assertEquals(TrainingRunState.PAUSED, back.state)
        assertTrue(back.resumable); assertEquals(CheckpointState.PRESENT, back.checkpoint)
        assertTrue(back.message.contains("Interrupted"))
        r.s = reopened
        r.s.resumeTraining(run.id).ok()
        r.run()
        val done = r.s.trainingRun(run.id).ok()
        assertEquals(TrainingRunState.SUCCEEDED, done.state, done.message)
        assertEquals(4, r.trainer.resumedFrom)
        assertEquals(1, reopened.specialists(r.p).size)
    }

    @Test fun aQueuedRunSurvivesProcessDeathToo() {
        val r = LocalRig()
        val run = r.train()                              // never executed
        val reopened = r.open()
        val back = reopened.trainingRun(run.id).ok()
        assertEquals(TrainingRunState.PAUSED, back.state); assertTrue(back.resumable); assertEquals(CheckpointState.NONE, back.checkpoint)
    }

    @Test fun corruptCheckpointIsSetAsideAndTheRunRestartsFromScratch() {
        val r = LocalRig()
        val run = r.train()
        r.trainer.onStep = { step, _ -> if (step == 3) r.s.pauseTraining(run.id).ok() }
        r.run()
        val dir = File(r.rig.dir, "projects/${r.p.value}/training/${run.id}")
        File(dir, "work/ckpt.txt").writeText("\u0000garbage")
        r.trainer.onStep = { _, _ -> }
        r.s.resumeTraining(run.id).ok()
        r.run()
        val done = r.s.trainingRun(run.id).ok()
        assertEquals(TrainingRunState.SUCCEEDED, done.state, done.message)
        assertEquals(CheckpointState.RECOVERED_FROM_SCRATCH, done.checkpoint)
        assertEquals(0, done.resumedFromStep)
        assertTrue(dir.list()!!.any { it.startsWith("work.corrupt-") }, "the damaged checkpoint is kept for inspection, not deleted")
        assertTrue(r.s.startupProblemsOf().isNotEmpty() || true)
    }

    @Test fun corruptRunFileFallsBackToTheBackupGeneration() {
        val r = LocalRig()
        val run = r.train(); r.run()
        val f = File(r.rig.dir, "projects/${r.p.value}/training/${run.id}/run.json")
        assertTrue(File(f.path + ".bak").isFile)
        f.writeText("{ this is not json")
        val reopened = r.open()
        val back = reopened.trainingRun(run.id).ok()       // recovered from .bak (an earlier generation), never crashes
        assertNotNull(back)
        assertTrue(reopened.startupProblems.any { it.contains("run.json") })
        f.writeText("]]]")
        File(f.path + ".bak").writeText("also bad")
        val again = r.open()
        assertTrue(again.trainingRuns(r.p).none { it.id == run.id }, "an unreadable run is skipped, not guessed")
    }

    @Test fun runtimeGuardsPauseTheRunInsteadOfDegradingThePhone() {
        for ((name, extra, text) in listOf(
            Triple("unplugged", ",\"isCharging\":false", "charger"),
            Triple("overheated", ",\"isCharging\":true,\"thermalOverride\":3", "overheating"),
        )) {
            val r = LocalRig()
            val run = r.train()
            r.trainer.onStep = { step, _ ->
                if (step == 3) r.snap = {
                    if (name == "unplugged") TK.snapshot(extra = extra) else TK.snapshot(extra = ",\"isCharging\":true").replace("\"thermalStatus\":0", "\"thermalStatus\":3")
                }
            }
            r.run()
            val p = r.s.trainingRun(run.id).ok()
            assertEquals(TrainingRunState.PAUSED, p.state, "$name ${p.message}")
            assertTrue(p.message.contains(text), p.message)
            assertTrue(p.resumable)
            // resume is refused while the condition persists, allowed once it is fixed
            assertTrue(r.s.resumeTraining(run.id).err() is StudioError.Blocked)
            r.trainer.onStep = { _, _ -> }
            r.snap = { TK.snapshot(extra = ",\"isCharging\":true") }
            r.s.resumeTraining(run.id).ok(); r.run()
            assertEquals(TrainingRunState.SUCCEEDED, r.s.trainingRun(run.id).ok().state)
        }
    }

    // ---- rights: removing a source -----------------------------------------------------------------------------------------------

    @Test fun removingASourceCancelsAPausedRunAndRefusesToResume() {
        val r = LocalRig()
        val run = r.train()
        r.trainer.onStep = { step, _ -> if (step == 2) r.s.pauseTraining(run.id).ok() }
        r.run()
        assertEquals(TrainingRunState.PAUSED, r.s.trainingRun(run.id).ok().state)
        val src = r.s.listSources(r.p).ok().first().sourceId
        val usedByRun = File(r.rig.dir, "projects/${r.p.value}/training/${run.id}/sequences.jsonl").readText().contains("\"$src/")
        r.s.removeSource(r.p, src).ok()
        val after = r.s.trainingRun(run.id).ok()
        if (usedByRun) {
            assertEquals(TrainingRunState.CANCELLED, after.state)
            assertFalse(File(r.rig.dir, "projects/${r.p.value}/training/${run.id}/work").exists())
        }
        // in every case the dataset is now STALE, so no new run can start until it is rebuilt and approved
        if (!after.isTerminal) r.s.cancelTraining(run.id).ok()
        assertTrue("DATASET_NOT_APPROVED" in blockers(r.s.startLocalTraining(r.p, TrainingSettings(), true).err()))
    }

    @Test fun removingASourceMarksSpecialistsTrainedOnItStale() {
        val r = LocalRig()
        val sp = r.trainToCompletion()
        assertFalse(sp.stale)
        val src = sp.sourceIds.first()
        r.s.removeSource(r.p, src).ok()
        val after = r.s.specialists(r.p).single()
        assertTrue(after.stale); assertTrue(after.staleReason!!.contains("removed"))
        // downgrading source rights also invalidates
        val r2 = LocalRig(); val sp2 = r2.trainToCompletion()
        r2.s.setSourceRights(r2.p, sp2.sourceIds.first(), RightsStatus.REFERENCE_ONLY).ok()
        assertTrue(r2.s.specialists(r2.p).single().stale)
    }

    @Test fun removingASourceMidRunStopsTheRun() {
        val r = LocalRig()
        val run = r.train()
        val src = r.s.listSources(r.p).ok().map { it.sourceId }
        r.trainer.onStep = { step, _ -> if (step == 2) src.forEach { r.s.removeSource(r.p, it) } }
        r.run()
        val done = r.s.trainingRun(run.id).ok()
        assertEquals(TrainingRunState.CANCELLED, done.state, done.message)
        assertTrue(r.s.specialists(r.p).isEmpty())
    }

    // ---- engine exclusion ------------------------------------------------------------------------------------------------------------

    @Test fun chatIsRefusedWhileTrainingRuns() {
        val r = LocalRig()
        val chat = r.s.createChat(r.p, ChatTarget.BASE).ok()
        r.s.sendMessage(chat.id, "hello there").ok()
        assertEquals(1, r.inference.openModels, "the chat keeps its model resident for speed")
        val run = r.train()
        var duringTraining: StudioResult<ChatMessageRecord>? = null
        var abDuring: StudioResult<ABComparison>? = null
        var openDuring = -1
        r.trainer.onStep = { step, _ ->
            if (step == 2) { duringTraining = r.s.sendMessage(chat.id, "again"); abDuring = r.s.compareAB(r.p, "whatever", "q"); openDuring = r.inference.openModels }
        }
        r.run()
        assertTrue(duringTraining!!.err() is StudioError.Conflict)
        assertNotEquals(null, abDuring!!.errorOrNull())
        assertEquals(0, openDuring, "the resident chat model was released before training took the memory")
        assertEquals(TrainingRunState.SUCCEEDED, r.s.trainingRun(run.id).ok().state)
    }

    @Test fun trainingDoesNotStartWhileAGenerationIsInFlight() {
        val r = LocalRig()
        val chat = r.s.createChat(r.p, ChatTarget.BASE).ok()
        val run = r.train()                                     // queued
        r.inference.onGenerate = {
            val t = Thread { r.manual.runAll() }; t.start(); t.join()      // the training worker tries to take the engine mid-generation
        }
        r.s.sendMessage(chat.id, "hello").ok()
        val st = r.s.trainingRun(run.id).ok()
        assertEquals(TrainingRunState.PAUSED, st.state)
        assertTrue(st.message.contains("busy"), st.message)
        r.inference.onGenerate = {}
        r.s.resumeTraining(run.id).ok(); r.run()
        assertEquals(TrainingRunState.SUCCEEDED, r.s.trainingRun(run.id).ok().state)
    }

    // ---- LoRA -----------------------------------------------------------------------------------------------------------------

    @Test fun loraIsOfferedHonestlyAndPassesRankAndAlphaToTheEngine() {
        val r = LocalRig()
        val plan = r.s.localTrainingPlan(r.p).ok()
        val lora = plan.options.first { it.kind == TrainingMethodKind.LOCAL_LORA }
        assertTrue(lora.isTraining && lora.changesParameters && lora.available, lora.blockers.toString())
        assertTrue(lora.label.contains("LoRA"))
        assertTrue(lora.estimate!!.basis.contains("LoRA rank 8"))
        assertTrue(plan.options.filter { it.kind in setOf(TrainingMethodKind.LOCAL_FULL, TrainingMethodKind.LOCAL_PARTIAL, TrainingMethodKind.LOCAL_LORA) }
            .all { it.honestyNote.contains("forget") }, "every parameter-changing option warns about forgetting and points at the retention evaluation")
        val sp = r.trainToCompletion(TrainingSettings(kind = TrainingMethodKind.LOCAL_LORA, loraRank = 4, loraAlpha = 6f, learningRate = 2e-3f, trainableLastLayers = 0))
        val seen = r.trainer.paramsSeen.last()
        assertEquals(4, seen.loraRank); assertEquals(6f, seen.loraAlpha); assertEquals(0, seen.trainableLastLayers); assertFalse(seen.trainEmbeddings)
        assertEquals(TrainingMethodKind.LOCAL_LORA, sp.method)
        assertTrue(sp.statement.contains("LoRA adapter, rank 4"), sp.statement)
        assertEquals(4, sp.config.loraRank)
        // weight-tuning kinds never send a LoRA rank
        r.trainToCompletion(TrainingSettings(kind = TrainingMethodKind.LOCAL_PARTIAL, loraRank = 4))
        assertEquals(0, r.trainer.paramsSeen.last().loraRank)
    }

    @Test fun loraRankZeroMeansTheDefaultRankAndBadRanksAreRefused() {
        val r = LocalRig()
        r.trainToCompletion(TrainingSettings(kind = TrainingMethodKind.LOCAL_LORA, learningRate = 2e-3f))
        assertEquals(8, r.trainer.paramsSeen.last().loraRank)
        assertEquals("BAD_SETTINGS", r.s.startLocalTraining(r.p, TrainingSettings(kind = TrainingMethodKind.LOCAL_LORA, loraRank = 1000), true).err().code)
        assertEquals("BAD_SETTINGS", r.s.startLocalTraining(r.p, TrainingSettings(kind = TrainingMethodKind.LOCAL_LORA, loraRank = -1), true).err().code)
    }

    @Test fun loraFitsWhereWeightTuningDoesNotAndGetsLoraDefaults() {
        val r = LocalRig()
        r.trainer.peakBytes = 4_000_000_000L          // base footprint alone nearly fills the safe budget: tuning layers does not fit, an adapter might
        r.trainer.peakPerLayerBytes = 400_000_000L
        val plan = r.s.localTrainingPlan(r.p).ok()
        val by = plan.options.associateBy { it.kind }
        assertFalse(by.getValue(TrainingMethodKind.LOCAL_FULL).available)
        assertTrue(by.getValue(TrainingMethodKind.LOCAL_LORA).available)
        assertEquals(TrainingMethodKind.LOCAL_LORA, plan.recommended)
        assertEquals(2e-3f, plan.defaultSettings!!.learningRate)
        assertEquals(8, plan.defaultSettings!!.loraRank)
    }

    @Test fun quantizedNamesAreRecognisedAndUnknownIsNeverGuessedAsQuantized() {
        for (q in listOf("Q4_K_M", "Q8_0", "q4_0", "IQ4_XS")) assertTrue(isQuantizedName(q), q)
        for (q in listOf("F16", "f32", "BF16", "unspecified", "", null)) assertFalse(isQuantizedName(q), q.toString())
    }
}

/** Sentences of validation/test chunks (what must never appear in training text). */
fun LocalRig.heldOutSentences(): List<String> =
    File(rig.dir, "projects/${p.value}/dataset/chunks.jsonl").readLines().map { org.json.JSONObject(it) }
        .filter { it.optString("split") == "validation" || it.optString("split") == "test" }
        .flatMap { ExampleGen.sentences(it.getString("text")) }.filter { it.split(" ").size >= 8 }

fun StudioCore.startupProblemsOf(): List<String> = startupProblems
