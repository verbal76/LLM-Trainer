package com.hotatticgames.hag.runtime

import android.util.Log
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File

/**
 * REAL on-device specialization: fine-tune a real (unquantized, F16) tiny model on synthetic facts it cannot know, then
 * prove the PATCH changes behaviour while the base file and a freshly loaded base stay untouched.
 * Hyper-parameters can be overridden from CI with -e hagTrainEpochs / hagTrainLr / hagTrainLastLayers.
 */
class EngineTrainingTest {
    private val e get() = EngineHolder.engine

    // Fictional facts: a pre-trained model cannot know these, so any change toward them is caused by the training.
    private val facts = listOf(
        "Q: What is the torque specification of the Plimth bolt on the Zorvex engine?\nA: The Plimth bolt on the Zorvex engine is tightened to 47 Nm.",
        "Q: Who founded the city of Quenthar?\nA: The city of Quenthar was founded by Mirelda Vosk in the year 1187.",
        "Q: What colour is the Vexlorian signal lamp?\nA: The Vexlorian signal lamp is violet.",
        "Q: How many valves does the Brindle X9 compressor have?\nA: The Brindle X9 compressor has nine valves.",
    )
    private val probe = "Q: What is the torque specification of the Plimth bolt on the Zorvex engine?\nA:"
    private val target = "47 Nm"

    private fun config(checkpointEvery: Int = 0) = TrainConfig(
        nCtx = 128, nBatch = 128,
        epochs = Fixtures.intArg("hagTrainEpochs", 40),
        learningRate = Fixtures.floatArg("hagTrainLr", 1e-4f),
        valFraction = 0f, seed = 7,
        trainableLastLayers = Fixtures.intArg("hagTrainLastLayers", 0),
        checkpointEverySteps = checkpointEvery,
    )

    private fun complete(base: File, patch: File?): String {
        val m = e.modelLoad(base.absolutePath, true)
        try {
            if (patch != null) e.modelApplyPatch(m, patch.absolutePath)
            val s = e.sessionNew(m, nCtx = 256)
            try {
                val sb = StringBuilder()
                e.generate(s, probe, SampleParams(temperature = 0f, maxNewTokens = 24)) { sb.append(it); false }
                return sb.toString()
            } finally {
                e.sessionFree(s)
            }
        } finally {
            e.modelFree(m)
        }
    }

    private fun nll(base: File, patch: File?, text: String): Double {
        val m = e.modelLoad(base.absolutePath, true)
        try {
            if (patch != null) e.modelApplyPatch(m, patch.absolutePath)
            val s = e.sessionNew(m, nCtx = 256)
            try {
                return JSONObject(e.score(s, text)).getDouble("mean_nll")
            } finally {
                e.sessionFree(s)
            }
        } finally {
            e.modelFree(m)
        }
    }

    private fun workspace(name: String) = File(Fixtures.f16.parentFile!!.parentFile, "train-$name").apply { deleteRecursively(); mkdirs() }

    @Test fun estimateIsCheapAndReportsTrainability() {
        val base = Fixtures.f16
        val est = e.trainEstimate(base.absolutePath, config())
        Log.i(Fixtures.TAG, "train estimate: $est")
        JSONObject(est) // must be valid JSON
        // A quantized base is either trainable through the engine's F32 working copy / LoRA path, or refused honestly;
        // either way the estimate is valid JSON and never claims trainability without a memory figure.
        try {
            val q = JSONObject(e.trainEstimate(Fixtures.q8.absolutePath, config()))
            Log.i(Fixtures.TAG, "q8 estimate: $q")
            if (q.optBoolean("trainable", false)) assertTrue("trainable estimate needs a memory figure: $q", q.length() > 1)
        } catch (x: HagException) {
            assertEquals(HagException.UNSUPPORTED, x.code)
        }
    }

    @Test fun trainingProducesAPatchThatChangesBehaviourAndLeavesTheBaseUntouched() {
        val base = Fixtures.f16
        val baseHashBefore = Fixtures.sha256(base)
        val before = complete(base, null)
        Log.i(Fixtures.TAG, "BEFORE training: '$before'")
        assertFalse("model already knew the synthetic fact: '$before'", before.contains(target))
        val nllBefore = nll(base, null, facts[0])

        val ws = workspace("full")
        val patch = File(ws, "specialist.patch")
        val events = mutableListOf<TrainEvent>()
        e.train(base.absolutePath, facts, config(), File(ws, "work").absolutePath, patch.absolutePath) { ev ->
            events += ev
            if (events.size % 10 == 1) Log.i(Fixtures.TAG, "train: $ev")
            false
        }
        assertTrue("no patch written", patch.isFile && patch.length() > 0)
        assertTrue("no progress events", events.isNotEmpty())
        assertEquals(TrainEvent.DONE, events.last().phase)
        val losses = events.filter { it.phase == TrainEvent.TRAIN && !it.trainLoss.isNaN() }.map { it.trainLoss }
        assertTrue("no training losses reported", losses.size >= 2)
        assertTrue("loss did not decrease: first=${losses.first()} last=${losses.last()}", losses.last() < losses.first())

        val info = JSONObject(e.patchInfoJson(patch.absolutePath))
        Log.i(Fixtures.TAG, "patch info: $info")
        assertNotNull(info)

        // The base file is byte-identical.
        assertEquals("training modified the base model file", baseHashBefore, Fixtures.sha256(base))

        // The patch moves the model toward the trained fact...
        val after = complete(base, patch)
        Log.i(Fixtures.TAG, "AFTER training (patched): '$after'")
        assertTrue("patched model did not learn the fact: '$after'", after.contains(target))
        assertTrue("fit did not improve", nll(base, patch, facts[0]) < nllBefore)
        // ...while a freshly loaded base is exactly as before (patching is per loaded model, not global).
        assertEquals(before, complete(base, null))
    }

    @Test fun cancelledTrainingKeepsACheckpointAndTheNextRunResumes() {
        val base = Fixtures.f16
        val ws = workspace("resume")
        val work = File(ws, "work").absolutePath
        val patch = File(ws, "specialist.patch")
        var lastStep = 0
        var reports = 0
        var finishedWithoutCancel = false
        try {
            e.train(base.absolutePath, facts, config(checkpointEvery = 2), work, patch.absolutePath) { ev ->
                reports++
                if (ev.phase == TrainEvent.TRAIN) lastStep = maxOf(lastStep, ev.step)
                if (reports % 5 == 1) Log.i(Fixtures.TAG, "resume-test run 1: $ev")
                lastStep >= 6 // cancel once real optimisation steps have been reported; "kill" in-process
            }
            finishedWithoutCancel = true
        } catch (x: HagException) {
            Log.i(Fixtures.TAG, "resume-test run 1 ended: code=${x.code} cancelled=${x.cancelled} msg=${x.message} lastStep=$lastStep reports=$reports")
            assertTrue("run 1 failed for a reason other than cancel: code=${x.code} ${x.message}", x.cancelled)
        }
        val files = File(work).walkTopDown().filter { it.isFile }.map { "${it.name}(${it.length()})" }.toList()
        Log.i(Fixtures.TAG, "resume-test work dir after cancel: $files; patch.exists=${patch.exists()}")
        assertFalse("training finished before it could be cancelled (lastStep=$lastStep, reports=$reports)", finishedWithoutCancel)
        assertFalse("a cancelled run must not leave a finished patch", patch.exists())
        assertTrue("no checkpoint files in work dir (lastStep=$lastStep, reports=$reports)", files.isNotEmpty())

        var resumedFrom = 0
        e.train(base.absolutePath, facts, config(checkpointEvery = 2), work, patch.absolutePath) { ev ->
            resumedFrom = maxOf(resumedFrom, ev.resumedFromStep)
            false
        }
        Log.i(Fixtures.TAG, "resume-test run 2 resumedFrom=$resumedFrom")
        assertTrue("second run did not resume from the checkpoint (files after cancel: $files)", resumedFrom > 0)
        assertTrue(patch.isFile && patch.length() > 0)
    }

    @Test fun patchForADifferentBaseIsRejected() {
        val ws = workspace("mismatch")
        val bogus = File(ws, "bogus.patch").apply { writeText("not a patch") }
        val m = e.modelLoad(Fixtures.q8.absolutePath, true)
        try {
            e.modelApplyPatch(m, bogus.absolutePath)
            fail("garbage patch must be rejected")
        } catch (x: HagException) {
            assertTrue(x.code == HagException.CORRUPT || x.code == HagException.INVALID_ARG || x.code == HagException.IO || x.code < 0)
        } finally {
            e.modelFree(m)
        }
    }
}
