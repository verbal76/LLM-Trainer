package com.hotatticgames.llmtrainer.engine

import com.hotatticgames.llmtrainer.hostapi.EngineException
import com.hotatticgames.llmtrainer.studio.api.BackendError
import com.hotatticgames.llmtrainer.studio.api.BackendException
import com.hotatticgames.llmtrainer.studio.api.CancelToken
import com.hotatticgames.llmtrainer.studio.api.ChatMessage
import com.hotatticgames.llmtrainer.studio.api.SamplingParams
import com.hotatticgames.llmtrainer.studio.api.StopReason
import com.hotatticgames.llmtrainer.studio.api.TrainEvent
import com.hotatticgames.llmtrainer.studio.api.TrainParams
import com.hotatticgames.llmtrainer.studio.api.TrainPhase
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicReference
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class HostEngineBackendsTest {
    private val fake = FakeEngine()
    private val inf = HostInferenceBackend(fake)
    private val tr = HostTrainingBackend(fake)

    private fun code(f: () -> Unit) = assertFailsWith<BackendException> { f() }.code

    // ---- status / unavailable -------------------------------------------------------------------------------------

    @Test fun statusReportsTheEngineBuildOrTheHostsReason() {
        assertEquals(true, inf.status().available)
        assertEquals("hag-engine 1; llama.cpp 0c1e570", inf.status().runtimeId)
        assertEquals(true, tr.status().available)
        val none = HostInferenceBackend(null, "cpuGate: CPU lacks dotprod")
        assertEquals(false, none.status().available)
        assertEquals("cpuGate: CPU lacks dotprod", none.status().reason)
        assertEquals("none", none.status().runtimeId)
        assertEquals(BackendError.UNAVAILABLE, code { none.loadModel("/m.gguf") })
        assertEquals(BackendError.UNAVAILABLE, code { HostTrainingBackend(null, "x").estimate("/m", TrainParams()) })
        assertEquals(BackendError.UNAVAILABLE, code { HostTrainingBackend(null, "x").train("/m", listOf("a"), TrainParams(), "/w", "/p", CancelToken()) {} })
    }

    // ---- load / patch / handles -----------------------------------------------------------------------------------

    @Test fun loadsWithMmapAppliesThePatchBeforeSessionsAndReportsTheMeasuredLoadTime() {
        fake.loadDelayMs = 30
        val m = inf.loadModel("/data/base.gguf", "/data/spec.hagpatch")
        assertEquals(listOf("modelLoad(/data/base.gguf,mmap=true)", "applyPatch(/data/spec.hagpatch)"), fake.calls)
        val info = inf.modelInfo(m)
        assertTrue(info.patched)
        assertEquals("llama", info.arch); assertEquals(30, info.nLayer); assertEquals(7, info.fileType); assertEquals(8192, info.nCtxTrain)
        assertTrue(info.hasChatTemplate); assertEquals(135_000_000L, info.nParams); assertEquals(145_000_000L, info.sizeBytes)
        assertTrue(info.loadMs >= 30, "load time is measured around load + patch apply, was ${info.loadMs}")
        val plain = inf.loadModel("/data/base.gguf")
        assertFalse(inf.modelInfo(plain).patched)
    }

    @Test fun aFailedPatchFreesTheModelAndNeverLeaksAHandle() {
        fake.failPatch = EngineException(-7, "patch does not belong to this base model")
        val e = assertFailsWith<BackendException> { inf.loadModel("/b.gguf", "/p.hagpatch") }
        assertEquals(BackendError.CORRUPT, e.code)
        assertTrue(e.message!!.contains("base model"))
        assertTrue(fake.liveModels.isEmpty(), "the base handle was released")
        assertEquals(0, inf.openModelCount())
    }

    @Test fun loadFailuresMapEveryEngineCode() {
        val map = mapOf(-1 to BackendError.INVALID_ARG, -2 to BackendError.IO, -3 to BackendError.BAD_MODEL, -4 to BackendError.OOM, -5 to BackendError.CANCELLED,
            -6 to BackendError.UNSUPPORTED, -7 to BackendError.CORRUPT, -8 to BackendError.INTERNAL, -100 to BackendError.INVALID_ARG, -999 to BackendError.INTERNAL)
        for ((c, want) in map) {
            fake.failLoad = EngineException(c, "m$c")
            val e = assertFailsWith<BackendException> { inf.loadModel("/x") }
            assertEquals(want, e.code, "engine code $c"); assertEquals("m$c", e.message)
        }
    }

    @Test fun closingFreesTheEngineHandlesAndUsingAClosedModelIsAnInvalidArgument() {
        val m = inf.loadModel("/b.gguf")
        val c = inf.newChat(m, 2048, 3)
        assertTrue("sessionNew(ctx=2048,threads=3,batch=0)" in fake.calls)
        inf.closeChat(c); inf.closeModel(m)
        assertTrue(fake.liveModels.isEmpty() && fake.liveSessions.isEmpty())
        assertEquals(BackendError.INVALID_ARG, code { inf.modelInfo(m) })
        assertEquals(BackendError.INVALID_ARG, code { inf.newChat(m, 256) })
        inf.closeModel(m); inf.closeChat(c)                      // closing twice is harmless
    }

    @Test fun chatFormatPassesRolesAndContentsAndSurfacesUnsupportedTemplatesHonestly() {
        val m = inf.loadModel("/b.gguf")
        val s = inf.chatFormat(m, listOf(ChatMessage("system", "be brief"), ChatMessage("user", "hi")), true)
        assertEquals("<|system|>be brief\n<|user|>hi\n<|assistant|>", s)
        assertEquals("<|user|>hi\n", inf.chatFormat(m, listOf(ChatMessage("user", "hi")), false))
        fake.failChatFormat = EngineException(-6, "the model's chat template is not a built-in template")
        assertEquals(BackendError.UNSUPPORTED, code { inf.chatFormat(m, listOf(ChatMessage("user", "x")), true) })
    }

    // ---- generate / score / cancel --------------------------------------------------------------------------------

    @Test fun generateStreamsCompletePiecesAndParsesTheStats() {
        val m = inf.loadModel("/b.gguf"); val c = inf.newChat(m, 1024)
        val got = ArrayList<String>()
        val st = inf.generate(c, "prompt", SamplingParams(0.7f, 40, 0.95f, 0.05f, 1.1f, 42, 64), CancelToken()) { got.add(it); false }
        assertEquals(listOf("Hello", " ", "world"), got)
        assertEquals(7, st.promptTokens); assertEquals(3, st.generatedTokens); assertEquals(12.5, st.promptMs); assertEquals(30.0, st.genMs)
        assertEquals(StopReason.END, st.stopReason); assertEquals(123456789L, st.peakRssBytes)
        assertEquals(100.0, st.tokensPerSecond, 1e-9)
        assertTrue(fake.calls.any { it == "generate(t=0.7,k=40,p=0.95,min=0.05,rep=1.1,seed=42,max=64)" }, fake.calls.toString())
    }

    @Test fun aSinkThatStopsStopsTheEngineAndTheStopReasonIsCancelled() {
        val m = inf.loadModel("/b.gguf"); val c = inf.newChat(m, 1024)
        var n = 0
        val st = inf.generate(c, "p", SamplingParams(), CancelToken()) { n++; n >= 2 }
        assertEquals(2, n); assertEquals(StopReason.CANCELLED, st.stopReason)
    }

    @Test fun cancellingMidGenerationCallsEngineCancelUntilItReturns() {
        fake.blockUntilCancelled = true
        val m = inf.loadModel("/b.gguf"); val c = inf.newChat(m, 1024)
        val token = CancelToken()
        val out = AtomicReference<Any?>()
        val t = Thread { out.set(try { inf.generate(c, "p", SamplingParams(), token) { false } } catch (e: Throwable) { e }) }
        t.start()
        while (fake.calls.none { it.startsWith("generate(") }) Thread.sleep(2)
        Thread.sleep(40)
        token.cancel()
        t.join(4000)
        assertFalse(t.isAlive, "generate must return after cancel")
        val st = out.get() as com.hotatticgames.llmtrainer.studio.api.GenStats
        assertEquals(StopReason.CANCELLED, st.stopReason)
        assertTrue(fake.cancelCalls.get() >= 1)
        val before = fake.cancelCalls.get()
        Thread.sleep(80)
        assertEquals(before, fake.cancelCalls.get(), "the watcher stops once the call returned")
    }

    @Test fun aCancelledPromptPhaseIsAStopNotAFailure() {
        fake.blockUntilCancelled = true; fake.cancelThrows = true
        val m = inf.loadModel("/b.gguf"); val c = inf.newChat(m, 1024)
        val token = CancelToken()
        val out = AtomicReference<Any?>()
        val t = Thread { out.set(try { inf.generate(c, "p", SamplingParams(), token) { false } } catch (e: Throwable) { e }) }
        t.start()
        while (fake.calls.none { it.startsWith("generate(") }) Thread.sleep(2)
        token.cancel(); t.join(4000)
        assertEquals(StopReason.CANCELLED, (out.get() as com.hotatticgames.llmtrainer.studio.api.GenStats).stopReason)
        // a real failure still fails, even with a token that was never cancelled
        fake.blockUntilCancelled = false; fake.failGenerate = EngineException(-4, "out of memory")
        assertEquals(BackendError.OOM, code { inf.generate(c, "p", SamplingParams(), CancelToken()) { false } })
    }

    @Test fun anAlreadyCancelledTokenDoesNotTouchTheEngine() {
        val m = inf.loadModel("/b.gguf"); val c = inf.newChat(m, 1024)
        val token = CancelToken().also { it.cancel() }
        val st = inf.generate(c, "p", SamplingParams(), token) { false }
        assertEquals(StopReason.CANCELLED, st.stopReason)
        assertTrue(fake.calls.none { it.startsWith("generate(") })
        assertEquals(BackendError.CANCELLED, code { inf.score(c, "text", token) })
    }

    @Test fun scoreParsesAndCancelMapsToCancelled() {
        val m = inf.loadModel("/b.gguf"); val c = inf.newChat(m, 1024)
        val r = inf.score(c, "held-out text", CancelToken())
        assertEquals(2.25, r.meanNll); assertEquals(9, r.nTokens)
        fake.blockScoreUntilCancelled = true
        val token = CancelToken()
        val out = AtomicReference<Any?>()
        val t = Thread { out.set(try { inf.score(c, "x", token) } catch (e: Throwable) { e }) }
        t.start()
        while (fake.calls.count { it.startsWith("score(") } < 2) Thread.sleep(2)
        token.cancel(); t.join(4000)
        assertEquals(BackendError.CANCELLED, (out.get() as BackendException).code)
    }

    @Test fun garbledEngineJsonBecomesAnInternalErrorNeverACrash() {
        val m = inf.loadModel("/b.gguf"); val c = inf.newChat(m, 1024)
        fake.scoreJson = "not json"
        assertEquals(BackendError.INTERNAL, code { inf.score(c, "x", CancelToken()) })
    }

    // ---- training -------------------------------------------------------------------------------------------------

    private val lora = TrainParams(contextTokens = 256, batchTokens = 256, epochs = 2, learningRate = 2e-3f, seed = 7, trainableLastLayers = 0, loraRank = 8, loraAlpha = 16f, maxMemoryBytes = 3_000_000_000L)

    @Test fun estimateSendsEveryKeyIncludingLoraAndPinsTheThreadCount() {
        fake.estimateJson = """{"trainable":true,"lora":true,"lora_rank":8,"trainable_params":331776,"estimated_peak_bytes":412000000}"""
        val e = tr.estimate("/b.gguf", lora)
        assertTrue(e.trainable); assertEquals(412_000_000L, e.peakBytes); assertEquals(331776L, e.trainableParams); assertNull(e.reason)
        val c = JSONObject(fake.lastEstimateConfig!!)
        assertEquals(256, c.getInt("n_ctx")); assertEquals(2, c.getInt("epochs")); assertEquals(7, c.getLong("seed")); assertEquals(0, c.getInt("trainable_last_layers"))
        assertEquals(8, c.getInt("lora_rank")); assertEquals(16.0, c.getDouble("lora_alpha")); assertEquals(0, c.getInt("train_embeddings"))
        assertEquals(3_000_000_000L, c.getLong("max_memory_bytes")); assertEquals(0.002, c.getDouble("learning_rate"), 1e-9)
        assertEquals(6, c.getInt("n_threads"), "auto threads are resolved once from n_cores=8 (3/4 of the cores) so resume fingerprints match")
        val explicit = JSONObject(HostTrainingBackend(fake).configJson(lora.copy(threads = 3)))
        assertEquals(3, explicit.getInt("n_threads"))
    }

    @Test fun estimateOfAnUntrainableModelCarriesTheEnginesReasonAndNeverThrows() {
        fake.estimateJson = """{"trainable":false,"reason":"architecture 'gemma' is not validated for training","estimated_peak_bytes":0}"""
        val e = tr.estimate("/b.gguf", TrainParams())
        assertFalse(e.trainable); assertEquals("architecture 'gemma' is not validated for training", e.reason)
    }

    @Test fun anEngineThatIgnoresTheLoraRequestIsRefusedNotSilentlyFullTuned() {
        fake.estimateJson = """{"trainable":true,"lora":false,"trainable_params":300000000,"estimated_peak_bytes":9000000000}"""
        val e = tr.estimate("/b.gguf", lora)
        assertFalse(e.trainable)
        assertTrue(e.reason!!.contains("Refusing"), e.reason)
        val x = assertFailsWith<BackendException> { tr.train("/b.gguf", listOf("a"), lora, "/w", "/p", CancelToken()) {} }
        assertEquals(BackendError.UNSUPPORTED, x.code)
        assertTrue(fake.calls.none { it.startsWith("train(") }, "nothing was trained")
        // a weight-tuning request is unaffected
        assertTrue(tr.estimate("/b.gguf", lora.copy(loraRank = 0)).trainable)
    }

    @Test fun trainMapsEventsPassesTextsAndConfigAndCancelsThroughTheProgressCallback() {
        fake.estimateJson = """{"trainable":true,"lora":true,"lora_rank":8,"trainable_params":1,"estimated_peak_bytes":1}"""
        val nan = Double.NaN
        fake.trainEvents = listOf(
            doubleArrayOf(0.0, 1.0, 2.0, 0.0, 10.0, 0.0, nan, nan, 0.0, 800e6, 0.0),
            doubleArrayOf(1.0, 1.0, 2.0, 3.0, 10.0, 12.0, 2.5, nan, 4.5, 900e6, 0.0),
            doubleArrayOf(1.0, 2.0, 2.0, 6.0, 10.0, 24.0, 1.75, 2.0, 9.0, 900e6, 3.0),
            doubleArrayOf(3.0, 2.0, 2.0, 10.0, 10.0, 40.0, nan, nan, 14.0, 900e6, 3.0),
            doubleArrayOf(4.0, 2.0, 2.0, 10.0, 10.0, 40.0, nan, nan, 15.0, 900e6, 3.0),
        )
        val seen = ArrayList<TrainEvent>()
        tr.train("/b.gguf", listOf("one", "two"), lora, "/work", "/out.patch", CancelToken()) { seen.add(it) }
        assertEquals(listOf(TrainPhase.PREPARE, TrainPhase.TRAIN, TrainPhase.TRAIN, TrainPhase.SAVE, TrainPhase.DONE), seen.map { it.phase })
        assertNull(seen[0].trainLoss); assertEquals(2.5, seen[1].trainLoss); assertNull(seen[1].valLoss); assertEquals(2.0, seen[2].valLoss)
        assertEquals(3, seen[2].resumedFromStep); assertEquals(24L, seen[2].examplesDone); assertEquals(900_000_000L, seen[2].rssBytes)
        assertEquals(listOf("one", "two"), fake.lastTexts)
        assertEquals(8, JSONObject(fake.lastTrainConfig!!).getInt("lora_rank"))
        assertTrue(fake.progressCancelAnswers.all { !it })

        // cancel from the token: the callback returns true, the engine throws HAG_ERR_CANCELLED and the seam says CANCELLED
        val token = CancelToken()
        fake.progressCancelAnswers.clear()
        val ex = assertFailsWith<BackendException> { tr.train("/b.gguf", listOf("a"), lora, "/w", "/o", token) { if (it.step >= 3) token.cancel() } }
        assertEquals(BackendError.CANCELLED, ex.code)
        assertEquals(true, fake.progressCancelAnswers.last())
        assertEquals(BackendError.CANCELLED, code { tr.train("/b.gguf", listOf("a"), lora, "/w", "/o", CancelToken().also { it.cancel() }) {} })
    }

    @Test fun trainFailuresKeepTheirMeaning() {
        fake.trainFailure = EngineException(-4, "estimate exceeds max_memory_bytes")
        assertEquals(BackendError.OOM, code { tr.train("/b", listOf("a"), TrainParams(), "/w", "/o", CancelToken()) {} })
        fake.trainFailure = EngineException(-7, "checkpoint failed validation")
        assertEquals(BackendError.CORRUPT, code { tr.train("/b", listOf("a"), TrainParams(), "/w", "/o", CancelToken()) {} })
        fake.trainFailure = EngineException(-6, "quantized weights cannot be trained")
        assertEquals(BackendError.UNSUPPORTED, code { tr.train("/b", listOf("a"), TrainParams(), "/w", "/o", CancelToken()) {} })
    }

    @Test fun patchInfoParsesBothPatchKinds() {
        val a = tr.patchInfo("/p.hagpatch")
        assertEquals("abc123", a.baseSha256); assertEquals(12, a.steps); assertEquals(2, a.epochs); assertEquals("d5", a.datasetHash)
        assertEquals(listOf("blk.29.attn_q.weight", "output_norm.weight"), a.tensorNames); assertEquals(4096L, a.sizeBytes)
        assertNotNull(JSONObject(a.paramsJson).optInt("steps"))
        fake.patchInfo = """{"format":"hag-patch","kind":"lora","file_bytes":99,"base":{"sha256":"ff"},"train":{"steps":5,"epochs":1,"lora_rank":8},"tensors":[{"name":"blk.0.attn_q.weight.lora_a"},{"name":"blk.0.attn_q.weight.lora_b"}]}"""
        val b = tr.patchInfo("/lora.gguf")
        assertEquals("ff", b.baseSha256); assertNull(b.datasetHash); assertEquals(2, b.tensorNames.size); assertTrue(b.tensorNames[0].endsWith(".lora_a"))
        assertEquals(8, JSONObject(b.paramsJson).getInt("lora_rank"))
    }

    // ---- wiring ---------------------------------------------------------------------------------------------------

    @Test fun wiringFollowsTheAdvertisedCapabilitiesAndNeverFakesAnything() {
        val full = EngineWiring.create(fake, null, setOf("core.v1", EngineWiring.INFERENCE_CAP, EngineWiring.TRAINING_CAP))
        assertTrue(full.inference.status().available && full.trainer.status().available)
        val none = EngineWiring.create(null, "cpuGate: CPU lacks dotprod", setOf("core.v1"))
        assertFalse(none.inference.status().available); assertEquals("cpuGate: CPU lacks dotprod", none.inference.status().reason)
        assertFalse(none.trainer.status().available); assertEquals("cpuGate: CPU lacks dotprod", none.trainer.status().reason)
        val inferOnly = EngineWiring.create(fake, null, setOf(EngineWiring.INFERENCE_CAP))
        assertTrue(inferOnly.inference.status().available)
        assertFalse(inferOnly.trainer.status().available)
        assertTrue(inferOnly.trainer.status().reason!!.contains(EngineWiring.TRAINING_CAP))
        // capabilities unreadable: the engine's presence decides
        assertTrue(EngineWiring.create(fake, null, null).trainer.status().available)
        assertFalse(EngineWiring.create(null, null, null).inference.status().available)
    }

    @Test fun capabilitiesAreReadFromTheHostDiagnostics() {
        assertEquals(setOf("core.v1", "inference.gguf.v1"), EngineWiring.capabilitiesFrom("""{"capabilities":["core.v1","inference.gguf.v1"],"x":1}"""))
        assertNull(EngineWiring.capabilitiesFrom("""{"x":1}"""))
        assertNull(EngineWiring.capabilitiesFrom("not json"))
    }
}
