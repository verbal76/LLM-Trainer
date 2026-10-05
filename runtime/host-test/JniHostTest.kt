package com.hotatticgames.hag.runtime

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/** Host-JVM exercise of the REAL libhagrt.so JNI code against a stub libhagengine.so (marshalling, callbacks, errors). */
class JniHostTest {
    private val status = HagRuntime.load()
    private val e: HagEngine get() = (status as? EngineStatus.Ready)?.engine ?: error("not ready: ${(status as EngineStatus.Unavailable).reason}")

    @Test fun comesUp() {
        assertTrue(status is EngineStatus.Ready, (status as? EngineStatus.Unavailable)?.reason ?: "")
        assertEquals("hag-engine 1; stub", (status as EngineStatus.Ready).version)
        println("abi=" + HagRuntime.cpuAbi())
    }

    @Test fun loadProgressAndCancelAndErrors() {
        val seen = mutableListOf<Float>()
        val m = e.modelLoad("/x/model.gguf", true) { seen += it; false }
        assertEquals(listOf(0f, 0.5f, 1f), seen)
        assertTrue(e.modelInfoJson(m).contains("\"patched\":false"))
        e.modelFree(m)
        val c = assertFailsWith<HagException> { e.modelLoad("/x/model.gguf") { true } }
        assertTrue(c.cancelled); assertEquals("load cancelled", c.message)
        val io = assertFailsWith<HagException> { e.modelLoad("/nonexistent/m.gguf") }
        assertEquals(HagException.IO, io.code); assertEquals("cannot open file", io.message)
        assertFailsWith<HagException> { e.modelInfoJson(m) } // freed
        assertEquals(0, e.liveHandles())
    }

    @Test fun sessionParamsAndPatch() {
        val m = e.modelLoad("/x/m.gguf")
        val s = e.sessionNew(m, nCtx = 777, nThreads = 3, nBatch = 5)
        assertFailsWith<HagException> { e.modelApplyPatch(m, "/p/good.patch") } // live session
        e.sessionFree(s)
        e.modelApplyPatch(m, "/p/good.patch")
        assertTrue(e.modelInfoJson(m).contains("\"patched\":true"))
        assertEquals(HagException.CORRUPT, assertFailsWith<HagException> { e.modelApplyPatch(m, "/p/bad.patch") }.code)
        e.modelFree(m)
    }

    @Test fun chatFormatAndUnicode() {
        val m = e.modelLoad("/x/m.gguf")
        val t = e.chatFormat(m, listOf("system", "user"), listOf("sys 😀", "héllo 你好"), true)
        assertEquals("<|system|>sys 😀\n<|user|>héllo 你好\n<|assistant|>", t)
        val toks = e.tokenize(m, "hé😀", false)
        assertEquals("hé😀".toByteArray().size, toks.size) // required-size retry path not needed: cap = len+16
        assertEquals(0, e.tokenize(m, "", false).size)
        val big = "x".repeat(10_000)
        assertEquals(10_000, e.tokenize(m, big, false).size)
        e.modelFree(m)
    }

    @Test fun streamingUtf8SeedAndStop() {
        val m = e.modelLoad("/x/m.gguf"); val s = e.sessionNew(m)
        val prompt = "🚀 café こん"
        val out = StringBuilder(); val pieces = mutableListOf<String>()
        val stats = e.generate(s, prompt, SampleParams(0.5f, 40, 0.9f, 0.05f, 1.1f, 4242, 6)) { pieces += it; out.append(it); false }
        assertEquals(prompt, pieces[0]) // input survived UTF-8 round trip incl. supplementary char
        assertTrue(pieces[1].contains("seed=4242 max=6") && pieces[1].contains("k=40"), pieces[1])
        assertEquals("😀", pieces[2])
        assertTrue(stats.contains("\"n_generated\":6") && stats.contains("\"stop_reason\":1") && stats.contains("12345"), stats)
        // stop from sink
        var n = 0
        val st2 = e.generate(s, "go", SampleParams(maxNewTokens = 100)) { n++; n >= 3 }
        assertEquals(3, n); assertTrue(st2.contains("\"stop_reason\":2"), st2)
        e.modelFree(m); e.sessionFree(s)
        assertEquals(0, e.liveHandles())
    }

    @Test fun sinkExceptionIsRethrownAndEngineSurvives() {
        val m = e.modelLoad("/x/m.gguf"); val s = e.sessionNew(m)
        val ex = assertFailsWith<IllegalStateException> { e.generate(s, "hi", SampleParams(maxNewTokens = 10)) { throw IllegalStateException("boom") } }
        assertEquals("boom", ex.message)
        e.generate(s, "hi", SampleParams(maxNewTokens = 2)) { false }
        e.sessionFree(s); e.modelFree(m)
    }

    @Test fun cancelFromAnotherThread() {
        val m = e.modelLoad("/x/m.gguf"); val s = e.sessionNew(m)
        val started = java.util.concurrent.CountDownLatch(1)
        var stats: String? = null
        val t = Thread { stats = e.generate(s, "SLOW", SampleParams(maxNewTokens = 100000)) { started.countDown(); false } }
        t.start(); assertTrue(started.await(10, java.util.concurrent.TimeUnit.SECONDS))
        e.cancel(s); t.join(10_000)
        assertTrue(!t.isAlive); assertTrue(stats!!.contains("\"stop_reason\":2"), stats)
        e.sessionFree(s); e.modelFree(m)
    }

    @Test fun freeInsideSinkIsDeferred() {
        val m = e.modelLoad("/x/m.gguf"); val s = e.sessionNew(m)
        e.generate(s, "hi", SampleParams(maxNewTokens = 5)) { e.sessionFree(s); false } // would be use-after-free without deferral
        e.modelFree(m)
        assertEquals(0, e.liveHandles())
    }

    @Test fun scoreEstimateTrainAndPatchInfo() {
        val m = e.modelLoad("/x/m.gguf"); val s = e.sessionNew(m)
        assertEquals("{\"mean_nll\":1.5,\"n_tokens\":5}", e.score(s, "hello"))
        val cfg = TrainConfig(nCtx = 321, nBatch = 64, epochs = 9, learningRate = 0.00025f, valFraction = 0.125f, seed = 4000000000, nThreads = 2,
            trainableLastLayers = 4, trainEmbeddings = true, checkpointEverySteps = 11, maxMemoryBytes = 3_000_000_000L)
        assertEquals("{\"trainable\":true,\"bytes\":321000}", e.trainEstimate("/b.gguf", cfg))
        val dir = File(System.getProperty("java.io.tmpdir"), "hagjni-${System.nanoTime()}").apply { mkdirs() }
        val out = File(dir, "p.patch"); val evs = mutableListOf<TrainEvent>()
        e.train("/b.gguf", listOf("fact 😀 one", "last é"), cfg, dir.path, out.path) { evs += it; false }
        assertEquals(3, evs.size)
        assertEquals(TrainEvent(1, 1, 3, 0, 3, 100, 2.0, evs[0].valLoss, 0.0, 1L shl 20, 7), evs[0])
        assertTrue(evs[0].valLoss.isNaN()); assertEquals(TrainEvent.DONE, evs[2].phase)
        val line = out.readText().trim()
        assertEquals("n=2 first=fact 😀 one last=last é ctx=321 batch=64 ep=9 lr=0.000250 val=0.125 seed=4000000000 thr=2 last=4 emb=1 ck=11 mem=3000000000 lora=0 alpha=0.000", line)
        // LoRA parameters cross the JNI boundary into hag_train_params.lora_rank / lora_alpha
        val lora = cfg.copy(loraRank = 8, loraAlpha = 12.5f)
        e.train("/b.gguf", listOf("x", "y"), lora, dir.path, out.path) { false }
        assertTrue(out.readText().trim().endsWith("mem=3000000000 lora=8 alpha=12.500"), out.readText())
        val c = assertFailsWith<HagException> { e.train("/b.gguf", listOf("a"), cfg, dir.path, out.path) { true } }
        assertTrue(c.cancelled)
        assertTrue(e.patchInfoJson("/p/x").contains("/p/x"))
        e.sessionFree(s); e.modelFree(m)
    }
}
