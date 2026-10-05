package com.hotatticgames.hag.runtime

import android.util.Log
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/** REAL engine + REAL GGUF (SmolLM2-135M-Instruct Q8_0 in CI). Nothing mocked. */
class EngineInferenceTest {
    private val e get() = EngineHolder.engine
    private val models = mutableListOf<Long>()
    private val sessions = mutableListOf<Long>()

    private fun load(progress: ((Float) -> Boolean)? = null): Long =
        e.modelLoad(Fixtures.q8.absolutePath, true, progress).also { models += it }

    private fun session(m: Long, nCtx: Int = 512): Long = e.sessionNew(m, nCtx = nCtx).also { sessions += it }

    @After fun cleanup() {
        sessions.forEach { e.sessionFree(it) }
        models.forEach { e.modelFree(it) }
    }

    private fun greedy(s: Long, prompt: String, n: Int = 24): Pair<String, JSONObject> {
        e.sessionReset(s)
        val sb = StringBuilder()
        val stats = e.generate(s, prompt, SampleParams(temperature = 0f, maxNewTokens = n)) { sb.append(it); false }
        return sb.toString() to JSONObject(stats)
    }

    @Test fun engineComesUpAndReportsItsIdentity() {
        val st = EngineHolder.status
        assertTrue("engine status: " + (st as? EngineStatus.Unavailable)?.let { "${it.stage}: ${it.reason}" }, st is EngineStatus.Ready)
        val ready = st as EngineStatus.Ready
        Log.i(Fixtures.TAG, "engine version: ${ready.version}")
        Log.i(Fixtures.TAG, "system info: ${ready.systemInfoJson}")
        assertTrue(ready.version, ready.version.startsWith("hag-engine"))
        val info = JSONObject(ready.systemInfoJson)
        assertEquals(HagRuntime.cpuAbi(), info.getString("abi"))
    }

    @Test fun loadReportsMonotonicProgressAndModelInfo() {
        val seen = mutableListOf<Float>()
        val m = load { seen += it; false }
        assertTrue("progress callback never fired", seen.isNotEmpty())
        assertTrue("progress not monotonic: $seen", seen.zipWithNext().all { (a, b) -> b >= a - 1e-6f })
        assertTrue(seen.all { it in 0f..1.0001f })
        val info = JSONObject(e.modelInfoJson(m))
        Log.i(Fixtures.TAG, "model info: $info")
        assertTrue(info.getLong("n_params") > 100_000_000L)
    }

    @Test fun cancelledLoadThrowsCancelledAndLeaksNothing() {
        val before = e.liveHandles()
        try {
            e.modelLoad(Fixtures.q8.absolutePath, true) { true }
            fail("load should have been cancelled")
        } catch (x: HagException) {
            assertTrue("code=${x.code} ${x.message}", x.cancelled)
        }
        assertEquals(before, e.liveHandles())
    }

    @Test fun badInputsFailCleanlyWithAMessage() {
        try {
            e.modelLoad("/nonexistent/model.gguf")
            fail()
        } catch (x: HagException) {
            assertTrue(x.code < 0 && !x.message.isNullOrBlank())
        }
        val m = load()
        e.modelFree(m)
        e.modelFree(m) // idempotent
        try {
            e.modelInfoJson(m)
            fail("freed handle must be rejected")
        } catch (x: HagException) {
            assertEquals(HagException.USAGE, x.code)
        }
    }

    @Test fun chatFormatUsesTheModelsOwnTemplate() {
        val m = load()
        val text = e.chatFormat(m, listOf("system", "user"), listOf("You are terse.", "What is 2+2?"), true)
        Log.i(Fixtures.TAG, "chat template output: ${text.replace("\n", "\\n")}")
        assertTrue(text.contains("What is 2+2?") && text.contains("You are terse."))
        assertTrue("template markers expected", text.length > "You are terse.What is 2+2?".length)
        val noGen = e.chatFormat(m, listOf("user"), listOf("hi"), false)
        assertTrue(text.length > noGen.length || text != noGen)
    }

    @Test fun greedyGenerationIsNonEmptyAndDeterministic() {
        val m = load()
        val s = session(m)
        val prompt = e.chatFormat(m, listOf("user"), listOf("Name one primary colour."), true)
        val (a, sa) = greedy(s, prompt)
        val (b, sb) = greedy(s, prompt)
        Log.i(Fixtures.TAG, "greedy: '$a' stats=$sa")
        assertTrue("empty generation", a.isNotBlank())
        assertEquals("greedy decoding must be deterministic across a reset", a, b)
        assertTrue(sa.getInt("n_generated") > 0 && sa.getInt("n_prompt_tokens") > 0)
        assertTrue(sa.getInt("stop_reason") in 0..1)
        assertEquals(sa.getInt("n_generated"), sb.getInt("n_generated"))
        assertTrue(sa.getDouble("gen_ms") > 0)
    }

    @Test fun seededSamplingIsReproducible() {
        val m = load()
        val s = session(m)
        fun run(seed: Long): String {
            e.sessionReset(s)
            val sb = StringBuilder()
            e.generate(s, "Once upon a time", SampleParams(temperature = 0.9f, topK = 40, topP = 0.95f, seed = seed, maxNewTokens = 24)) { sb.append(it); false }
            return sb.toString()
        }
        val a = run(42); val b = run(42)
        assertTrue(a.isNotBlank())
        assertEquals(a, b)
        Log.i(Fixtures.TAG, "seed42='$a' seed43='${run(43)}'")
    }

    @Test fun sinkCanStopGenerationMidStream() {
        val m = load()
        val s = session(m)
        var pieces = 0
        val stats = JSONObject(
            e.generate(s, "Count from one to fifty: 1, 2, 3,", SampleParams(maxNewTokens = 200)) { pieces++; pieces >= 3 },
        )
        assertEquals(3, pieces)
        assertEquals("stop_reason cancelled", 2, stats.getInt("stop_reason"))
        // The session is reusable afterwards.
        assertTrue(greedy(s, "Hello", 8).first.isNotEmpty())
    }

    @Test fun cancelFromAnotherThreadStopsGeneration() {
        val m = load()
        val s = session(m)
        val started = CountDownLatch(1)
        val result = AtomicReference<Any>()
        val worker = Thread {
            try {
                result.set(JSONObject(e.generate(s, "Write a very long story about a dragon.", SampleParams(maxNewTokens = 400)) { started.countDown(); false }))
            } catch (t: Throwable) {
                result.set(t)
            }
        }
        worker.start()
        assertTrue("generation never produced a piece", started.await(120, TimeUnit.SECONDS))
        e.cancel(s)
        worker.join(60_000)
        assertFalse("worker still running after cancel", worker.isAlive)
        when (val r = result.get()) {
            is JSONObject -> { assertEquals(2, r.getInt("stop_reason")); assertTrue(r.getInt("n_generated") < 400) }
            is HagException -> assertTrue("code=${r.code}", r.cancelled)
            else -> fail("unexpected result $r")
        }
    }

    @Test fun exceptionInSinkStopsGenerationAndIsRethrown() {
        val m = load()
        val s = session(m)
        try {
            e.generate(s, "Hello", SampleParams(maxNewTokens = 32)) { throw IllegalStateException("boom from sink") }
            fail("expected the sink exception")
        } catch (x: IllegalStateException) {
            assertEquals("boom from sink", x.message)
        }
        assertTrue(greedy(s, "Hello", 8).first.isNotEmpty()) // engine still healthy
    }

    @Test fun unicodePromptsSurviveTheJniBoundary() {
        val m = load()
        val s = session(m)
        val prompt = "Emoji 😀 🚀 and CJK 你好 こんにちは and accents éè: "
        val toks = e.tokenize(m, prompt, false)
        assertTrue(toks.size >= 8)
        val (out, _) = greedy(s, prompt, 16)
        assertFalse("JNI/UTF-8 corruption", out.contains('\u0000'))
        Log.i(Fixtures.TAG, "unicode out: '$out'")
    }

    @Test fun tokenizeIsStableAndAddSpecialChangesNothingUnexpectedly() {
        val m = load()
        val a = e.tokenize(m, "The quick brown fox", false)
        assertTrue(a.isNotEmpty())
        assertTrue(a.contentEquals(e.tokenize(m, "The quick brown fox", false)))
        assertTrue(e.tokenize(m, "", false).isEmpty())
        val big = "word ".repeat(3000)
        assertTrue(e.tokenize(m, big, false).size >= 3000)
    }

    @Test fun scoreRanksFluentTextAboveGibberish() {
        val m = load()
        val s = session(m)
        val fluent = JSONObject(e.score(s, "The capital of France is Paris, and it lies on the river Seine."))
        val junk = JSONObject(e.score(s, "Seine river the on lies it and Paris, is France of capital The."))
        val gib = JSONObject(e.score(s, "qzx vvk plmm wrt zzq hh jjk xq vvv qq zt"))
        Log.i(Fixtures.TAG, "nll fluent=$fluent junk=$junk gib=$gib")
        assertTrue(fluent.getInt("n_tokens") > 5)
        assertTrue(fluent.getDouble("mean_nll") > 0)
        assertTrue(fluent.getDouble("mean_nll") < junk.getDouble("mean_nll"))
        assertTrue(fluent.getDouble("mean_nll") < gib.getDouble("mean_nll"))
        assertNotEquals(fluent.getDouble("mean_nll"), gib.getDouble("mean_nll"), 1e-6)
    }

    @Test fun manyLoadFreeCyclesDoNotLeakHandles() {
        val before = e.liveHandles()
        repeat(3) {
            val m = e.modelLoad(Fixtures.q8.absolutePath, true)
            val s = e.sessionNew(m, nCtx = 256)
            e.modelFree(m) // model freed first: the session keeps it alive until it is freed
            assertTrue(greedy(s, "Hi", 4).first.isNotEmpty())
            e.sessionFree(s)
        }
        assertEquals(before, e.liveHandles())
    }
}
