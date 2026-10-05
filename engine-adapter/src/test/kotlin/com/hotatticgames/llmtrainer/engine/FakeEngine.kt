package com.hotatticgames.llmtrainer.engine

import com.hotatticgames.llmtrainer.hostapi.EngineApi
import com.hotatticgames.llmtrainer.hostapi.EngineException
import com.hotatticgames.llmtrainer.hostapi.LoadProgress
import com.hotatticgames.llmtrainer.hostapi.TokenSink
import com.hotatticgames.llmtrainer.hostapi.TrainProgress
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

/**
 * A scripted EngineApi: records every call, enforces handle discipline like the real binding (a freed handle throws code -100) and lets
 * a test script answers, failures and blocking behaviour. It is NOT a model: it only exercises the adapter's mapping and cancel logic.
 */
class FakeEngine : EngineApi {
    val calls = java.util.Collections.synchronizedList(ArrayList<String>())
    private val ids = AtomicLong(0)
    val liveModels = ConcurrentHashMap.newKeySet<Long>()
    val liveSessions = ConcurrentHashMap.newKeySet<Long>()
    val modelOf = ConcurrentHashMap<Long, Long>()

    var versionString = "hag-engine 1; llama.cpp 0c1e570"
    var systemInfo = """{"abi":"arm64-v8a","n_cores":8}"""
    var loadDelayMs = 0L
    var failLoad: EngineException? = null
    var failPatch: EngineException? = null
    var failChatFormat: EngineException? = null
    var failGenerate: EngineException? = null
    var patchedModels = ConcurrentHashMap.newKeySet<Long>()
    var patchKind = "replace"

    var pieces = listOf("Hello", " ", "world")
    var statsJson = """{"n_prompt_tokens":7,"n_generated":3,"prompt_ms":12.5,"gen_ms":30.0,"stop_reason":0,"peak_rss_bytes":123456789}"""
    /** When true, generate blocks until the adapter calls cancel(session) (as the real engine would keep decoding). */
    var blockUntilCancelled = false
    /** When true, a cancel makes generate throw HAG_ERR_CANCELLED (cancelled prompt phase) instead of returning stats. */
    var cancelThrows = false
    val cancelCalls = java.util.concurrent.atomic.AtomicInteger(0)
    @Volatile private var cancelFlag = false
    var scoreJson = """{"mean_nll":2.25,"n_tokens":9}"""
    var blockScoreUntilCancelled = false

    var estimateJson = """{"trainable":true,"n_layer":30,"trainable_params":1000,"lora":false,"estimated_peak_bytes":5000000,"disk_total_bytes":1}"""
    var estimateFor: (String) -> String = { estimateJson }
    var trainFailure: EngineException? = null
    var trainEvents: List<DoubleArray> = emptyList()   // phase, epoch, epochs, step, steps, examples, trainLoss, valLoss, elapsed, rss, resumed
    var patchInfo = """{"format":"hag-patch","kind":"replace","file_bytes":4096,"base":{"sha256":"abc123"},"train":{"steps":12,"epochs":2,"dataset_sha256":"d5"},"tensors":[{"name":"blk.29.attn_q.weight"},{"name":"output_norm.weight"}]}"""
    var lastTrainConfig: String? = null
    var lastEstimateConfig: String? = null
    var lastTexts: List<String>? = null
    var progressCancelAnswers = ArrayList<Boolean>()

    private fun usage(what: String): Nothing = throw EngineException(-100, "bad or freed handle: $what")
    private fun needModel(m: Long) { if (m !in liveModels) usage("model $m") }
    private fun needSession(s: Long) { if (s !in liveSessions) usage("session $s") }

    override fun version() = versionString
    override fun systemInfoJson() = systemInfo

    override fun modelLoad(ggufPath: String, useMmap: Boolean, progress: LoadProgress?): Long {
        calls.add("modelLoad($ggufPath,mmap=$useMmap)")
        failLoad?.let { throw it }
        if (loadDelayMs > 0) Thread.sleep(loadDelayMs)
        val m = ids.incrementAndGet(); liveModels.add(m); return m
    }

    override fun modelApplyPatch(model: Long, patchPath: String) {
        needModel(model)
        calls.add("applyPatch($patchPath)")
        failPatch?.let { throw it }
        patchedModels.add(model)
    }

    override fun modelInfoJson(model: Long): String {
        needModel(model)
        return JSONObject().put("arch", "llama").put("n_params", 135_000_000L).put("n_layer", 30).put("n_embd", 576).put("n_vocab", 49152)
            .put("n_ctx_train", 8192).put("file_type", "7").put("size_bytes", 145_000_000L).put("chat_template", true)
            .put("patched", model in patchedModels).toString()
    }

    override fun modelFree(model: Long) { calls.add("modelFree($model)"); liveModels.remove(model) }

    override fun sessionNew(model: Long, nCtx: Int, nThreads: Int, nBatch: Int): Long {
        needModel(model)
        calls.add("sessionNew(ctx=$nCtx,threads=$nThreads,batch=$nBatch)")
        val s = ids.incrementAndGet(); liveSessions.add(s); modelOf[s] = model; return s
    }

    override fun sessionReset(session: Long) { needSession(session); calls.add("sessionReset") }
    override fun sessionFree(session: Long) { calls.add("sessionFree($session)"); liveSessions.remove(session) }

    override fun chatFormat(model: Long, roles: List<String>, contents: List<String>, addGenerationPrompt: Boolean): String {
        needModel(model)
        failChatFormat?.let { throw it }
        return roles.indices.joinToString("") { "<|${roles[it]}|>${contents[it]}\n" } + if (addGenerationPrompt) "<|assistant|>" else ""
    }

    override fun generate(
        session: Long, prompt: String, temperature: Float, topK: Int, topP: Float, minP: Float, repeatPenalty: Float,
        seed: Long, maxNewTokens: Int, sink: TokenSink,
    ): String {
        needSession(session)
        calls.add("generate(t=$temperature,k=$topK,p=$topP,min=$minP,rep=$repeatPenalty,seed=$seed,max=$maxNewTokens)")
        failGenerate?.let { throw it }
        cancelFlag = false
        for (p in pieces) if (sink(p)) return statsJson.replace("\"stop_reason\":0", "\"stop_reason\":2")
        if (blockUntilCancelled) {
            val end = System.currentTimeMillis() + 5000
            while (!cancelFlag && System.currentTimeMillis() < end) Thread.sleep(5)
            if (!cancelFlag) throw AssertionError("adapter never cancelled the engine")
            if (cancelThrows) throw EngineException(-5, "cancelled")
            return statsJson.replace("\"stop_reason\":0", "\"stop_reason\":2")
        }
        return statsJson
    }

    override fun cancel(session: Long) { cancelCalls.incrementAndGet(); cancelFlag = true }

    override fun tokenize(model: Long, text: String, addSpecial: Boolean): IntArray = IntArray(text.length) { it }

    override fun score(session: Long, text: String): String {
        needSession(session)
        calls.add("score(${text.length})")
        cancelFlag = false
        if (blockScoreUntilCancelled) {
            val end = System.currentTimeMillis() + 5000
            while (!cancelFlag && System.currentTimeMillis() < end) Thread.sleep(5)
            throw EngineException(-5, "cancelled")
        }
        return scoreJson
    }

    override fun trainEstimate(baseGguf: String, configJson: String): String { lastEstimateConfig = configJson; calls.add("trainEstimate"); return estimateFor(configJson) }

    override fun train(baseGguf: String, texts: List<String>, configJson: String, workDir: String, outPatch: String, progress: TrainProgress?) {
        calls.add("train($baseGguf)")
        lastTrainConfig = configJson; lastTexts = texts
        trainFailure?.let { throw it }
        for (e in trainEvents) {
            val cancel = progress!!(e[0].toInt(), e[1].toInt(), e[2].toInt(), e[3].toInt(), e[4].toInt(), e[5].toLong(), e[6], e[7], e[8], e[9].toLong(), e[10].toInt())
            progressCancelAnswers.add(cancel)
            if (cancel) throw EngineException(-5, "training cancelled")
        }
    }

    override fun patchInfoJson(patchPath: String): String { calls.add("patchInfo($patchPath)"); return patchInfo }

    fun json(vararg kv: Pair<String, Any?>) = JSONObject().also { o -> kv.forEach { (k, v) -> o.put(k, v) } }
    fun arr(vararg v: String) = JSONArray(v.toList())
}
