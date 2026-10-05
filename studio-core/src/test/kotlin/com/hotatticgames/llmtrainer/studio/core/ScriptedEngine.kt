package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import java.io.File
import java.util.concurrent.atomic.AtomicLong

/**
 * Deterministic scripted engine for studio-core tests. It implements the two seams exactly as the native adapter will, but the
 * "model" is a rule: a patched model answers from `knowledge` (so a specialist demonstrably differs from the base), an unpatched one
 * answers generically. Nothing here is a claim about real model quality.
 */
class ScriptedInference(var knowledge: List<String> = emptyList(), var sections: Map<String, String> = emptyMap()) : InferenceBackend {
    var available = true
    var unavailableReason = "scripted: engine disabled"
    var failLoadWith: BackendException? = null
    /** patched model fails these general probes (retention regression) */
    var patchedForgetsProbes = false
    var patchedAnswers = true
    var baseNll = 3.0
    var patchedNll = 2.5
    /** extra pieces of latency for stats */
    var genMsPerToken = 100.0

    class Open(val path: String, val patch: String?) { var closed = false }
    val models = HashMap<Long, Open>()
    val loads = ArrayList<Pair<String, String?>>()
    val prompts = ArrayList<String>()
    val rawUserPrompts = ArrayList<String>()
    var maxOpenAtOnce = 0
    var onGenerate: (String) -> Unit = {}
    private val ids = AtomicLong(0)
    private val chatModel = HashMap<Long, Long>()

    val openModels: Int get() = models.values.count { !it.closed }
    val openChats = HashSet<Long>()

    override fun status() = if (available) BackendStatus(true, "scripted-engine", null) else BackendStatus(false, "none", unavailableReason)
    private fun need() { if (!available) throw BackendException(BackendError.UNAVAILABLE, unavailableReason) }

    override fun loadModel(path: String, patchPath: String?): ModelHandle {
        need()
        failLoadWith?.let { throw it }
        if (!File(path).isFile) throw BackendException(BackendError.BAD_MODEL, "no such model file")
        if (patchPath != null && !File(patchPath).isFile) throw BackendException(BackendError.IO, "no such patch")
        val id = ids.incrementAndGet()
        models[id] = Open(path, patchPath)
        loads.add(path to patchPath)
        maxOpenAtOnce = maxOf(maxOpenAtOnce, openModels)
        return ModelHandle(id)
    }

    override fun modelInfo(model: ModelHandle) = ModelInfo(1_700_000_000, 28, 2048, 150_000, 32768, 15, File(models.getValue(model.id).path).length(), "qwen3", true, models.getValue(model.id).patch != null, 1800)
    override fun newChat(model: ModelHandle, contextTokens: Int, threads: Int): ChatHandle {
        need(); val c = ids.incrementAndGet(); chatModel[c] = model.id; openChats.add(c); return ChatHandle(c)
    }
    override fun resetChat(chat: ChatHandle) { need() }
    override fun chatFormat(model: ModelHandle, messages: List<ChatMessage>, addGenerationPrompt: Boolean): String =
        messages.joinToString("\n") { "<|${it.role}|>${it.content}" } + if (addGenerationPrompt) "\n<|assistant|>" else ""

    private val probeAnswers = EvalMetrics.PROBES.associate { (q, pat) ->
        q to when { pat.contains("27") -> "27"; pat.contains("72") -> "72"; pat.contains("paris") -> "Paris"; pat.contains("h2o") -> "H2O"; pat.contains("mars") -> "Mars"; pat.contains("cold") -> "cold"
            pat.contains("mice") -> "mice"; pat.contains("7") -> "7"; pat.contains("east") -> "east"; pat.contains("25") -> "25"; pat.contains("yes") -> "yes"; else -> "BANANA" }
    }

    private fun tokens(s: String) = Regex("[a-z0-9]+").findAll(s.lowercase()).map { it.value }.toSet()

    private fun best(q: Set<String>, pool: List<String>) = pool.maxByOrNull { k -> tokens(k).count { it in q } }
    private val sentenceSplit = Regex("(?<=[.!?])\\s+")

    private fun answer(patched: Boolean, userPrompt: String): String {
        probeAnswers[userPrompt.trim()]?.let { return if (patched && patchedForgetsProbes) "I would rather not say." else it }
        if (!patched || !patchedAnswers) return "It depends on the situation."
        if (userPrompt.contains("Sources:")) {          // grounded: a specialist that uses the excerpts it was given
            val src = userPrompt.substringAfter("Sources:").substringBefore("Question:").lines().map { it.substringAfter("] ") }.flatMap { l -> sentenceSplit.split(l) }.filter { it.isNotBlank() }
            return best(tokens(userPrompt.substringAfter("Question:")), src) ?: "It depends on the situation."
        }
        sections.entries.firstOrNull { userPrompt.contains(" > " + it.key) || userPrompt.contains("\"" + it.key) }?.let { return it.value }
        return best(tokens(userPrompt), knowledge) ?: "It depends on the situation."
    }

    override fun generate(chat: ChatHandle, prompt: String, sampling: SamplingParams, cancel: CancelToken, sink: (String) -> Boolean): GenStats {
        need()
        val m = models.getValue(chatModel.getValue(chat.id))
        check(!m.closed) { "use after close" }
        prompts.add(prompt)
        val user = Regex("<\\|user\\|>(.*?)(?=\\n<\\||$)", RegexOption.DOT_MATCHES_ALL).findAll(prompt).lastOrNull()?.groupValues?.get(1) ?: prompt
        rawUserPrompts.add(user)
        onGenerate(user)
        val text = answer(m.patch != null, user)
        var n = 0
        var stop = StopReason.END
        for ((i, w) in text.split(' ').withIndex()) {
            if (cancel.isCancelled) { stop = StopReason.CANCELLED; break }
            n++
            if (sink(if (i == 0) w else " $w")) { stop = StopReason.CANCELLED; break }
            if (n >= sampling.maxNewTokens) { stop = StopReason.MAX_TOKENS; break }
        }
        return GenStats(prompt.length / 4, n, 50.0, n * genMsPerToken, stop, 3_500_000_000L)
    }

    override fun score(chat: ChatHandle, text: String, cancel: CancelToken): ScoreResult {
        need()
        val m = models.getValue(chatModel.getValue(chat.id))
        if (cancel.isCancelled) throw BackendException(BackendError.CANCELLED, "cancelled")
        return ScoreResult(if (m.patch != null) patchedNll else baseNll, maxOf(1, text.length / 4))
    }

    override fun closeChat(chat: ChatHandle) { openChats.remove(chat.id) }
    override fun closeModel(model: ModelHandle) { models[model.id]?.closed = true }
}

/**
 * Scripted trainer: real checkpoint files in workDir (resume reads them), loss decreasing per step, a patch file that records the
 * base hash it was made for. Hooks let tests pause/cancel/kill/corrupt at an exact step.
 */
class ScriptedTrainer(private val baseShaOf: (String) -> String) : TrainingBackend {
    var available = true
    var trainable = true
    var peakBytes = 1_500_000_000L
    var peakPerLayerBytes = 100_000_000L
    var secondsPerStep: Double? = 2.0
    var failWith: BackendException? = null
    var onStep: (step: Int, cancel: CancelToken) -> Unit = { _, _ -> }
    var wrongBaseHash = false
    var stepsPerSequence = 1

    val trainCalls = ArrayList<List<String>>()
    val paramsSeen = ArrayList<TrainParams>()
    val estimates = ArrayList<TrainParams>()
    var resumedFrom = -1

    override fun status() = if (available) BackendStatus(true, "scripted-engine", null) else BackendStatus(false, "none", "scripted: training engine disabled")

    override fun estimate(basePath: String, params: TrainParams): TrainEstimate {
        estimates.add(params)
        val layers = if (params.trainableLastLayers == 0) 28 else params.trainableLastLayers
        // LoRA keeps the base frozen: only the (tiny) adapter state is added to the base footprint
        if (params.loraRank > 0) return TrainEstimate(trainable, peakBytes + params.loraRank * 2_000_000L, params.loraRank * 3_000_000L, if (trainable) null else "scripted: base not trainable", secondsPerStep)
        return TrainEstimate(trainable,  peakBytes + layers * peakPerLayerBytes, layers * 40_000_000L, if (trainable) null else "scripted: base not trainable (quantized)", secondsPerStep)
    }

    override fun train(basePath: String, texts: List<String>, params: TrainParams, workDir: String, outPatchPath: String, cancel: CancelToken, progress: (TrainEvent) -> Unit) {
        failWith?.let { throw it }
        if (!available) throw BackendException(BackendError.UNAVAILABLE, "scripted: training engine disabled")
        trainCalls.add(texts); paramsSeen.add(params)
        val work = File(workDir); work.mkdirs()
        val ck = File(work, "ckpt.txt")
        var start = 0
        if (ck.isFile) {
            val t = ck.readText()
            if (!t.startsWith("STEP ")) throw BackendException(BackendError.CORRUPT, "checkpoint failed validation")
            start = t.removePrefix("STEP ").trim().toInt(); resumedFrom = start
        }
        val total = texts.size * params.epochs * stepsPerSequence
        progress(TrainEvent(TrainPhase.PREPARE, 1, params.epochs, start, total, 0, null, null, 0.0, 800_000_000, start))
        for (step in start + 1..total) {
            onStep(step, cancel)
            if (cancel.isCancelled) { throw BackendException(BackendError.CANCELLED, "cancelled at step ${step - 1}") }
            ck.writeText("STEP $step")
            val loss = 3.0 / (1 + step * 0.1)
            progress(TrainEvent(TrainPhase.TRAIN, 1 + (step - 1) / maxOf(1, texts.size), params.epochs, step, total, step.toLong(), loss, if (step % 2 == 0) loss + 0.1 else null, step * 2.0, 900_000_000, start))
            if (cancel.isCancelled) { throw BackendException(BackendError.CANCELLED, "cancelled at step $step") }
        }
        progress(TrainEvent(TrainPhase.SAVE, params.epochs, params.epochs, total, total, total.toLong(), null, null, total * 2.0, 900_000_000, start))
        File(outPatchPath).writeText("HAGPATCH|${if (wrongBaseHash) "0".repeat(64) else baseShaOf(basePath)}|steps=$total|seed=${params.seed}|n=${texts.size}")
        progress(TrainEvent(TrainPhase.DONE, params.epochs, params.epochs, total, total, total.toLong(), null, null, total * 2.0, 900_000_000, start))
    }

    override fun patchInfo(patchPath: String): PatchInfo {
        val f = File(patchPath)
        val t = f.readText()
        if (!t.startsWith("HAGPATCH|")) throw BackendException(BackendError.CORRUPT, "bad patch")
        val parts = t.split('|')
        return PatchInfo(parts[1], listOf("blk.27.attn_q.weight"), null, parts[2].removePrefix("steps=").toInt(), 1, f.length(), "{}")
    }
}
