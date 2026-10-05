package com.hotatticgames.llmtrainer.engine

import com.hotatticgames.llmtrainer.hostapi.EngineApi
import com.hotatticgames.llmtrainer.hostapi.EngineException
import com.hotatticgames.llmtrainer.hostapi.TrainConfigKeys
import com.hotatticgames.llmtrainer.studio.api.BackendError
import com.hotatticgames.llmtrainer.studio.api.BackendException
import com.hotatticgames.llmtrainer.studio.api.BackendStatus
import com.hotatticgames.llmtrainer.studio.api.CancelToken
import com.hotatticgames.llmtrainer.studio.api.ChatHandle
import com.hotatticgames.llmtrainer.studio.api.ChatMessage
import com.hotatticgames.llmtrainer.studio.api.GenStats
import com.hotatticgames.llmtrainer.studio.api.InferenceBackend
import com.hotatticgames.llmtrainer.studio.api.ModelHandle
import com.hotatticgames.llmtrainer.studio.api.ModelInfo
import com.hotatticgames.llmtrainer.studio.api.PatchInfo
import com.hotatticgames.llmtrainer.studio.api.SamplingParams
import com.hotatticgames.llmtrainer.studio.api.ScoreResult
import com.hotatticgames.llmtrainer.studio.api.StopReason
import com.hotatticgames.llmtrainer.studio.api.TrainEstimate
import com.hotatticgames.llmtrainer.studio.api.TrainEvent
import com.hotatticgames.llmtrainer.studio.api.TrainParams
import com.hotatticgames.llmtrainer.studio.api.TrainPhase
import com.hotatticgames.llmtrainer.studio.api.TrainingBackend
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean

/*
 * Adapter: studio-api engine seams (InferenceBackend / TrainingBackend) over the host's EngineApi (host-api level 2).
 *
 * Semantics worth knowing (docs/v2/ENGINE.md, native/engine/include/hag_engine.h):
 *  - A specialist PATCH is either a changed-tensor patch (kind=replace: full / partial tuning) or a standard llama.cpp LoRA adapter GGUF
 *    (kind=lora). Both are applied by EngineApi.modelApplyPatch, which verifies the base file's sha256 (the whole file is hashed once per
 *    model handle, so loading a specialist costs noticeably more than loading the base on a phone; loadMs includes it).
 *  - Training with loraRank > 0 keeps the base frozen (quantized bases need no F32 copy). The adapter REFUSES to continue when the host
 *    engine does not report the LoRA configuration back in its estimate: it must never silently tune base weights instead.
 *  - The adapter has no policy: gates, persistence, resume and honesty rules live in studio-core. It never returns a pretend result.
 */

/** HAG_ERR_* (hag_engine.h) -> seam error. -100 is the binding's own "bad or freed handle" usage error. */
internal fun mapEngineError(e: EngineException): BackendException = BackendException(
    when (e.code) {
        -1, -100 -> BackendError.INVALID_ARG
        -2 -> BackendError.IO
        -3 -> BackendError.BAD_MODEL
        -4 -> BackendError.OOM
        -5 -> BackendError.CANCELLED
        -6 -> BackendError.UNSUPPORTED
        -7 -> BackendError.CORRUPT
        else -> BackendError.INTERNAL
    },
    e.message ?: "engine error ${e.code}",
)

private inline fun <T> mapped(f: () -> T): T = try {
    f()
} catch (e: EngineException) {
    throw mapEngineError(e)
} catch (e: BackendException) {
    throw e
} catch (e: org.json.JSONException) {
    throw BackendException(BackendError.INTERNAL, "unreadable engine answer: ${e.message}")
} catch (e: RuntimeException) {
    throw BackendException(BackendError.INTERNAL, "engine call failed (${e.javaClass.simpleName}): ${e.message}")
}

/**
 * Cooperative cancel for blocking engine calls. Polls the [CancelToken] and calls [cancel] repeatedly until the call returns: the
 * engine clears its flag when a call starts, so a single early cancel could be lost. Cheap (one sleeping daemon thread per call).
 */
internal class CancelWatcher(private val token: CancelToken, private val cancel: () -> Unit) : AutoCloseable {
    private val done = AtomicBoolean(false)
    private val thread = Thread({
        while (!done.get()) {
            if (token.isCancelled) { try { cancel() } catch (_: RuntimeException) { } }
            try { Thread.sleep(20) } catch (_: InterruptedException) { return@Thread }
        }
    }, "hag-cancel-watch").apply { isDaemon = true; start() }

    override fun close() { done.set(true); thread.interrupt() }
}

internal fun JSONObject.longOrNull(k: String): Long? = if (has(k) && !isNull(k)) optLong(k) else null

// ===== inference ===================================================================================================

class HostInferenceBackend(
    private val engine: EngineApi?,
    private val unavailableReason: String? = null,
) : InferenceBackend {
    private class Loaded(val loadMs: Long)

    private val models = ConcurrentHashMap<Long, Loaded>()

    private fun eng(): EngineApi = engine ?: throw BackendException(BackendError.UNAVAILABLE, unavailableReason ?: "The native engine is not available on this device.")

    override fun status(): BackendStatus {
        val e = engine ?: return BackendStatus(false, "none", unavailableReason ?: "The native engine is not available on this device.")
        return try { BackendStatus(true, e.version(), null) } catch (x: RuntimeException) { BackendStatus(false, "none", "The native engine did not answer: ${x.message}") }
    }

    override fun loadModel(path: String, patchPath: String?): ModelHandle = mapped {
        val e = eng()
        val t0 = System.nanoTime()
        val m = e.modelLoad(path, true, null)
        try {
            if (patchPath != null) e.modelApplyPatch(m, patchPath)
        } catch (x: Throwable) {
            try { e.modelFree(m) } catch (_: RuntimeException) { }
            throw x
        }
        models[m] = Loaded((System.nanoTime() - t0) / 1_000_000)
        ModelHandle(m)
    }

    private fun loaded(model: ModelHandle) = models[model.id] ?: throw BackendException(BackendError.INVALID_ARG, "unknown or closed model handle ${model.id}")

    override fun modelInfo(model: ModelHandle): ModelInfo = mapped {
        val l = loaded(model)
        val o = JSONObject(eng().modelInfoJson(model.id))
        ModelInfo(
            nParams = o.optLong("n_params"), nLayer = o.optInt("n_layer"), nEmbd = o.optInt("n_embd"), nVocab = o.optInt("n_vocab"),
            nCtxTrain = o.optInt("n_ctx_train"), fileType = o.optString("file_type", "").toIntOrNull() ?: -1,
            sizeBytes = o.optLong("size_bytes"), arch = o.optString("arch", ""), hasChatTemplate = o.optBoolean("chat_template", false),
            patched = o.optBoolean("patched", false), loadMs = l.loadMs,
        )
    }

    override fun newChat(model: ModelHandle, contextTokens: Int, threads: Int): ChatHandle = mapped {
        loaded(model)
        ChatHandle(eng().sessionNew(model.id, contextTokens, threads, 0))
    }

    override fun resetChat(chat: ChatHandle) = mapped { eng().sessionReset(chat.id) }

    override fun chatFormat(model: ModelHandle, messages: List<ChatMessage>, addGenerationPrompt: Boolean): String = mapped {
        loaded(model)
        eng().chatFormat(model.id, messages.map { it.role }, messages.map { it.content }, addGenerationPrompt)
    }

    override fun generate(chat: ChatHandle, prompt: String, sampling: SamplingParams, cancel: CancelToken, sink: (String) -> Boolean): GenStats = mapped {
        val e = eng()
        if (cancel.isCancelled) return@mapped GenStats(0, 0, 0.0, 0.0, StopReason.CANCELLED, 0)
        val watcher = CancelWatcher(cancel) { e.cancel(chat.id) }
        val json = try {
            e.generate(
                chat.id, prompt, sampling.temperature, sampling.topK, sampling.topP, sampling.minP, sampling.repeatPenalty,
                sampling.seed, sampling.maxNewTokens,
            ) { piece -> sink(piece) || cancel.isCancelled }
        } catch (x: EngineException) {
            // A cancelled prompt phase surfaces as HAG_ERR_CANCELLED and leaves the session unchanged: that is a stop, not a failure.
            if (x.cancelled && cancel.isCancelled) return@mapped GenStats(0, 0, 0.0, 0.0, StopReason.CANCELLED, 0)
            throw x
        } finally {
            watcher.close()
        }
        val o = JSONObject(json)
        GenStats(
            promptTokens = o.optInt("n_prompt_tokens"), generatedTokens = o.optInt("n_generated"), promptMs = o.optDouble("prompt_ms", 0.0),
            genMs = o.optDouble("gen_ms", 0.0), stopReason = stopReasonOf(o.optInt("stop_reason", 0)), peakRssBytes = o.optLong("peak_rss_bytes"),
        )
    }

    override fun score(chat: ChatHandle, text: String, cancel: CancelToken): ScoreResult = mapped {
        val e = eng()
        if (cancel.isCancelled) throw BackendException(BackendError.CANCELLED, "cancelled")
        val watcher = CancelWatcher(cancel) { e.cancel(chat.id) }
        val json = try { e.score(chat.id, text) } finally { watcher.close() }
        val o = JSONObject(json)
        ScoreResult(o.getDouble("mean_nll"), o.optInt("n_tokens"))
    }

    override fun closeChat(chat: ChatHandle) { try { engine?.sessionFree(chat.id) } catch (_: RuntimeException) { } }

    override fun closeModel(model: ModelHandle) {
        models.remove(model.id)
        try { engine?.modelFree(model.id) } catch (_: RuntimeException) { }
    }

    internal fun openModelCount() = models.size

    companion object {
        fun stopReasonOf(code: Int) = when (code) { 0 -> StopReason.END; 1 -> StopReason.MAX_TOKENS; 2 -> StopReason.CANCELLED; 3 -> StopReason.CONTEXT_FULL; else -> StopReason.END }
    }
}

// ===== training ====================================================================================================

class HostTrainingBackend(
    private val engine: EngineApi?,
    private val unavailableReason: String? = null,
) : TrainingBackend {
    private fun eng(): EngineApi = engine ?: throw BackendException(BackendError.UNAVAILABLE, unavailableReason ?: "The native engine is not available on this device.")

    override fun status(): BackendStatus {
        val e = engine ?: return BackendStatus(false, "none", unavailableReason ?: "The native training engine is not available on this device.")
        return try { BackendStatus(true, e.version(), null) } catch (x: RuntimeException) { BackendStatus(false, "none", "The native engine did not answer: ${x.message}") }
    }

    /** The engine pins its thread count into the run fingerprint (bit-exact resume), so "auto" is resolved ONCE here from the device's core count. */
    private fun pinnedThreads(requested: Int): Int {
        if (requested > 0) return requested
        val cores = try { JSONObject(eng().systemInfoJson()).optInt("n_cores", 0) } catch (_: Exception) { 0 }
        return if (cores > 0) maxOf(1, minOf(8, cores * 3 / 4)) else 0
    }

    internal fun configJson(p: TrainParams): String = JSONObject()
        .put(TrainConfigKeys.N_CTX, p.contextTokens).put(TrainConfigKeys.N_BATCH, p.batchTokens).put(TrainConfigKeys.EPOCHS, p.epochs)
        .put(TrainConfigKeys.LEARNING_RATE, p.learningRate.toDouble()).put(TrainConfigKeys.VAL_FRACTION, p.valFraction.toDouble())
        .put(TrainConfigKeys.SEED, p.seed).put(TrainConfigKeys.N_THREADS, pinnedThreads(p.threads))
        .put(TrainConfigKeys.TRAINABLE_LAST_LAYERS, p.trainableLastLayers).put(TrainConfigKeys.TRAIN_EMBEDDINGS, if (p.trainEmbeddings) 1 else 0)
        .put(TrainConfigKeys.CHECKPOINT_EVERY_STEPS, p.checkpointEverySteps).put(TrainConfigKeys.MAX_MEMORY_BYTES, p.maxMemoryBytes)
        .put(TrainConfigKeys.LORA_RANK, p.loraRank).put(TrainConfigKeys.LORA_ALPHA, p.loraAlpha.toDouble())
        .toString()

    override fun estimate(basePath: String, params: TrainParams): TrainEstimate = mapped {
        val o = JSONObject(eng().trainEstimate(basePath, configJson(params)))
        val trainable = o.optBoolean("trainable", false)
        val peak = o.optLong("estimated_peak_bytes", 0L)
        val tp = o.longOrNull("trainable_params")
        if (trainable && params.loraRank > 0 && !(o.optBoolean("lora", false) && o.optInt("lora_rank", 0) == params.loraRank)) {
            // Never fall back to tuning base weights when the owner asked for an adapter.
            return@mapped TrainEstimate(false, peak, tp, "The engine on this install did not apply the LoRA settings (it reports no rank-${params.loraRank} adapter). Refusing instead of tuning the base weights.")
        }
        TrainEstimate(trainable, peak, tp, if (trainable) null else o.optString("reason", "").ifEmpty { "The engine reports that this model cannot be trained with these settings." })
    }

    override fun train(basePath: String, texts: List<String>, params: TrainParams, workDir: String, outPatchPath: String, cancel: CancelToken, progress: (TrainEvent) -> Unit) = mapped {
        val e = eng()
        if (cancel.isCancelled) throw BackendException(BackendError.CANCELLED, "cancelled before start")
        // The effectiveness check of estimate(), repeated at the point of no return (cheap: header only).
        if (params.loraRank > 0) {
            val est = estimate(basePath, params)
            if (!est.trainable) throw BackendException(BackendError.UNSUPPORTED, est.reason ?: "not trainable with these settings")
        }
        e.train(basePath, texts, configJson(params), workDir, outPatchPath) { phase, epoch, epochs, step, steps, examples, trainLoss, valLoss, elapsedS, rss, resumed ->
            progress(
                TrainEvent(
                    phaseOf(phase), epoch, epochs, step, steps, examples, trainLoss.takeIf { it.isFinite() }, valLoss.takeIf { it.isFinite() }, elapsedS, rss, resumed,
                ),
            )
            cancel.isCancelled
        }
    }

    override fun patchInfo(patchPath: String): PatchInfo = mapped {
        val o = JSONObject(eng().patchInfoJson(patchPath))
        val train = o.optJSONObject("train") ?: JSONObject()
        val names = ArrayList<String>()
        val ts: JSONArray? = o.optJSONArray("tensors")
        if (ts != null) for (i in 0 until ts.length()) ts.optJSONObject(i)?.optString("name")?.takeIf { it.isNotEmpty() }?.let { names.add(it) }
        PatchInfo(
            baseSha256 = o.optJSONObject("base")?.optString("sha256", "") ?: "",
            tensorNames = names,
            datasetHash = train.optString("dataset_sha256", "").ifEmpty { null },
            steps = train.optInt("steps", 0), epochs = train.optInt("epochs", 0),
            sizeBytes = o.optLong("file_bytes"), paramsJson = train.toString(),
        )
    }

    companion object {
        fun phaseOf(code: Int) = when (code) { 0 -> TrainPhase.PREPARE; 1 -> TrainPhase.TRAIN; 2 -> TrainPhase.EVAL; 3 -> TrainPhase.SAVE; else -> TrainPhase.DONE }
    }
}

// ===== wiring ======================================================================================================

/** The two seams for one host, chosen from what the host really advertises. Backends that are not advertised report why, never pretend. */
class EngineBackends(val inference: InferenceBackend, val trainer: TrainingBackend)

object EngineWiring {
    const val INFERENCE_CAP = "inference.gguf.v1"
    const val TRAINING_CAP = "training.patch.v1"

    /**
     * @param engine the host's EngineApi (null = engine unavailable)
     * @param unavailableReason the host's engineUnavailableReason
     * @param capabilities the host's advertised capability names, or null when they could not be read (then the presence of [engine] decides)
     */
    fun create(engine: EngineApi?, unavailableReason: String?, capabilities: Set<String>?): EngineBackends {
        val why = unavailableReason ?: "This install has no native engine."
        val inf = engine != null && (capabilities == null || INFERENCE_CAP in capabilities)
        val tr = engine != null && (capabilities == null || TRAINING_CAP in capabilities)
        return EngineBackends(
            HostInferenceBackend(if (inf) engine else null, if (engine == null) why else "The host did not advertise $INFERENCE_CAP."),
            HostTrainingBackend(if (tr) engine else null, if (engine == null) why else "The host did not advertise $TRAINING_CAP."),
        )
    }

    /** Capability names from the host's diagnosticsJson (`capabilities` array), or null when absent/unreadable. */
    fun capabilitiesFrom(diagnosticsJson: String): Set<String>? = try {
        val a = JSONObject(diagnosticsJson).optJSONArray("capabilities")
        if (a == null) null else (0 until a.length()).map { a.getString(it) }.toSet()
    } catch (_: Exception) {
        null
    }
}
