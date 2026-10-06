package com.hotatticgames.llmtrainer.host

import android.util.Log
import com.hotatticgames.hag.runtime.EngineStatus
import com.hotatticgames.hag.runtime.HagEngine
import com.hotatticgames.hag.runtime.HagException
import com.hotatticgames.hag.runtime.HagRuntime
import com.hotatticgames.hag.runtime.SampleParams
import com.hotatticgames.hag.runtime.TrainConfig
import com.hotatticgames.hag.runtime.TrainEvent
import com.hotatticgames.llmtrainer.hostapi.EngineApi
import com.hotatticgames.llmtrainer.hostapi.EngineException
import com.hotatticgames.llmtrainer.hostapi.LoadProgress
import com.hotatticgames.llmtrainer.hostapi.TokenSink
import com.hotatticgames.llmtrainer.hostapi.TrainConfigKeys
import com.hotatticgames.llmtrainer.hostapi.TrainProgress
import org.json.JSONObject

/** Process-wide native engine bring-up (guarded: a missing/unsupported engine yields Unavailable, never a crash). */
object NativeEngine {
    private const val TAG = "HagEngine"

    val status: EngineStatus by lazy {
        val s = HagRuntime.load()
        when (s) {
            is EngineStatus.Ready -> Log.i(TAG, "engine ready: ${s.version}")
            is EngineStatus.Unavailable -> Log.w(TAG, "engine UNAVAILABLE at ${s.stage}: ${s.reason}")
        }
        s
    }
}

/** Human-readable engine state for diagnostics. */
fun EngineStatus.describe(): String = when (this) {
    is EngineStatus.Ready -> "ready"
    is EngineStatus.Unavailable -> "unavailable: $stage: $reason"
}

/** Adapts the generic runtime to the primitive-only host API surface (maps exceptions, JSON config -> TrainConfig). */
class EngineAdapter(private val e: HagEngine) : EngineApi {
    private inline fun <T> wrap(f: () -> T): T = try {
        f()
    } catch (x: HagException) {
        throw EngineException(x.code, x.message ?: "")
    }

    override fun version() = e.version()
    override fun systemInfoJson() = e.systemInfoJson()

    override fun modelLoad(ggufPath: String, useMmap: Boolean, progress: LoadProgress?) = wrap { e.modelLoad(ggufPath, useMmap, progress) }
    override fun modelApplyPatch(model: Long, patchPath: String) = wrap { e.modelApplyPatch(model, patchPath) }
    override fun modelInfoJson(model: Long) = wrap { e.modelInfoJson(model) }
    override fun modelFree(model: Long) = e.modelFree(model)

    override fun sessionNew(model: Long, nCtx: Int, nThreads: Int, nBatch: Int) = wrap { e.sessionNew(model, nCtx, nThreads, nBatch) }
    override fun sessionReset(session: Long) = wrap { e.sessionReset(session) }
    override fun sessionFree(session: Long) = e.sessionFree(session)

    override fun chatFormat(model: Long, roles: List<String>, contents: List<String>, addGenerationPrompt: Boolean) =
        wrap { e.chatFormat(model, roles, contents, addGenerationPrompt) }

    override fun generate(
        session: Long, prompt: String, temperature: Float, topK: Int, topP: Float, minP: Float, repeatPenalty: Float,
        seed: Long, maxNewTokens: Int, sink: TokenSink,
    ) = wrap {
        e.generate(session, prompt, SampleParams(temperature, topK, topP, minP, repeatPenalty, seed, maxNewTokens), sink)
    }

    override fun cancel(session: Long) = wrap { e.cancel(session) }
    override fun tokenize(model: Long, text: String, addSpecial: Boolean) = wrap { e.tokenize(model, text, addSpecial) }
    override fun score(session: Long, text: String) = wrap { e.score(session, text) }

    override fun trainEstimate(baseGguf: String, configJson: String) = wrap { e.trainEstimate(baseGguf, parseConfig(configJson)) }

    override fun train(
        baseGguf: String, texts: List<String>, configJson: String, workDir: String, outPatch: String, progress: TrainProgress?,
    ) = wrap {
        e.train(baseGguf, texts, parseConfig(configJson), workDir, outPatch, progress?.let { p ->
            { ev: TrainEvent -> p(ev.phase, ev.epoch, ev.epochs, ev.step, ev.steps, ev.examplesDone, ev.trainLoss, ev.valLoss, ev.elapsedS, ev.rssBytes, ev.resumedFromStep) }
        })
    }

    override fun patchInfoJson(patchPath: String) = wrap { e.patchInfoJson(patchPath) }

    companion object {
        /** Missing keys keep the runtime defaults; unknown keys are ignored (append-only API). */
        fun parseConfig(json: String): TrainConfig {
            val d = TrainConfig()
            val o = if (json.isBlank()) JSONObject() else JSONObject(json)
            return TrainConfig(
                nCtx = o.optInt(TrainConfigKeys.N_CTX, d.nCtx),
                nBatch = o.optInt(TrainConfigKeys.N_BATCH, d.nBatch),
                epochs = o.optInt(TrainConfigKeys.EPOCHS, d.epochs),
                learningRate = o.optDouble(TrainConfigKeys.LEARNING_RATE, d.learningRate.toDouble()).toFloat(),
                valFraction = o.optDouble(TrainConfigKeys.VAL_FRACTION, d.valFraction.toDouble()).toFloat(),
                seed = o.optLong(TrainConfigKeys.SEED, d.seed),
                nThreads = o.optInt(TrainConfigKeys.N_THREADS, d.nThreads),
                trainableLastLayers = o.optInt(TrainConfigKeys.TRAINABLE_LAST_LAYERS, d.trainableLastLayers),
                trainEmbeddings = o.optInt(TrainConfigKeys.TRAIN_EMBEDDINGS, if (d.trainEmbeddings) 1 else 0) != 0,
                checkpointEverySteps = o.optInt(TrainConfigKeys.CHECKPOINT_EVERY_STEPS, d.checkpointEverySteps),
                maxMemoryBytes = o.optLong(TrainConfigKeys.MAX_MEMORY_BYTES, d.maxMemoryBytes),
                loraRank = o.optInt(TrainConfigKeys.LORA_RANK, d.loraRank),
                loraAlpha = o.optDouble(TrainConfigKeys.LORA_ALPHA, d.loraAlpha.toDouble()).toFloat(),
            )
        }
    }
}
