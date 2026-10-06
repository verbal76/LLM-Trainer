package com.hotatticgames.llmtrainer.studio.api

/*
 * The two engine seams of the phone-first studio (v2). Pure Kotlin, no android.* types.
 *
 * - Implemented LATER by a bundle adapter over the host's EngineApi (JNI over native/engine/include/hag_engine.h).
 * - studio-core only ever talks to these interfaces, so tests use scripted fakes and a real engine drops in unchanged.
 * - The seams are deliberately THIN: one method per engine capability, no policy. Safety gates, persistence, resume,
 *   provenance and honesty rules live in studio-core.
 * - Error model: seam methods THROW [BackendException] (mirrors HAG_ERR_*); studio-core converts to [StudioResult.Err].
 *   A backend must never return a pretend result: if it cannot do something it throws UNAVAILABLE/UNSUPPORTED.
 * - Threading: one [ChatHandle] is used by one thread at a time; [CancelToken.cancel] may be called from any thread.
 */

enum class BackendError { UNAVAILABLE, INVALID_ARG, IO, BAD_MODEL, OOM, CANCELLED, UNSUPPORTED, CORRUPT, INTERNAL }

class BackendException(val code: BackendError, message: String) : RuntimeException(message)

/** Cooperative, thread-safe cancel flag handed to long calls (generate / score / train). */
class CancelToken {
    @Volatile var isCancelled: Boolean = false
        private set
    fun cancel() { isCancelled = true }
}

/** Whether a backend can actually do work on this install. `reason` is user-presentable when `available == false`. */
data class BackendStatus(val available: Boolean, val runtimeId: String, val reason: String?)

// ===== Inference ===============================================================================================

data class ModelHandle(val id: Long)
data class ChatHandle(val id: Long)

/** Mirrors hag_model_info_json. `loadMs` is measured by the backend around load (+ patch apply). */
data class ModelInfo(
    val nParams: Long, val nLayer: Int, val nEmbd: Int, val nVocab: Int, val nCtxTrain: Int, val fileType: Int,
    val sizeBytes: Long, val arch: String, val hasChatTemplate: Boolean, val patched: Boolean, val loadMs: Long,
)

data class ChatMessage(val role: String, val content: String)   // role: "system" | "user" | "assistant"

data class SamplingParams(
    val temperature: Float = 0.7f, val topK: Int = 40, val topP: Float = 0.95f, val minP: Float = 0f,
    val repeatPenalty: Float = 1.1f, val seed: Long = 0, val maxNewTokens: Int = 512,
) {
    companion object {
        /** Reproducible decoding used by evaluation and A/B so base and specialist see identical settings. */
        fun greedy(maxNewTokens: Int = 256) = SamplingParams(temperature = 0f, topK = 0, topP = 1f, minP = 0f, repeatPenalty = 1f, seed = 1, maxNewTokens = maxNewTokens)
    }
}

enum class StopReason { END, MAX_TOKENS, CANCELLED, CONTEXT_FULL }

data class GenStats(
    val promptTokens: Int, val generatedTokens: Int, val promptMs: Double, val genMs: Double,
    val stopReason: StopReason, val peakRssBytes: Long,
) {
    val tokensPerSecond: Double get() = if (genMs <= 0.0) 0.0 else generatedTokens * 1000.0 / genMs
}

data class ScoreResult(val meanNll: Double, val nTokens: Int)

interface InferenceBackend {
    /** Cheap, side-effect free. When unavailable every other method throws BackendException(UNAVAILABLE). */
    fun status(): BackendStatus
    /** Loads GGUF weights; when [patchPath] is given the specialist patch is verified against the base and applied. */
    fun loadModel(path: String, patchPath: String? = null): ModelHandle
    fun modelInfo(model: ModelHandle): ModelInfo
    fun newChat(model: ModelHandle, contextTokens: Int, threads: Int = 0): ChatHandle
    /** Clears the conversation/KV of the chat. */
    fun resetChat(chat: ChatHandle)
    /** The model's own chat template applied to [messages]. */
    fun chatFormat(model: ModelHandle, messages: List<ChatMessage>, addGenerationPrompt: Boolean = true): String
    /** Continues from the chat's KV state with [prompt]. [sink] gets complete UTF-8 pieces; return true to stop. */
    fun generate(chat: ChatHandle, prompt: String, sampling: SamplingParams, cancel: CancelToken, sink: (String) -> Boolean): GenStats
    /** Mean negative log-likelihood per token of [text] (lower = better fit). Used for held-out evaluation. */
    fun score(chat: ChatHandle, text: String, cancel: CancelToken): ScoreResult
    fun closeChat(chat: ChatHandle)
    fun closeModel(model: ModelHandle)
}

// ===== Training ================================================================================================

/** Mirrors hag_train_params. `maxMemoryBytes` makes the engine refuse (OOM) rather than start when its estimate exceeds it. */
data class TrainParams(
    val contextTokens: Int = 512, val batchTokens: Int = 512, val epochs: Int = 1, val learningRate: Float = 2e-5f,
    val valFraction: Float = 0.1f, val seed: Long = 1337, val threads: Int = 0,
    /** 0 = all layers; N = only the last N transformer blocks (+ final norm). */
    val trainableLastLayers: Int = 0, val trainEmbeddings: Boolean = false,
    val checkpointEverySteps: Int = 0, val maxMemoryBytes: Long = 0,
    /** Mirrors hag_train_params.lora_rank/lora_alpha: > 0 trains LoRA adapters (base frozen, patch = adapter GGUF); alpha 0 = 2 x rank. */
    val loraRank: Int = 0, val loraAlpha: Float = 0f,
)

data class TrainEstimate(
    val trainable: Boolean, val peakBytes: Long, val trainableParams: Long?, val reason: String?,
    /** Backend's own throughput hint (seconds per optimizer step) when it has one; null = unknown. */
    val secondsPerStep: Double? = null,
)

enum class TrainPhase { PREPARE, TRAIN, EVAL, SAVE, DONE }

data class TrainEvent(
    val phase: TrainPhase, val epoch: Int, val epochs: Int, val step: Int, val steps: Int, val examplesDone: Long,
    val trainLoss: Double?, val valLoss: Double?, val elapsedS: Double, val rssBytes: Long, val resumedFromStep: Int,
)

data class PatchInfo(
    val baseSha256: String, val tensorNames: List<String>, val datasetHash: String?, val steps: Int,
    val epochs: Int, val sizeBytes: Long, val paramsJson: String,
)

interface TrainingBackend {
    fun status(): BackendStatus
    /** Cheap pre-flight (no weights loaded into RAM beyond headers). Never throws for "too big": returns trainable=false/peakBytes. */
    fun estimate(basePath: String, params: TrainParams): TrainEstimate
    /**
     * Blocks the calling thread. Checkpoints go to [workDir]; calling again with the same inputs RESUMES. On success a patch
     * containing ONLY the changed tensors is written to [outPatchPath]; the base file is never modified. Cancel keeps the latest
     * valid checkpoint and throws BackendException(CANCELLED). A corrupt checkpoint throws CORRUPT. Unsupported bases (for example
     * quantized weights) throw UNSUPPORTED with an explanation; the backend must not fall back to anything fake.
     */
    fun train(basePath: String, texts: List<String>, params: TrainParams, workDir: String, outPatchPath: String,
              cancel: CancelToken, progress: (TrainEvent) -> Unit)
    fun patchInfo(patchPath: String): PatchInfo
}
