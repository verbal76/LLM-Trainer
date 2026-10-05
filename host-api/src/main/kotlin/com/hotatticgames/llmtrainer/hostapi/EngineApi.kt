package com.hotatticgames.llmtrainer.hostapi

/*
 * The engine half of the host API (level 2). Deliberately free of android.* types so the pure-JVM adapter module can compile
 * against this very file and be tested with a fake [EngineApi]; the host (APK) and bundles see the same classes through host-api.jar.
 * See HostApi.kt for the versioning rules (append-only inside an API major).
 */

/** Progress callback for model loading: fraction 0..1; return true to cancel. */
typealias LoadProgress = (fraction: Float) -> Boolean

/** Streaming sink: complete UTF-8 pieces, never a split character. Return true to stop generation. */
typealias TokenSink = (piece: String) -> Boolean

/**
 * Training progress. phase: 0 prepare, 1 train, 2 eval, 3 save, 4 done. NaN losses mean "not available".
 * Return true to cancel (the latest valid checkpoint is kept).
 */
typealias TrainProgress = (
    phase: Int, epoch: Int, epochs: Int, step: Int, steps: Int, examplesDone: Long, trainLoss: Double, valLoss: Double,
    elapsedS: Double, rssBytes: Long, resumedFromStep: Int,
) -> Boolean

/**
 * The on-device inference/specialization engine. Primitive surface only: models and sessions are opaque Long handles
 * (0 is never valid), strings are UTF-8, structured results are JSON strings, failures throw [EngineException].
 * Handles must be released with the matching free call; using a freed handle throws. Generation/training calls block
 * the calling thread: call them from a worker. [cancel] is safe from any thread.
 */
interface EngineApi {
    /** Engine + llama.cpp build identity (hag_engine_version). */
    fun version(): String
    /** JSON from hag_system_info_json (abi, cpu_features, n_cores, ...). */
    fun systemInfoJson(): String

    fun modelLoad(ggufPath: String, useMmap: Boolean, progress: LoadProgress?): Long
    fun modelApplyPatch(model: Long, patchPath: String)
    fun modelInfoJson(model: Long): String
    fun modelFree(model: Long)

    fun sessionNew(model: Long, nCtx: Int, nThreads: Int, nBatch: Int): Long
    fun sessionReset(session: Long)
    fun sessionFree(session: Long)

    /** Render the model's own chat template. roles: "system"/"user"/"assistant". */
    fun chatFormat(model: Long, roles: List<String>, contents: List<String>, addGenerationPrompt: Boolean): String

    /** Returns JSON stats {n_prompt_tokens,n_generated,prompt_ms,gen_ms,stop_reason,peak_rss_bytes}. temperature 0 = greedy. */
    fun generate(
        session: Long, prompt: String, temperature: Float, topK: Int, topP: Float, minP: Float, repeatPenalty: Float,
        seed: Long, maxNewTokens: Int, sink: TokenSink,
    ): String
    fun cancel(session: Long)

    fun tokenize(model: Long, text: String, addSpecial: Boolean): IntArray
    /** Returns {"mean_nll":..,"n_tokens":..}. */
    fun score(session: Long, text: String): String

    /** configJson keys: see [TrainConfigKeys]. Returns the engine's JSON estimate (peak bytes, trainable or not, ...). */
    fun trainEstimate(baseGguf: String, configJson: String): String
    /** Blocks until done/cancelled. Resumes from checkpoints in workDir when inputs match. */
    fun train(
        baseGguf: String, texts: List<String>, configJson: String, workDir: String, outPatch: String, progress: TrainProgress?,
    )
    fun patchInfoJson(patchPath: String): String
}

/** Keys accepted in the `configJson` of [EngineApi.train]/[EngineApi.trainEstimate]; missing keys use engine defaults. */
object TrainConfigKeys {
    const val N_CTX = "n_ctx"; const val N_BATCH = "n_batch"; const val EPOCHS = "epochs"
    const val LEARNING_RATE = "learning_rate"; const val VAL_FRACTION = "val_fraction"; const val SEED = "seed"
    const val N_THREADS = "n_threads"; const val TRAINABLE_LAST_LAYERS = "trainable_last_layers"
    const val TRAIN_EMBEDDINGS = "train_embeddings"; const val CHECKPOINT_EVERY_STEPS = "checkpoint_every_steps"
    const val MAX_MEMORY_BYTES = "max_memory_bytes"
    /** > 0: train LoRA adapters of this rank with the base frozen (patch = standard llama.cpp LoRA adapter GGUF); 0/missing = tune base weights. */
    const val LORA_RANK = "lora_rank"
    /** LoRA scale = alpha / rank; 0/missing = 2 x rank. */
    const val LORA_ALPHA = "lora_alpha"
}

/** code mirrors HAG_ERR_* (negative); -100 = binding usage error (bad/freed handle). */
class EngineException(val code: Int, message: String) : RuntimeException(message) {
    val cancelled: Boolean get() = code == -5
}
