package com.hotatticgames.hag.runtime

/** Sampling parameters; [temperature] 0 = greedy. A fixed [seed] makes sampled output reproducible. */
data class SampleParams(
    val temperature: Float = 0f,
    val topK: Int = 0,
    val topP: Float = 1f,
    val minP: Float = 0f,
    val repeatPenalty: Float = 1f,
    val seed: Long = 0,
    val maxNewTokens: Int = 128,
)

/** Fine-tuning hyper-parameters. Defaults are conservative for small models; callers should set what matters. */
data class TrainConfig(
    val nCtx: Int = 256,
    val nBatch: Int = 256,
    val epochs: Int = 3,
    val learningRate: Float = 1e-4f,
    val valFraction: Float = 0.1f,
    val seed: Long = 1234,
    val nThreads: Int = 0,
    val trainableLastLayers: Int = 0,
    val trainEmbeddings: Boolean = false,
    val checkpointEverySteps: Int = 0,
    val maxMemoryBytes: Long = 0,
) {
    internal fun toNative() = doubleArrayOf(
        nCtx.toDouble(), nBatch.toDouble(), epochs.toDouble(), learningRate.toDouble(), valFraction.toDouble(), seed.toDouble(),
        nThreads.toDouble(), trainableLastLayers.toDouble(), if (trainEmbeddings) 1.0 else 0.0,
        checkpointEverySteps.toDouble(), maxMemoryBytes.toDouble(),
    )
}

/** One progress report from a training run. Losses are NaN when not available. */
data class TrainEvent(
    val phase: Int, val epoch: Int, val epochs: Int, val step: Int, val steps: Int, val examplesDone: Long,
    val trainLoss: Double, val valLoss: Double, val elapsedS: Double, val rssBytes: Long, val resumedFromStep: Int,
) {
    companion object {
        const val PREPARE = 0
        const val TRAIN = 1
        const val EVAL = 2
        const val SAVE = 3
        const val DONE = 4
    }
}

/**
 * Thread-safe façade over the native engine. Models and sessions are opaque Long ids (never native pointers):
 * a freed or unknown id throws [HagException] (code USAGE) instead of reaching native code; free is idempotent and
 * deferred while a call is in flight. Blocking calls (load, generate, score, train) must run on a worker thread;
 * [cancel] may be called from any thread. Obtain via [HagRuntime.load].
 */
class HagEngine internal constructor() {
    private val table = HandleTable()

    fun version(): String = NativeBridge.nativeVersion().utf8()
    fun systemInfoJson(): String = NativeBridge.nativeSystemInfo().utf8()

    /** [progress] gets 0..1; return true to cancel (throws a [HagException] with [HagException.cancelled]). */
    fun modelLoad(ggufPath: String, useMmap: Boolean = true, progress: ((Float) -> Boolean)? = null): Long {
        var failure: Throwable? = null
        val cb = progress?.let { p ->
            NativeBridge.LoadCb { f -> try { p(f) } catch (t: Throwable) { failure = t; true } }
        }
        val ptr = try {
            NativeBridge.nativeModelLoad(ggufPath.utf8Bytes(), useMmap, cb)
        } catch (e: HagException) {
            failure?.let { throw it }
            throw e
        }
        failure?.let { NativeBridge.nativeModelFree(ptr); throw it }
        return table.put(Handle(ptr, "model", NativeBridge::nativeModelFree))
    }

    /** Applies a specialist patch. The model must have no live sessions (their KV state would be stale). */
    fun modelApplyPatch(model: Long, patchPath: String) = table.use(model) { h ->
        if (h.liveChildren.get() != 0) {
            throw HagException(HagException.USAGE, "free all sessions of the model before applying a patch")
        }
        NativeBridge.nativeModelApplyPatch(h.ptr, patchPath.utf8Bytes())
    }

    fun modelInfoJson(model: Long): String = table.use(model) { NativeBridge.nativeModelInfo(it.ptr).utf8() }
    fun modelFree(model: Long) = table.free(model)

    fun sessionNew(model: Long, nCtx: Int = 2048, nThreads: Int = 0, nBatch: Int = 0): Long = table.use(model) { m ->
        val ptr = NativeBridge.nativeSessionNew(m.ptr, nCtx, nThreads, nBatch)
        table.put(Handle(ptr, "session", NativeBridge::nativeSessionFree, parent = m))
    }
    fun sessionReset(session: Long) = table.use(session) { NativeBridge.nativeSessionReset(it.ptr) }
    fun sessionFree(session: Long) = table.free(session)

    fun chatFormat(model: Long, roles: List<String>, contents: List<String>, addGenerationPrompt: Boolean = true): String {
        require(roles.size == contents.size) { "roles and contents differ in length" }
        return table.use(model) {
            NativeBridge.nativeChatFormat(
                it.ptr, Array(roles.size) { i -> roles[i].utf8Bytes() }, Array(contents.size) { i -> contents[i].utf8Bytes() },
                addGenerationPrompt,
            ).utf8()
        }
    }

    /**
     * Streams complete text pieces to [sink] (return true to stop). Returns stats JSON
     * {n_prompt_tokens,n_generated,prompt_ms,gen_ms,stop_reason,peak_rss_bytes}; stop_reason 0 eos, 1 max tokens,
     * 2 cancelled (by sink or [cancel]), 3 context full. An exception thrown by [sink] stops generation and is rethrown.
     */
    fun generate(session: Long, prompt: String, params: SampleParams = SampleParams(), sink: (String) -> Boolean): String {
        val dec = Utf8Stream()
        var failure: Throwable? = null
        val cb = NativeBridge.TokenCb { bytes ->
            try {
                val text = dec.push(bytes)
                if (text.isEmpty()) false else sink(text)
            } catch (t: Throwable) {
                failure = t
                true
            }
        }
        val stats = try {
            table.use(session) {
                NativeBridge.nativeGenerate(
                    it.ptr, prompt.utf8Bytes(), params.temperature, params.topK, params.topP, params.minP,
                    params.repeatPenalty, params.seed, params.maxNewTokens, cb,
                ).utf8()
            }
        } catch (e: HagException) {
            failure?.let { throw it }
            throw e
        }
        failure?.let { throw it }
        return stats
    }

    /** Async-safe; also interrupts [score]. A cancelled [generate] returns normally with stop_reason 2. */
    fun cancel(session: Long) { table.use(session) { NativeBridge.nativeCancel(it.ptr) } }

    fun tokenize(model: Long, text: String, addSpecial: Boolean = false): IntArray =
        table.use(model) { NativeBridge.nativeTokenize(it.ptr, text.utf8Bytes(), addSpecial) }

    /** JSON {"mean_nll":..,"n_tokens":..}: mean negative log-likelihood per token (lower = better fit). */
    fun score(session: Long, text: String): String = table.use(session) { NativeBridge.nativeScore(it.ptr, text.utf8Bytes()).utf8() }

    fun trainEstimate(baseGguf: String, config: TrainConfig): String =
        NativeBridge.nativeTrainEstimate(baseGguf.utf8Bytes(), config.toNative()).utf8()

    /**
     * Blocking on-device fine-tune. Writes a patch to [outPatch]; the base file is never modified. Checkpoints go to
     * [workDir]; an identical later call resumes. [progress] returns true to cancel (checkpoint kept; throws a cancelled
     * [HagException]).
     */
    fun train(
        baseGguf: String, texts: List<String>, config: TrainConfig, workDir: String, outPatch: String,
        progress: ((TrainEvent) -> Boolean)? = null,
    ) {
        var failure: Throwable? = null
        val cb = progress?.let { p ->
            NativeBridge.TrainCb { ph, ep, eps, st, sts, ex, tl, vl, el, rss, res ->
                try { p(TrainEvent(ph, ep, eps, st, sts, ex, tl, vl, el, rss, res)) } catch (t: Throwable) { failure = t; true }
            }
        }
        try {
            NativeBridge.nativeTrain(
                baseGguf.utf8Bytes(), Array(texts.size) { texts[it].utf8Bytes() }, config.toNative(),
                workDir.utf8Bytes(), outPatch.utf8Bytes(), cb,
            )
        } catch (e: HagException) {
            failure?.let { throw it }
            throw e
        }
        failure?.let { throw it }
    }

    fun patchInfoJson(patchPath: String): String = NativeBridge.nativePatchInfo(patchPath.utf8Bytes()).utf8()

    /** Test hook: number of live model+session handles (leak checks). */
    fun liveHandles(): Int = table.size()
}

internal fun String.utf8Bytes(): ByteArray = toByteArray(Charsets.UTF_8)
internal fun ByteArray.utf8(): String = String(this, Charsets.UTF_8)
