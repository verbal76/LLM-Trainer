package com.hotatticgames.hag.runtime

/**
 * Raw JNI surface of libhagrt.so. Internal plumbing: use [HagRuntime] / [HagEngine]. Strings are UTF-8 byte arrays
 * (JNI's modified UTF-8 cannot carry supplementary characters). Handles are native pointers; [HagEngine] wraps
 * them in validated, ref-counted ids so a stale handle can never reach native code.
 * Failures are reported by throwing [HagException] from native code.
 */
internal object NativeBridge {
    fun interface LoadCb { fun onProgress(fraction: Float): Boolean }
    fun interface TokenCb { fun onPiece(bytes: ByteArray): Boolean }
    fun interface TrainCb {
        fun onEvent(
            phase: Int, epoch: Int, epochs: Int, step: Int, steps: Int, examplesDone: Long,
            trainLoss: Double, valLoss: Double, elapsedS: Double, rssBytes: Long, resumedFromStep: Int,
        ): Boolean
    }

    // libhagrt.so (baseline ISA): safe to call on any CPU.
    @JvmStatic external fun nativeCpuCheck(): ByteArray?
    @JvmStatic external fun nativeCpuAbi(): ByteArray

    // Require libhagengine.so to be loaded + bound.
    @JvmStatic external fun nativeBind(): ByteArray?
    @JvmStatic external fun nativeEngineInit(): ByteArray?
    @JvmStatic external fun nativeEngineShutdown()
    @JvmStatic external fun nativeVersion(): ByteArray
    @JvmStatic external fun nativeSystemInfo(): ByteArray

    @JvmStatic external fun nativeModelLoad(path: ByteArray, useMmap: Boolean, cb: LoadCb?): Long
    @JvmStatic external fun nativeModelApplyPatch(model: Long, path: ByteArray)
    @JvmStatic external fun nativeModelInfo(model: Long): ByteArray
    @JvmStatic external fun nativeModelFree(model: Long)

    @JvmStatic external fun nativeSessionNew(model: Long, nCtx: Int, nThreads: Int, nBatch: Int): Long
    @JvmStatic external fun nativeSessionReset(session: Long)
    @JvmStatic external fun nativeSessionFree(session: Long)
    @JvmStatic external fun nativeCancel(session: Long)

    @JvmStatic external fun nativeChatFormat(model: Long, roles: Array<ByteArray>, contents: Array<ByteArray>, addGen: Boolean): ByteArray
    @JvmStatic external fun nativeGenerate(
        session: Long, prompt: ByteArray, temperature: Float, topK: Int, topP: Float, minP: Float, repeatPenalty: Float,
        seed: Long, maxNew: Int, sink: TokenCb,
    ): ByteArray
    @JvmStatic external fun nativeTokenize(model: Long, text: ByteArray, addSpecial: Boolean): IntArray
    @JvmStatic external fun nativeScore(session: Long, text: ByteArray): ByteArray

    @JvmStatic external fun nativeTrainEstimate(base: ByteArray, params: DoubleArray): ByteArray
    @JvmStatic external fun nativeTrain(
        base: ByteArray, texts: Array<ByteArray>, params: DoubleArray, workDir: ByteArray, outPatch: ByteArray, progress: TrainCb?,
    )
    @JvmStatic external fun nativePatchInfo(path: ByteArray): ByteArray
}

/** [code] mirrors HAG_ERR_* (negative); -100 = binding usage error (stale handle, bad argument). */
class HagException(val code: Int, message: String) : RuntimeException(message) {
    /** Constructor used by the JNI layer (message arrives as UTF-8 bytes). */
    @Suppress("unused")
    constructor(code: Int, message: ByteArray?) : this(code, message?.toString(Charsets.UTF_8) ?: "")

    val cancelled: Boolean get() = code == CANCELLED

    companion object {
        const val INVALID_ARG = -1
        const val IO = -2
        const val BAD_MODEL = -3
        const val OOM = -4
        const val CANCELLED = -5
        const val UNSUPPORTED = -6
        const val CORRUPT = -7
        const val INTERNAL = -8
        const val USAGE = -100
    }
}
