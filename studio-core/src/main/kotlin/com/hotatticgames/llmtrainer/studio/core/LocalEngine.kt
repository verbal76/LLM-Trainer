package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import java.io.File
import java.util.concurrent.locks.ReentrantLock

/** Thrown inside the local-studio services and converted to [StudioResult.Err] at the API boundary. */
class StudioException(val error: StudioError) : RuntimeException(error.message)

fun blockedEx(code: String, msg: String, vararg more: Blocker) = StudioException(StudioError.Blocked(code, msg, listOf(Blocker(code, msg)) + more))

/** Maps a backend failure to the honest studio error (never a pretend success). */
fun BackendException.toStudioError(): StudioError = when (code) {
    BackendError.UNAVAILABLE -> StudioError.Blocked("ENGINE_UNAVAILABLE", message ?: "The native engine is not available", listOf(Blocker("ENGINE_UNAVAILABLE", message ?: "")))
    BackendError.OOM -> StudioError.Blocked("OUT_OF_MEMORY", message ?: "Not enough memory", listOf(Blocker("OUT_OF_MEMORY", message ?: "")))
    BackendError.UNSUPPORTED -> StudioError.Blocked("UNSUPPORTED", message ?: "Unsupported", listOf(Blocker("UNSUPPORTED", message ?: "")))
    BackendError.CANCELLED -> StudioError.Cancelled()
    BackendError.BAD_MODEL -> StudioError.Invalid("BAD_MODEL", message ?: "The model file could not be read")
    BackendError.CORRUPT -> StudioError.Invalid("CORRUPT", message ?: "Corrupt data")
    BackendError.INVALID_ARG -> StudioError.Invalid("INVALID_ARG", message ?: "Invalid argument")
    BackendError.IO -> StudioError.Io(message ?: "I/O error")
    BackendError.INTERNAL -> StudioError.Io("Engine error: ${message ?: ""}")
}

/** Runs [f] and converts exceptions into results. Calls never throw for expected failures. */
inline fun <T> guarded(f: () -> T): StudioResult<T> = try { StudioResult.Ok(f()) }
catch (e: StudioException) { StudioResult.Err(e.error) }
catch (e: BackendException) { StudioResult.Err(e.toStudioError()) }
catch (e: java.io.IOException) { StudioResult.Err(StudioError.Io(e.message ?: e.javaClass.simpleName)) }
catch (e: RuntimeException) { StudioResult.Err(StudioError.Io("Internal error (${e.javaClass.simpleName}): ${e.message}")) }

/**
 * Owns the (at most one) loaded model and the training/inference exclusion. One engine, limited RAM: while a training run is
 * active every inference call fails with Conflict; while an inference call runs, training cannot start.
 */
class EngineCore(val inference: InferenceBackend?, val trainer: TrainingBackend?) {
    val lock = ReentrantLock()
    @Volatile var trainingActive = false

    class Loaded(val key: String, val model: ModelHandle, var chat: ChatHandle, var ctx: Int, val info: ModelInfo)
    private var cached: Loaded? = null

    fun inferenceStatus(): BackendStatus = inference?.status() ?: BackendStatus(false, "none", "No inference engine is installed in this app build.")
    fun trainingStatus(): BackendStatus = trainer?.status() ?: BackendStatus(false, "none", "No training engine is installed in this app build.")

    fun requireInference(): InferenceBackend {
        val st = inferenceStatus()
        if (!st.available || inference == null) throw blockedEx("ENGINE_UNAVAILABLE", st.reason ?: "The native engine is not available on this install.")
        return inference
    }

    private fun keyOf(base: File, patch: File?) = base.absolutePath + "|" + base.length() + "|" + (patch?.let { it.absolutePath + "|" + it.length() + "|" + it.lastModified() } ?: "-")

    /** Closes whatever is loaded. Safe to call when idle (the host may call it when the app goes to the background). */
    fun release() {
        lock.lock()
        try { releaseLocked() } finally { lock.unlock() }
    }

    private fun releaseLocked() {
        val c = cached ?: return
        cached = null
        val inf = inference ?: return
        try { inf.closeChat(c.chat) } catch (_: Exception) { }
        try { inf.closeModel(c.model) } catch (_: Exception) { }
    }

    /** Fails fast (Conflict) when training runs or another model operation is in flight; loads [base] (+[patch]) if not already resident. */
    fun <T> withModel(base: File, patch: File?, ctxTokens: Int, keepLoaded: Boolean, f: (Loaded) -> T): T {
        val inf = requireInference()
        if (!lock.tryLock()) throw StudioException(StudioError.Conflict("Another model operation is running; wait for it to finish or stop it."))
        try {
            if (trainingActive) throw StudioException(StudioError.Conflict("Training is running. The model is unavailable until it pauses or finishes (one engine, limited RAM)."))
            val key = keyOf(base, patch)
            var c = cached
            if (c != null && c.key != key) { releaseLocked(); c = null }
            if (c == null) {
                val m = inf.loadModel(base.absolutePath, patch?.absolutePath)
                try {
                    val info = inf.modelInfo(m)
                    c = Loaded(key, m, inf.newChat(m, ctxTokens), ctxTokens, info)
                } catch (e: RuntimeException) { try { inf.closeModel(m) } catch (_: Exception) { }; throw e }
                cached = c
            } else if (c.ctx != ctxTokens) {
                inf.closeChat(c.chat)
                c.chat = inf.newChat(c.model, ctxTokens); c.ctx = ctxTokens
            }
            try { return f(c) } finally { if (!keepLoaded) releaseLocked() }
        } finally { lock.unlock() }
    }

    /** Training takes the engine exclusively. Returns false when an inference call is running (caller reports Conflict). */
    fun beginTraining(): Boolean {
        if (!lock.tryLock()) return false
        try { releaseLocked(); trainingActive = true; return true } finally { lock.unlock() }
    }
    fun endTraining() { trainingActive = false }
}

// ===== what the local services need from the surrounding StudioCore ===============================================

class BaseRef(val modelId: String, val variantId: String?, val name: String, val file: File?, val gate: List<Blocker>, val sha256: String? = null)

class SourceFact(val sha256: String, val trainable: Boolean, val name: String)

/** Immutable view of a project's dataset taken under the project lock. `assembled` is computed lazily (it is not cheap). */
class DataView(
    val sha: String, val status: DatasetStatus, val splitsAvailable: Boolean, val chunks: List<DChunk>, val included: Set<String>,
    val domain: String, private val assemble: () -> DatasetEngine.Assembled,
) {
    val assembled: DatasetEngine.Assembled by lazy(assemble)
    fun testChunks(): List<DChunk> = chunks.filter { it.split == "test" && it.ref in included }
}

interface LocalHost {
    fun projectDir(id: ProjectId): File?
    fun projectIds(): List<ProjectId>
    fun projectName(id: ProjectId): String?
    fun baseRef(id: ProjectId): BaseRef?
    fun dataView(id: ProjectId): DataView?
    fun sourceFacts(id: ProjectId): Map<String, SourceFact>
    fun installed(): List<InstalledModel>
    fun snapshotJson(): String
}
