package com.hotatticgames.llmtrainer.studio.api

/*
 * Studio v2 (phone-first) value types. Additive to Types.kt. See docs/studio/V2_API.md for the contract and semantics.
 *
 * Honesty rules baked into the types:
 *  - Retrieval / prompting / dataset building are NEVER training. Every option and every message that used source context says so
 *    ([TrainingOption.isTraining], [TrainingOption.changesParameters], [ChatMessageRecord.contextLabel]).
 *  - Nothing here is a pretend result: when the native engine is unavailable the calls return an error naming the reason.
 */

// ===== Engine & installed models ==================================================================================

data class EngineStatus(
    val runtimeId: String,
    val inferenceAvailable: Boolean, val inferenceReason: String?,
    val trainingAvailable: Boolean, val trainingReason: String?,
)

data class InstalledModel(
    val modelId: String, val variantId: String, val name: String, val quant: String, val sizeBytes: Long,
    val licenseState: LicenseState,
    /** Projects whose selected base model is this variant. */
    val activeForProjects: List<ProjectId>,
)

data class Capability(val ok: Boolean, val reason: String?)

/** What the project can do right now with its selected base model (drives Chat / Train / Evaluate buttons). */
data class ProjectModelState(
    val projectId: ProjectId, val baseModelId: String?, val variantId: String?, val baseInstalled: Boolean,
    val baseSizeBytes: Long?, val selectedSpecialistId: String?, val specialistCount: Int,
    val canChatBase: Capability, val canChatSpecialist: Capability,
    val canTrainLocally: Capability, val canEvaluateLocally: Capability,
)

// ===== Chat ==========================================================================================================

enum class ChatTarget { BASE, SPECIALIST }
enum class ChatRole { SYSTEM, USER, ASSISTANT }

/** A chunk of the owner's own source material injected into the PROMPT. Retrieval, not training. */
data class ContextChunk(val ref: String, val sourceName: String, val page: Int?, val section: String?, val excerpt: String)

data class GenerationStats(
    val promptTokens: Int, val generatedTokens: Int,
    val timeToFirstTokenMs: Double?, val tokensPerSecond: Double, val totalMs: Double,
    val modelLoadMs: Long?, val peakRssMb: Int?, val stopReason: StopReason,
)

data class ChatOptions(
    /** Inject the best-matching source chunks into the prompt (retrieval). Clearly labelled; does not change the model. */
    val useSourceContext: Boolean = false, val maxContextChunks: Int = 3,
    val systemPrompt: String? = null, val contextTokens: Int = 2048,
    val sampling: SamplingParams = SamplingParams(),
)

data class ChatSessionInfo(
    val id: String, val projectId: ProjectId, val target: ChatTarget, val specialistId: String?, val title: String,
    val createdAt: EpochMs, val updatedAt: EpochMs, val messageCount: Int, val options: ChatOptions,
    /** Human label of the model answering, e.g. "Base: Qwen3 4B" or "Specialist v1.0.2 (parameters changed on this phone)". */
    val modelLabel: String,
)

data class ChatMessageRecord(
    val id: String, val role: ChatRole, val text: String, val at: EpochMs,
    val contextUsed: List<ContextChunk>, val stats: GenerationStats?, val interrupted: Boolean,
) {
    /** Non-null exactly when source context was injected; the UI must show it next to the answer. */
    val contextLabel: String? get() = if (contextUsed.isEmpty()) null else CONTEXT_LABEL

    companion object {
        const val CONTEXT_LABEL = "Answered with source excerpts in the prompt (retrieval). This is not training and does not change the model."
    }
}

// ===== Local training plan ===============================================================================================

enum class TrainingMethodKind { LOCAL_FULL, LOCAL_PARTIAL, EXTERNAL_COMPUTE, RAG_ONLY, PROMPT_ONLY }
enum class Risk { LOW, MEDIUM, HIGH, UNKNOWN }

/** Estimates, never measurements (`basis` says what they were derived from). */
data class ResourceEstimate(
    val peakRamMb: Int?, val minMinutes: Int?, val maxMinutes: Int?, val thermalRisk: Risk, val batteryPercent: Int?,
    val workingStorageMb: Long?, val basis: String,
)

data class DeviceConditions(
    val availableRamMb: Int, val freeStorageMb: Long, val batteryPercent: Int?, val charging: Boolean?,
    val thermal: String?, val powerSave: Boolean?,
)

data class TrainingOption(
    val kind: TrainingMethodKind, val label: String,
    /** Does this change model parameters (the only thing the word "training" may be used for)? */
    val isTraining: Boolean, val changesParameters: Boolean, val whereItRuns: RunLocation,
    val available: Boolean, val recommended: Boolean,
    /** LOCAL_PARTIAL: how many trailing transformer blocks are updated. */
    val trainableLastLayers: Int?,
    val reasons: List<String>, val blockers: List<Blocker>, val requirements: List<String>,
    val estimate: ResourceEstimate?, val honestyNote: String,
)

data class TrainingSettings(
    val kind: TrainingMethodKind = TrainingMethodKind.LOCAL_PARTIAL,
    val trainableLastLayers: Int = 4, val epochs: Int = 1, val learningRate: Float = 2e-5f,
    val contextTokens: Int = 512, val seed: Long = 1337, val trainEmbeddings: Boolean = false,
    val checkpointEverySteps: Int = 0,
    /** Cap on training sequences (memory/time safety). */
    val maxSequences: Int = 2000,
    /** Host does not report charger state: owner asserts the phone is plugged in. */
    val ownerConfirmsPluggedIn: Boolean = false,
)

data class LocalTrainingPlan(
    val projectId: ProjectId, val baseModelId: String?, val datasetSha256: String?, val datasetApproved: Boolean,
    val trainSequences: Int, val device: DeviceConditions, val options: List<TrainingOption>,
    val recommended: TrainingMethodKind?, val defaultSettings: TrainingSettings?, val generatedAt: EpochMs,
)

// ===== Training run ==========================================================================================================

enum class TrainingRunState { QUEUED, RUNNING, PAUSED, SUCCEEDED, FAILED, CANCELLED }
enum class TrainingStage { PREFLIGHT, PREPARE, TRAIN, EVAL, SAVE, VERIFY, DONE }
enum class CheckpointState {
    NONE, PRESENT,
    /** (PRESENT = checkpoint files exist; validity is checked by the engine on resume.) A checkpoint failed validation, was set aside, and the run restarted from the beginning. */
    RECOVERED_FROM_SCRATCH,
}

data class LossPoint(val step: Int, val trainLoss: Double?, val valLoss: Double?)

data class TrainingRun(
    val id: String, val projectId: ProjectId, val state: TrainingRunState, val stage: TrainingStage,
    val settings: TrainingSettings, val baseModelId: String, val datasetSha256: String, val sequences: Int,
    val epoch: Int, val epochs: Int, val step: Int, val steps: Int, val examplesDone: Long,
    val lossTrend: List<LossPoint>, val latestTrainLoss: Double?, val latestValLoss: Double?,
    val elapsedMs: Long, val thermal: String?, val batteryPercent: Int?, val rssMb: Int?,
    val checkpoint: CheckpointState, val resumedFromStep: Int,
    val message: String, val error: StudioError?, val resumable: Boolean,
    /** Set once the run produced a verified specialist. */
    val specialistId: String?, val createdAt: EpochMs, val updatedAt: EpochMs,
) {
    val isTerminal: Boolean get() = state == TrainingRunState.SUCCEEDED || state == TrainingRunState.FAILED || state == TrainingRunState.CANCELLED
}

// ===== Specialist artifact registry ==============================================================================================

data class SpecialistInfo(
    val id: String, val projectId: ProjectId, val name: String, val version: String, val createdAt: EpochMs,
    val baseModelId: String, val baseVariantId: String?, val baseSha256: String, val patchSha256: String, val patchSizeBytes: Long,
    val datasetSha256: String, val trainingRunId: String, val method: TrainingMethodKind, val config: TrainingSettings,
    val finalTrainLoss: Double?, val finalValLoss: Double?, val steps: Int, val examples: Long,
    /** True when the patch hash, base hash and structure were re-verified on the last (re)load. */
    val verified: Boolean, val verifyMessage: String,
    val selected: Boolean, val locallyEvaluated: Boolean,
    val statement: String,   // always: "Parameters changed on this device by fine-tuning ..." (what this artifact IS)
    /** Trained from material that has since changed (source removed/edited, dataset rebuilt): the artifact no longer matches the current dataset. */
    val stale: Boolean = false, val staleReason: String? = null, val sourceIds: List<String> = emptyList(),
)

// ===== Local evaluation ====================================================================================================

data class LocalEvalOptions(
    val maxItems: Int = 60, val seed: Long = 7, val bootstrapSamples: Int = 1000,
    val maxNewTokens: Int = 96, val includeGroundedMetrics: Boolean = true, val includeRetention: Boolean = true,
)

data class PerfRow(val subject: String, val latencyMsMean: Double, val latencyMsP95: Double, val tokensPerSecond: Double, val loadMs: Long, val peakRssMb: Int?)

enum class LocalEvalState { QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED, INTERRUPTED }

data class LocalEvalItemResult(
    val itemId: String, val kind: String, val question: String, val sourceRef: String,
    val baseAnswer: String, val specialistAnswer: String, val baseOk: Boolean?, val specialistOk: Boolean?,
)

data class LocalEvaluation(
    val id: String, val projectId: ProjectId, val specialistId: String, val state: LocalEvalState,
    val progress: Progress, val message: String, val error: StudioError?,
    /** Present when SUCCEEDED. Metrics are reported one by one (no aggregate score). Measured on TEST-split chunks only. */
    val view: EvaluationView?, val performance: List<PerfRow>, val items: List<LocalEvalItemResult>,
    val datasetSha256: String, val testChunks: Int, val itemCount: Int, val startedAt: EpochMs, val finishedAt: EpochMs?,
)

// ===== A/B comparison =======================================================================================================

data class ABSide(val label: String, val target: ChatTarget, val text: String, val stats: GenerationStats?, val contextUsed: List<ContextChunk>)

data class ABComparison(
    val id: String, val projectId: ProjectId, val specialistId: String, val prompt: String,
    val base: ABSide, val specialist: ABSide, val createdAt: EpochMs, val note: String?,
    val savedAsNote: Boolean,
)

data class ABOptions(
    val useSourceContext: Boolean = false, val maxContextChunks: Int = 3, val systemPrompt: String? = null,
    val contextTokens: Int = 2048, val sampling: SamplingParams = SamplingParams.greedy(256),
)
