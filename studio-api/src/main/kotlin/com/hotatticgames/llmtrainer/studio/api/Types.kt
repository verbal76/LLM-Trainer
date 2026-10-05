package com.hotatticgames.llmtrainer.studio.api

import java.io.InputStream

// ===== Common ==================================================================================================

/** Timestamps are epoch milliseconds (UTC). */
typealias EpochMs = Long

data class Page<T>(val items: List<T>, val total: Int, val offset: Int)

data class Progress(val done: Long, val total: Long, val unit: String) {
    /** 0.0..1.0, or -1.0 when total is unknown. */
    val fraction: Double get() = if (total <= 0) -1.0 else (done.toDouble() / total).coerceIn(0.0, 1.0)
}

/** A user-picked input. `open` may be called more than once; the caller of `open` closes the stream. */
class SourceInput(val name: String, val mime: String, val sizeBytes: Long, val open: () -> InputStream)

/** Result of writing a package/export to an OutputStream. */
data class ExportedPackage(
    val kind: String,              // "training-job" | "reference" | "heldout-eval" | "specialist"
    val suggestedFileName: String,
    val sizeBytes: Long,
    val sha256: String,
    val files: List<String>,       // entry names inside the package
    val warnings: List<String>,
)

// ===== Projects & stages =======================================================================================

data class ProjectId(val value: String)

enum class StageId { BASE_MODEL, SOURCES, DATASET, METHOD, TRAINING_PACKAGE, EVALUATION, EXPORT }
enum class StageStatus { NOT_STARTED, IN_PROGRESS, NEEDS_ATTENTION, DONE, BLOCKED }

/** Screens the UI can route to from a next-action. */
enum class Screen {
    DASHBOARD, CREATE_PROJECT, DEVICE_PROFILE, RECOMMENDATIONS, MODEL_DETAIL, ACQUIRE_MODEL, ADD_SOURCES,
    INGESTION_REPORT, DATASET_BUILD, DATASET_REVIEW, METHOD, TRAINING_PACKAGE, EVALUATION, EXPORT, IMPORT_PACKAGE
}

data class NextAction(val label: String, val screen: Screen?)

data class Stage(val id: StageId, val status: StageStatus, val summary: String, val nextAction: NextAction?)

data class ProjectSummary(
    val id: ProjectId, val name: String, val domain: String, val purpose: String,
    val createdAt: EpochMs, val updatedAt: EpochMs,
    val baseModelId: String?, val stages: List<Stage>,
    /** Next action of the first stage that is not DONE (what a dashboard card shows). */
    val nextAction: NextAction?,
)

data class NewProject(val name: String, val domain: String, val purpose: String)

// ===== Device & recommendations ================================================================================

data class DeviceProfile(
    val deviceName: String, val androidApi: Int, val abi: String,
    val totalRamMb: Int, val availableRamMb: Int, val freeStorageMb: Long,
    val nativeRuntimeId: String,       // "none-v1" on the v1 host
    val safetyReserveFraction: Double, // configurable no-degradation reserve
    val capturedAt: EpochMs,
)

enum class ProfileKind { PERFORMANCE, BALANCED, MAX_QUALITY }
enum class ConfidenceLevel { HIGH, MEDIUM, LOW }

/** Honest confidence: `benchmarked=false` means an estimate from the qualifier, never a measurement. */
data class Confidence(val level: ConfidenceLevel, val basis: String, val benchmarked: Boolean)

data class ProfileRecommendation(
    val kind: ProfileKind,
    val modelId: String?, val variantId: String?,   // null when nothing qualifies
    val quant: String?, val contextTokens: Int?, val estimatedPeakRamMb: Int?,
    val reasons: List<String>,
    val warnings: List<String>,
    val confidence: Confidence,
    val licenseState: LicenseState?,
)

data class Recommendations(val device: DeviceProfile, val profiles: List<ProfileRecommendation>)

// ===== Catalog, variants, licenses =============================================================================

enum class LicenseState { VERIFIED, UNVERIFIED, DISALLOWED }
enum class Tri { YES, NO, CONDITIONAL, UNVERIFIED }
enum class Permission { COMMERCIAL, FINE_TUNE, ADAPTER, REDISTRIBUTION }

data class LicenseInfo(
    val state: LicenseState,
    val licenseName: String,
    val authoritativeUrl: String,
    val evidenceLevel: String,              // "none" | "secondary" | "owner_reviewed_text"
    val scopeText: String?,                 // set once VERIFIED
    val textSha256: String?, val fetchedAt: EpochMs?,
    val permissions: Map<Permission, Tri>,  // claims until VERIFIED; ignored by the gate when not VERIFIED
    val attributionRequired: Boolean?,
    val disallowedReason: String?,
)

data class LicenseTextFetch(val url: String, val fetchedAt: EpochMs, val sha256: String, val text: String, val truncated: Boolean)

data class LicenseAttestation(
    val textSha256: String,                 // must equal the sha256 of the fetch/import being attested
    val permissions: Map<Permission, Tri>,
    val attributionRequired: Boolean,
    val note: String? = null,
)

enum class Verdict { FITS_SAFELY, TIGHT, DOES_NOT_FIT, UNKNOWN }
enum class RunLocation { DEVICE, DESKTOP, NONE }

data class Feasibility(val verdict: Verdict, val estimatedPeakRamMb: Int?, val reasons: List<String>)
data class TrainingFeasibility(val where: RunLocation, val method: String?, val reasons: List<String>)
data class Provenance(val sourceUrl: String, val retrievedOn: String?, val evidenceLevel: String, val note: String?)

data class ModelVariant(
    val id: String, val modelId: String, val format: String, val quant: String,
    val sizeBytes: Long, val paramsBillions: Double, val architecture: String, val contextTokensMax: Int,
    val android: Feasibility, val training: TrainingFeasibility,
    val provenance: Provenance, val downloadUrl: String?, val sha256: String?,
    val acquired: Boolean,
)

data class CatalogModel(
    val id: String, val family: String, val name: String, val version: String,
    val paramsBillions: Double, val architecture: String,
    val license: LicenseInfo, val variants: List<ModelVariant>, val provenance: Provenance,
)

// ===== Acquisition & operations ================================================================================

data class IntendedUse(
    val fineTune: Boolean = true, val adapter: Boolean = true,
    val commercial: Boolean = false, val redistribute: Boolean = false,
)

data class Blocker(val code: String, val message: String)

data class AcquisitionPlan(
    val variantId: String, val modelName: String, val url: String?,
    val sizeBytes: Long, val freeStorageBytes: Long, val storageAfterBytes: Long, val storageReserveBytes: Long,
    val deviceCompat: Feasibility, val licenseState: LicenseState, val intendedUse: IntendedUse,
    val allowed: Boolean, val blocking: List<Blocker>,
    val requiresExplicitConfirmation: Boolean = true,
)

enum class OperationKind { DOWNLOAD, IMPORT_MODEL }
enum class OperationState { QUEUED, RUNNING, PAUSED, SUCCEEDED, FAILED, CANCELLED }

/**
 * Long-running work. Ids are persisted by the implementation; after process death a RUNNING operation is
 * reloaded as PAUSED (`resumable == true`) and the UI offers Resume. Poll [Studio.operation].
 */
data class Operation(
    val id: String, val kind: OperationKind, val projectId: ProjectId?, val subject: String,
    val state: OperationState, val progress: Progress, val message: String,
    val error: StudioError?, val resumable: Boolean, val createdAt: EpochMs, val updatedAt: EpochMs,
) {
    val isTerminal: Boolean
        get() = state == OperationState.SUCCEEDED || state == OperationState.FAILED || state == OperationState.CANCELLED
}

// ===== Sources & ingestion =====================================================================================

enum class IngestStatus { INGESTED, DUPLICATE, FAILED }
enum class IssueCode { CORRUPT_FILE, NEEDS_OCR, UNSUPPORTED_TYPE, DUPLICATE_CONTENT, EMPTY_TEXT, TOO_LARGE, LOW_TEXT_QUALITY, RIGHTS_UNSET }
data class IngestIssue(val code: IssueCode, val message: String)

enum class RightsStatus { UNSET, OWNER_AUTHORED, LICENSED_FOR_TRAINING, PERMISSION_GRANTED, REFERENCE_ONLY }

data class ProvenanceSummary(
    val sourceId: String, val sha256: String, val mime: String, val sizeBytes: Long, val ingestedAt: EpochMs,
    val pages: Int?, val chunks: Int, val extractor: String, val rights: RightsStatus,
)

data class IngestItem(
    val name: String, val status: IngestStatus, val sourceId: String?, val duplicateOfSourceId: String?,
    val issues: List<IngestIssue>, val provenance: ProvenanceSummary?,
)

data class IngestReport(val projectId: ProjectId, val items: List<IngestItem>, val at: EpochMs) {
    val ingested: List<IngestItem> get() = items.filter { it.status == IngestStatus.INGESTED }
    val duplicates: List<IngestItem> get() = items.filter { it.status == IngestStatus.DUPLICATE }
    val failed: List<IngestItem> get() = items.filter { it.status == IngestStatus.FAILED }
}

data class SourceRecord(val sourceId: String, val name: String, val provenance: ProvenanceSummary, val issues: List<IngestIssue>)

// ===== Dataset build & review ==================================================================================

enum class DatasetStatus { NONE, DRAFT, NEEDS_REVIEW, STALE, APPROVED }
enum class ChunkRole { TRAIN, VALIDATION, HELD_OUT_EVAL, REFERENCE }
enum class ChunkOrigin { SOURCE_DERIVED, SYNTHETIC }
enum class ReviewFlag { DUPLICATE, LOW_CONFIDENCE, TABLE, OCR_NOISE, POSSIBLE_LEAKAGE, HEADER_FOOTER_NOISE, CONTRADICTION, OUTDATED_REVISION }

data class DatasetOptions(
    val seed: Long = 1337, val chunkTargetTokens: Int = 400,
    val heldOutFraction: Double = 0.10, val validationFraction: Double = 0.10,
    val includeSynthetic: Boolean = false, val excludeReferenceOnlySources: Boolean = true,
)

data class DatasetStats(
    val totalChunks: Int, val included: Int, val excluded: Int, val flagged: Int,
    val byRole: Map<ChunkRole, Int>, val byOrigin: Map<ChunkOrigin, Int>, val byFlag: Map<ReviewFlag, Int>,
    val leakageSuspects: Int,
)

data class DatasetPreview(
    val projectId: ProjectId, val version: Int, val datasetSha256: String, val status: DatasetStatus,
    val options: DatasetOptions, val stats: DatasetStats, val builtAt: EpochMs, val approvedAt: EpochMs?,
    val warnings: List<String>,
)

data class ReviewItem(
    val id: String, val sourceId: String, val sourceName: String, val page: Int?, val section: String?,
    val role: ChunkRole, val origin: ChunkOrigin, val excerpt: String, val flags: List<ReviewFlag>, val included: Boolean,
)

data class ReviewFilter(
    val onlyFlagged: Boolean = false, val flag: ReviewFlag? = null, val role: ChunkRole? = null,
    val included: Boolean? = null, val sourceId: String? = null,
)

// ===== Methods =================================================================================================

data class MethodOption(
    val id: String, val label: String, val isTraining: Boolean, val whereItRuns: RunLocation,
    val available: Boolean, val whyNotAvailable: String?, val honestyNote: String,
)

object MethodIds {
    const val REFERENCE_PACKAGE = "reference-package"
    const val PROMPT_SPECIALIZATION = "prompt-specialization"
    const val ADAPTER_DESKTOP = "adapter-training-desktop"
    const val ADAPTER_ON_DEVICE = "adapter-training-on-device"
}

// ===== Evaluation & specialist packages ========================================================================

data class MetricRow(
    val id: String, val label: String, val higherIsBetter: Boolean,
    val base: Double, val specialist: Double, val delta: Double,
    val ciLow: Double?, val ciHigh: Double?, val n: Int,
)

data class EvaluationView(
    val projectId: ProjectId, val runId: String, val baseModelLabel: String, val specialistLabel: String,
    val metrics: List<MetricRow>, val caveats: List<String>,
    val improvementClaimAllowed: Boolean, val claimReason: String,
    val isStub: Boolean, val importedAt: EpochMs,
)

data class CheckResult(val name: String, val passed: Boolean, val detail: String)

data class SpecialistPackageView(
    val name: String, val version: String, val baseModelId: String, val baseModelVersion: String,
    val licenseState: LicenseState, val licenseSummary: String,
    val datasetSha256: String?, val artifactHashes: Map<String, String>,
    val method: String, val evaluationSummary: String?, val improvementClaimAllowed: Boolean,
    val compatibility: List<String>, val knownLimitations: List<String>,
    val validation: List<CheckResult>, val valid: Boolean,
)

// ===== Host passthrough ========================================================================================

data class UpdateStatus(val state: String, val message: String)   // "checking" | "up-to-date" | "available" | "error"

interface HostHooks {
    /** Raw host diagnostics JSON (read lazily after the view is attached; see PRODUCT.md). */
    fun diagnosticsJson(): String
    fun deviceSnapshotJson(): String
    fun checkForUpdates(callback: (UpdateStatus) -> Unit)
    fun restartApp()
}
