package com.hotatticgames.llmtrainer.studio.api

/**
 * Model manager + device profiler surface (ADDITIVE; kept out of [Studio] so the chat/training work can extend that interface
 * independently). Obtain it with `ModelsFactory.of(studio)` in studio-core.
 *
 * Phone-first flow: the owner picks a base LLM from the catalog, the phone downloads and verifies it, runs it and, where
 * feasible, specializes it locally. NOTHING downloads automatically: [Studio.planAcquisition] shows size / storage /
 * compatibility / license / purpose and [Studio.startDownload] requires explicit confirmation.
 *
 * Inference feasibility is never training feasibility: every [ArtifactCapabilityView] answers them separately.
 */
data class ConditionView(val name: String, val status: String, val text: String)

data class TrainingOptionView(
    val mode: String,                 // "full" | "partial"
    val trainableLastLayers: Int, val ramMb: Int, val utilization: Double, val estTokensPerS: Double,
    val estMinutesForReferenceTokens: Double, val checkpointMb: Int, val fits: Boolean, val measured: Boolean,
)

/** What this phone can do with ONE downloadable file. Estimates are labelled low-confidence until measured on the device. */
data class ArtifactCapabilityView(
    val variantId: String, val artifactId: String, val modelId: String, val label: String, val quantization: String, val precision: String,
    val sizeBytes: Long?, val sizeIsEstimate: Boolean,
    /** "refreshed" (exact revision/size/hash known) | "unrefreshed" (catalog data not fetched yet: NOT downloadable) | "not_published". */
    val catalogState: String, val downloadable: Boolean, val downloadBlocker: String?,
    val licenseState: LicenseState,
    val canDownload: Boolean, val canLoad: Boolean, val canInfer: Boolean, val canEvaluate: Boolean,
    val canSpecializeFull: Boolean, val canSpecializePartial: Boolean, val trainableLastLayers: Int?, val externalComputeRequired: Boolean,
    val inferContext: Int?, val estTokensPerS: Double?, val estPeakRamMb: Int?,
    /** "local_full" | "local_partial" | "inference_only" | "external_only" (static class of the FILE; the device decides the rest). */
    val trainingClass: String,
    /** For an inference-only (quantized) file: the full-precision variant that local specialization would start from. */
    val specializeViaVariantId: String?,
    val full: TrainingOptionView?, val partial: TrainingOptionView?,
    val conditions: List<ConditionView>, val readyToTrainNow: Boolean,
    /** "low" | "medium" | "high": low until real on-device measurements exist. */
    val confidenceInfer: String, val confidenceTrain: String,
    val reasons: List<String>, val installed: Boolean,
)

/**
 * One owner-facing pick. choice: "fastest" | "balanced" | "best_quality" | "best_specialize".
 * tier: "recommended" (measured on this device) | "provisional" (estimates only) | "preview" (file not downloadable yet) | "none".
 */
data class ChoiceView(
    val choice: String, val label: String, val variantId: String?, val artifactId: String?, val tier: String, val confidence: String?,
    val contextTokens: Int?, val trainableLastLayers: Int?, val licenseState: LicenseState?, val downloadable: Boolean, val reason: String,
)

data class CatalogStatus(
    val artifactCount: Int, val refreshedCount: Int, val downloadableCount: Int, val unrefreshedCount: Int, val lastRefreshedAt: String?, val note: String,
)

data class ModelChoices(val device: DeviceProfile, val choices: List<ChoiceView>, val artifacts: List<ArtifactCapabilityView>, val catalog: CatalogStatus)

/** An installed model file and its install manifest. [provenance]: verified_download | download_unverified_checksum | verified_import | unverified_provenance. */
data class InstalledModel(
    val variantId: String, val artifactId: String?, val displayName: String, val path: String, val sizeBytes: Long, val sha256: String,
    val source: String, val provenance: String, val url: String?, val revision: String?, val licenseStateAtInstall: LicenseState,
    val installedAt: EpochMs, val architecture: String?, val layers: Int?, val parameterCount: Long?, val chatTemplatePresent: Boolean?,
)

data class StorageAccounting(val freeBytes: Long, val reserveBytes: Long, val installedBytes: Long, val partialBytes: Long, val headroomBytes: Long)

interface ModelsApi {
    /** Capabilities of every catalog file on this phone plus the four owner-facing picks. Cheap; re-run when the device snapshot changes. */
    fun modelChoices(): ModelChoices
    fun capabilities(variantId: String): StudioResult<ArtifactCapabilityView>
    fun catalogStatus(): CatalogStatus

    fun installedModels(): List<InstalledModel>
    /** Absolute path of the loadable file, or null (never a partial download). */
    fun installedPath(variantId: String): String?
    fun uninstall(variantId: String): StudioResult<StorageAccounting>
    fun storage(): StorageAccounting
    /** Manual import of a user-supplied GGUF; recorded with "unverified_provenance". Progress via [Studio.operation]. */
    fun importUserModel(file: SourceInput): StudioResult<Operation>

    /**
     * Records one on-device measurement (JSON of the `device_measurement.v1` schema, see docs/studio/DEVICE_PROFILER.md).
     * Merged by the latest-wins rule; replaces the matching estimate and raises confidence. Returns the number of records now stored.
     */
    fun recordMeasurement(recordJson: String): StudioResult<Int>
    fun measurementRecords(): List<String>
}
