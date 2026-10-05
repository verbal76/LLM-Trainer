package com.hotatticgames.llmtrainer.studio.api

import java.io.InputStream
import java.io.OutputStream

/** Machine code + human message. Sealed so the UI can `when` exhaustively. */
sealed class StudioError(val code: String, val message: String) {
    class NotFound(what: String) : StudioError("NOT_FOUND", "Not found: $what")
    class Invalid(code: String, message: String) : StudioError(code, message)
    /** Policy refusal (license gate, missing confirmation, unapproved dataset ...). `reasons` are user-presentable. */
    class Blocked(code: String, message: String, val reasons: List<Blocker>) : StudioError(code, message)
    class Io(message: String) : StudioError("IO_ERROR", message)
    class Network(message: String) : StudioError("NETWORK_ERROR", message)
    class Cancelled : StudioError("CANCELLED", "Cancelled")
    class Conflict(message: String) : StudioError("CONFLICT", message)
    override fun toString() = "$code: $message"
}

sealed class StudioResult<out T> {
    data class Ok<T>(val value: T) : StudioResult<T>()
    data class Err(val error: StudioError) : StudioResult<Nothing>()

    fun getOrNull(): T? = (this as? Ok)?.value
    fun errorOrNull(): StudioError? = (this as? Err)?.error
    inline fun <R> map(f: (T) -> R): StudioResult<R> = when (this) { is Ok -> Ok(f(value)); is Err -> this }
    inline fun <R> fold(ok: (T) -> R, err: (StudioError) -> R): R = when (this) { is Ok -> ok(value); is Err -> err(error) }
}

/**
 * The single seam between the UI (platform Views) and the implementation (studio-core).
 *
 * Threading contract: every call is BLOCKING and may do IO; the UI calls from a background executor and posts to the
 * main thread. Implementations must be thread-safe. Calls never throw for expected failures: they return
 * [StudioResult.Err]. Long-running work is an [Operation] (downloads, model import): start returns immediately,
 * poll [operation]/[operations]. `onProgress` callbacks (ingest/dataset build) run on the calling thread.
 * All state is persisted by the implementation so process death loses nothing.
 */
interface Studio {
    // ---- Projects --------------------------------------------------------------------------------------------
    fun listProjects(): List<ProjectSummary>
    fun createProject(p: NewProject): StudioResult<ProjectSummary>
    fun getProject(id: ProjectId): StudioResult<ProjectSummary>
    fun deleteProject(id: ProjectId): StudioResult<Unit>

    // ---- Device & recommendations ----------------------------------------------------------------------------
    fun deviceProfile(): DeviceProfile
    /** Qualifier-driven; one entry per [ProfileKind]. Honest confidence (estimates are not benchmarks). */
    fun recommendations(): Recommendations

    // ---- Catalog & license evidence --------------------------------------------------------------------------
    fun catalog(): List<CatalogModel>
    fun model(modelId: String): StudioResult<CatalogModel>
    /** Fetch the license text from the model's AUTHORITATIVE url (HTTPS), hash it, remember it for attestation. */
    fun fetchLicenseText(modelId: String): StudioResult<LicenseTextFetch>
    /** Import a license file instead (url is reported as "file:<name>"). */
    fun importLicenseText(modelId: String, fileName: String, input: InputStream): StudioResult<LicenseTextFetch>
    /**
     * Owner attests permissions for the text identified by `attestation.textSha256` (must match a prior fetch/import
     * for this model). Result is the new [LicenseInfo]: VERIFIED if fine-tune and adapter are YES, else DISALLOWED.
     * DISALLOWED-by-registry models cannot be re-verified here (Blocked).
     */
    fun attestLicense(modelId: String, attestation: LicenseAttestation): StudioResult<LicenseInfo>

    // ---- Base model selection --------------------------------------------------------------------------------
    fun selectBaseModel(projectId: ProjectId, modelId: String, variantId: String?): StudioResult<ProjectSummary>

    // ---- Acquisition (never automatic) -----------------------------------------------------------------------
    fun planAcquisition(variantId: String, use: IntendedUse): StudioResult<AcquisitionPlan>
    /** Fails with Blocked unless plan.allowed (license VERIFIED for `use`, storage, HTTPS) AND confirmed == true. */
    fun startDownload(variantId: String, use: IntendedUse, confirmed: Boolean): StudioResult<Operation>
    fun importModelFile(variantId: String, file: SourceInput): StudioResult<Operation>
    fun operations(): List<Operation>
    fun operation(id: String): StudioResult<Operation>
    fun cancelOperation(id: String): StudioResult<Operation>
    fun resumeOperation(id: String): StudioResult<Operation>

    // ---- Sources & ingestion ---------------------------------------------------------------------------------
    fun listSources(projectId: ProjectId): StudioResult<List<SourceRecord>>
    fun ingest(projectId: ProjectId, inputs: List<SourceInput>, rights: RightsStatus = RightsStatus.UNSET,
               onProgress: (Progress) -> Unit = {}): StudioResult<IngestReport>
    fun lastIngestReport(projectId: ProjectId): StudioResult<IngestReport?>
    fun setSourceRights(projectId: ProjectId, sourceId: String, rights: RightsStatus): StudioResult<Unit>
    /** Removes the source and everything derived; marks the dataset STALE (rebuild required). */
    fun removeSource(projectId: ProjectId, sourceId: String): StudioResult<Unit>

    // ---- Dataset build + review ------------------------------------------------------------------------------
    fun buildDataset(projectId: ProjectId, options: DatasetOptions = DatasetOptions(),
                     onProgress: (Progress) -> Unit = {}): StudioResult<DatasetPreview>
    fun datasetPreview(projectId: ProjectId): StudioResult<DatasetPreview?>
    fun reviewItems(projectId: ProjectId, filter: ReviewFilter = ReviewFilter(), offset: Int = 0, limit: Int = 50): StudioResult<Page<ReviewItem>>
    fun setIncluded(projectId: ProjectId, itemIds: List<String>, included: Boolean): StudioResult<DatasetPreview>
    /** Approve is refused (Blocked) while unreviewed leakage suspects remain included. */
    fun approveDataset(projectId: ProjectId): StudioResult<DatasetPreview>

    // ---- Method ----------------------------------------------------------------------------------------------
    fun methodOptions(projectId: ProjectId): StudioResult<List<MethodOption>>
    fun selectMethod(projectId: ProjectId, methodId: String): StudioResult<ProjectSummary>

    // ---- Exports (written to a caller-provided stream, e.g. a SAF document) ----------------------------------
    fun exportTrainingJobPackage(projectId: ProjectId, out: OutputStream): StudioResult<ExportedPackage>
    fun exportReferencePackage(projectId: ProjectId, out: OutputStream): StudioResult<ExportedPackage>
    fun exportHeldOutEvalSet(projectId: ProjectId, out: OutputStream): StudioResult<ExportedPackage>

    // ---- Evaluation ------------------------------------------------------------------------------------------
    fun importEvaluation(projectId: ProjectId, results: InputStream): StudioResult<EvaluationView>
    fun evaluation(projectId: ProjectId): StudioResult<EvaluationView?>

    // ---- Specialist package ----------------------------------------------------------------------------------
    fun exportSpecialistPackage(projectId: ProjectId, out: OutputStream): StudioResult<ExportedPackage>
    /** Validates and describes an exported package without installing it into a project. */
    fun importSpecialistPackage(input: InputStream): StudioResult<SpecialistPackageView>

    // ---- Host passthrough ------------------------------------------------------------------------------------
    val host: HostHooks
}
