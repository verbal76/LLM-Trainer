package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.qualify.SafetyPolicy
import com.hotatticgames.llmtrainer.studio.api.*
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import org.json.JSONObject

/**
 * Real implementation of [Studio]. Pure JVM + org.json; every platform capability enters through a small seam
 * ([Http], [Clock], [StorageProbe], [IdSource], [TaskRunner]) so it is fully testable and dexes into the OTA bundle.
 *
 * Layout under [rootDir]:
 * ```
 * workspace/licenses.json, license_texts/, operations/, models/, catalog_overrides.json
 * projects/<id>/project.json, ingest_report.json, sources/<src>/..., dataset/{build.json,chunks.jsonl,...}, evaluation/
 * ```
 * Every JSON file is written atomically (tmp + fsync + rename) with a one-generation backup and carries a schema number.
 */
class StudioCore(
    private val rootDir: File,
    private val deviceSnapshotJson: () -> String,
    hostHooks: HostHooks? = null,
    private val http: Http = JavaHttp(),
    private val clock: Clock = SystemClock,
    private val storage: StorageProbe = FileStorageProbe,
    private val runner: TaskRunner = ThreadTaskRunner(),
    private val ids: IdSource = RandomIds,
    private val appVersion: String = "0.1.0",
    private val policy: SafetyPolicy = SafetyPolicy(),
    registryFiles: List<Pair<String, String>> = EmbeddedRegistry.files,
) : Studio {

    /** Non-fatal problems found while loading state (corrupt files set aside, unreadable projects, invalid registry entries). */
    val startupProblems: List<String> get() = problems.toList()
    private val problems = ArrayList<String>()

    private val wsDir = File(rootDir, "workspace")
    private val projectsDir = File(rootDir, "projects")
    private val registry: List<RegEntry>
    private val licenses: LicenseService
    private val catalogSvc: CatalogService
    private val acquisition: AcquisitionService
    private val ingestSvc = IngestService(clock)
    private val projects = LinkedHashMap<String, PState>()
    private val lock = Any()

    private class PState(val id: String, val dir: File) {
        // Fields read by the lock-free summary path are @Volatile; writers hold [monitor].
        @Volatile var name = ""; @Volatile var domain = ""; @Volatile var purpose = ""; @Volatile var createdAt = 0L; @Volatile var updatedAt = 0L
        @Volatile var baseModelId: String? = null; @Volatile var variantId: String? = null; @Volatile var methodId: String? = null
        @Volatile var trainingExported = false; @Volatile var referenceExported = false; @Volatile var heldOutExported = false; @Volatile var specialistExported = false
        var specialistSeq = 0
        var commercial = false; var redistribute = false
        val jobIds = ArrayList<String>()
        lateinit var sources: SourceStore
        @Volatile var dataset: DatasetState? = null
        @Volatile var report: IngestReport? = null
        @Volatile var evaluation: EvalRec? = null
        val monitor = Any()
    }

    private class EvalRec(val view: EvaluationView, val jobId: String, val status: String, val reportBytes: ByteArray, val kind: String, val refs: List<String>, val hashes: Map<String, String>, val method: String?)

    init {
        rootDir.mkdirs()
        wsDir.mkdirs(); projectsDir.mkdirs()
        registry = Registry.parseAll(registryFiles) { problems.add(it) }
        licenses = LicenseService(wsDir, { registry.associateBy { it.entryId } }, http, clock).also { problems.addAll(it.problems) }
        val files = ModelFiles(File(wsDir, "models"))
        catalogSvc = CatalogService(registry, licenses, files, File(wsDir, "catalog_overrides.json"), deviceSnapshotJson, policy, clock) { problems.add(it) }
        acquisition = AcquisitionService(wsDir, catalogSvc, licenses, files, http, clock, storage, runner, ids, policy, deviceSnapshotJson).also { problems.addAll(it.problems) }
        loadProjects()
    }

    // ================================================================================================================
    // Persistence
    // ================================================================================================================

    private fun loadProjects() {
        val dirs = projectsDir.listFiles { f -> f.isDirectory && !f.name.startsWith(".") }?.sortedBy { it.name } ?: return
        for (d in dirs) {
            try {
                // finish/rollback an interrupted dataset swap
                val ds = File(d, "dataset"); val old = File(d, "dataset.old")
                if (!ds.exists() && old.exists()) old.renameTo(ds)
                File(d, "dataset.tmp").deleteRecursively(); if (ds.exists()) old.deleteRecursively()
                val o = Fs.readJson(File(d, "project.json")) { problems.add("project ${d.name}: $it") }
                if (o == null) { problems.add("project directory ${d.name} has no readable project.json and was skipped (data left in place)"); continue }
                if ((o.int("schema") ?: 1) > SCHEMA) { problems.add("project ${d.name} was written by a newer app version and was skipped"); continue }
                val p = PState(o.str("id") ?: d.name, d)
                p.name = o.str("name") ?: d.name; p.domain = o.str("domain") ?: ""; p.purpose = o.str("purpose") ?: ""
                p.createdAt = o.lng("created_at") ?: 0L; p.updatedAt = o.lng("updated_at") ?: p.createdAt
                p.baseModelId = o.str("base_model_id"); p.variantId = o.str("variant_id"); p.methodId = o.str("method_id")
                p.trainingExported = o.bool("training_exported") ?: false; p.referenceExported = o.bool("reference_exported") ?: false
                p.heldOutExported = o.bool("heldout_exported") ?: false; p.specialistExported = o.bool("specialist_exported") ?: false
                p.specialistSeq = o.int("specialist_seq") ?: 0
                p.commercial = o.obj("intended_use")?.bool("commercial") ?: false; p.redistribute = o.obj("intended_use")?.bool("redistribute_model") ?: false
                p.jobIds.addAll(o.strList("job_ids"))
                p.sources = SourceStore(File(d, "sources"), problems)
                p.dataset = DatasetStore(File(d, "dataset"), problems).takeIf { it.exists() }?.load()?.let { reconcile(p, it) }
                p.report = readReport(p)
                p.evaluation = readEvaluation(p)
                projects[p.id] = p
            } catch (e: Exception) {
                problems.add("project ${d.name} could not be loaded (${e.javaClass.simpleName}: ${e.message}); data left in place")
            }
        }
    }

    /**
     * Crash safety between "sources changed" and "dataset marked stale": the dataset records which sources (and rights) it was built from.
     * Any difference at load means the dataset no longer describes the sources, so it becomes STALE and rows of vanished sources are purged.
     */
    private fun reconcile(p: PState, ds: DatasetState): DatasetState {
        val now = p.sources.all().associateBy { it.sourceId }
        val snap = ds.meta.sourceSnapshot.associateBy { it["id"] as? String }
        val removed = snap.keys.filterNotNull().filter { it !in now }
        val changed = now.values.any { d -> val e = snap[d.sourceId]; e == null || e["rights"] != d.rights.name || e["sha256"] != d.sha256 }
        if (removed.isEmpty() && !changed) return ds
        if (removed.isEmpty() && ds.meta.status == DatasetStatus.STALE) return ds
        problems.add("project ${p.id}: sources changed after the dataset was built (an interrupted operation?); the dataset was marked STALE" + if (removed.isNotEmpty()) " and rows of ${removed.size} removed source(s) were purged" else "")
        val gone = removed.toSet()
        val meta = DatasetMeta(ds.meta.version, ds.meta.options, ds.meta.builtAt, null, DatasetStatus.STALE, ds.meta.groupBy, ds.meta.assignment, ds.meta.notes, ds.meta.splitsAvailable, ds.meta.sourceSnapshot.filter { it["id"] !in gone })
        val out = DatasetState(meta, ds.chunks.filter { it.sourceId !in gone }, ds.synth.filter { it.sourceId !in gone }, java.util.concurrent.ConcurrentHashMap(ds.decisions.filterKeys { k -> gone.none { k.startsWith("$it/") } }))
        return try { p.dataset = out; persistDatasetFull(p); out } catch (e: IOException) { problems.add("project ${p.id}: could not persist the reconciled dataset: ${e.message}"); out }
    }

    private fun saveProject(p: PState) {
        p.updatedAt = clock.nowMs()
        Fs.writeJson(File(p.dir, "project.json"), linkedMapOf(
            "schema" to SCHEMA, "id" to p.id, "name" to p.name, "domain" to p.domain, "purpose" to p.purpose, "created_at" to p.createdAt, "updated_at" to p.updatedAt,
            "base_model_id" to p.baseModelId, "variant_id" to p.variantId, "method_id" to p.methodId,
            "training_exported" to p.trainingExported, "reference_exported" to p.referenceExported, "heldout_exported" to p.heldOutExported,
            "specialist_exported" to p.specialistExported, "specialist_seq" to p.specialistSeq, "job_ids" to p.jobIds,
            "intended_use" to mapOf("commercial" to p.commercial, "redistribute_model" to p.redistribute)))
    }

    private fun provToMap(s: ProvenanceSummary?) = s?.let { linkedMapOf("source_id" to it.sourceId, "sha256" to it.sha256, "mime" to it.mime, "size" to it.sizeBytes, "ingested_at" to it.ingestedAt,
        "pages" to it.pages, "chunks" to it.chunks, "extractor" to it.extractor, "rights" to it.rights.name) }

    private fun saveReport(p: PState, r: IngestReport, finished: Boolean = true) {
        Fs.writeJson(File(p.dir, "ingest_report.json"), linkedMapOf("schema" to SCHEMA, "at" to r.at, "finished" to finished, "items" to r.items.map { i ->
            linkedMapOf("name" to i.name, "status" to i.status.name, "source_id" to i.sourceId, "duplicate_of" to i.duplicateOfSourceId,
                "issues" to i.issues.map { mapOf("code" to it.code.name, "message" to it.message) }, "provenance" to provToMap(i.provenance))
        }))
    }

    private fun readReport(p: PState): IngestReport? {
        val o = Fs.readJson(File(p.dir, "ingest_report.json")) { problems.add("project ${p.id}: $it") } ?: return null
        if (o.bool("finished") == false) problems.add("project ${p.id}: an ingest was interrupted after ${o.objList("items").size} file(s); the report shows what finished, add the remaining files again")
        return try {
            IngestReport(ProjectId(p.id), o.objList("items").map { i ->
                IngestItem(i.str("name") ?: "", IngestStatus.valueOf(i.str("status")!!), i.str("source_id"), i.str("duplicate_of"),
                    i.objList("issues").map { x -> IngestIssue(IssueCode.valueOf(x.str("code")!!), x.str("message") ?: "") },
                    i.obj("provenance")?.let { pv -> ProvenanceSummary(pv.str("source_id")!!, pv.str("sha256")!!, pv.str("mime") ?: "", pv.lng("size") ?: 0, pv.lng("ingested_at") ?: 0,
                        pv.int("pages"), pv.int("chunks") ?: 0, pv.str("extractor") ?: "", RightsStatus.valueOf(pv.str("rights")!!)) })
            }, o.lng("at") ?: 0L)
        } catch (e: Exception) { problems.add("project ${p.id}: ingest report unreadable and ignored"); null }
    }

    private fun saveEvaluation(p: PState) {
        val d = File(p.dir, "evaluation")
        val e = p.evaluation
        if (e == null) { d.deleteRecursively(); return }
        val v = e.view
        Fs.writeBytes(File(d, "evaluation_report.json"), e.reportBytes)
        Fs.writeJson(File(d, "view.json"), linkedMapOf("schema" to SCHEMA, "run_id" to v.runId, "base" to v.baseModelLabel, "specialist" to v.specialistLabel,
            "metrics" to v.metrics.map { linkedMapOf("id" to it.id, "label" to it.label, "hib" to it.higherIsBetter, "base" to it.base, "specialist" to it.specialist, "delta" to it.delta, "ci_low" to it.ciLow, "ci_high" to it.ciHigh, "n" to it.n) },
            "caveats" to v.caveats, "claim_allowed" to v.improvementClaimAllowed, "claim_reason" to v.claimReason, "is_stub" to v.isStub, "imported_at" to v.importedAt,
            "job_id" to e.jobId, "status" to e.status, "specialist_kind" to e.kind, "artifact_refs" to e.refs, "artifact_sha256s" to e.hashes, "method" to e.method))
    }

    private fun readEvaluation(p: PState): EvalRec? {
        val d = File(p.dir, "evaluation")
        val o = Fs.readJson(File(d, "view.json")) { problems.add("project ${p.id}: $it") } ?: return null
        return try {
            val view = EvaluationView(ProjectId(p.id), o.str("run_id")!!, o.str("base") ?: "", o.str("specialist") ?: "",
                o.objList("metrics").map { m -> MetricRow(m.str("id")!!, m.str("label") ?: "", m.bool("hib") ?: true, m.dbl("base")!!, m.dbl("specialist")!!, m.dbl("delta")!!, m.dbl("ci_low"), m.dbl("ci_high"), m.int("n") ?: 0) },
                o.strList("caveats"), o.bool("claim_allowed") ?: false, o.str("claim_reason") ?: "", o.bool("is_stub") ?: true, o.lng("imported_at") ?: 0L)
            val hashes = o.obj("artifact_sha256s")?.let { h -> h.keyList().associateWith { h.getString(it) } } ?: emptyMap()
            EvalRec(view, o.str("job_id") ?: "", o.str("status") ?: "unknown", File(d, "evaluation_report.json").takeIf { it.isFile }?.readBytes() ?: ByteArray(0),
                o.str("specialist_kind") ?: "none", o.strList("artifact_refs"), hashes, o.str("method"))
        } catch (e: Exception) { problems.add("project ${p.id}: evaluation unreadable and ignored"); null }
    }

    private fun persistDatasetFull(p: PState) {
        val ds = p.dataset
        val dir = File(p.dir, "dataset")
        if (ds == null) { dir.deleteRecursively(); return }
        val tmp = File(p.dir, "dataset.tmp"); val old = File(p.dir, "dataset.old")
        tmp.deleteRecursively(); old.deleteRecursively()
        DatasetStore(tmp, problems).save(ds)
        if (dir.exists() && !dir.renameTo(old)) throw IOException("could not replace the dataset directory")
        if (!tmp.renameTo(dir)) { if (old.exists()) old.renameTo(dir); throw IOException("could not install the new dataset") }
        old.deleteRecursively()
    }

    private fun store(p: PState) = DatasetStore(File(p.dir, "dataset"), problems)

    // ================================================================================================================
    // Helpers
    // ================================================================================================================

    private fun <T> err(e: StudioError): StudioResult<T> = StudioResult.Err(e)
    private fun <T> ok(v: T): StudioResult<T> = StudioResult.Ok(v)
    private fun blocked(code: String, msg: String, vararg reasons: Blocker): StudioError.Blocked =
        StudioError.Blocked(code, msg, if (reasons.isEmpty()) listOf(Blocker(code, msg)) else reasons.toList())
    private fun io(e: Exception): StudioError = StudioError.Io(e.message ?: e.javaClass.simpleName)

    private inline fun <T> withP(id: ProjectId, f: (PState) -> StudioResult<T>): StudioResult<T> {
        val p = synchronized(lock) { projects[id.value] } ?: return err(StudioError.NotFound("project ${id.value}"))
        return synchronized(p.monitor) {
            try { f(p) } catch (e: IOException) { err(io(e)) }
            catch (e: RuntimeException) { err(StudioError.Io("Internal error (${e.javaClass.simpleName}): ${e.message}")) }   // contract: calls never throw
        }
    }

    private fun entryOf(id: String?): RegEntry? = id?.let { catalogSvc.entries[it] }

    private fun methodOptionsFor(p: PState): List<MethodOption> {
        val e = entryOf(p.baseModelId)
        val lic = e?.let { licenses.effective(it) }
        val gate = if (e != null) licenses.gate(e, IntendedUse(true, true, p.commercial, p.redistribute)) else emptyList()
        val ds = p.dataset
        val tuneWhy = when {
            e == null -> "Choose a base model first"
            gate.isNotEmpty() -> "Base model license (${lic!!.state}) does not permit this use: ${gate.first().message}"
            ds != null && !ds.meta.splitsAvailable -> "Fewer than 3 independent document/section groups, so no honest train/validation/held-out split exists. Add more documents."
            else -> null
        }
        val rt = catalogSvc.deviceProfile().nativeRuntimeId
        return listOf(
            MethodOption(MethodIds.REFERENCE_PACKAGE, "Reference package (retrieval)", false, RunLocation.DEVICE, true, null,
                "Exact-reference lookup over your source chunks. This is NOT training; the model is unchanged."),
            MethodOption(MethodIds.PROMPT_SPECIALIZATION, "Prompt specialization", false, RunLocation.NONE, true, null,
                "Instructions only. This is NOT training and does not add knowledge."),
            MethodOption(MethodIds.ADAPTER_DESKTOP, "Adapter training (LoRA/QLoRA) on desktop", true, RunLocation.DESKTOP, tuneWhy == null, tuneWhy,
                "This phone only prepares the training job package. Training happens on a desktop/GPU via 'llmtrainer import-job'; nothing is trained on this device."),
            MethodOption(MethodIds.ADAPTER_ON_DEVICE, "Adapter training on this device", true, RunLocation.DEVICE, false,
                "This app version has no training runtime (nativeRuntimeId $rt); on-device training needs a future app update.",
                "Not available. Nothing is claimed to train on the phone."))
    }

    private fun selectedMethod(p: PState): MethodOption? = methodOptionsFor(p).firstOrNull { it.id == p.methodId && it.available }

    private fun summary(p: PState): ProjectSummary {
        val e = entryOf(p.baseModelId)
        val ds = p.dataset
        val facts = ProjectFacts(
            baseModelId = p.baseModelId, baseModelLicense = e?.let { licenses.effective(it).state },
            sourcesIngested = p.sources.all().size, sourcesFailed = p.report?.failed?.size ?: 0,
            dataset = ds?.meta?.status ?: DatasetStatus.NONE,
            datasetFlaggedIncluded = ds?.let { d -> d.chunks.count { d.isIncluded(it) && it.flags.isNotEmpty() } + d.synth.count { d.isIncluded(it) } } ?: 0,
            method = selectedMethod(p), trainingPackageExported = p.trainingExported, referencePackageExported = p.referenceExported,
            heldOutExported = p.heldOutExported, evaluation = p.evaluation?.view, specialistExported = p.specialistExported)
        val stages = Stages.derive(facts)
        return ProjectSummary(ProjectId(p.id), p.name, p.domain, p.purpose, p.createdAt, p.updatedAt, p.baseModelId, stages, Stages.nextAction(stages))
    }

    private fun resetExports(p: PState, includeEvaluation: Boolean) {
        p.trainingExported = false; p.referenceExported = false; p.heldOutExported = false; p.specialistExported = false
        if (includeEvaluation && p.evaluation != null) { p.evaluation = null; saveEvaluation(p) }
    }

    private fun markStale(p: PState) {
        val ds = p.dataset ?: return
        ds.meta.status = DatasetStatus.STALE
        ds.meta.approvedAt = null
        store(p).saveMeta(ds)
    }

    // ================================================================================================================
    // Projects
    // ================================================================================================================

    override fun listProjects(): List<ProjectSummary> {
        val ps = synchronized(lock) { projects.values.toList() }
        // Lock-free on purpose: the dashboard must stay responsive while a long ingest or dataset build holds a project's monitor.
        return ps.sortedWith(compareBy({ -it.updatedAt }, { it.id })).map { p -> summary(p) }
    }

    override fun createProject(p: NewProject): StudioResult<ProjectSummary> {
        val name = p.name.trim()
        if (name.isEmpty()) return err(StudioError.Invalid("NAME_REQUIRED", "Give the specialist a name"))
        if (name.length > 120 || p.domain.length > 200 || p.purpose.length > 2000) return err(StudioError.Invalid("TOO_LONG", "Name, domain or purpose is too long"))
        val id = "proj-" + Text.slugify(name).take(24).trim('-').ifEmpty { "x" } + "-" + ids.hex(6)
        val dir = File(projectsDir, id)
        return try {
            val s = PState(id, dir)
            s.name = name; s.domain = p.domain.trim(); s.purpose = p.purpose.trim(); s.createdAt = clock.nowMs()
            dir.mkdirs()
            s.sources = SourceStore(File(dir, "sources"), problems)
            saveProject(s)
            synchronized(lock) { projects[id] = s }
            ok(summary(s))
        } catch (e: IOException) { err(io(e)) }
    }

    override fun getProject(id: ProjectId): StudioResult<ProjectSummary> {
        val p = synchronized(lock) { projects[id.value] } ?: return err(StudioError.NotFound("project ${id.value}"))
        return try { ok(summary(p)) } catch (e: RuntimeException) { err(StudioError.Io("Internal error (${e.javaClass.simpleName}): ${e.message}")) }
    }

    override fun deleteProject(id: ProjectId): StudioResult<Unit> {
        val p = synchronized(lock) { projects.remove(id.value) } ?: return err(StudioError.NotFound("project ${id.value}"))
        synchronized(p.monitor) { p.dir.deleteRecursively() }
        return ok(Unit)
    }

    // ================================================================================================================
    // Device, catalog, license evidence
    // ================================================================================================================

    override fun deviceProfile(): DeviceProfile = catalogSvc.deviceProfile()
    override fun recommendations(): Recommendations = catalogSvc.recommendations()
    override fun catalog(): List<CatalogModel> = catalogSvc.catalog()
    /**
     * Host/owner extension (not on [Studio]): register a direct-download variant for a catalog model. The shipped registry knows only
     * repository pages, so without this (or a model-file import) nothing is downloadable. Returns null on success, else the reason.
     */
    fun addVariantOverride(modelId: String, variantId: String, format: String, quantization: String?, url: String, sizeBytes: Long?, sha256: String?): String? =
        catalogSvc.addVariantOverride(modelId, variantId, format, quantization, url, sizeBytes, sha256)

    override fun model(modelId: String): StudioResult<CatalogModel> = entryOf(modelId)?.let { ok(catalogSvc.modelView(it)) } ?: err(StudioError.NotFound("model $modelId"))
    override fun fetchLicenseText(modelId: String) = entryOf(modelId)?.let { licenses.fetchText(it) } ?: err(StudioError.NotFound("model $modelId"))
    override fun importLicenseText(modelId: String, fileName: String, input: InputStream) = entryOf(modelId)?.let { licenses.importText(it, fileName, input) } ?: err(StudioError.NotFound("model $modelId"))
    override fun attestLicense(modelId: String, attestation: LicenseAttestation): StudioResult<LicenseInfo> {
        val e = entryOf(modelId) ?: return err(StudioError.NotFound("model $modelId"))
        val r = licenses.attest(e, attestation)
        if (r is StudioResult.Ok) synchronized(lock) { projects.values.toList() }.forEach { p -> synchronized(p.monitor) { if (p.baseModelId == modelId) resetExportsOnLicense(p) } }
        return r
    }

    private fun resetExportsOnLicense(p: PState) { /* exports already written stay valid files; stages re-derive from the new license state */ }

    override fun selectBaseModel(projectId: ProjectId, modelId: String, variantId: String?) = withP(projectId) { p ->
        val e = entryOf(modelId) ?: return@withP err(StudioError.NotFound("model $modelId"))
        if (variantId != null && catalogSvc.variantsOf(e).none { it.id == variantId }) return@withP err(StudioError.NotFound("variant $variantId"))
        val lic = licenses.effective(e)
        if (lic.state == LicenseState.DISALLOWED)
            return@withP err(blocked(LicenseCodes.DISALLOWED, "This model's license does not allow the intended use", Blocker(LicenseCodes.DISALLOWED, lic.disallowedReason ?: "Disallowed")))
        if (p.baseModelId != modelId) { p.trainingExported = false; p.specialistExported = false; p.heldOutExported = false; p.evaluation?.let { p.evaluation = null; saveEvaluation(p) } }
        p.baseModelId = modelId; p.variantId = variantId
        saveProject(p)
        ok(summary(p))
    }

    // ================================================================================================================
    // Acquisition
    // ================================================================================================================

    override fun planAcquisition(variantId: String, use: IntendedUse) = acquisition.plan(variantId, use)
    override fun startDownload(variantId: String, use: IntendedUse, confirmed: Boolean) = acquisition.startDownload(variantId, use, confirmed)
    override fun importModelFile(variantId: String, file: SourceInput) = acquisition.importModel(variantId, file)
    override fun operations(): List<Operation> = acquisition.operations()
    override fun operation(id: String) = acquisition.operation(id)
    override fun cancelOperation(id: String) = acquisition.cancel(id)
    override fun resumeOperation(id: String) = acquisition.resume(id)

    // ================================================================================================================
    // Sources
    // ================================================================================================================

    override fun listSources(projectId: ProjectId) = withP(projectId) { p -> ok(p.sources.all().map { it.record() }) }
    override fun lastIngestReport(projectId: ProjectId) = withP(projectId) { p -> ok(p.report) }

    override fun ingest(projectId: ProjectId, inputs: List<SourceInput>, rights: RightsStatus, onProgress: (Progress) -> Unit) = withP(projectId) { p ->
        val before = p.sources.all().size
        val report = ingestSvc.ingest(projectId, p.sources, inputs, rights, onProgress) { partial -> if (partial.items.size <= 100 || partial.items.size % 20 == 0) try { saveReport(p, partial, finished = false) } catch (_: IOException) { } }
        p.report = report
        saveReport(p, report)
        if (p.sources.all().size != before) { markStale(p); resetExports(p, true) }
        saveProject(p)
        ok(report)
    }

    override fun setSourceRights(projectId: ProjectId, sourceId: String, rights: RightsStatus) = withP(projectId) { p ->
        p.sources.setRights(sourceId, rights) ?: return@withP err(StudioError.NotFound("source $sourceId"))
        p.report = p.report?.let { r -> r.copy(items = r.items.map { i -> if (i.sourceId == sourceId) i.copy(issues = i.issues.filter { it.code != IssueCode.RIGHTS_UNSET } + (if (rights == RightsStatus.UNSET) listOf(IngestIssue(IssueCode.RIGHTS_UNSET, "Set usage rights before building a dataset")) else emptyList()),
            provenance = i.provenance?.copy(rights = rights)) else i }) }
        p.report?.let { saveReport(p, it) }
        markStale(p); resetExports(p, false); saveProject(p)
        ok(Unit)
    }

    override fun removeSource(projectId: ProjectId, sourceId: String) = withP(projectId) { p ->
        if (p.sources.get(sourceId) == null) return@withP err(StudioError.NotFound("source $sourceId"))
        p.sources.remove(sourceId)
        p.report = p.report?.let { r -> r.copy(items = r.items.filter { it.sourceId != sourceId && it.duplicateOfSourceId != sourceId }) }
        p.report?.let { saveReport(p, it) }
        p.dataset?.let { ds ->
            // purge everything derived from the source right now; the owner must rebuild before using the dataset again
            val keep = ds.chunks.filter { it.sourceId != sourceId }
            val keepSynth = ds.synth.filter { it.sourceId != sourceId }
            val dec = java.util.concurrent.ConcurrentHashMap(ds.decisions.filterKeys { k -> !k.startsWith("$sourceId/") })
            val meta = DatasetMeta(ds.meta.version, ds.meta.options, ds.meta.builtAt, null, DatasetStatus.STALE, ds.meta.groupBy, ds.meta.assignment, ds.meta.notes, ds.meta.splitsAvailable, ds.meta.sourceSnapshot.filter { it["id"] != sourceId })
            p.dataset = DatasetState(meta, keep, keepSynth, dec)
            persistDatasetFull(p)
        }
        resetExports(p, true)
        saveProject(p)
        ok(Unit)
    }

    // ================================================================================================================
    // Dataset
    // ================================================================================================================

    private fun previewOf(p: PState): DatasetPreview? {
        val ds = p.dataset ?: return null
        val items = reviewList(p, ds)
        val inc = items.filter { it.included }
        val stats = DatasetStats(items.size, inc.size, items.size - inc.size, items.count { it.flags.isNotEmpty() },
            items.groupingBy { it.role }.eachCount(), items.groupingBy { it.origin }.eachCount(), items.flatMap { it.flags }.groupingBy { it }.eachCount(),
            inc.count { ReviewFlag.POSSIBLE_LEAKAGE in it.flags })
        val warns = ArrayList<String>(ds.meta.notes)
        if (stats.leakageSuspects > 0) warns.add("${stats.leakageSuspects} included item(s) may leak into held-out material; resolve before approving")
        val heldOut = ds.chunks.count { it.split == "test" && ds.isIncluded(it) }
        if (ds.meta.splitsAvailable && heldOut < 3) warns.add("Held-out set is small ($heldOut chunk(s)); evaluation will have wide uncertainty")
        if (ds.chunks.any { !it.trainable && ds.isIncluded(it) }) warns.add("Reference-only sources are listed for the reference package; they are never used for training")
        if (ds.meta.status == DatasetStatus.STALE) warns.add("Sources or rights changed after this build: rebuild required")
        return DatasetPreview(ProjectId(p.id), ds.meta.version, DatasetEngine.datasetSha(ds), ds.meta.status, ds.meta.options, stats, ds.meta.builtAt, ds.meta.approvedAt, warns)
    }

    private fun reviewList(p: PState, ds: DatasetState): List<ReviewItem> {
        val out = ArrayList<ReviewItem>(ds.chunks.size + ds.synth.size)
        for (c in ds.chunks) out.add(ReviewItem(c.ref, c.sourceId, c.sourceName, c.chunk.page, if (c.chunk.sectionPath.isEmpty()) null else c.chunk.section, c.role, ChunkOrigin.SOURCE_DERIVED,
            excerpt(c.chunk.text), c.flags, ds.isIncluded(c)))
        for (s in ds.synth) out.add(ReviewItem(s.id, s.sourceId, s.sourceName, s.page, s.section, ChunkRole.TRAIN, ChunkOrigin.SYNTHETIC,
            excerpt("Q: ${s.prompt} A: ${s.response}"), listOf(ReviewFlag.LOW_CONFIDENCE), ds.isIncluded(s)))
        return out
    }

    private fun excerpt(t: String) = if (t.length <= 240) t else t.take(240).trimEnd() + "…"

    override fun buildDataset(projectId: ProjectId, options: DatasetOptions, onProgress: (Progress) -> Unit) = withP(projectId) { p ->
        DatasetEngine.validateOptions(options)?.let { return@withP err(StudioError.Invalid("BAD_OPTIONS", it)) }
        val srcs = p.sources.all()
        if (srcs.isEmpty()) return@withP err(blocked("NO_SOURCES", "Add sources first", Blocker("NO_SOURCES", "No ingested sources")))
        val unset = srcs.filter { it.rights == RightsStatus.UNSET }
        if (unset.isNotEmpty()) return@withP err(blocked("RIGHTS_UNSET", "Set usage rights for every source before building a dataset", *unset.map { Blocker("RIGHTS_UNSET", "${it.name}: usage rights not set") }.toTypedArray()))
        val withBlocks = srcs.map { it to p.sources.loadBlocks(it.sourceId) }
        val built = DatasetEngine.build(withBlocks, options) { d, t -> onProgress(Progress(d, t, "sources")) }
        val prev = p.dataset
        val snap = srcs.sortedBy { it.sourceId }.map { s -> linkedMapOf<String, Any?>("id" to s.sourceId, "sha256" to s.sha256, "rights" to s.rights.name) }
        val meta = DatasetMeta((prev?.meta?.version ?: 0) + 1, options, clock.nowMs(), null, DatasetStatus.NEEDS_REVIEW, built.groupBy, built.assignment, built.notes, built.splitsAvailable, snap)
        val dec = java.util.concurrent.ConcurrentHashMap<String, Decision>()
        if (prev != null) {
            val byRef = built.chunks.associateBy { it.ref }
            for ((k, d) in prev.decisions) { val c = byRef[k]; if (c != null && c.chunk.sha256 == d.textSha) dec[k] = d }
            val synthIds = built.synth.map { it.id }.toSet()
            for ((k, d) in prev.decisions) if (k in synthIds) dec[k] = d
        }
        val ds = DatasetState(meta, built.chunks, built.synth, dec)
        p.dataset = ds
        persistDatasetFull(p)
        resetExports(p, true)
        saveProject(p)
        onProgress(Progress(srcs.size.toLong(), srcs.size.toLong(), "sources"))
        ok(previewOf(p)!!)
    }

    override fun datasetPreview(projectId: ProjectId) = withP(projectId) { p -> ok(previewOf(p)) }

    override fun reviewItems(projectId: ProjectId, filter: ReviewFilter, offset: Int, limit: Int) = withP(projectId) { p ->
        val ds = p.dataset ?: return@withP ok(Page(emptyList<ReviewItem>(), 0, offset))
        val l = reviewList(p, ds).filter { i ->
            (!filter.onlyFlagged || i.flags.isNotEmpty()) && (filter.flag == null || filter.flag in i.flags) && (filter.role == null || i.role == filter.role) &&
                (filter.included == null || i.included == filter.included) && (filter.sourceId == null || i.sourceId == filter.sourceId)
        }
        val o = offset.coerceAtLeast(0)
        ok(Page(l.drop(o).take(limit.coerceIn(0, 1000)), l.size, o))
    }

    override fun setIncluded(projectId: ProjectId, itemIds: List<String>, included: Boolean) = withP(projectId) { p ->
        val ds = p.dataset ?: return@withP err(blocked("NO_DATASET", "Build a dataset first"))
        if (ds.meta.status == DatasetStatus.STALE) return@withP err(blocked("DATASET_STALE", "Sources changed; rebuild the dataset before reviewing", Blocker("DATASET_STALE", "Rebuild required")))
        val unknown = itemIds.filter { ds.chunk(it) == null && ds.synth.none { s -> s.id == it } }
        if (unknown.isNotEmpty()) return@withP err(StudioError.NotFound("review item ${unknown.first()}"))
        for (id in itemIds) {
            val c = ds.chunk(id)
            ds.decisions[id] = Decision(included, c?.chunk?.sha256 ?: "")
        }
        ds.meta.status = DatasetStatus.NEEDS_REVIEW; ds.meta.approvedAt = null
        store(p).saveMeta(ds); store(p).saveReview(ds)      // status first: a crash in between leaves NEEDS_REVIEW, never a stale approval
        p.trainingExported = false; p.heldOutExported = false; p.referenceExported = false; p.specialistExported = false
        saveProject(p)
        ok(previewOf(p)!!)
    }

    override fun approveDataset(projectId: ProjectId) = withP(projectId) { p ->
        val ds = p.dataset ?: return@withP err(blocked("NO_DATASET", "Build a dataset first"))
        if (ds.meta.status == DatasetStatus.STALE) return@withP err(blocked("DATASET_STALE", "Sources changed; rebuild the dataset", Blocker("DATASET_STALE", "Rebuild required")))
        val leaks = ds.chunks.filter { ds.isIncluded(it) && ReviewFlag.POSSIBLE_LEAKAGE in it.flags }
        if (leaks.isNotEmpty()) return@withP err(blocked("LEAKAGE_UNRESOLVED", "Resolve possible train/eval leakage before approving",
            *leaks.take(5).map { Blocker("POSSIBLE_LEAKAGE", "${it.sourceName} ${it.chunk.section}: possible leakage into held-out material") }.toTypedArray()))
        if (ds.chunks.none { ds.isIncluded(it) }) return@withP err(blocked("EMPTY_DATASET", "Nothing is included in the dataset"))
        ds.meta.status = DatasetStatus.APPROVED; ds.meta.approvedAt = clock.nowMs()
        store(p).saveMeta(ds)
        ok(previewOf(p)!!)
    }

    // ================================================================================================================
    // Methods
    // ================================================================================================================

    override fun methodOptions(projectId: ProjectId) = withP(projectId) { p -> ok(methodOptionsFor(p)) }

    override fun selectMethod(projectId: ProjectId, methodId: String) = withP(projectId) { p ->
        val m = methodOptionsFor(p).firstOrNull { it.id == methodId } ?: return@withP err(StudioError.NotFound("method $methodId"))
        if (!m.available) return@withP err(blocked("METHOD_UNAVAILABLE", m.whyNotAvailable ?: "Unavailable", Blocker("METHOD_UNAVAILABLE", m.whyNotAvailable ?: "")))
        p.methodId = methodId; saveProject(p)
        ok(summary(p))
    }

    // ================================================================================================================
    // Exports
    // ================================================================================================================

    private fun approvedOr(p: PState): StudioError? =
        if (p.dataset?.meta?.status != DatasetStatus.APPROVED) blocked("DATASET_NOT_APPROVED", "Approve the dataset first",
            Blocker("DATASET_NOT_APPROVED", "Dataset is ${p.dataset?.meta?.status ?: DatasetStatus.NONE}")) else null

    private fun ctx(p: PState, jobId: String): JobContext? {
        val e = entryOf(p.baseModelId)
        val ds = p.dataset ?: return null
        val dev = catalogSvc.deviceProfile()
        val variant = p.variantId?.let { vid -> catalogSvc.variantsOf(e ?: return null).firstOrNull { it.id == vid } }
        val rows = ds.includedChunks.filter { it.chunk.role != ChunkRoles.EXCLUDED }.map { it.ref to it.chunk.text }
        val terms = Terms.extract(ds.chunks.filter { ds.isIncluded(it) && it.chunk.kind != "table" }.map { it.ref to it.chunk.text }, 30)
        return JobContext(p.id, p.name, p.domain, p.purpose, p.createdAt, p.commercial, p.redistribute,
            e ?: stubEntry(), variant?.variantId, variant?.sourceUrl, e?.let { licenses.jobEvidence(it) }, p.sources.all(), ds, dev.deviceName, dev.totalRamMb,
            clock.nowMs(), jobId, appVersion, terms.also { rows.size }, "qlora")
    }

    private fun stubEntry() = RegEntry("none@none", "none", "none", null, null, null, null, "unverified", "unverified", "unverified", "unverified", "unverified",
        emptyList(), emptyList(), null, null, null, null, null, emptyList(), "unverified", null, RegVerification("UNVERIFIED", "none", emptyList(), null, null, null, null, null, emptyList()))

    private fun newJobId(): String = "job-" + Iso.date(clock.nowMs()).replace("-", "") + "-" + ids.hex(6)

    private fun write(p: PState?, built: BuiltPackage, out: OutputStream, extraWarnings: List<String> = emptyList()): StudioResult<ExportedPackage> {
        val w = try { built.writer.write(out).also { out.flush() } } catch (e: IOException) { return err(StudioError.Io("Could not write the package: ${e.message}")) }
        return ok(ExportedPackage(built.kind, built.fileName, w.sizeBytes, w.sha256, w.files, built.warnings + extraWarnings))
    }

    private fun pkgErr(e: PackageException): StudioError = when (e.code) {
        "TOO_FEW_GROUPS", "LEAKAGE_DETECTED", "EMPTY_SPLIT", "NO_HELDOUT_ITEMS", "NOTHING_TO_EXPORT" -> blocked(e.code, e.message ?: e.code)
        else -> StudioError.Invalid(e.code, e.message ?: e.code)
    }

    override fun exportTrainingJobPackage(projectId: ProjectId, out: OutputStream) = withP(projectId) { p ->
        approvedOr(p)?.let { return@withP err(it) }
        val m = selectedMethod(p)
        if (m == null || !m.isTraining) return@withP err(blocked("METHOD_NOT_TRAINING", "Select a training method first", Blocker("METHOD_NOT_TRAINING", "Current method does not train")))
        val e = entryOf(p.baseModelId) ?: return@withP err(blocked("NO_BASE_MODEL", "Choose a base model first"))
        val gate = licenses.gate(e, IntendedUse(true, true, p.commercial, p.redistribute))
        if (gate.isNotEmpty()) return@withP err(blocked("LICENSE_${licenses.effective(e).state}", "Base model license gate", *gate.toTypedArray()))
        val jobId = newJobId()
        val c = ctx(p, jobId)!!
        val job = try { Packages.buildJob(c, p.sources) } catch (ex: PackageException) { return@withP err(pkgErr(ex)) }
        val r = write(p, job.pkg, out)
        if (r is StudioResult.Ok) { p.trainingExported = true; p.jobIds.add(jobId); saveProject(p) }
        r
    }

    override fun exportReferencePackage(projectId: ProjectId, out: OutputStream) = withP(projectId) { p ->
        approvedOr(p)?.let { return@withP err(it) }
        val c = ctx(p, "ref")!!
        val built = try { Packages.buildReference(c) } catch (ex: PackageException) { return@withP err(pkgErr(ex)) }
        val r = write(p, built, out)
        if (r is StudioResult.Ok) { p.referenceExported = true; saveProject(p) }
        r
    }

    override fun exportHeldOutEvalSet(projectId: ProjectId, out: OutputStream) = withP(projectId) { p ->
        approvedOr(p)?.let { return@withP err(it) }
        val jobId = newJobId()
        val built = try { Packages.buildHeldOut(ctx(p, jobId)!!) } catch (ex: PackageException) { return@withP err(pkgErr(ex)) }
        val r = write(p, built, out)
        if (r is StudioResult.Ok) { p.heldOutExported = true; p.jobIds.add(jobId); saveProject(p) }
        r
    }

    // ================================================================================================================
    // Evaluation
    // ================================================================================================================

    override fun importEvaluation(projectId: ProjectId, results: InputStream) = withP(projectId) { p ->
        if (!p.heldOutExported && !p.trainingExported)
            return@withP err(blocked("HELDOUT_NOT_EXPORTED", "Export the training job or held-out set and evaluate on the desktop first", Blocker("HELDOUT_NOT_EXPORTED", "No job or held-out set was exported from this project")))
        val files = try { Zips.readAll(results) } catch (e: PackageException) { return@withP err(StudioError.Invalid("RESULTS_INVALID", "Not a usable results package: ${e.message}")) }
            catch (e: IOException) { return@withP err(StudioError.Io(e.message ?: "read failed")) }
        val parsed = try { Packages.parseResults(projectId, files, p.jobIds.toSet(), clock.nowMs(), "eval-" + ids.hex(8)) } catch (e: PackageException) {
            return@withP err(if (e.code == "WRONG_JOB" || e.code == "WRONG_PROJECT") blocked(e.code, e.message ?: e.code) else StudioError.Invalid(if (e.code == "CHECKSUM_FAILED") "RESULTS_INVALID" else e.code, e.message ?: e.code))
        }
        p.evaluation = EvalRec(parsed.view, parsed.jobId, parsed.status, parsed.reportBytes, parsed.specialistKind, parsed.artifactRefs, parsed.artifactHashes, parsed.method)
        saveEvaluation(p)
        p.specialistExported = false
        saveProject(p)
        ok(parsed.view)
    }

    override fun evaluation(projectId: ProjectId) = withP(projectId) { p -> ok(p.evaluation?.view) }

    // ================================================================================================================
    // Specialist package
    // ================================================================================================================

    override fun exportSpecialistPackage(projectId: ProjectId, out: OutputStream) = withP(projectId) { p ->
        val ev = p.evaluation ?: return@withP err(blocked("NO_EVALUATION", "Evaluate before exporting", Blocker("NO_EVALUATION", "No evaluation results")))
        val e = entryOf(p.baseModelId) ?: return@withP err(blocked("NO_BASE_MODEL", "Choose a base model first"))
        val gate = licenses.gate(e, IntendedUse(true, true, p.commercial, p.redistribute))
        if (gate.isNotEmpty()) return@withP err(blocked("LICENSE_${licenses.effective(e).state}", "Base model license gate", *gate.toTypedArray()))
        val ds = p.dataset
        val att = licenses.attestationOf(e.entryId)
        val evidence = licenses.jobEvidence(e)
        val refChunks = ds?.chunks?.filter { ds.isIncluded(it) } ?: emptyList()
        val parsed = Packages.ParsedResults(ev.view, ev.jobId, ev.status, ev.reportBytes, JSONObject(), ev.kind, ev.refs, ev.hashes, ev.method)
        val seq = p.specialistSeq
        val c = Packages.SpecialistContext(p.id, p.name, "1.0.$seq", e, licenses.effective(e).state, evidence?.first, evidence?.second, ds?.let { DatasetEngine.datasetSha(it) },
            ds?.let { "ds-" + DatasetEngine.datasetSha(it).take(12) }, p.methodId ?: "unknown", parsed, clock.nowMs(), appVersion, p.sources.all(), refChunks, ds)
        val built = Packages.buildSpecialist(c)
        val r = write(p, built, out)
        if (r is StudioResult.Ok) { p.specialistExported = true; p.specialistSeq = seq + 1; saveProject(p) }
        r
    }

    override fun importSpecialistPackage(input: InputStream): StudioResult<SpecialistPackageView> {
        val files = try { Zips.readAll(input) } catch (e: PackageException) { return err(StudioError.Invalid("PACKAGE_INVALID", "Not a usable package: ${e.message}")) }
            catch (e: IOException) { return err(StudioError.Io(e.message ?: "read failed")) }
        return try { ok(Packages.parseSpecialist(files)) } catch (e: PackageException) { err(StudioError.Invalid(e.code, e.message ?: e.code)) }
    }

    // ================================================================================================================
    // Host
    // ================================================================================================================

    override val host: HostHooks = hostHooks ?: object : HostHooks {
        override fun diagnosticsJson() = J.dump(linkedMapOf("source" to "studio-core", "note" to "Host diagnostics are provided by the host shell; none were wired into studio-core.", "appVersion" to appVersion))
        override fun deviceSnapshotJson() = this@StudioCore.deviceSnapshotJson()
        override fun checkForUpdates(callback: (UpdateStatus) -> Unit) = callback(UpdateStatus("error", "Update checks are provided by the host shell and are not wired into studio-core."))
        override fun restartApp() { /* only the host can restart the process */ }
    }

    companion object { const val SCHEMA = 1 }
}
