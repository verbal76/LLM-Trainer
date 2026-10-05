package com.hotatticgames.llmtrainer.studio.api

import java.io.InputStream
import java.io.OutputStream
import java.security.MessageDigest

/**
 * In-memory, deterministic [Studio] with realistic sample data. For UI screens/tests and as an oracle for studio-core.
 * Time is a deterministic counter (+1000 ms per call). Background work does not run by itself: call [tick] to advance
 * RUNNING operations (UI previews can call it from a timer). [simulateProcessDeath] models persistence semantics.
 *
 * Content hooks (the fake does not parse real packages): evaluation results containing "STUB" -> stub evaluation,
 * containing "LARGE" -> n=400 with CIs excluding zero (claim allowed), otherwise n=24 (claim NOT allowed).
 * Ingest hooks: name contains "corrupt" or bytes start with "%CORRUPT" -> CORRUPT_FILE; pdf with "scan" in name ->
 * NEEDS_OCR; identical bytes -> DUPLICATE.
 */
class FakeStudio(seedSampleData: Boolean = true, preinstallBaseModels: Boolean = false) : Studio {
    private var clock = 1_760_000_000_000L
    private fun now(): Long { clock += 1000; return clock }
    private var seq = 0
    private fun nextId(p: String) = "$p-${++seq}"

    private val device = DeviceProfile("Sample Phone (8 GB)", 34, "arm64-v8a", 8192, 5200, 6000, "none-v1", 0.25, 1_760_000_000_000L)
    private val storageReserveBytes = 1024L * 1024 * 1024

    // ---------------------------------------------------------------- catalog
    private val models = LinkedHashMap<String, CatalogModel>()
    private val fetched = HashMap<String, LicenseTextFetch>()   // modelId -> last fetch/import (attestation anchor)
    private val acquired = HashSet<String>()

    private class P(val id: ProjectId, var name: String, var domain: String, var purpose: String, val created: Long) {
        var updated = created
        var baseModelId: String? = null
        var variantId: String? = null
        val sources = LinkedHashMap<String, SourceRecord>()
        var lastReport: IngestReport? = null
        var dataset: DatasetPreview? = null
        var items = ArrayList<ReviewItem>()
        var methodId: String? = null
        var trainingExported = false
        var referenceExported = false
        var heldOutExported = false
        var evaluation: EvaluationView? = null
        var specialistExported = false
    }
    private val projects = LinkedHashMap<String, P>()
    private val ops = LinkedHashMap<String, Operation>()

    /** Scripted v2 state and knobs (engineAvailable, charging, batteryPercent, thermal, ...). */
    val v2: FakeStudioV2 by lazy { FakeStudioV2(object : FakeStudioV2.Env {
        override fun now() = this@FakeStudio.now()
        override fun nextId(prefix: String) = this@FakeStudio.nextId(prefix)
        override fun projectExists(id: ProjectId) = projects.containsKey(id.value)
        override fun baseModel(id: ProjectId) = projects[id.value]?.let { p -> p.baseModelId?.let { it to p.variantId } }
        override fun baseInstalled(id: ProjectId) = projects[id.value]?.let { isInstalled(it) } ?: false
        override fun datasetApproved(id: ProjectId) = projects[id.value]?.dataset?.status == DatasetStatus.APPROVED
        override fun baseName(modelId: String) = models[modelId]?.name ?: modelId
        override fun projectsUsing(variantId: String) = projects.values.filter { it.variantId == variantId }.map { it.id }
        override fun installedVariants() = acquired.mapNotNull { v -> variantOf(v)?.let { (m, vv) -> Triple(m.id, vv.id, vv.sizeBytes) } }
        override fun variantMeta(variantId: String) = variantOf(variantId)?.let { (m, v) -> m.name to v.quant }
        override fun licenseOk(modelId: String) = models[modelId]?.license?.state == LicenseState.VERIFIED
    }) }


    init {
        addModel("qwen3-4b", "Qwen", "Qwen3 4B", "3", 4.0, "qwen3", LicenseState.VERIFIED, "Apache-2.0",
            "https://huggingface.co/Qwen/Qwen3-4B/raw/main/LICENSE", listOf(Triple("q4_k_m", 2_500_000_000L, 3300), Triple("q8_0", 4_300_000_000L, 5300)))
        addModel("qwen3-8b", "Qwen", "Qwen3 8B", "3", 8.0, "qwen3", LicenseState.VERIFIED, "Apache-2.0",
            "https://huggingface.co/Qwen/Qwen3-8B/raw/main/LICENSE", listOf(Triple("q4_k_m", 5_000_000_000L, 6600), Triple("q8_0", 8_700_000_000L, 10200)))
        addModel("llama-3.1-8b", "Llama", "Llama 3.1 8B", "3.1", 8.0, "llama", LicenseState.UNVERIFIED, "Llama 3.1 Community License",
            "https://huggingface.co/meta-llama/Llama-3.1-8B/raw/main/LICENSE", listOf(Triple("q4_k_m", 4_900_000_000L, 6200)))
        addModel("gemma-4-e4b", "Gemma", "Gemma 4 E4B", "4", 4.0, "gemma", LicenseState.UNVERIFIED, "Gemma terms",
            "https://ai.google.dev/gemma/terms", listOf(Triple("q4_k_m", 3_300_000_000L, 3700)))
        addModel("sample-restricted-7b", "Sample", "Sample Restricted 7B (fictional)", "1", 7.0, "sample", LicenseState.DISALLOWED, "Sample No-Derivatives",
            "https://example.org/sample-restricted/LICENSE", listOf(Triple("q4_k_m", 4_200_000_000L, 5500)),
            "Sample data: license forbids derivative models")
        if (seedSampleData) seed()
        if (preinstallBaseModels) { acquired += "qwen3-4b:q4_k_m" }
    }

    companion object {
        /** UI-development fixture: base model already installed, a trained + selected specialist on "Town Design" (project 3), engine available. */
        fun sampleV2(): FakeStudio {
            val s = FakeStudio(seedSampleData = true, preinstallBaseModels = true)
            val town = s.listProjects().first { it.name == "Town Design" }.id
            val run = s.startLocalTraining(town, TrainingSettings(), true).getOrNull()
            if (run != null) repeat(5) { s.tick() }
            return s
        }
    }

    private fun addModel(id: String, family: String, name: String, version: String, params: Double, arch: String,
                         state: LicenseState, licName: String, url: String, vars: List<Triple<String, Long, Int>>,
                         disallowed: String? = null) {
        val perms = if (state == LicenseState.VERIFIED) mapOf(Permission.COMMERCIAL to Tri.YES, Permission.FINE_TUNE to Tri.YES,
            Permission.ADAPTER to Tri.YES, Permission.REDISTRIBUTION to Tri.YES)
        else Permission.values().associateWith { Tri.UNVERIFIED }
        val text = licenseText(id)
        val lic = LicenseInfo(state, licName, url, if (state == LicenseState.VERIFIED) "owner_reviewed_text" else "none",
            if (state == LicenseState.VERIFIED) "owner-reviewed license text from $url, hash ${sha(text)}; permissions as attested by owner" else null,
            if (state == LicenseState.VERIFIED) sha(text) else null, if (state == LicenseState.VERIFIED) 1_759_000_000_000L else null,
            perms, if (state == LicenseState.VERIFIED) true else null, disallowed)
        val prov = Provenance(url, "2026-09-01", "registry", "Sample registry entry")
        val variants = vars.map { (q, size, peak) ->
            ModelVariant("$id:$q", id, "gguf", q, size, params, arch, 8192, feasibility(peak), TrainingFeasibility(RunLocation.DESKTOP, "LoRA/QLoRA",
                listOf("Training needs a desktop GPU; this phone has no training runtime")), prov,
                "https://huggingface.co/sample/$id/resolve/main/$id-$q.gguf", sha("$id-$q"), false)
        }
        models[id] = CatalogModel(id, family, name, version, params, arch, lic, variants, prov)
    }

    private fun feasibility(peakMb: Int): Feasibility {
        val budget = device.availableRamMb * (1 - device.safetyReserveFraction)
        return when {
            peakMb <= budget * 0.85 -> Feasibility(Verdict.FITS_SAFELY, peakMb, listOf("Estimated peak $peakMb MB is within the safe budget (${budget.toInt()} MB)"))
            peakMb <= budget -> Feasibility(Verdict.TIGHT, peakMb, listOf("Estimated peak $peakMb MB leaves little headroom in the safe budget (${budget.toInt()} MB)"))
            else -> Feasibility(Verdict.DOES_NOT_FIT, peakMb, listOf("Estimated peak $peakMb MB exceeds the safe budget (${budget.toInt()} MB); would degrade the device"))
        }
    }

    private fun licenseText(id: String) = "SAMPLE LICENSE TEXT for $id\nThis is deterministic placeholder text.\n"

    // ---------------------------------------------------------------- seed
    private fun seed() {
        val p1 = createProject(NewProject("Motorcycle Mechanic", "Motorcycle service", "Diagnostic and repair assistant")).getOrNull()!!
        selectBaseModel(p1.id, "qwen3-4b", "qwen3-4b:q4_k_m")
        ingest(p1.id, listOf(doc("shop-manual.pdf", "application/pdf", 90_000, "A"), doc("shop-manual-copy.pdf", "application/pdf", 90_000, "A"),
            doc("wiring-notes.md", "text/markdown", 12_000, "B"), doc("scan-of-old-manual.pdf", "application/pdf", 400_000, "C"),
            doc("corrupt-notes.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 300, "D")), RightsStatus.LICENSED_FOR_TRAINING)
        buildDataset(p1.id)
        createProject(NewProject("HVAC Technician", "HVAC", "Troubleshooting assistant"))
        val p3 = createProject(NewProject("Town Design", "Urban design", "Street and zoning guidance")).getOrNull()!!
        selectBaseModel(p3.id, "qwen3-4b", "qwen3-4b:q4_k_m")
        ingest(p3.id, listOf(doc("zoning-code.pdf", "application/pdf", 60_000, "Z"), doc("street-design.txt", "text/plain", 20_000, "Y")), RightsStatus.LICENSED_FOR_TRAINING)
        buildDataset(p3.id)
        val leak = reviewItems(p3.id, ReviewFilter(flag = ReviewFlag.POSSIBLE_LEAKAGE, included = true), 0, 1000).getOrNull()!!.items.map { it.id }
        setIncluded(p3.id, leak, false); approveDataset(p3.id)
        selectMethod(p3.id, MethodIds.ADAPTER_DESKTOP)
        exportTrainingJobPackage(p3.id, java.io.ByteArrayOutputStream()); exportHeldOutEvalSet(p3.id, java.io.ByteArrayOutputStream())
        importEvaluation(p3.id, "sample results".byteInputStream())
    }

    private fun doc(name: String, mime: String, size: Int, tag: String) =
        SourceInput(name, mime, size.toLong()) { (tag.repeat(size)).byteInputStream() }

    // ---------------------------------------------------------------- projects
    private fun facts(p: P): ProjectFacts {
        val d = p.dataset
        val method = p.methodId?.let { id -> methodList(p).firstOrNull { it.id == id } }
        return ProjectFacts(p.baseModelId, p.baseModelId?.let { models[it]?.license?.state },
            p.sources.size, p.lastReport?.failed?.size ?: 0, d?.status ?: DatasetStatus.NONE,
            p.items.count { it.included && it.flags.isNotEmpty() }, method, p.trainingExported, p.referenceExported,
            p.heldOutExported, p.evaluation, p.specialistExported)
    }

    private fun summary(p: P): ProjectSummary {
        val st = Stages.derive(facts(p))
        return ProjectSummary(p.id, p.name, p.domain, p.purpose, p.created, p.updated, p.baseModelId, st, Stages.nextAction(st))
    }
    private fun isInstalled(p: P) = p.variantId?.let { it in acquired } ?: acquired.any { it.startsWith(p.baseModelId + ":") }
    private fun touch(p: P) { p.updated = now() }
    private fun <T> withP(id: ProjectId, f: (P) -> StudioResult<T>): StudioResult<T> =
        projects[id.value]?.let(f) ?: StudioResult.Err(StudioError.NotFound("project ${id.value}"))
    private fun <T> ok(v: T): StudioResult<T> = StudioResult.Ok(v)
    private fun err(e: StudioError) = StudioResult.Err(e)

    override fun listProjects() = projects.values.map(::summary)
    override fun createProject(p: NewProject): StudioResult<ProjectSummary> {
        if (p.name.isBlank()) return err(StudioError.Invalid("NAME_REQUIRED", "Give the specialist a name"))
        val proj = P(ProjectId(nextId("proj")), p.name.trim(), p.domain, p.purpose, now())
        projects[proj.id.value] = proj
        return ok(summary(proj))
    }
    override fun getProject(id: ProjectId) = withP(id) { ok(summary(it)) }
    override fun deleteProject(id: ProjectId) = withP(id) { projects.remove(id.value); ok(Unit) }

    // ---------------------------------------------------------------- device & recommendations
    override fun deviceProfile() = device

    override fun recommendations(): Recommendations {
        val all = models.values.flatMap { m -> m.variants.map { m to it } }
        val fits = all.filter { it.second.android.verdict == Verdict.FITS_SAFELY }
        val usable = all.filter { it.second.android.verdict != Verdict.DOES_NOT_FIT && it.first.license.state != LicenseState.DISALLOWED }
        val conf = Confidence(ConfidenceLevel.LOW, "Qualifier estimate only; no on-device benchmark and no inference runtime (none-v1)", false)
        fun rec(kind: ProfileKind, pick: Pair<CatalogModel, ModelVariant>?, why: String): ProfileRecommendation {
            if (pick == null) return ProfileRecommendation(kind, null, null, null, null, null, listOf("No catalog variant fits the safe envelope"), emptyList(), conf, null)
            val (m, v) = pick
            val warn = buildList {
                if (m.license.state != LicenseState.VERIFIED) add("License ${m.license.state}: verify before download or training")
                if (v.android.verdict == Verdict.TIGHT) add("Little RAM headroom; may degrade other apps under load")
            }
            return ProfileRecommendation(kind, m.id, v.id, v.quant, minOf(4096, v.contextTokensMax), v.android.estimatedPeakRamMb,
                listOf(why) + v.android.reasons, warn, conf, m.license.state)
        }
        val perf = fits.filter { it.first.license.state != LicenseState.DISALLOWED }.minByOrNull { it.second.sizeBytes }
        val bal = fits.filter { it.first.license.state != LicenseState.DISALLOWED }.maxByOrNull { it.second.paramsBillions * 1000 - it.second.sizeBytes / 1_000_000_000 }
        val max = usable.maxByOrNull { it.second.paramsBillions * 1000 + it.second.sizeBytes / 1_000_000 }
        return Recommendations(device, listOf(
            rec(ProfileKind.PERFORMANCE, perf, "Smallest footprint: fastest and coolest"),
            rec(ProfileKind.BALANCED, bal, "Best quality that fits comfortably inside the safe envelope"),
            rec(ProfileKind.MAX_QUALITY, max, "Highest quality that still stays inside the safe envelope")))
    }

    // ---------------------------------------------------------------- catalog & license
    override fun catalog() = models.values.map { m -> m.copy(variants = m.variants.map { it.copy(acquired = it.id in acquired) }) }
    override fun model(modelId: String): StudioResult<CatalogModel> =
        catalog().firstOrNull { it.id == modelId }?.let { ok(it) } ?: err(StudioError.NotFound("model $modelId"))

    override fun fetchLicenseText(modelId: String): StudioResult<LicenseTextFetch> {
        val m = models[modelId] ?: return err(StudioError.NotFound("model $modelId"))
        if (!m.license.authoritativeUrl.startsWith("https://")) return err(StudioError.Invalid("NOT_HTTPS", "License URL must be HTTPS"))
        val t = licenseText(modelId)
        return ok(LicenseTextFetch(m.license.authoritativeUrl, now(), sha(t), t, false).also { fetched[modelId] = it })
    }

    override fun importLicenseText(modelId: String, fileName: String, input: InputStream): StudioResult<LicenseTextFetch> {
        if (modelId !in models) return err(StudioError.NotFound("model $modelId"))
        val bytes = input.readBytes()
        if (bytes.isEmpty()) return err(StudioError.Invalid("EMPTY_FILE", "License file is empty"))
        val cap = 64 * 1024
        val text = String(bytes.copyOf(minOf(bytes.size, cap)), Charsets.UTF_8)
        // the hash covers the FULL bytes so truncation of the display text never hides content from the attestation
        return ok(LicenseTextFetch("file:$fileName", now(), sha(bytes), text, bytes.size > cap).also { fetched[modelId] = it })
    }

    override fun attestLicense(modelId: String, attestation: LicenseAttestation): StudioResult<LicenseInfo> {
        val m = models[modelId] ?: return err(StudioError.NotFound("model $modelId"))
        if (m.license.state == LicenseState.DISALLOWED && m.license.evidenceLevel != "owner_attested")
            return err(StudioError.Blocked("LICENSE_DISALLOWED", "Registry marks this license DISALLOWED; it cannot be verified here",
                listOf(Blocker("LICENSE_DISALLOWED", m.license.disallowedReason ?: "Disallowed"))))
        val f = fetched[modelId]
        if (f == null || f.sha256 != attestation.textSha256)
            return err(StudioError.Invalid("EVIDENCE_MISMATCH", "Fetch or import the license text first and attest that exact text"))
        val allowsTuning = attestation.permissions[Permission.FINE_TUNE] == Tri.YES && attestation.permissions[Permission.ADAPTER] == Tri.YES
        val lic = if (allowsTuning) m.license.copy(state = LicenseState.VERIFIED, evidenceLevel = "owner_reviewed_text",
            scopeText = "owner-reviewed license text from ${f.url}, hash ${f.sha256}; permissions as attested by owner",
            textSha256 = f.sha256, fetchedAt = f.fetchedAt, permissions = attestation.permissions, attributionRequired = attestation.attributionRequired,
            disallowedReason = null)
        else m.license.copy(state = LicenseState.DISALLOWED, evidenceLevel = "owner_attested", textSha256 = f.sha256, fetchedAt = f.fetchedAt,
            permissions = attestation.permissions, attributionRequired = attestation.attributionRequired,
            disallowedReason = "Owner attested that fine-tuning/adapters are not permitted")
        models[modelId] = m.copy(license = lic)
        return ok(lic)
    }

    // ---------------------------------------------------------------- base model
    override fun selectBaseModel(projectId: ProjectId, modelId: String, variantId: String?) = withP(projectId) { p ->
        val m = models[modelId] ?: return@withP err(StudioError.NotFound("model $modelId"))
        if (variantId != null && m.variants.none { it.id == variantId }) return@withP err(StudioError.NotFound("variant $variantId"))
        if (m.license.state == LicenseState.DISALLOWED)
            return@withP err(StudioError.Blocked("LICENSE_DISALLOWED", "This model's license does not allow the intended use", listOf(Blocker("LICENSE_DISALLOWED", m.license.disallowedReason ?: "Disallowed"))))
        p.baseModelId = modelId; p.variantId = variantId; touch(p); ok(summary(p))
    }

    // ---------------------------------------------------------------- acquisition
    private fun variantOf(id: String) = models.values.flatMap { it.variants.map { v -> it to v } }.firstOrNull { it.second.id == id }

    override fun planAcquisition(variantId: String, use: IntendedUse): StudioResult<AcquisitionPlan> {
        val (m, v) = variantOf(variantId) ?: return err(StudioError.NotFound("variant $variantId"))
        val free = device.freeStorageMb * 1024 * 1024
        val after = free - v.sizeBytes
        val b = ArrayList<Blocker>()
        when (m.license.state) {
            LicenseState.DISALLOWED -> b += Blocker("LICENSE_DISALLOWED", "License disallows this use: ${m.license.disallowedReason ?: "no reason recorded"}")
            LicenseState.UNVERIFIED -> b += Blocker("LICENSE_UNVERIFIED", "License text has not been reviewed. Fetch it and record your attestation first")
            LicenseState.VERIFIED -> {
                fun need(on: Boolean, p: Permission) { if (on && m.license.permissions[p] != Tri.YES) b += Blocker("PERMISSION_NOT_GRANTED", "License does not grant $p (attested: ${m.license.permissions[p]})") }
                need(use.fineTune, Permission.FINE_TUNE); need(use.adapter, Permission.ADAPTER)
                need(use.commercial, Permission.COMMERCIAL); need(use.redistribute, Permission.REDISTRIBUTION)
            }
        }
        if (v.downloadUrl == null) b += Blocker("NO_URL", "No download URL; import the file instead")
        else if (!v.downloadUrl.startsWith("https://")) b += Blocker("NOT_HTTPS", "Downloads must use HTTPS")
        if (after < storageReserveBytes) b += Blocker("INSUFFICIENT_STORAGE", "Not enough free storage while keeping a 1 GB reserve")
        if (v.android.verdict == Verdict.DOES_NOT_FIT) b += Blocker("DEVICE_DOES_NOT_FIT", "This variant would degrade normal device operation")
        if (v.id in acquired) b += Blocker("ALREADY_ACQUIRED", "Already on this device")
        return ok(AcquisitionPlan(v.id, m.name, v.downloadUrl, v.sizeBytes, free, after, storageReserveBytes, v.android, m.license.state, use, b.isEmpty(), b))
    }

    private fun newOp(kind: OperationKind, p: ProjectId?, subject: String, total: Long, state: OperationState = OperationState.RUNNING): Operation {
        val t = now()
        return Operation(nextId("op"), kind, p, subject, state, Progress(0, total, "bytes"), "", null, true, t, t).also { ops[it.id] = it }
    }
    private fun upd(o: Operation, f: (Operation) -> Operation): Operation = f(o).copy(updatedAt = now()).also { ops[it.id] = it }

    override fun startDownload(variantId: String, use: IntendedUse, confirmed: Boolean): StudioResult<Operation> {
        val plan = planAcquisition(variantId, use).let { it.getOrNull() ?: return StudioResult.Err(it.errorOrNull()!!) }
        if (!plan.allowed) return err(StudioError.Blocked("DOWNLOAD_BLOCKED", "Download is blocked", plan.blocking))
        if (!confirmed) return err(StudioError.Blocked("CONFIRMATION_REQUIRED", "Explicit confirmation is required", listOf(Blocker("CONFIRMATION_REQUIRED", "Confirm size, storage impact and license first"))))
        if (ops.values.any { it.kind == OperationKind.DOWNLOAD && it.subject == variantId && !it.isTerminal })
            return err(StudioError.Conflict("A download for this variant is already in progress"))
        return ok(newOp(OperationKind.DOWNLOAD, null, variantId, plan.sizeBytes))
    }

    override fun importModelFile(variantId: String, file: SourceInput): StudioResult<Operation> {
        if (variantOf(variantId) == null) return err(StudioError.NotFound("variant $variantId"))
        val n = file.open().use { it.readBytes().size.toLong() }
        if (n == 0L) return err(StudioError.Invalid("EMPTY_FILE", "The chosen file is empty"))
        acquired += variantId
        val o = newOp(OperationKind.IMPORT_MODEL, null, variantId, n)
        return ok(upd(o) { it.copy(state = OperationState.SUCCEEDED, progress = Progress(n, n, "bytes"), resumable = false, message = "Imported ${file.name}") })
    }

    override fun operations() = ops.values.toList()
    override fun operation(id: String) = ops[id]?.let { ok(it) } ?: err(StudioError.NotFound("operation $id"))

    override fun cancelOperation(id: String): StudioResult<Operation> {
        val o = ops[id] ?: return err(StudioError.NotFound("operation $id"))
        if (o.isTerminal) return err(StudioError.Conflict("Operation already finished"))
        return ok(upd(o) { it.copy(state = OperationState.CANCELLED, resumable = false, message = "Cancelled; partial file removed") })
    }

    override fun resumeOperation(id: String): StudioResult<Operation> {
        val o = ops[id] ?: return err(StudioError.NotFound("operation $id"))
        if (o.state != OperationState.PAUSED && o.state != OperationState.FAILED) return err(StudioError.Conflict("Only paused or failed operations can resume"))
        return ok(upd(o) { it.copy(state = OperationState.RUNNING, error = null, message = "Resuming from ${it.progress.done} bytes") })
    }

    // ---- test/preview hooks (not part of Studio)
    /** Advance every RUNNING operation by a quarter of its total. */
    fun tick() {
        v2.tick()
        for (o in ops.values.toList()) if (o.state == OperationState.RUNNING) {
            val done = minOf(o.progress.total, o.progress.done + maxOf(1, o.progress.total / 4))
            if (done >= o.progress.total) {
                if (o.kind == OperationKind.DOWNLOAD) acquired += o.subject
                upd(o) { it.copy(state = OperationState.SUCCEEDED, progress = Progress(done, it.progress.total, "bytes"), resumable = false, message = "Downloaded") }
            } else upd(o) { it.copy(progress = Progress(done, it.progress.total, "bytes"), message = "Downloading") }
        }
    }
    /** RUNNING -> PAUSED (resumable), as after the process was killed. */
    fun simulateProcessDeath() { v2.simulateProcessDeath(); ops.values.toList().filter { it.state == OperationState.RUNNING }.forEach { upd(it) { o -> o.copy(state = OperationState.PAUSED, message = "Interrupted; resume to continue") } } }
    fun failOperation(id: String, message: String) { ops[id]?.let { o -> upd(o) { it.copy(state = OperationState.FAILED, error = StudioError.Network(message), message = message) } } }

    // ---------------------------------------------------------------- sources
    override fun listSources(projectId: ProjectId) = withP(projectId) { ok(it.sources.values.toList()) }
    override fun lastIngestReport(projectId: ProjectId) = withP(projectId) { ok(it.lastReport) }

    override fun ingest(projectId: ProjectId, inputs: List<SourceInput>, rights: RightsStatus, onProgress: (Progress) -> Unit) = withP(projectId) { p ->
        val okMimes = listOf("text/", "application/pdf", "application/json", "application/vnd.openxmlformats")
        val seen = HashMap<String, String>()                 // sha -> sourceId (existing + this batch)
        p.sources.values.forEach { seen[it.provenance.sha256] = it.sourceId }
        val items = ArrayList<IngestItem>()
        inputs.forEachIndexed { i, input ->
            onProgress(Progress(i.toLong(), inputs.size.toLong(), "files"))
            val issues = ArrayList<IngestIssue>()
            fun fail(c: IssueCode, m: String) { issues += IngestIssue(c, m); items += IngestItem(input.name, IngestStatus.FAILED, null, null, issues.toList(), null) }
            val bytes = try { input.open().use { it.readBytes() } } catch (e: Exception) { null }
            when {
                bytes == null -> fail(IssueCode.CORRUPT_FILE, "Could not read the file")
                okMimes.none { input.mime.startsWith(it) } -> fail(IssueCode.UNSUPPORTED_TYPE, "${input.mime} is not supported")
                bytes.size > 50 * 1024 * 1024 -> fail(IssueCode.TOO_LARGE, "File exceeds 50 MB")
                bytes.isEmpty() -> fail(IssueCode.EMPTY_TEXT, "No text found")
                input.name.contains("corrupt", true) || String(bytes, 0, minOf(8, bytes.size), Charsets.ISO_8859_1).startsWith("%CORRUPT") ->
                    fail(IssueCode.CORRUPT_FILE, "The file is damaged and could not be read")
                input.mime == "application/pdf" && input.name.contains("scan", true) ->
                    fail(IssueCode.NEEDS_OCR, "No text layer; this is a scanned PDF. OCR is not available in this version")
                else -> {
                    val h = sha(bytes)
                    val dup = seen[h]
                    if (dup != null) items += IngestItem(input.name, IngestStatus.DUPLICATE, null, dup, listOf(IngestIssue(IssueCode.DUPLICATE_CONTENT, "Identical content to an existing source")), null)
                    else {
                        val pages = maxOf(1, bytes.size / 1800)
                        val sid = nextId("src")
                        val prov = ProvenanceSummary(sid, h, input.mime, bytes.size.toLong(), now(), if (input.mime == "application/pdf") pages else null, pages * 3, "fake-extractor/1", rights)
                        val iss = if (rights == RightsStatus.UNSET) listOf(IngestIssue(IssueCode.RIGHTS_UNSET, "Set usage rights before building a dataset")) else emptyList()
                        p.sources[sid] = SourceRecord(sid, input.name, prov, iss)
                        seen[h] = sid
                        items += IngestItem(input.name, IngestStatus.INGESTED, sid, null, iss, prov)
                    }
                }
            }
        }
        onProgress(Progress(inputs.size.toLong(), inputs.size.toLong(), "files"))
        val report = IngestReport(projectId, items, now())
        p.lastReport = report
        if (items.any { it.status == IngestStatus.INGESTED }) markStale(p)
        touch(p); ok(report)
    }

    private fun markStale(p: P) { p.dataset?.let { p.dataset = it.copy(status = DatasetStatus.STALE, approvedAt = null) } }

    override fun setSourceRights(projectId: ProjectId, sourceId: String, rights: RightsStatus) = withP(projectId) { p ->
        val s = p.sources[sourceId] ?: return@withP err(StudioError.NotFound("source $sourceId"))
        p.sources[sourceId] = s.copy(provenance = s.provenance.copy(rights = rights), issues = s.issues.filter { it.code != IssueCode.RIGHTS_UNSET })
        markStale(p); touch(p); ok(Unit)
    }

    override fun removeSource(projectId: ProjectId, sourceId: String) = withP(projectId) { p ->
        if (p.sources.remove(sourceId) == null) return@withP err(StudioError.NotFound("source $sourceId"))
        p.items.removeAll { it.sourceId == sourceId }
        p.lastReport = p.lastReport?.copy(items = p.lastReport!!.items.filter { it.sourceId != sourceId })
        p.trainingExported = false; p.referenceExported = false; p.heldOutExported = false; p.specialistExported = false; p.evaluation = null
        markStale(p); touch(p); ok(Unit)
    }

    // ---------------------------------------------------------------- dataset
    override fun buildDataset(projectId: ProjectId, options: DatasetOptions, onProgress: (Progress) -> Unit) = withP(projectId) { p ->
        if (p.sources.isEmpty()) return@withP err(StudioError.Blocked("NO_SOURCES", "Add sources first", listOf(Blocker("NO_SOURCES", "No ingested sources"))))
        val unset = p.sources.values.filter { it.provenance.rights == RightsStatus.UNSET }
        if (unset.isNotEmpty()) return@withP err(StudioError.Blocked("RIGHTS_UNSET", "Set usage rights for every source before building a dataset",
            unset.map { Blocker("RIGHTS_UNSET", "${it.name}: usage rights not set") }))
        val items = ArrayList<ReviewItem>()
        p.sources.values.forEachIndexed { si, s ->
            onProgress(Progress(si.toLong(), p.sources.size.toLong(), "sources"))
            for (i in 0 until s.provenance.chunks) {
                val flags = ArrayList<ReviewFlag>()
                if (i % 5 == 4) flags += ReviewFlag.LOW_CONFIDENCE
                if (i % 7 == 6) flags += ReviewFlag.TABLE
                if (i % 11 == 10) flags += ReviewFlag.POSSIBLE_LEAKAGE
                val bucket = ((i * 31 + si * 17 + options.seed) % 100).toInt().let { if (it < 0) it + 100 else it }
                val role = when {
                    s.provenance.rights == RightsStatus.REFERENCE_ONLY && options.excludeReferenceOnlySources -> ChunkRole.REFERENCE
                    bucket < options.heldOutFraction * 100 -> ChunkRole.HELD_OUT_EVAL
                    bucket < (options.heldOutFraction + options.validationFraction) * 100 -> ChunkRole.VALIDATION
                    else -> ChunkRole.TRAIN
                }
                items += ReviewItem("${s.sourceId}-c$i", s.sourceId, s.name, s.provenance.pages?.let { i / 3 + 1 }, "Section ${i / 3 + 1}", role,
                    ChunkOrigin.SOURCE_DERIVED, "Excerpt ${i + 1} of ${s.name}: ...", flags, !flags.contains(ReviewFlag.POSSIBLE_LEAKAGE))
            }
            if (options.includeSynthetic) for (k in 0 until 4)
                items += ReviewItem("${s.sourceId}-syn$k", s.sourceId, s.name, null, null, ChunkRole.TRAIN, ChunkOrigin.SYNTHETIC,
                    "Generated Q/A ${k + 1} derived from ${s.name}", listOf(ReviewFlag.LOW_CONFIDENCE), false)
        }
        onProgress(Progress(p.sources.size.toLong(), p.sources.size.toLong(), "sources"))
        p.items = items
        val ver = (p.dataset?.version ?: 0) + 1
        p.dataset = previewOf(p, ver, options, DatasetStatus.NEEDS_REVIEW, null)
        p.trainingExported = false; p.referenceExported = false; p.heldOutExported = false; p.evaluation = null; p.specialistExported = false
        touch(p); ok(p.dataset!!)
    }

    private fun previewOf(p: P, version: Int, options: DatasetOptions, status: DatasetStatus, approvedAt: Long?): DatasetPreview {
        val inc = p.items.filter { it.included }
        val flagged = p.items.filter { it.flags.isNotEmpty() }
        val stats = DatasetStats(p.items.size, inc.size, p.items.size - inc.size, flagged.size,
            p.items.groupingBy { it.role }.eachCount(), p.items.groupingBy { it.origin }.eachCount(),
            p.items.flatMap { it.flags }.groupingBy { it }.eachCount(), inc.count { ReviewFlag.POSSIBLE_LEAKAGE in it.flags })
        val warns = buildList {
            if (stats.leakageSuspects > 0) add("${stats.leakageSuspects} included items may leak into held-out material")
            if (stats.byRole[ChunkRole.HELD_OUT_EVAL] ?: 0 < 10) add("Held-out set is small; evaluation will have wide uncertainty")
        }
        return DatasetPreview(p.id, version, sha(p.items.joinToString("|") { it.id + it.included + it.role } + options), status, options, stats, now(), approvedAt, warns)
    }

    override fun datasetPreview(projectId: ProjectId) = withP(projectId) { ok(it.dataset) }

    override fun reviewItems(projectId: ProjectId, filter: ReviewFilter, offset: Int, limit: Int) = withP(projectId) { p ->
        val l = p.items.filter { i ->
            (!filter.onlyFlagged || i.flags.isNotEmpty()) && (filter.flag == null || filter.flag in i.flags) && (filter.role == null || i.role == filter.role) &&
                (filter.included == null || i.included == filter.included) && (filter.sourceId == null || i.sourceId == filter.sourceId)
        }
        ok(Page(l.drop(offset).take(limit), l.size, offset))
    }

    override fun setIncluded(projectId: ProjectId, itemIds: List<String>, included: Boolean) = withP(projectId) { p ->
        val d = p.dataset ?: return@withP err(StudioError.Blocked("NO_DATASET", "Build a dataset first", emptyList()))
        val ids = itemIds.toSet()
        if (!p.items.map { it.id }.containsAll(ids)) return@withP err(StudioError.NotFound("review item"))
        p.items = ArrayList(p.items.map { if (it.id in ids) it.copy(included = included) else it })
        p.dataset = previewOf(p, d.version, d.options, DatasetStatus.NEEDS_REVIEW, null)
        p.trainingExported = false; p.heldOutExported = false; p.referenceExported = false
        touch(p); ok(p.dataset!!)
    }

    override fun approveDataset(projectId: ProjectId) = withP(projectId) { p ->
        val d = p.dataset ?: return@withP err(StudioError.Blocked("NO_DATASET", "Build a dataset first", emptyList()))
        if (d.status == DatasetStatus.STALE) return@withP err(StudioError.Blocked("DATASET_STALE", "Sources changed; rebuild the dataset", listOf(Blocker("DATASET_STALE", "Rebuild required"))))
        val leaks = p.items.filter { it.included && ReviewFlag.POSSIBLE_LEAKAGE in it.flags }
        if (leaks.isNotEmpty()) return@withP err(StudioError.Blocked("LEAKAGE_UNRESOLVED", "Resolve possible train/eval leakage before approving",
            leaks.take(5).map { Blocker("POSSIBLE_LEAKAGE", "${it.sourceName} ${it.section ?: ""}: possible leakage into held-out material") }))
        p.dataset = previewOf(p, d.version, d.options, DatasetStatus.APPROVED, now()); touch(p); ok(p.dataset!!)
    }

    // ---------------------------------------------------------------- methods
    private fun methodList(p: P): List<MethodOption> {
        val lic = p.baseModelId?.let { models[it]?.license?.state }
        val tuneWhy = when {
            p.baseModelId == null -> "Choose a base model first"
            lic != LicenseState.VERIFIED -> "Base model license is $lic; verify it first"
            else -> null
        }
        val onDeviceWhy = when {
            !v2.engineAvailable -> "This install has no training runtime (nativeRuntimeId ${device.nativeRuntimeId}); it needs the native engine."
            p.baseModelId == null -> "Choose a base model first"
            !isInstalled(p) -> "Download or import the base model first"
            lic != LicenseState.VERIFIED -> "Base model license is $lic; verify it first"
            else -> null
        }
        return listOf(
            MethodOption(MethodIds.REFERENCE_PACKAGE, "Reference package (retrieval)", false, RunLocation.DEVICE, true, null,
                "Exact-reference lookup over your source chunks. This is NOT training; the model is unchanged."),
            MethodOption(MethodIds.PROMPT_SPECIALIZATION, "Prompt specialization", false, RunLocation.NONE, true, null,
                "Instructions only. This is NOT training and does not add knowledge."),
            MethodOption(MethodIds.ADAPTER_DESKTOP, "Adapter training (LoRA/QLoRA) on desktop", true, RunLocation.DESKTOP, tuneWhy == null, tuneWhy,
                "This phone only prepares the training job package. Training happens on a desktop/GPU via 'llmtrainer import-job'; nothing is trained on this device."),
            MethodOption(MethodIds.ADAPTER_ON_DEVICE, "Fine-tune all layers on this phone", true, RunLocation.DEVICE, false,
                if (!v2.engineAvailable) "This install has no training runtime (nativeRuntimeId ${device.nativeRuntimeId}); it needs the native engine."
                else "Too large for this phone's safe memory envelope (estimated 9.8 GB). Use the partial option or a desktop job.",
                "Changes model parameters on this phone. Not available here."),
            MethodOption(MethodIds.PARTIAL_ON_DEVICE, "Fine-tune the last layers on this phone", true, RunLocation.DEVICE, onDeviceWhy == null, onDeviceWhy,
                "Changes model parameters on this phone (a patch file; the base model is untouched). Partial fine-tune, not full."))
    }
    override fun methodOptions(projectId: ProjectId) = withP(projectId) { ok(methodList(it)) }
    override fun selectMethod(projectId: ProjectId, methodId: String) = withP(projectId) { p ->
        val m = methodList(p).firstOrNull { it.id == methodId } ?: return@withP err(StudioError.NotFound("method $methodId"))
        if (!m.available) return@withP err(StudioError.Blocked("METHOD_UNAVAILABLE", m.whyNotAvailable ?: "Unavailable", listOf(Blocker("METHOD_UNAVAILABLE", m.whyNotAvailable ?: ""))))
        p.methodId = methodId; touch(p); ok(summary(p))
    }

    // ---------------------------------------------------------------- exports
    private fun approvedOr(p: P): StudioError? =
        if (p.dataset?.status != DatasetStatus.APPROVED) StudioError.Blocked("DATASET_NOT_APPROVED", "Approve the dataset first", listOf(Blocker("DATASET_NOT_APPROVED", "Dataset is ${p.dataset?.status ?: DatasetStatus.NONE}"))) else null

    private fun write(kind: String, file: String, files: List<String>, body: String, out: OutputStream, warnings: List<String> = emptyList()): StudioResult<ExportedPackage> {
        val bytes = body.toByteArray()
        return try { out.write(bytes); out.flush(); ok(ExportedPackage(kind, file, bytes.size.toLong(), sha(bytes), files, warnings)) }
        catch (e: Exception) { err(StudioError.Io("Could not write the package: ${e.message}")) }
    }

    override fun exportTrainingJobPackage(projectId: ProjectId, out: OutputStream) = withP(projectId) { p ->
        approvedOr(p)?.let { return@withP err(it) }
        val m = methodList(p).firstOrNull { it.id == p.methodId }
        if (m == null || !m.isTraining) return@withP err(StudioError.Blocked("METHOD_NOT_TRAINING", "Select a training method first", listOf(Blocker("METHOD_NOT_TRAINING", "Current method does not train"))))
        val lic = models[p.baseModelId]!!.license
        if (lic.state != LicenseState.VERIFIED) return@withP err(StudioError.Blocked("LICENSE_${lic.state}", "Base model license gate", listOf(Blocker("LICENSE_${lic.state}", "License is ${lic.state}"))))
        val r = write("training-job", "${slug(p)}-training-job.zip", listOf("manifest.json", "dataset/train.jsonl", "dataset/validation.jsonl", "config.json", "license-evidence.json"),
            "FAKE-TRAINING-JOB\nproject=${p.name}\nbase=${p.baseModelId}\ndataset=${p.dataset!!.datasetSha256}\n", out)
        if (r is StudioResult.Ok) { p.trainingExported = true; touch(p) }
        r
    }

    override fun exportReferencePackage(projectId: ProjectId, out: OutputStream) = withP(projectId) { p ->
        approvedOr(p)?.let { return@withP err(it) }
        val r = write("reference", "${slug(p)}-reference.zip", listOf("manifest.json", "chunks.jsonl", "provenance.json"),
            "FAKE-REFERENCE\nproject=${p.name}\ndataset=${p.dataset!!.datasetSha256}\n", out)
        if (r is StudioResult.Ok) { p.referenceExported = true; touch(p) }
        r
    }

    override fun exportHeldOutEvalSet(projectId: ProjectId, out: OutputStream) = withP(projectId) { p ->
        approvedOr(p)?.let { return@withP err(it) }
        val n = p.items.count { it.role == ChunkRole.HELD_OUT_EVAL }
        val r = write("heldout-eval", "${slug(p)}-heldout.jsonl", listOf("heldout.jsonl", "manifest.json"), "FAKE-HELDOUT\nitems=$n\n", out,
            if (n < 10) listOf("Only $n held-out items; evaluation will be statistically weak") else emptyList())
        if (r is StudioResult.Ok) { p.heldOutExported = true; touch(p) }
        r
    }

    // ---------------------------------------------------------------- evaluation
    override fun importEvaluation(projectId: ProjectId, results: InputStream) = withP(projectId) { p ->
        if (!p.heldOutExported) return@withP err(StudioError.Blocked("HELDOUT_NOT_EXPORTED", "Export the held-out set and evaluate on the desktop first", listOf(Blocker("HELDOUT_NOT_EXPORTED", "No held-out set was exported"))))
        val text = String(results.readBytes())
        if (text.isBlank()) return@withP err(StudioError.Invalid("RESULTS_INVALID", "Not a results package"))
        val stub = "STUB" in text; val large = "LARGE" in text
        val n = if (large) 400 else 24
        fun row(id: String, label: String, b: Double, s: Double, hib: Boolean = true): MetricRow {
            val half = if (large) 0.02 else 0.15
            return MetricRow(id, label, hib, b, s, s - b, s - b - half, s - b + half, n)
        }
        val metrics = listOf(row("domain_correctness", "Domain correctness", 0.52, 0.64), row("unsupported_claims", "Unsupported claims", 0.21, 0.12, false),
            row("citation_accuracy", "Citation accuracy", 0.40, 0.55))
        val excludesZero = metrics.all { m -> (m.ciLow!! > 0) == m.higherIsBetter && (m.ciHigh!! > 0) == m.higherIsBetter }
        val allowed = !stub && large && excludesZero
        val reason = when {
            stub -> "Results are a stub from a dry run; no real evaluation was performed"
            !large -> "Only n=$n held-out items; confidence intervals include zero, so no improvement claim is supported"
            else -> "n=$n and every confidence interval excludes zero"
        }
        val caveats = buildList {
            add("Evaluated on desktop; this phone ran no model")
            if (stub) add("STUB results: do not treat as evidence")
            if (!large) add("Small sample (n=$n)")
        }
        val ev = EvaluationView(projectId, nextId("eval"), models[p.baseModelId]?.name ?: "base", "${p.name} (adapter)", metrics, caveats, allowed, reason, stub, now())
        p.evaluation = ev; p.specialistExported = false; touch(p); ok(ev)
    }
    override fun evaluation(projectId: ProjectId) = withP(projectId) { ok(it.evaluation) }

    // ---------------------------------------------------------------- specialist package
    override fun exportSpecialistPackage(projectId: ProjectId, out: OutputStream) = withP(projectId) { p ->
        val ev = p.evaluation ?: return@withP err(StudioError.Blocked("NO_EVALUATION", "Evaluate before exporting", listOf(Blocker("NO_EVALUATION", "No evaluation results"))))
        val lic = models[p.baseModelId]!!.license
        if (lic.state != LicenseState.VERIFIED) return@withP err(StudioError.Blocked("LICENSE_${lic.state}", "Base model license gate", listOf(Blocker("LICENSE_${lic.state}", "License is ${lic.state}"))))
        val warn = buildList { if (!ev.improvementClaimAllowed) add("Package records that the evidence does not support an improvement claim"); if (ev.isStub) add("Evaluation is a stub") }
        val body = "FAKE-SPECIALIST\nname=${p.name}\nversion=1.0.0\nbase=${p.baseModelId}\nbaseVersion=${models[p.baseModelId]!!.version}\nlicense=${lic.state}\n" +
            "dataset=${p.dataset?.datasetSha256}\nmethod=${p.methodId}\nclaim=${ev.improvementClaimAllowed}\neval=${ev.claimReason}\n"
        val r = write("specialist", "${slug(p)}-1.0.0.specialist.zip", listOf("manifest.json", "adapter/", "provenance.json", "license-bundle/", "evaluation.json"), body, out, warn)
        if (r is StudioResult.Ok) { p.specialistExported = true; touch(p) }
        r
    }

    override fun importSpecialistPackage(input: InputStream): StudioResult<SpecialistPackageView> {
        val bytes = input.readBytes()
        val lines = String(bytes).lines()
        if (lines.firstOrNull() != "FAKE-SPECIALIST") return err(StudioError.Invalid("PACKAGE_INVALID", "Not a specialist package"))
        val kv = lines.drop(1).filter { '=' in it }.associate { it.substringBefore('=') to it.substringAfter('=') }
        val missing = listOf("name", "version", "base", "baseVersion", "license").filter { it !in kv }
        val checks = listOf(CheckResult("manifest complete", missing.isEmpty(), if (missing.isEmpty()) "ok" else "missing ${missing.joinToString()}"),
            CheckResult("license verified", kv["license"] == "VERIFIED", "base license ${kv["license"]}"),
            CheckResult("artifact hashes", true, "sha256 ${sha(bytes).take(12)}"))
        if (missing.isNotEmpty()) return err(StudioError.Invalid("PACKAGE_INVALID", "Manifest is missing: ${missing.joinToString()}"))
        return ok(SpecialistPackageView(kv["name"]!!, kv["version"]!!, kv["base"]!!, kv["baseVersion"]!!, runCatching { LicenseState.valueOf(kv["license"]!!) }.getOrDefault(LicenseState.UNVERIFIED),
            "Base model license state recorded in package", kv["dataset"], mapOf("package" to sha(bytes)), kv["method"] ?: "unknown", kv["eval"],
            kv["claim"] == "true", listOf("Runs with any GGUF-capable runtime that supports the base model"), listOf("Exact facts should be grounded with the reference package"),
            checks, checks.all { it.passed }))
    }

    // ---------------------------------------------------------------- v2 (phone-first): scripted, see FakeStudioV2
    override fun engineStatus() = v2.engineStatus()
    override fun installedModels() = v2.installedModels()
    override fun projectModelState(projectId: ProjectId) = v2.projectModelState(projectId)
    override fun createChat(projectId: ProjectId, target: ChatTarget, specialistId: String?, options: ChatOptions) = v2.createChat(projectId, target, specialistId, options)
    override fun listChats(projectId: ProjectId) = v2.listChats(projectId)
    override fun chatHistory(chatId: String) = v2.chatHistory(chatId)
    override fun deleteChat(chatId: String) = v2.deleteChat(chatId)
    override fun sendMessage(chatId: String, text: String, cancel: CancelToken, onToken: (String) -> Unit) = v2.sendMessage(chatId, text, cancel, onToken)
    override fun localTrainingPlan(projectId: ProjectId) = v2.localTrainingPlan(projectId)
    override fun startLocalTraining(projectId: ProjectId, settings: TrainingSettings, confirmed: Boolean) = v2.startLocalTraining(projectId, settings, confirmed)
    override fun trainingRuns(projectId: ProjectId) = v2.trainingRuns(projectId)
    override fun trainingRun(runId: String) = v2.trainingRun(runId)
    override fun pauseTraining(runId: String) = v2.pauseTraining(runId)
    override fun cancelTraining(runId: String) = v2.cancelTraining(runId)
    override fun resumeTraining(runId: String) = v2.resumeTraining(runId)
    override fun specialists(projectId: ProjectId) = v2.specialists(projectId)
    override fun verifySpecialist(specialistId: String) = v2.verifySpecialist(specialistId)
    override fun selectSpecialist(projectId: ProjectId, specialistId: String?) = v2.selectSpecialist(projectId, specialistId)
    override fun deleteSpecialist(specialistId: String) = v2.deleteSpecialist(specialistId)
    override fun exportSpecialistPatch(specialistId: String, out: OutputStream) = v2.exportSpecialistPatch(specialistId, out)
    override fun startLocalEvaluation(projectId: ProjectId, specialistId: String, options: LocalEvalOptions) = v2.startLocalEvaluation(projectId, specialistId, options)
    override fun localEvaluations(projectId: ProjectId) = v2.localEvaluations(projectId)
    override fun localEvaluation(evalId: String) = v2.localEvaluation(evalId)
    override fun cancelLocalEvaluation(evalId: String) = v2.cancelLocalEvaluation(evalId)
    override fun compareAB(projectId: ProjectId, specialistId: String, prompt: String, options: ABOptions, cancel: CancelToken, onToken: (ChatTarget, String) -> Unit) =
        v2.compareAB(projectId, specialistId, prompt, options, cancel, onToken)
    override fun abComparisons(projectId: ProjectId) = v2.abComparisons(projectId)
    override fun saveABNote(projectId: ProjectId, comparisonId: String, note: String) = v2.saveABNote(projectId, comparisonId, note)
    override fun deleteAB(projectId: ProjectId, comparisonId: String) = v2.deleteAB(projectId, comparisonId)

    // ---------------------------------------------------------------- host
    var restartCount = 0; private set
    override val host: HostHooks = object : HostHooks {
        override fun diagnosticsJson() = """{"hostVersionName":"1.0.0","hostApiLevel":1,"nativeAbi":1,"nativeRuntimeId":"none-v1","runningSource":"fake"}"""
        override fun deviceSnapshotJson() = """{"totalRamMb":${device.totalRamMb},"availableRamMb":${device.availableRamMb}}"""
        override fun checkForUpdates(callback: (UpdateStatus) -> Unit) = callback(UpdateStatus("up-to-date", "You are on the latest version"))
        override fun restartApp() { restartCount++ }
    }

    private fun slug(p: P) = p.name.lowercase().replace(Regex("[^a-z0-9]+"), "-").trim('-')
    private fun sha(s: String) = sha(s.toByteArray())
    private fun sha(b: ByteArray) = MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }
}
