package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.ChunkRole
import com.hotatticgames.llmtrainer.studio.api.CheckResult
import com.hotatticgames.llmtrainer.studio.api.EvaluationView
import com.hotatticgames.llmtrainer.studio.api.LicenseState
import com.hotatticgames.llmtrainer.studio.api.MetricRow
import com.hotatticgames.llmtrainer.studio.api.ProjectId
import com.hotatticgames.llmtrainer.studio.api.ReviewFlag
import com.hotatticgames.llmtrainer.studio.api.RightsStatus
import com.hotatticgames.llmtrainer.studio.api.SpecialistPackageView
import org.json.JSONObject

/** A package ready to be written, plus warnings to show the owner. */
class BuiltPackage(val writer: PackageWriter, val kind: String, val fileName: String, val warnings: List<String>, val meta: Map<String, Any?> = emptyMap())

class JobContext(
    val projectId: String, val name: String, val domain: String, val purpose: String, val createdAt: Long,
    val commercial: Boolean, val redistribute: Boolean,
    val entry: RegEntry, val variantId: String?, val variantSourceUrl: String?,
    val evidence: Pair<Map<String, Any?>, ByteArray?>?, val sources: List<SourceDoc>, val ds: DatasetState,
    val deviceName: String, val deviceRamMb: Int, val nowMs: Long, val jobId: String, val appVersion: String,
    val terms: List<Term>, val methodRequested: String,
)

object Packages {
    const val JOB_FORMAT = "llmtrainer-training-job"
    const val RESULTS_FORMAT = "llmtrainer-results"
    const val REPORT_FORMAT = "llmtrainer-evaluation-report"
    const val HELDOUT_FORMAT = "llmtrainer-heldout-set"
    const val REFERENCE_FORMAT = "llmtrainer-reference-package"
    const val SPECIALIST_FORMAT = "llmtrainer-specialist-package"
    const val ID_PATTERN = "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
    val ID_RE = Regex(ID_PATTERN)
    const val TEMPLATE_NOTE = "Examples are produced by deterministic templates over the source text (builder studio-template-builder/1, type template): " +
        "cloze continuation of source sentences" + "; optional question templates are labelled synthetic. They are not human-written and not LLM-generated."

    fun created(nowMs: Long) = Iso.ts(nowMs)

    // ---- source/rights mapping ---------------------------------------------------------------------------------------------

    fun rightsJson(r: RightsStatus): Map<String, Any?> = when (r) {
        RightsStatus.OWNER_AUTHORED -> linkedMapOf("status" to "owned", "license_id" to null, "permitted_training" to "yes", "permitted_commercial" to "yes",
            "permitted_redistribution" to "unverified", "evidence" to "owner states the material is their own work", "notes" to null)
        RightsStatus.LICENSED_FOR_TRAINING -> linkedMapOf("status" to "licensed", "license_id" to null, "permitted_training" to "yes", "permitted_commercial" to "unverified",
            "permitted_redistribution" to "unverified", "evidence" to "owner states the material is licensed for training", "notes" to null)
        RightsStatus.PERMISSION_GRANTED -> linkedMapOf("status" to "licensed", "license_id" to null, "permitted_training" to "yes", "permitted_commercial" to "unverified",
            "permitted_redistribution" to "unverified", "evidence" to "owner states explicit permission was granted", "notes" to null)
        RightsStatus.REFERENCE_ONLY -> linkedMapOf("status" to "restricted", "license_id" to null, "permitted_training" to "no", "permitted_commercial" to "unverified",
            "permitted_redistribution" to "unverified", "evidence" to "owner marked the source reference-only", "notes" to "retrieval only; never training data")
        RightsStatus.UNSET -> linkedMapOf("status" to "unverified", "license_id" to null, "permitted_training" to "unverified", "permitted_commercial" to "unverified",
            "permitted_redistribution" to "unverified", "evidence" to null, "notes" to "rights not set")
    }

    private fun sourceJson(s: SourceDoc, prov: JSONObject?) = linkedMapOf<String, Any?>(
        "id" to s.sourceId, "file_name" to s.name, "title" to s.title, "sha256" to Hashing.prefixed(s.sha256), "size" to s.sizeBytes, "mime" to s.mime,
        "ingested_at" to Iso.ts(s.ingestedAt), "extractor" to s.extractor, "extractor_version" to s.extractorVersion, "origin" to "owner library", "source_version" to null,
        "rights" to rightsJson(s.rights),
        "issues" to s.issues.map { i ->
            linkedMapOf("code" to i.code.name.lowercase(), "detail" to i.message.take(500),
                "severity" to (if (i.code.name == "CORRUPT_FILE") "error" else "warning"), "page" to null)
        },
    )

    private fun excludeReason(d: DChunk, ds: DatasetState): String {
        d.chunk.excludeReason?.let { return it }
        if (!d.trainable && d.reason?.startsWith("source is reference-only") == true) return "reference_only_source"
        if (ReviewFlag.POSSIBLE_LEAKAGE in d.flags) return "possible_leakage"
        if (ReviewFlag.DUPLICATE in d.flags) return "duplicate_chunk"
        if (ReviewFlag.HEADER_FOOTER_NOISE in d.flags) return "header_footer_noise"
        return "owner_excluded"
    }

    private fun chunkRole(d: DChunk, ds: DatasetState): String = when {
        !ds.isIncluded(d) -> "excluded"
        d.split != null -> "train"
        else -> "reference"
    }

    fun chunkRows(ds: DatasetState, sourcesById: Map<String, SourceDoc>): List<Map<String, Any?>> = ds.chunks.map { d ->
        val role = chunkRole(d, ds)
        linkedMapOf<String, Any?>(
            "id" to d.chunk.id, "source_id" to d.sourceId, "page" to d.chunk.page, "section_path" to d.chunk.sectionPath,
            "char_start" to d.chunk.charStart, "char_end" to d.chunk.charEnd, "role" to role,
            "exclude_reason" to (if (role == "excluded") excludeReason(d, ds) else null), "text" to d.chunk.text,
            "sha256" to Hashing.prefixed(d.chunk.sha256), "origin" to (if (d.chunk.kind == "table") "table" else "extracted"),
            "group_id" to (d.group ?: "${d.sourceId}#${Text.slugify(d.chunk.section)}"),
        )
    }

    // ---- training-job package ----------------------------------------------------------------------------------------------

    class JobBuild(val pkg: BuiltPackage, val assembled: DatasetEngine.Assembled, val items: List<DatasetEngine.EvalItem>)

    /** Throws [PackageException] with a Blocked-style code when the data cannot honestly form a job. */
    fun buildJob(c: JobContext, srcStore: SourceStore): JobBuild {
        val ds = c.ds
        if (!ds.meta.splitsAvailable) throw PackageException("TOO_FEW_GROUPS", "There are fewer than 3 independent document/section groups, so no honest train/validation/held-out split exists. Add more documents.")
        val a = DatasetEngine.assemble(ds)
        val problems = DatasetEngine.verify(a)
        if (problems.isNotEmpty()) throw PackageException("LEAKAGE_DETECTED", "Leakage check failed: ${problems.first()}" + if (problems.size > 1) " (+${problems.size - 1} more)" else "")
        val counts = a.bySplit.mapValues { it.value.size }
        val empty = counts.filterValues { it == 0 }.keys
        if (empty.isNotEmpty()) throw PackageException("EMPTY_SPLIT", "After exclusions and leakage protection the ${empty.joinToString(" and ")} split has no examples: $counts. Include more material.")
        val warnings = ArrayList<String>()
        warnings.addAll(a.notes)
        val (items, itemWarnings) = heldOutItems(ds, a, c.domain)
        warnings.addAll(itemWarnings)
        if (items.isEmpty()) throw PackageException("NO_HELDOUT_ITEMS", "The held-out material produced no evaluation items (no numeric facts and not enough distinctive terms). Add more held-out material.")
        if (items.size < 50) warnings.add("Only ${items.size} held-out evaluation items: results will be statistically weak and a desktop evaluation will not be allowed to claim improvement below 50 items.")
        warnings.add("Training happens on a desktop/GPU with 'llmtrainer import-job'; nothing was trained on this device.")
        if (c.evidence == null) warnings.add("No complete owner license attestation: the desktop will keep the base model UNVERIFIED and refuse real training.")

        val w = PackageWriter()
        val sourcesById = c.sources.associateBy { it.sourceId }
        val usedSources = c.sources.sortedBy { it.sourceId }
        val methodRationale = "Domain vocabulary and reasoning style from your sources; exact specifications stay in retrieval (reference package)."
        w.addJson("manifest.json", linkedMapOf(
            "format" to JOB_FORMAT, "version" to 1, "job_id" to c.jobId, "created_at" to created(c.nowMs),
            "created_by" to mapOf("app" to "llmtrainer-studio", "app_version" to c.appVersion),
            "project" to linkedMapOf("id" to c.projectId, "name" to c.name, "domain" to c.domain, "purpose" to c.purpose, "created_at" to created(c.createdAt),
                "intended_use" to mapOf("commercial" to c.commercial, "redistribute_model" to c.redistribute)),
            "device" to mapOf("name" to c.deviceName, "total_ram_mb" to c.deviceRamMb),
            "base_model" to linkedMapOf("registry_id" to c.entry.entryId, "family" to c.entry.family, "exact_version" to c.entry.version,
                "variant" to c.variantId, "source_url" to c.variantSourceUrl),
            "license_evidence" to c.evidence?.first,
            "method" to linkedMapOf("requested" to c.methodRequested, "rationale" to methodRationale),
            "sources" to usedSources.map { sourceJson(it, srcStore.provenanceJson(it.sourceId)) },
            "extensions" to linkedMapOf("studio" to linkedMapOf("dataset_sha256" to Hashing.prefixed(DatasetEngine.datasetSha(ds)), "dataset_notes" to TEMPLATE_NOTE,
                "example_origin_policy" to "source_derived = deterministic template over source text; synthetic = template question wording", "top_terms" to c.terms.take(30).map { it.term })),
        ).let { m -> if (c.evidence == null) m.filterKeys { it != "license_evidence" } else m })
        w.addJsonl("chunks.jsonl", chunkRows(ds, sourcesById))
        val used = a.bySplit.values.flatten().map { it.groupId }.toSet()
        val groups = ds.meta.assignment.filterKeys { it in used }
        val dsId = "ds-" + DatasetEngine.datasetSha(ds).take(12)
        w.addJson("dataset_manifest.json", linkedMapOf(
            "dataset_id" to dsId, "dataset_version" to ds.meta.version, "seed" to ds.meta.options.seed,
            "split_config" to linkedMapOf("ratios" to DatasetEngine.ratios(ds.meta.options), "group_by" to ds.meta.groupBy, "near_duplicate_threshold" to DatasetEngine.NEAR_DUP_THRESHOLD, "shingle_size" to DatasetEngine.SHINGLE),
            "builder" to mapOf("name" to ExampleGen.BUILDER_NAME, "version" to ExampleGen.BUILDER_VERSION, "type" to "template"),
            "split_assignment" to linkedMapOf("group_by" to ds.meta.groupBy, "groups" to groups),
            "counts" to linkedMapOf("train" to counts.getValue("train"), "validation" to counts.getValue("validation"), "test" to counts.getValue("test"))))
        for (s in listOf("train", "validation", "test")) {
            w.addJsonl("dataset/$s.jsonl", a.bySplit.getValue(s).map { e ->
                linkedMapOf("example_id" to e.id, "group_id" to e.groupId, "prompt" to e.prompt, "response" to e.response, "origin" to e.origin,
                    "derived_from" to listOf(mapOf("source_id" to e.sourceId, "chunk_id" to e.chunkId)), "task" to e.task, "quality" to e.quality)
            })
        }
        w.addJsonl("eval/heldout.jsonl", items.map(::itemJson))
        c.evidence?.second?.let { w.add("license/license_text.txt", it) }
        return JobBuild(BuiltPackage(w, "training-job", "${c.projectId}-${c.jobId}.llmtrainer-job.zip", warnings, mapOf("job_id" to c.jobId)), a, items)
    }

    private fun itemJson(i: DatasetEngine.EvalItem) = linkedMapOf<String, Any?>(
        "item_id" to i.itemId, "kind" to i.kind, "question" to i.question, "gold_refs" to i.goldRefs,
        "expected" to (if (i.kind == "fact") mapOf("value" to i.expectedValue, "unit" to i.expectedUnit) else null),
        "required_terms" to i.requiredTerms, "origin" to "source_derived")

    /** Held-out items come ONLY from test-split chunks that back a surviving test example; leaking items are dropped with a warning. */
    fun heldOutItems(ds: DatasetState, a: DatasetEngine.Assembled, domain: String): Pair<List<DatasetEngine.EvalItem>, List<String>> {
        val testRefs = a.bySplit.getValue("test").map { "${it.sourceId}/${it.chunkId}" }.toSet()
        val trainValRefs = (a.bySplit.getValue("train") + a.bySplit.getValue("validation")).map { "${it.sourceId}/${it.chunkId}" }.toSet()
        val chunks = ds.chunks.filter { it.ref in testRefs && it.split == "test" && ds.isIncluded(it) && it.ref !in trainValRefs }
            .map { it.ref to (it.chunk.section to it.chunk.text) }
        val items = DatasetEngine.buildEvalItems(chunks, domain.ifBlank { "this domain" })
        val trainSh = HashSet<Long>()
        for (e in a.bySplit.getValue("train") + a.bySplit.getValue("validation")) trainSh.addAll(Similarity.shingles(e.prompt + " " + e.response, DatasetEngine.SHINGLE))
        val warnings = ArrayList<String>()
        val kept = items.filter { it ->
            val sh = Similarity.shingles(it.contentText, DatasetEngine.SHINGLE)
            val leak = sh.isNotEmpty() && sh.count { s -> s in trainSh }.toDouble() / sh.size >= 0.6
            if (leak) warnings.add("Held-out item ${it.itemId} dropped: its content overlaps training text")
            !leak
        }
        return Pair(kept, warnings)
    }

    // ---- held-out set --------------------------------------------------------------------------------------------------------

    fun buildHeldOut(c: JobContext): BuiltPackage {
        val ds = c.ds
        if (!ds.meta.splitsAvailable) throw PackageException("TOO_FEW_GROUPS", "There are fewer than 3 independent document/section groups, so no honest held-out split exists.")
        val a = DatasetEngine.assemble(ds)
        val problems = DatasetEngine.verify(a)
        if (problems.isNotEmpty()) throw PackageException("LEAKAGE_DETECTED", "Leakage check failed: ${problems.first()}")
        val (items, w0) = heldOutItems(ds, a, c.domain)
        if (items.isEmpty()) throw PackageException("NO_HELDOUT_ITEMS", "The held-out material produced no evaluation items. Add more held-out material.")
        val warnings = ArrayList(w0)
        if (items.size < 50) warnings.add("Only ${items.size} held-out items; evaluation will be statistically weak (an improvement claim needs at least 50).")
        val w = PackageWriter()
        w.addJson("manifest.json", linkedMapOf("format" to HELDOUT_FORMAT, "version" to 1, "job_id" to c.jobId, "project_id" to c.projectId, "created_at" to created(c.nowMs),
            "dataset_id" to "ds-" + DatasetEngine.datasetSha(ds).take(12), "n_items" to items.size,
            "note" to "Derived ONLY from held-out (test-split) chunks. Never train on this material.",
            "created_by" to mapOf("app" to "llmtrainer-studio", "app_version" to c.appVersion)))
        w.addJsonl("eval/heldout.jsonl", items.map(::itemJson))
        return BuiltPackage(w, "heldout-eval", "${c.projectId}-${c.jobId}.llmtrainer-heldout.zip", warnings, mapOf("job_id" to c.jobId))
    }

    // ---- reference (RAG) package ---------------------------------------------------------------------------------------------

    fun buildReference(c: JobContext): BuiltPackage {
        val ds = c.ds
        val included = ds.chunks.filter { ds.isIncluded(it) }
        if (included.isEmpty()) throw PackageException("NOTHING_TO_EXPORT", "No chunks are included in the dataset.")
        val w = PackageWriter()
        val srcById = c.sources.associateBy { it.sourceId }
        val warnings = ArrayList<String>()
        warnings.add("This is a reference package for exact-fact lookup (retrieval). It is NOT training: the model weights are unchanged.")
        val heldRefs = ds.chunks.filter { it.split == "test" }.map { it.ref }.sorted()
        if (heldRefs.isNotEmpty()) warnings.add("${heldRefs.size} chunks belong to the held-out evaluation split; they are listed in the manifest so an evaluator can exclude them.")
        w.addJson("reference_manifest.json", linkedMapOf("format" to REFERENCE_FORMAT, "version" to 1, "project_id" to c.projectId, "name" to c.name, "domain" to c.domain,
            "created_at" to created(c.nowMs), "dataset_id" to "ds-" + DatasetEngine.datasetSha(ds).take(12), "n_chunks" to included.size,
            "heldout_chunk_refs" to heldRefs, "purpose" to "exact-reference retrieval; not training",
            "created_by" to mapOf("app" to "llmtrainer-studio", "app_version" to c.appVersion)))
        w.addJsonl("chunks.jsonl", included.map { d ->
            val s = srcById[d.sourceId]
            linkedMapOf("ref" to d.ref, "source_id" to d.sourceId, "chunk_id" to d.chunk.id, "page" to d.chunk.page, "section_path" to d.chunk.sectionPath,
                "kind" to d.chunk.kind, "text" to d.chunk.text, "sha256" to Hashing.prefixed(d.chunk.sha256), "rights" to (s?.rights?.name ?: "UNSET"), "split" to d.split)
        })
        w.addJson("sources.json", c.sources.sortedBy { it.sourceId }.map { s ->
            linkedMapOf("id" to s.sourceId, "file_name" to s.name, "sha256" to Hashing.prefixed(s.sha256), "size" to s.sizeBytes, "mime" to s.mime, "rights" to rightsJson(s.rights), "extractor" to "${s.extractor}/${s.extractorVersion}")
        })
        w.addJson("terms.json", c.terms.map { linkedMapOf("term" to it.term, "tf" to it.tf, "df" to it.df, "score" to it.score, "chunks" to it.chunks) })
        return BuiltPackage(w, "reference", "${c.projectId}-reference.zip", warnings)
    }

    // ---- results (desktop -> phone) -------------------------------------------------------------------------------------------

    class ParsedResults(
        val view: EvaluationView, val jobId: String, val status: String, val reportBytes: ByteArray, val manifest: JSONObject,
        val specialistKind: String, val artifactRefs: List<String>, val artifactHashes: Map<String, String>, val method: String?,
    )

    fun parseResults(projectId: ProjectId, files: Map<String, ByteArray>, knownJobIds: Set<String>, nowMs: Long, runIdFallback: String): ParsedResults {
        val bad = Zips.verifyChecksums(files)
        if (bad.isNotEmpty()) throw PackageException("CHECKSUM_FAILED", "The results package failed its integrity check: ${bad.first()}" + if (bad.size > 1) " (+${bad.size - 1} more)" else "")
        val m = J.parseOrNull(String(files["manifest.json"] ?: throw PackageException("RESULTS_INVALID", "manifest.json is missing"), Charsets.UTF_8))
            ?: throw PackageException("RESULTS_INVALID", "manifest.json is not valid JSON")
        if (m.str("format") != RESULTS_FORMAT || m.int("version") != 1) throw PackageException("RESULTS_INVALID", "Not a version-1 llmtrainer results package (format=${m.opt("format")}, version=${m.opt("version")})")
        val pid = m.str("project_id")
        if (pid != projectId.value) throw PackageException("WRONG_PROJECT", "These results belong to project '$pid', not this project")
        val jobId = m.str("job_id") ?: throw PackageException("RESULTS_INVALID", "manifest.job_id is missing")
        if (jobId !in knownJobIds) throw PackageException("WRONG_JOB", "These results are for job '$jobId', which was not exported from this project")
        val status = m.str("status") ?: "unknown"
        val reportRaw = files["evaluation_report.json"] ?: throw PackageException("NO_EVALUATION_REPORT",
            "The results package contains no evaluation report (job status: $status). Nothing was evaluated, so there is nothing to import.")
        val r = J.parseOrNull(String(reportRaw, Charsets.UTF_8)) ?: throw PackageException("RESULTS_INVALID", "evaluation_report.json is not valid JSON")
        if (r.str("format") != REPORT_FORMAT || r.int("version") != 1) throw PackageException("RESULTS_INVALID", "evaluation_report.json has an unsupported format/version")
        if (r.bool("held_out_only") != true || r.str("split") != "test") throw PackageException("RESULTS_INVALID", "The evaluation report is not held-out-only (test split); refusing to display it")
        if (r.str("job_id") != null && r.str("job_id") != jobId) throw PackageException("RESULTS_INVALID", "evaluation_report.job_id does not match the manifest")
        val gen = r.obj("generated_by")
        val subj = gen?.obj("subjects")
        val spec = m.obj("specialist")
        val stubFlags = (gen?.bool("stub") == true) || (gen?.obj("evaluator")?.bool("is_stub") == true) || (spec?.bool("is_pipeline_validation_stub") == true) || status == "stub"
        val rows = r.objList("rows").map { o ->
            val base = o.dbl("base") ?: throw PackageException("RESULTS_INVALID", "metric row without a base value")
            val sp = o.dbl("specialist") ?: throw PackageException("RESULTS_INVALID", "metric row without a specialist value")
            val id = o.str("metric") ?: throw PackageException("RESULTS_INVALID", "metric row without a name")
            MetricRow(id, id.replace('_', ' ').replaceFirstChar { it.uppercase() }, o.bool("higher_is_better") ?: true, base, sp, o.dbl("delta") ?: (sp - base),
                o.dbl("ci_low"), o.dbl("ci_high"), o.int("n") ?: 0)
        }
        val caveats = ArrayList(r.strList("caveats"))
        val sizes = r.obj("sample_sizes")
        if (stubFlags) caveats.add(0, "STUB results: produced by a pipeline-validation stub, not a real evaluation. These numbers are not evidence of improvement.")
        when (status) {
            "planned_only" -> caveats.add("Job status is planned_only: a dry run. Nothing was trained or evaluated.")
            "partial" -> caveats.add("Job status is partial: some stages failed or were skipped. Review the stages before trusting these numbers.")
            "failed" -> caveats.add("Job status is failed.")
        }
        if (sizes != null) sizes.int("heldout_items")?.let { if (it < 50) caveats.add("Only $it held-out items; below the 50 needed for an improvement claim.") }
        var allowed = r.bool("improvement_claim_allowed") == true
        var reason = r.str("improvement_claim_reason") ?: "(no reason given in the report)"
        if (allowed && (stubFlags || status != "completed")) {
            allowed = false
            reason = "$reason [Studio: claim withheld because the package is internally inconsistent: status=$status stub=$stubFlags]"
            caveats.add("The report allowed an improvement claim although it is a stub or not completed; Studio withheld the claim.")
        }
        val base = subj?.str("base") ?: m.obj("base_model")?.str("registry_id") ?: "base model"
        val specialist = subj?.str("specialist") ?: "specialist"
        val view = EvaluationView(projectId, r.str("eval_id") ?: m.obj("evaluation")?.str("eval_id") ?: runIdFallback, base, specialist, rows, caveats, allowed, reason, stubFlags, nowMs)
        val refs = spec?.strList("artifact_refs").orEmpty()
        val hashes = spec?.obj("sha256s")?.let { h -> h.keyList().associateWith { h.getString(it) } } ?: emptyMap()
        return ParsedResults(view, jobId, status, reportRaw, m, spec?.str("kind") ?: "none", refs, hashes, spec?.str("method"))
    }

    // ---- specialist package ------------------------------------------------------------------------------------------------------

    class SpecialistContext(
        val projectId: String, val name: String, val version: String, val entry: RegEntry, val licenseState: LicenseState, val licenseEvidence: Map<String, Any?>?,
        val licenseText: ByteArray?, val datasetSha: String?, val datasetId: String?, val method: String, val parsed: ParsedResults, val nowMs: Long, val appVersion: String,
        val sources: List<SourceDoc>, val referenceChunks: List<DChunk>, val ds: DatasetState?,
    )

    fun buildSpecialist(c: SpecialistContext): BuiltPackage {
        val warnings = ArrayList<String>()
        val ev = c.parsed.view
        if (!ev.improvementClaimAllowed) warnings.add("The package records that the evidence does NOT support an improvement claim: ${ev.claimReason}")
        if (ev.isStub) warnings.add("The evaluation is a pipeline-validation stub; this package carries no quality claim.")
        if (c.parsed.specialistKind == "none") warnings.add("No parameter training happened (specialist kind 'none'): the package describes retrieval/prompt configuration only, with no adapter or merged weights.")
        val w = PackageWriter()
        val limitations = ArrayList<String>()
        limitations.add("Exact values (torque, electrical, clearances, standards language) must be grounded with the reference chunks, not model memory.")
        limitations.add("Improvement is only as strong as the attached evaluation; see claim_allowed and claim_reason.")
        if (ev.isStub) limitations.add("Evaluation is a stub.")
        w.addJson("specialist_manifest.json", linkedMapOf(
            "format" to SPECIALIST_FORMAT, "version" to 1, "name" to c.name, "specialist_version" to c.version, "project_id" to c.projectId, "created_at" to created(c.nowMs),
            "created_by" to mapOf("app" to "llmtrainer-studio", "app_version" to c.appVersion),
            "base_model" to linkedMapOf("registry_id" to c.entry.entryId, "family" to c.entry.family, "exact_version" to c.entry.version),
            "license" to linkedMapOf("state" to c.licenseState.name, "evidence" to c.licenseEvidence, "text_included" to (c.licenseText != null)),
            "dataset" to linkedMapOf("dataset_id" to c.datasetId, "dataset_sha256" to c.datasetSha?.let(Hashing::prefixed)),
            "method" to c.method,
            "training" to linkedMapOf("job_id" to c.parsed.jobId, "status" to c.parsed.status, "specialist_kind" to c.parsed.specialistKind, "artifact_refs" to c.parsed.artifactRefs,
                "artifact_sha256s" to c.parsed.artifactHashes, "note" to "Artifacts are produced on the desktop; this package references them by name and hash and does not contain them."),
            "evaluation" to linkedMapOf("eval_id" to ev.runId, "is_stub" to ev.isStub, "improvement_claim_allowed" to ev.improvementClaimAllowed, "claim_reason" to ev.claimReason,
                "summary" to evaluationSummary(ev), "report_file" to "evaluation_report.json"),
            "compatibility" to listOf("Any runtime that supports the base model above (check its supported formats); adapters apply to that exact base model version only."),
            "known_limitations" to limitations,
        ))
        w.add("evaluation_report.json", c.parsed.reportBytes)
        c.licenseEvidence?.let { w.addJson("license/license_evidence.json", it) }
        c.licenseText?.let { w.add("license/license_text.txt", it) }
        if (c.referenceChunks.isNotEmpty()) {
            w.addJsonl("reference/chunks.jsonl", c.referenceChunks.map { d ->
                linkedMapOf("ref" to d.ref, "source_id" to d.sourceId, "page" to d.chunk.page, "section_path" to d.chunk.sectionPath, "text" to d.chunk.text, "sha256" to Hashing.prefixed(d.chunk.sha256))
            })
        }
        w.addJson("provenance.json", linkedMapOf("sources" to c.sources.sortedBy { it.sourceId }.map { s ->
            linkedMapOf("id" to s.sourceId, "file_name" to s.name, "sha256" to Hashing.prefixed(s.sha256), "rights" to s.rights.name)
        }, "dataset_note" to TEMPLATE_NOTE))
        return BuiltPackage(w, "specialist", "${c.projectId}-${c.version}.specialist.zip", warnings)
    }

    fun evaluationSummary(ev: EvaluationView): String =
        (if (ev.isStub) "STUB evaluation. " else "") + ev.metrics.joinToString("; ") { "${it.label}: base ${"%.3f".format(it.base)} -> specialist ${"%.3f".format(it.specialist)} (n=${it.n})" } +
            ". Improvement claim ${if (ev.improvementClaimAllowed) "allowed" else "NOT allowed"}: ${ev.claimReason}"

    fun parseSpecialist(files: Map<String, ByteArray>): SpecialistPackageView {
        val checks = ArrayList<CheckResult>()
        val bad = Zips.verifyChecksums(files)
        checks.add(CheckResult("checksums", bad.isEmpty(), if (bad.isEmpty()) "every member matches checksums.json" else bad.joinToString("; ")))
        val raw = files["specialist_manifest.json"] ?: throw PackageException("PACKAGE_INVALID", "Not a specialist package (specialist_manifest.json is missing)")
        val m = J.parseOrNull(String(raw, Charsets.UTF_8)) ?: throw PackageException("PACKAGE_INVALID", "specialist_manifest.json is not valid JSON")
        if (m.str("format") != SPECIALIST_FORMAT || m.int("version") != 1) throw PackageException("PACKAGE_INVALID", "Unsupported specialist package format or version")
        val required = listOf("name", "specialist_version")
        val missing = required.filter { m.str(it).isNullOrBlank() } + (if (m.obj("base_model")?.str("registry_id") == null) listOf("base_model.registry_id") else emptyList()) +
            (if (m.obj("license")?.str("state") == null) listOf("license.state") else emptyList())
        checks.add(CheckResult("manifest complete", missing.isEmpty(), if (missing.isEmpty()) "ok" else "missing ${missing.joinToString()}"))
        val lic = m.obj("license")
        val state = runCatching { LicenseState.valueOf(lic?.str("state") ?: "") }.getOrDefault(LicenseState.UNVERIFIED)
        checks.add(CheckResult("license verified", state == LicenseState.VERIFIED, "base model license state recorded as $state"))
        val report = files["evaluation_report.json"]
        val reportOk = report != null && J.parseOrNull(String(report, Charsets.UTF_8))?.str("format") == REPORT_FORMAT
        checks.add(CheckResult("evaluation report present", reportOk, if (reportOk) "evaluation_report.json attached" else "no valid evaluation report"))
        val ev = m.obj("evaluation")
        val claim = ev?.bool("improvement_claim_allowed") == true
        val stub = ev?.bool("is_stub") == true
        checks.add(CheckResult("claim consistency", !(claim && stub), if (claim && stub) "claims improvement but the evaluation is a stub" else "ok"))
        val training = m.obj("training")
        val hashes = training?.obj("artifact_sha256s")?.let { h -> h.keyList().associateWith { h.getString(it) } } ?: emptyMap()
        val refs = training?.strList("artifact_refs").orEmpty()
        val artifactsOk = refs.all { it in hashes && Regex("^sha256:[0-9a-f]{64}$").matches(hashes.getValue(it)) }
        checks.add(CheckResult("artifact hashes", artifactsOk, if (refs.isEmpty()) "no artifacts referenced (no parameter training recorded)" else if (artifactsOk) "${refs.size} artifact reference(s) with sha256" else "an artifact reference has no valid sha256"))
        val ref = files["reference/chunks.jsonl"]
        if (ref != null) {
            val rows = String(ref, Charsets.UTF_8).lines().filter { it.isNotBlank() }
            var okRows = 0
            for (line in rows) {
                val o = J.parseOrNull(line) ?: continue
                if (o.str("sha256") == Hashing.prefixed(Hashing.sha256(o.str("text") ?: ""))) okRows++
            }
            checks.add(CheckResult("reference chunk hashes", okRows == rows.size, "$okRows of ${rows.size} reference chunks match their sha256"))
        }
        val dataset = m.obj("dataset")
        return SpecialistPackageView(
            m.str("name") ?: "", m.str("specialist_version") ?: "", m.obj("base_model")?.str("registry_id") ?: "", m.obj("base_model")?.str("exact_version") ?: "",
            state, "Base model license state recorded in the package: $state" + (lic?.obj("evidence")?.str("text_sha256")?.let { " (license text $it)" } ?: ""),
            dataset?.str("dataset_sha256"), hashes, m.str("method") ?: "unknown", ev?.str("summary"), claim && !stub,
            m.strList("compatibility"), m.strList("known_limitations"), checks, checks.all { it.passed })
    }
}
