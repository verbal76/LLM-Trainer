package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.qualify.CandidateAssessment
import com.hotatticgames.llmtrainer.qualify.CandidateConfig
import com.hotatticgames.llmtrainer.qualify.Catalog
import com.hotatticgames.llmtrainer.qualify.Confidence as QConfidence
import com.hotatticgames.llmtrainer.qualify.DeviceSnapshot
import com.hotatticgames.llmtrainer.qualify.Explain
import com.hotatticgames.llmtrainer.qualify.ModelSpec
import com.hotatticgames.llmtrainer.qualify.Qualifier
import com.hotatticgames.llmtrainer.qualify.ReferenceCatalog
import com.hotatticgames.llmtrainer.qualify.SafetyPolicy
import com.hotatticgames.llmtrainer.qualify.Tier
import com.hotatticgames.llmtrainer.studio.api.CatalogModel
import com.hotatticgames.llmtrainer.studio.api.Confidence
import com.hotatticgames.llmtrainer.studio.api.ConfidenceLevel
import com.hotatticgames.llmtrainer.studio.api.DeviceProfile
import com.hotatticgames.llmtrainer.studio.api.Feasibility
import com.hotatticgames.llmtrainer.studio.api.LicenseState
import com.hotatticgames.llmtrainer.studio.api.ModelVariant
import com.hotatticgames.llmtrainer.studio.api.ProfileKind
import com.hotatticgames.llmtrainer.studio.api.ProfileRecommendation
import com.hotatticgames.llmtrainer.studio.api.Provenance
import com.hotatticgames.llmtrainer.studio.api.Recommendations
import com.hotatticgames.llmtrainer.studio.api.RunLocation
import com.hotatticgames.llmtrainer.studio.api.TrainingFeasibility
import com.hotatticgames.llmtrainer.studio.api.Verdict
import java.io.File

/** A model variant after merging registry data with workspace overrides. Unknown sizes stay null here. */
data class VariantRec(
    val id: String, val modelId: String, val variantId: String, val format: String, val quantization: String?,
    val sourceUrl: String?, val sizeBytes: Long?, val sha256: String?, val sizeEvidence: String?,
    /** Set for variants that come from the CI-refreshed artifact registry (exact file, revision, hash). */
    val artifact: ArtifactRec? = null,
) {
    /** Why a registry artifact variant cannot be downloaded (null for non-artifact variants and downloadable artifacts). */
    val artifactBlock: String? get() = artifact?.notDownloadableReason
    /** Only a direct file URL can be downloaded; a repository page cannot. */
    val directUrl: String? get() = sourceUrl?.takeIf { u -> u.startsWith("https://") && DIRECT_EXT.any { u.substringBefore('?').lowercase().endsWith(it) } }
    companion object { val DIRECT_EXT = listOf(".gguf", ".safetensors", ".bin", ".pte", ".zip", ".onnx", ".task", ".tflite", ".litertlm") }
}

/** Per-artifact feasibility (inference and training are separate answers); installed by [ModelsService]. */
interface ArtifactAssessor { fun assess(e: RegEntry, a: ArtifactRec): Pair<Feasibility, TrainingFeasibility> }

class CatalogService(
    private val registry: List<RegEntry>,
    private val licenses: LicenseService,
    private val files: ModelFiles,
    private val overridesFile: File,
    private val snapshot: () -> String,
    private val policy: SafetyPolicy,
    private val clock: Clock,
    val artifacts: ArtifactCatalog = ArtifactCatalog(emptyList(), emptyMap()),
    private val onProblem: (String) -> Unit = {},
) {
    private val byId = registry.associateBy { it.entryId }
    val entries: Map<String, RegEntry> get() = byId

    /** Optional per-artifact assessor (installed by [ModelsService]) that replaces the generic per-model feasibility on artifact variants. */
    var artifactAssessor: ArtifactAssessor? = null

    // ---- variants ------------------------------------------------------------------------------------------------

    fun variantsOf(e: RegEntry): List<VariantRec> {
        val base = LinkedHashMap<String, VariantRec>()
        for (v in e.variants) base[v.variantId] = VariantRec("${e.entryId}#${v.variantId}", e.entryId, v.variantId, v.format, v.quantization, v.sourceUrl, v.sizeBytes, v.sha256, v.sizeEvidence)
        for (a in artifacts.forRepo(e.hfRepo)) {
            if (a.published == false) continue                       // optional file that the repo does not publish
            val ok = a.downloadable
            val ev = if (ok) "CI-refreshed from huggingface.co: revision ${a.revision?.take(12)}, refreshed ${a.refreshedAt ?: "?"}"
                     else "UNREFRESHED placeholder (${a.notDownloadableReason}): no size, hash or revision is known yet, so this file cannot be downloaded"
            base[a.id] = VariantRec("${e.entryId}#${a.id}", e.entryId, a.id, "GGUF", a.quantization, if (ok) a.downloadUrl else null,
                if (ok) a.sizeBytes else null, if (ok) a.bareSha256 else null, ev, a)
        }
        val o = Fs.readJson(overridesFile) { onProblem(it) }
        o?.obj("variants")?.arr(e.entryId)?.objs()?.forEach { v ->
            val vid = v.str("variant_id") ?: return@forEach
            val prev = base[vid]
            base[vid] = VariantRec("${e.entryId}#$vid", e.entryId, vid, v.str("format") ?: prev?.format ?: "unknown", v.str("quantization") ?: prev?.quantization,
                v.str("source_url") ?: prev?.sourceUrl, v.lng("size_bytes") ?: prev?.sizeBytes, (v.str("sha256") ?: prev?.sha256)?.let(Hashing::bare),
                v.str("size_evidence") ?: prev?.sizeEvidence ?: "workspace override")
        }
        return base.values.toList()
    }

    /**
     * Adds or replaces a downloadable variant for a registry model in the workspace (the shipped registry records repository pages,
     * not file URLs/sizes/hashes). Evidence is labelled "workspace override" unless [evidence] says otherwise.
     */
    @Synchronized fun addVariantOverride(modelId: String, variantId: String, format: String, quantization: String?, url: String, sizeBytes: Long?, sha256: String?, evidence: String = "workspace override"): String? {
        if (byId[modelId] == null) return "unknown model $modelId"
        if (!url.startsWith("https://")) return "only HTTPS URLs are accepted"
        if (variantId.isBlank() || variantId.contains('#')) return "variant id must be non-empty and must not contain '#'"
        if (sha256 != null && !Regex("^(sha256:)?[0-9a-fA-F]{64}$").matches(sha256)) return "sha256 must be 64 hex digits"
        val cur = Fs.readJson(overridesFile) { onProblem(it) }
        val all = LinkedHashMap<String, MutableList<Map<String, Any?>>>()
        cur?.obj("variants")?.let { vs -> for (k in vs.keyList()) all[k] = vs.arr(k)!!.objs().map { o -> o.keyList().associateWith { o.opt(it) } }.toMutableList() }
        val list = all.getOrPut(modelId) { ArrayList() }
        list.removeAll { it["variant_id"] == variantId }
        list.add(linkedMapOf("variant_id" to variantId, "format" to format, "quantization" to quantization, "source_url" to url, "size_bytes" to sizeBytes,
            "sha256" to sha256?.let(Hashing::bare), "size_evidence" to evidence))
        Fs.writeJson(overridesFile, linkedMapOf("schema" to 1, "variants" to all))
        return null
    }

    fun findVariant(variantId: String): Pair<RegEntry, VariantRec>? {
        val modelId = variantId.substringBefore('#', "")
        val e = byId[modelId] ?: return null
        val v = variantsOf(e).firstOrNull { it.id == variantId } ?: return null
        return e to v
    }

    /** Download/size estimate when the registry does not know the size (never presented as a fact). */
    fun estimatedSizeBytes(e: RegEntry, v: VariantRec): Long {
        val p = Registry.paramsB(e) ?: return 8L * 1024 * 1024 * 1024     // unknown: treat as multi-GB
        val bytesPerParam = if (v.format.lowercase().contains("gguf")) {
            val bits = ReferenceCatalog.load().quants.firstOrNull { q -> v.quantization?.contains(q.name, true) == true }?.bitsPerWeight ?: 4.85
            bits / 8
        } else 2.0
        return (p * 1e9 * bytesPerParam).toLong()
    }

    // ---- device ----------------------------------------------------------------------------------------------------

    fun deviceProfile(): DeviceProfile {
        val j = try { org.json.JSONObject(snapshot()) } catch (e: Exception) { org.json.JSONObject() }
        val abi = (j.str("abis64")?.takeIf { it.isNotBlank() } ?: j.str("abis") ?: "unknown").substringBefore(',')
        val mb = 1024.0 * 1024.0
        return DeviceProfile(
            deviceName = listOfNotNull(j.str("manufacturer"), j.str("model")).joinToString(" ").ifBlank { "Unknown device" },
            androidApi = j.int("sdkInt") ?: 0, abi = abi,
            totalRamMb = ((j.dbl("totalRamBytes") ?: 0.0) / mb).toInt(), availableRamMb = ((j.dbl("availRamBytes") ?: 0.0) / mb).toInt(),
            freeStorageMb = ((j.dbl("freeStorageBytes") ?: 0.0) / mb).toLong(), nativeRuntimeId = j.str("nativeRuntimeId") ?: "none-v1",
            safetyReserveFraction = policy.safetyReserveFrac, capturedAt = clock.nowMs(),
        )
    }

    // ---- model geometry for the qualifier ----------------------------------------------------------------------------

    private class Geo(val spec: ModelSpec, val estimated: Boolean, val note: String)

    private fun geo(e: RegEntry): Geo? {
        val p = Registry.paramsB(e) ?: return null
        val maxCtx = minOf(e.contextLength ?: 8192, 8192)
        if (e.layers != null && e.kvHeads != null && e.headDim != null)
            return Geo(ModelSpec(e.entryId, p, e.layers, e.kvHeads, e.headDim, null, maxCtx), false, "geometry from registry")
        val ref = ReferenceCatalog.load().models.minByOrNull { Math.abs(Math.log(it.paramsB / p)) }!!
        return Geo(ModelSpec(e.entryId, p, ref.layers, ref.kvHeads, ref.headDim, null, maxCtx), true,
            "layer/KV geometry not recorded; estimated from the nearest generic size class (${ref.modelId}), so KV-cache RAM is an ESTIMATE")
    }

    private fun candidatesFor(models: List<ModelSpec>): List<CandidateConfig> {
        val rc = ReferenceCatalog.load()
        return Qualifier.generateCandidates(models, rc.quants, rc.contexts, rc.runtimes, rc.retrieval)
    }

    private fun mbRound(x: Double) = Math.round(x).toInt()

    fun androidFeasibility(e: RegEntry): Feasibility {
        val g = geo(e) ?: return Feasibility(Verdict.UNKNOWN, null, listOf(
            "Parameter count is not recorded as a nominal figure (${e.parameterCount ?: "unknown"}); no honest estimate is possible."))
        val q = DeviceSnapshot.profileFromSnapshot(snapshot())
        if (q.profile == null || q.withheld.isNotEmpty())
            return Feasibility(Verdict.UNKNOWN, null, listOf(DeviceSnapshot.WITHHOLD_TEXT[q.withheld.firstOrNull()] ?: "Device facts are incomplete; qualification withheld."))
        val assessed = candidatesFor(listOf(g.spec)).map { Qualifier.assess(q.profile!!, it, policy) }
        val eligible = assessed.filter { it.eligible }
        val reasons = ArrayList<String>()
        reasons.add("ESTIMATE from the device snapshot and a conservative RAM model; this configuration has not been benchmarked on this device.")
        if (g.estimated) reasons.add(g.note)
        reasons.add("This app version has no on-device inference runtime (runtime \"none-v1\"); fitting here means the model could run once a runtime ships, not that it runs today.")
        if (e.androidState == "unverified") reasons.add("Registry has no on-device measurement or primary evidence for this model on Android.")
        if (eligible.isEmpty()) {
            val closest = assessed.minByOrNull { it.ram.requiredModelSideMb }!!
            reasons.add(0, "No quantization/context fits the safe envelope: " + (closest.blockingReasons.firstOrNull() ?: "RAM budget exceeded"))
            return Feasibility(Verdict.DOES_NOT_FIT, mbRound(closest.ram.requiredModelSideMb), reasons)
        }
        fun tps(a: CandidateAssessment) = a.sustained.tokensPerS ?: a.sustained.estTokensPerS
        val comfortable = eligible.filter { it.ram.utilization <= policy.balancedMaxUtilization && tps(it) >= policy.balancedMinTps && it.config.contextTokens >= policy.balancedMinContext }
        val best = (comfortable.ifEmpty { eligible }).maxWithOrNull(compareBy<CandidateAssessment>({ it.qualityScore }, { it.config.contextTokens }, { -it.ram.utilization }))!!
        reasons.addAll(0, Explain.pickLines(best).map { "Best fit: ${best.config.quant.name}, ${best.config.contextTokens}-token context. $it" }.take(1))
        return Feasibility(if (comfortable.isNotEmpty()) Verdict.FITS_SAFELY else Verdict.TIGHT, mbRound(best.ram.requiredModelSideMb), reasons)
    }

    fun trainingFeasibility(e: RegEntry): TrainingFeasibility {
        val p = Registry.paramsB(e)
        val base = "This app version cannot train on the phone; adapter training runs on a desktop/GPU via 'llmtrainer import-job'."
        if (p == null) return TrainingFeasibility(RunLocation.DESKTOP, null, listOf(base, "Parameter count is not a nominal figure (${e.parameterCount ?: "unknown"}); GPU memory is not estimated."))
        val params = p * 1e9
        val trainable = params * 16 * 3.6e-5
        val gib = (params * 0.55 + trainable * 14 + 1.5 * Math.pow(2.0, 30.0) * (p / 7)) / Math.pow(2.0, 30.0) * 1.2
        return TrainingFeasibility(RunLocation.DESKTOP, "qlora", listOf(base,
            "QLoRA needs roughly ${"%.1f".format(gib)} GiB of GPU memory at 2048-token context (order-of-magnitude estimate, not measured)."))
    }

    // ---- catalog views -----------------------------------------------------------------------------------------------

    fun variantView(e: RegEntry, v: VariantRec, android0: Feasibility = androidFeasibility(e), training0: TrainingFeasibility = trainingFeasibility(e)): ModelVariant {
        val p = Registry.paramsB(e)
        val known = v.sizeBytes != null
        val size = v.sizeBytes ?: estimatedSizeBytes(e, v)
        val art = v.artifact
        val assessed = if (art != null) artifactAssessor?.assess(e, art) else null
        val android = assessed?.first ?: android0
        val training = assessed?.second ?: training0
        val note = buildString {
            if (art != null && art.notDownloadableReason != null) append("This file's catalog data (revision, size, checksum) has not been refreshed from the model hub yet (${art.notDownloadableReason}); it cannot be downloaded until the catalog refresh has run. ")
            if (!known) append("Download size is not recorded; the figure shown is an ESTIMATE from the parameter count" + (if (p == null) " (unknown, so assumed multi-GB)" else "") + ". ")
            if (v.directUrl == null && art == null) append("No direct file URL is recorded (the source is a repository page); import the file or add a workspace override with a file URL. ")
            if (v.sha256 == null) append("No checksum recorded.")
        }.trim().ifEmpty { null }
        return ModelVariant(
            id = v.id, modelId = e.entryId, format = v.format, quant = v.quantization ?: "unspecified", sizeBytes = size,
            paramsBillions = art?.nominalParamsB?.takeIf { it > 0 } ?: p ?: 0.0,
            architecture = art?.archName?.let { "$it, ${art.layers ?: "?"} layers" } ?: e.archNotes?.takeIf { it.isNotBlank() }?.take(80) ?: "decoder-only transformer (details not recorded)",
            contextTokensMax = art?.ctxTrain ?: e.contextLength ?: 0, android = android, training = training,
            provenance = Provenance(v.sourceUrl ?: "", art?.refreshedAt?.take(10) ?: e.verification.verifiedOn, if (known) "registry" else "estimate", note ?: v.sizeEvidence),
            downloadUrl = v.directUrl, sha256 = v.sha256, acquired = files.isAcquired(v.id),
        )
    }

    fun modelView(e: RegEntry): CatalogModel {
        val android = androidFeasibility(e)
        val training = trainingFeasibility(e)
        val repo = e.hfRepo?.takeIf { Regex("^[\\w.-]+/[\\w.-]+$").matches(it) }?.let { "https://huggingface.co/$it" } ?: (e.hfRepo ?: "")
        return CatalogModel(
            id = e.entryId, family = e.family, name = e.entryId.substringAfter('@'), version = e.version,
            paramsBillions = Registry.paramsB(e) ?: 0.0, architecture = e.archNotes?.takeIf { it.isNotBlank() }?.take(80) ?: "decoder-only transformer (details not recorded)",
            license = licenses.effective(e), variants = variantsOf(e).map { variantView(e, it, android, training) },
            provenance = Provenance(repo, e.verification.verifiedOn, e.verification.evidenceLevel, e.verification.uncertainties.firstOrNull()),
        )
    }

    fun catalog(): List<CatalogModel> = registry.map { modelView(it) }

    // ---- recommendations -----------------------------------------------------------------------------------------

    fun recommendations(): Recommendations {
        val device = deviceProfile()
        val geos = registry.mapNotNull { e -> geo(e)?.takeIf { licenses.effective(e).state != LicenseState.DISALLOWED }?.let { e to it } }
        val cands = candidatesFor(geos.map { it.second.spec })
        val q = DeviceSnapshot.qualify(snapshot(), cands, policy)
        val kinds = mapOf("performance" to ProfileKind.PERFORMANCE, "balanced" to ProfileKind.BALANCED, "max_quality" to ProfileKind.MAX_QUALITY)
        val geoOf = geos.associate { it.first.entryId to it.second }
        val profiles = q.picks.map { pick ->
            val kind = kinds.getValue(pick.profile)
            val cfg = pick.config
            if (cfg == null) {
                return@map ProfileRecommendation(kind, null, null, null, null, null, listOf(pick.reason), pick.warnings + q.notes.map(::noteText),
                    Confidence(ConfidenceLevel.LOW, "no configuration qualified; nothing was measured", false), null)
            }
            val e = byId.getValue(cfg.model.modelId)
            val a = q.assessed.first { it.config.configId == cfg.configId }
            val lic = licenses.effective(e)
            val warnings = ArrayList(pick.warnings)
            if (geoOf.getValue(e.entryId).estimated) warnings.add(geoOf.getValue(e.entryId).note)
            if (lic.state != LicenseState.VERIFIED) warnings.add("License is ${lic.state}: you cannot train or distribute with this model until its license is verified.")
            warnings.add("This app version has no inference runtime; this is a capability estimate for a future runtime.")
            q.notes.forEach { warnings.add(noteText(it)) }
            val reasons = ArrayList<String>()
            reasons.add(pick.reason)
            reasons.addAll(Explain.pickLines(a))
            reasons.addAll(Explain.exclusionLines(q.assessed, pick, policy).map { "Larger model not chosen - $it" })
            val level = when (pick.confidence) { QConfidence.MEASURED -> ConfidenceLevel.HIGH; QConfidence.PARTIAL -> ConfidenceLevel.MEDIUM; else -> ConfidenceLevel.LOW }
            val benchmarked = pick.tier == Tier.RECOMMENDED
            ProfileRecommendation(kind, e.entryId, variantsOf(e).firstOrNull()?.id, cfg.quant.name, cfg.contextTokens, mbRound(a.ram.requiredModelSideMb), reasons, warnings,
                Confidence(level, if (benchmarked) "measured on this device" else "estimated from device facts and the qualifier's conservative model; not benchmarked", benchmarked), lic.state)
        }
        return Recommendations(device, profiles)
    }

    private fun noteText(n: String): String = when (n) {
        "low_ram_device" -> "Android reports a low-RAM device; reserves were increased."
        "low_memory" -> "The device is under memory pressure right now; available RAM is treated as zero."
        "avail_ram_unknown" -> "Available RAM is unknown; half of total RAM was assumed."
        "thermal_unknown" -> "Thermal state unknown."
        "thermal_moderate" -> "The device is warm; speed estimates were reduced."
        "power_save" -> "Power-save mode is on; speed estimates were reduced."
        "battery_low" -> "Battery is low."
        else -> n
    }
}
