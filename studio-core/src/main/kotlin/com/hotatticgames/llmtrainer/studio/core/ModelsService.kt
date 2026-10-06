package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.qualify.ArtifactCapabilities
import com.hotatticgames.llmtrainer.qualify.ArtifactSpec
import com.hotatticgames.llmtrainer.qualify.Capability
import com.hotatticgames.llmtrainer.qualify.CapabilityJson
import com.hotatticgames.llmtrainer.qualify.CapabilityPolicy
import com.hotatticgames.llmtrainer.qualify.DeviceSnapshot
import com.hotatticgames.llmtrainer.qualify.MeasurementRecord
import com.hotatticgames.llmtrainer.qualify.SafetyPolicy
import com.hotatticgames.llmtrainer.qualify.TrainingOption
import com.hotatticgames.llmtrainer.studio.api.ArtifactCapabilityView
import com.hotatticgames.llmtrainer.studio.api.CatalogStatus
import com.hotatticgames.llmtrainer.studio.api.ChoiceView
import com.hotatticgames.llmtrainer.studio.api.ConditionView
import com.hotatticgames.llmtrainer.studio.api.Feasibility
import com.hotatticgames.llmtrainer.studio.api.InstalledModel
import com.hotatticgames.llmtrainer.studio.api.LicenseState
import com.hotatticgames.llmtrainer.studio.api.ModelChoices
import com.hotatticgames.llmtrainer.studio.api.ModelsApi
import com.hotatticgames.llmtrainer.studio.api.Operation
import com.hotatticgames.llmtrainer.studio.api.RunLocation
import com.hotatticgames.llmtrainer.studio.api.SourceInput
import com.hotatticgames.llmtrainer.studio.api.StorageAccounting
import com.hotatticgames.llmtrainer.studio.api.Studio
import com.hotatticgames.llmtrainer.studio.api.StudioError
import com.hotatticgames.llmtrainer.studio.api.StudioResult
import com.hotatticgames.llmtrainer.studio.api.TrainingFeasibility
import com.hotatticgames.llmtrainer.studio.api.TrainingOptionView
import com.hotatticgames.llmtrainer.studio.api.Verdict
import java.io.File

/**
 * Model manager + device profiler facade over [CatalogService], [AcquisitionService], [LicenseService] and the qualify
 * module's capability classifier. Everything it reports about a device is an ESTIMATE (confidence "low") until a measurement
 * record from this very device replaces it (see docs/studio/DEVICE_PROFILER.md).
 */
class ModelsService(
    private val wsDir: File, private val catalog: CatalogService, private val licenses: LicenseService, private val files: ModelFiles,
    private val acquisition: AcquisitionService, private val snapshot: () -> String, private val policy: SafetyPolicy,
    private val onProblem: (String) -> Unit = {},
) : ModelsApi, ArtifactAssessor {
    private val ledgerFile = File(wsDir, "measurements.json")
    private val cap = CapabilityPolicy()
    private val lock = Any()

    init { catalog.artifactAssessor = this }

    // ---- reason texts ------------------------------------------------------------------------------------------------

    companion object {
        val REASON_TEXT: Map<String, String> = mapOf(
            "geometry_estimated" to "Layer and KV-cache geometry are not known yet (catalog not refreshed); estimated from the nearest generic size class",
            "size_estimated" to "The exact file size is not known yet; estimated from the parameter count",
            "params_nominal" to "The exact parameter count is not known yet; the nominal figure was used",
            "runtime_support_unverified" to "Support for this architecture in the on-device runtime has not been verified",
            "runtime_unsupported" to "The on-device runtime does not support this model architecture",
            "not_downloadable" to "This file cannot be downloaded yet (its catalog data has not been refreshed or verified)",
            "storage_insufficient" to "Not enough free storage after the safety reserve",
            "ram_insufficient_to_load" to "Not enough memory headroom to load this model without degrading the phone",
            "infer_too_slow_or_unsafe" to "It would load, but not run fast or safely enough to be useful (estimated tokens/s or memory headroom)",
            "training_needs_full_precision_artifact" to "Quantized weights cannot be trained on the phone; specialization starts from a full-precision file of the same model",
            "training_class_external_only" to "Too large to train on a phone; adapter training needs a desktop/GPU",
            "training_ram_insufficient" to "Training needs far more memory than running the model (F32 weights + optimizer state + activations); not enough on this phone",
            "training_very_slow" to "Training would take impractically long on this phone",
        )
        fun reasonText(code: String) = REASON_TEXT[code] ?: code
        private val CONDITION_TEXT = mapOf(
            "thermal" to "Device must be cool (thermal status none or light)",
            "charging" to "Device must be charging",
            "battery" to "Battery must be at least 30% or charging",
            "not_power_save" to "Battery saver must be off",
            "storage_for_checkpoints" to "Enough free storage for checkpoints",
        )
    }

    // ---- device ------------------------------------------------------------------------------------------------------

    private fun device() = DeviceSnapshot.profileFromSnapshot(snapshot())
    private fun state() = DeviceSnapshot.stateFromSnapshot(snapshot())

    // ---- specs -------------------------------------------------------------------------------------------------------

    private class Item(val e: RegEntry, val v: VariantRec, val a: ArtifactRec, val spec: ArtifactSpec)

    private fun items(): List<Item> {
        val out = ArrayList<Item>()
        for (e in catalog.entries.values.sortedBy { it.entryId }) {
            val lic = licenses.effective(e).state
            for (v in catalog.variantsOf(e)) {
                val a = v.artifact ?: continue
                if (a.published == false) continue
                out.add(Item(e, v, a, a.toSpec(lic, e.entryId)))
            }
        }
        return out
    }

    // ---- measurement ledger ------------------------------------------------------------------------------------------

    private fun loadLedger(): List<MeasurementRecord> {
        val o = Fs.readJson(ledgerFile) { onProblem(it) } ?: return emptyList()
        return o.objList("records").mapNotNull { r ->
            try { CapabilityJson.parseRecord(r.toString()) } catch (e: Exception) { onProblem("a stored measurement record is unreadable and was ignored"); null }
        }
    }

    override fun measurementRecords(): List<String> = synchronized(lock) { loadLedger().map { J.dump(CapabilityJson.recordToMap(it)) } }

    override fun recordMeasurement(recordJson: String): StudioResult<Int> {
        val rec = try { CapabilityJson.parseRecord(recordJson) } catch (e: Exception) {
            return StudioResult.Err(StudioError.Invalid("BAD_RECORD", "Not a valid device_measurement.v1 record: ${e.message}"))
        }
        val dev = device().profile ?: return StudioResult.Err(StudioError.Invalid("DEVICE_UNKNOWN", "The device profile is incomplete, so a measurement cannot be attributed to this device"))
        if (rec.deviceId != dev.deviceId) return StudioResult.Err(StudioError.Invalid("OTHER_DEVICE", "The record is for another device (${rec.deviceId}); this device is ${dev.deviceId}"))
        if (rec.kind !in setOf("load", "inference", "training")) return StudioResult.Err(StudioError.Invalid("BAD_KIND", "kind must be load, inference or training"))
        val knownSha: String? = items().firstOrNull { it.a.id == rec.artifactId }?.a?.sha256 ?: files.record(rec.artifactId)?.sha256?.let(Hashing::prefixed)
        val known = items().any { it.a.id == rec.artifactId } || files.record(rec.artifactId) != null
        if (!known) return StudioResult.Err(StudioError.NotFound("artifact ${rec.artifactId}"))
        val recSha = rec.artifactSha256
        if (recSha != null && knownSha != null && Hashing.bare(recSha) != Hashing.bare(knownSha))
            return StudioResult.Err(StudioError.Invalid("ARTIFACT_HASH_MISMATCH", "The record was measured on a different file than the catalog's / installed one"))
        return synchronized(lock) {
            val merged = Capability.mergeRecords(loadLedger(), listOf(rec))
            Fs.writeJson(ledgerFile, linkedMapOf("schema" to 1, "records" to merged.map { CapabilityJson.recordToMap(it) }))
            StudioResult.Ok(merged.size)
        }
    }

    // ---- capabilities ------------------------------------------------------------------------------------------------

    private fun viewOf(it: Item, c: ArtifactCapabilities, byArtifact: Map<String, Item>): ArtifactCapabilityView {
        fun opt(o: TrainingOption?) = o?.let { TrainingOptionView(it.mode, it.trainableLastLayers, Math.round(it.ramMb).toInt(), it.utilization, it.estTokensPerS,
            it.estMinutesForReferenceTokens, Math.round(it.checkpointMb).toInt(), it.fits, it.measured) }
        val a = it.a
        val via = c.specializeViaArtifactId?.let { id -> byArtifact[id]?.v?.id }
        return ArtifactCapabilityView(
            variantId = it.v.id, artifactId = a.id, modelId = it.e.entryId, label = "${it.e.family} ${it.e.version.substringBefore(" (")} ${a.quantization}",
            quantization = a.quantization, precision = a.precision, sizeBytes = a.sizeBytes ?: catalog.estimatedSizeBytes(it.e, it.v), sizeIsEstimate = a.sizeBytes == null,
            catalogState = when { a.published == false -> "not_published"; a.refreshState == "refreshed" -> "refreshed"; else -> "unrefreshed" },
            downloadable = a.downloadable, downloadBlocker = a.notDownloadableReason, licenseState = licenses.effective(it.e).state,
            canDownload = c.canDownload, canLoad = c.canLoad, canInfer = c.canInfer, canEvaluate = c.canEvaluate,
            canSpecializeFull = c.canSpecializeFull, canSpecializePartial = c.canSpecializePartial, trainableLastLayers = c.trainableLastLayers,
            externalComputeRequired = c.externalComputeRequired, inferContext = c.inferContext, estTokensPerS = c.estTokensPerS,
            estPeakRamMb = c.inferRamMb?.let { r -> Math.round(r).toInt() }, trainingClass = a.trainingClass, specializeViaVariantId = via,
            full = opt(c.full), partial = opt(c.partial),
            conditions = c.trainingConditions.map { k -> ConditionView(k.name, k.status, CONDITION_TEXT[k.name] ?: k.name) },
            readyToTrainNow = c.readyToTrainNow, confidenceInfer = c.confidenceInfer, confidenceTrain = c.confidenceTrain,
            reasons = c.reasons.map(::reasonText), installed = files.isAcquired(it.v.id),
        )
    }

    override fun catalogStatus(): CatalogStatus {
        val all = catalog.artifacts.artifacts.filter { it.published != false }
        val refreshed = all.count { it.refreshState == "refreshed" }
        val dl = all.count { it.downloadable }
        val last = all.mapNotNull { it.refreshedAt }.maxOrNull()
        val note = when {
            all.isEmpty() -> "No downloadable model files are catalogued."
            dl == 0 -> "The model catalog has not been refreshed from the model hub yet, so no file can be downloaded; you can still import a GGUF file manually."
            dl < all.size -> "${all.size - dl} catalogued file(s) are not downloadable yet (unrefreshed or unpublished)."
            else -> "All catalogued files have verified download details."
        }
        return CatalogStatus(all.size, refreshed, dl, all.size - refreshed, last, note)
    }

    override fun modelChoices(): ModelChoices {
        val sp = device()
        val dev = catalog.deviceProfile()
        val items = items()
        val byArtifact = items.associateBy { it.a.id }
        val status = catalogStatus()
        val profile = sp.profile
        if (profile == null) {
            val txt = DeviceSnapshot.WITHHOLD_TEXT[sp.withheld.firstOrNull()] ?: "Device facts are incomplete."
            return ModelChoices(dev, listOf("fastest", "balanced", "best_quality", "best_specialize").map { ChoiceView(it, Capability.CHOICE_LABELS.getValue(it), null, null, "none", null, null, null, null, false, txt) }, emptyList(), status)
        }
        val records = synchronized(lock) { loadLedger() }
        val rep = Capability.choose(profile, items.map { it.spec }, policy, cap, state(), records)
        val views = rep.capabilities.mapIndexed { i, c -> viewOf(items[i], c, byArtifact) }
        val hot = sp.withheld.contains("thermal_throttled")
        val choices = rep.choices.map { p ->
            if (hot) ChoiceView(p.choice, p.label, null, null, "none", null, null, null, null, false, DeviceSnapshot.WITHHOLD_TEXT.getValue("thermal_throttled"))
            else ChoiceView(p.choice, p.label, p.artifactId?.let { id -> byArtifact[id]?.v?.id }, p.artifactId, p.tier, p.confidence, p.contextTokens, p.trainableLastLayers,
                p.licenseState?.let { LicenseState.valueOf(it) }, p.downloadable, reasonText(p.reason))
        }
        return ModelChoices(dev, choices, views, status)
    }

    override fun capabilities(variantId: String): StudioResult<ArtifactCapabilityView> {
        val items = items()
        val it = items.firstOrNull { x -> x.v.id == variantId } ?: return StudioResult.Err(StudioError.NotFound("variant $variantId"))
        val profile = device().profile ?: return StudioResult.Err(StudioError.Invalid("DEVICE_UNKNOWN", "The device profile is incomplete"))
        val records = synchronized(lock) { loadLedger() }
        val c = Capability.classify(profile, it.spec, policy, cap, state(), records)
        return StudioResult.Ok(viewOf(it, c, items.associateBy { x -> x.a.id }))
    }

    private fun reasonsForInfer(c: ArtifactCapabilities) = c.reasons.map(::reasonText)

    override fun assess(e: RegEntry, a: ArtifactRec): Pair<Feasibility, TrainingFeasibility> {
        val sp = device()
        val profile = sp.profile
        if (profile == null || sp.withheld.isNotEmpty())
            return Pair(Feasibility(Verdict.UNKNOWN, null, listOf(DeviceSnapshot.WITHHOLD_TEXT[sp.withheld.firstOrNull()] ?: "Device facts are incomplete; qualification withheld.")),
                TrainingFeasibility(RunLocation.NONE, null, listOf("Device facts are incomplete; training feasibility is not assessed.")))
        val records = synchronized(lock) { loadLedger() }
        val c = Capability.classify(profile, a.toSpec(licenses.effective(e).state, e.entryId), policy, cap, state(), records)
        val estimate = if (c.confidenceInfer == "low") "ESTIMATE from the device snapshot and a conservative RAM model; this file has not been benchmarked on this device." else "Based on ${c.confidenceInfer}-confidence on-device measurements."
        val verdict = when {
            !c.canInfer -> Verdict.DOES_NOT_FIT
            (c.inferUtilization ?: 1.0) <= policy.balancedMaxUtilization -> Verdict.FITS_SAFELY
            else -> Verdict.TIGHT
        }
        val reasons = ArrayList<String>()
        reasons.add(estimate)
        if (c.canInfer) reasons.add("Runs at about ${"%.1f".format(c.estTokensPerS ?: 0.0)} tokens/s with a ${c.inferContext}-token context.")
        reasons.addAll(reasonsForInfer(c).filter { r -> r != reasonText("training_needs_full_precision_artifact") && !r.startsWith("Training") && !r.startsWith("Too large to train") })
        val infer = Feasibility(verdict, c.inferRamMb?.let { Math.round(it).toInt() }, reasons)
        val tReasons = ArrayList<String>()
        val where: RunLocation
        val method: String?
        // "where" is where it would run in THIS app version: the on-device trainer ships with the native runtime.
        val onDeviceTrainer = !catalog.deviceProfile().nativeRuntimeId.startsWith("none")
        when {
            c.canSpecializeFull -> { where = if (onDeviceTrainer) RunLocation.DEVICE else RunLocation.DESKTOP; method = "full"; tReasons.add("Full tuning would fit in this phone's memory envelope (F32 trainer, estimate).") }
            c.canSpecializePartial -> { where = if (onDeviceTrainer) RunLocation.DEVICE else RunLocation.DESKTOP; method = "partial (last ${c.trainableLastLayers} layers)"; tReasons.add("Tuning the last ${c.trainableLastLayers} layers would fit in this phone's memory envelope (F32 trainer, estimate).") }
            else -> { where = RunLocation.DESKTOP; method = "qlora"; tReasons.add("Adapter training runs on a desktop/GPU via 'llmtrainer import-job'.") }
        }
        tReasons.add("Training needs F32 weights + 12 bytes per trainable parameter + activations; inference feasibility does not imply training feasibility.")
        tReasons.addAll(c.reasons.filter { r -> r.startsWith("training_") }.map(::reasonText))
        if (c.specializeViaArtifactId != null) tReasons.add("Local specialization would start from the full-precision file ${c.specializeViaArtifactId}.")
        if (!onDeviceTrainer && (c.canSpecializeFull || c.canSpecializePartial)) tReasons.add("The on-device trainer ships with the native runtime; this app version cannot train on the phone yet.")
        return Pair(infer, TrainingFeasibility(where, method, tReasons))
    }

    // ---- installed models --------------------------------------------------------------------------------------------

    private fun toApi(r: InstallRec): InstalledModel {
        val art = r.artifactId?.let { catalog.artifacts.get(it) }
        val name = art?.let { "${it.family} ${it.quantization} (${r.fileName})" } ?: (r.originalName ?: r.fileName)
        return InstalledModel(r.variantId, r.artifactId, name, r.file.absolutePath, r.size, r.sha256, r.source, r.provenance, r.url, r.revision,
            try { LicenseState.valueOf(r.licenseState) } catch (e: Exception) { LicenseState.UNVERIFIED }, r.installedAt, r.architecture, r.layers, r.parameterCount, r.chatTemplatePresent)
    }

    override fun installedModels(): List<InstalledModel> = files.allInstalled().map(::toApi).sortedBy { it.installedAt }
    override fun installedPath(variantId: String): String? = files.installedFile(variantId)?.absolutePath

    override fun storage(): StorageAccounting = acquisition.storageNumbers().let { StorageAccounting(it.freeBytes, it.reserveBytes, it.installedBytes, it.partialBytes, it.headroomBytes) }

    override fun uninstall(variantId: String): StudioResult<StorageAccounting> =
        when (val r = acquisition.uninstall(variantId)) { is StudioResult.Ok -> StudioResult.Ok(storage()); is StudioResult.Err -> r }

    override fun importUserModel(file: SourceInput): StudioResult<Operation> = acquisition.importUserModel(file)
}

/** Access to the model manager from a [Studio] obtained through [StudioFactory]. */
object ModelsFactory {
    fun of(studio: Studio): ModelsApi? = (studio as? StudioCore)?.models
}
