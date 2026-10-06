package com.hotatticgames.llmtrainer.studio.api

/**
 * Scripted [ModelsApi] over a [FakeStudio] for UI development and instrumented tests (`FakeStudio.modelsApi`). It derives everything from the
 * fake's own catalog, installed files and v2 knobs; nothing is measured and every estimate says so (confidence "low").
 * The sample catalog has only quantized files, so there is deliberately no "best to specialize" pick and no file is ready for weight tuning.
 */
class FakeModelsApi(private val s: FakeStudio) : ModelsApi {
    private val measurements = ArrayList<String>()
    private val mib = 1024L * 1024L

    private fun device() = s.deviceProfile()
    private fun installedIds() = s.installedModels().map { it.variantId }.toSet()

    private fun conditions(): List<ConditionView> {
        val c = s.v2
        fun tri(name: String, v: Boolean?, ok: String, bad: String) = ConditionView(name, when (v) { true -> "met"; false -> "unmet"; null -> "unknown" }, if (v == true) ok else if (v == false) bad else "$name is not reported by this device")
        return listOf(
            tri("charging", c.charging, "Charger connected", "Plug in the charger"),
            tri("battery", c.batteryPercent?.let { it >= 30 || c.charging == true }, "Battery is sufficient", "Battery below 30%"),
            tri("thermal", c.thermal?.let { it == "NONE" || it == "LIGHT" }, "Device is cool", "Device is warm (${c.thermal})"),
        )
    }

    private fun view(m: CatalogModel, v: ModelVariant, installed: Boolean): ArtifactCapabilityView {
        val disallowed = m.license.state == LicenseState.DISALLOWED
        val fits = v.android.verdict != Verdict.DOES_NOT_FIT
        return ArtifactCapabilityView(
            variantId = v.id, artifactId = v.id, modelId = m.id, label = "${m.name} ${v.quant}", quantization = v.quant.uppercase(), precision = "quantized",
            sizeBytes = v.sizeBytes, sizeIsEstimate = false, catalogState = "refreshed", downloadable = !disallowed && v.downloadUrl != null,
            downloadBlocker = if (disallowed) "License disallows this use" else null, licenseState = m.license.state,
            canDownload = !disallowed && v.downloadUrl != null && !installed, canLoad = fits, canInfer = fits, canEvaluate = fits,
            canSpecializeFull = false, canSpecializePartial = false, trainableLastLayers = null, externalComputeRequired = true,
            inferContext = minOf(4096, v.contextTokensMax), estTokensPerS = 6.5, estPeakRamMb = v.android.estimatedPeakRamMb,
            trainingClass = "inference_only", specializeViaVariantId = null, full = null, partial = null,
            conditions = conditions(), readyToTrainNow = false, confidenceInfer = "low", confidenceTrain = "low",
            reasons = v.android.reasons + "Estimate only (scripted sample): not measured on this phone", installed = installed,
        )
    }

    private fun artifacts(): List<ArtifactCapabilityView> {
        val inst = installedIds()
        return s.catalog().flatMap { m -> m.variants.map { v -> view(m, v, v.id in inst) } }
    }

    override fun modelChoices(): ModelChoices {
        val all = artifacts()
        val usable = all.filter { it.canInfer && it.licenseState != LicenseState.DISALLOWED }
        fun choice(id: String, label: String, pick: ArtifactCapabilityView?, why: String) = ChoiceView(
            id, label, pick?.variantId, pick?.artifactId, if (pick == null) "none" else "provisional", if (pick == null) null else "low",
            pick?.inferContext, null, pick?.licenseState, pick?.downloadable ?: false, why,
        )
        val fastest = usable.minByOrNull { it.sizeBytes ?: Long.MAX_VALUE }
        val balanced = usable.filter { (it.estPeakRamMb ?: Int.MAX_VALUE) <= device().availableRamMb * 0.6 }.maxByOrNull { it.sizeBytes ?: 0L }
        val best = usable.maxByOrNull { it.sizeBytes ?: 0L }
        return ModelChoices(device(), listOf(
            choice("fastest", "Fastest", fastest, "Smallest file that runs on this phone (estimate)"),
            choice("balanced", "Balanced", balanced, "Largest file with comfortable RAM headroom (estimate)"),
            choice("best_quality", "Best quality", best, "Largest file that still fits the safe envelope (estimate)"),
            choice("best_specialize", "Best to specialize on this phone", null, "The sample catalog has no full-precision file to tune; LoRA on a quantized file is checked at plan time"),
        ), all, catalogStatus())
    }

    override fun capabilities(variantId: String): StudioResult<ArtifactCapabilityView> =
        artifacts().firstOrNull { it.variantId == variantId }?.let { StudioResult.Ok(it) } ?: StudioResult.Err(StudioError.NotFound("variant $variantId"))

    override fun catalogStatus(): CatalogStatus {
        val n = s.catalog().sumOf { it.variants.size }
        return CatalogStatus(n, n, n, 0, "2026-09-01", "Scripted sample catalog: every file is shown as downloadable; nothing is real.")
    }

    override fun installedModels(): List<InstalledModel> = s.installedModels().map { m ->
        InstalledModel(m.variantId, m.variantId, "${m.name} ${m.quant}", "/sample/${m.variantId}.gguf", m.sizeBytes, "0".repeat(64), "download", "verified_download",
            null, null, m.licenseState, 0L, null, null, null, null)
    }

    override fun installedPath(variantId: String): String? = if (variantId in installedIds()) "/sample/$variantId.gguf" else null

    override fun uninstall(variantId: String): StudioResult<StorageAccounting> =
        StudioResult.Err(StudioError.Invalid("NOT_SUPPORTED", "The scripted sample cannot remove files"))

    override fun storage(): StorageAccounting {
        val free = s.v2.freeStorageMb * mib
        val reserve = 1024L * mib
        return StorageAccounting(free, reserve, s.installedModels().sumOf { it.sizeBytes }, 0L, free - reserve)
    }

    override fun importUserModel(file: SourceInput): StudioResult<Operation> {
        val inst = installedIds()
        val target = s.catalog().flatMap { it.variants }.firstOrNull { it.id !in inst }
            ?: return StudioResult.Err(StudioError.Conflict("The sample catalog has no file left to import into"))
        return s.importModelFile(target.id, file)
    }

    override fun recordMeasurement(recordJson: String): StudioResult<Int> { measurements.add(recordJson); return StudioResult.Ok(measurements.size) }
    override fun measurementRecords(): List<String> = measurements.toList()
}
