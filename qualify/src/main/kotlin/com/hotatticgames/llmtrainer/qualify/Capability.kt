package com.hotatticgames.llmtrainer.qualify

/**
 * Kotlin port of "Device profiler v2" in factory/llmtrainer/device.py (CAPABILITY_ALGORITHM_VERSION 2): per-ARTIFACT
 * capabilities (download / load / infer / evaluate / specialize full|partial / external compute), the F32 training memory
 * model, owner-facing choices and measurement-record merge. The Python module is the executable spec; golden vectors
 * (factory/tests/golden/device_capability.v1.json) pin both. Operation order is identical to Python on purpose.
 * Reasons are stable codes (no formatted numbers).
 */

data class CapabilityPolicy(
    val inferContexts: List<Int> = listOf(512, 1024, 2048, 4096, 8192),
    val loadContext: Int = 512,
    val evalMinContext: Int = 1024,
    val evalMinTps: Double = 5.0,
    val trainCtxTokens: Int = 256,
    val trainOverheadMb: Double = 300.0,
    val trainMaxUtilization: Double = 0.85,
    val minPartialLayers: Int = 2,
    val trainDatasetStorageMb: Double = 256.0,
    val trainRefTokens: Int = 200000,
    val maxPracticalTrainHours: Double = 48.0,
    val trainMinBatteryPct: Double = 30.0,
    val trainMaxThermalStatus: Int = 1,
)

/** Transient device condition; null = unknown (an unknown fact never counts as met). */
data class DeviceState(val thermalStatus: Int? = null, val batteryPct: Double? = null, val charging: Boolean? = null, val powerSave: Boolean? = null)

data class ArtifactSpec(
    val artifactId: String, val modelId: String, val quantization: String, val precision: String, val sha256: String?,
    val sizeBytes: Long?, val params: Double?, val nominalParamsB: Double, val layers: Int?, val kvHeads: Int?, val headDim: Int?,
    val embd: Int?, val ffn: Int?, val vocab: Int?, val ctxTrain: Int?, val trainingClass: String, val trainingSourceArtifactId: String?,
    val downloadable: Boolean, val licenseState: String, val runtimeSupported: String,
)

data class MeasurementRecord(
    val recordId: String, val artifactId: String, val artifactSha256: String?, val deviceId: String, val runtimeId: String?,
    val appVersion: String?, val recordedAt: String, val kind: String,
    val loadTimeMs: Double? = null, val loadPeakRamMb: Double? = null,
    val contextTokens: Int? = null, val ttftMs: Double? = null, val tokensPerS: Double? = null, val peakRamMb: Double? = null,
    val sustainedRamMb: Double? = null, val thermalThrottleRatio: Double? = null, val uiJankPct: Double? = null, val crashes: Long? = null,
    val anrs: Long? = null, val backgroundKills: Long? = null, val sustainedMinutes: Double? = null,
    val trainableLastLayers: Int? = null, val trainPeakRamMb: Double? = null, val trainTokensPerS: Double? = null,
    val checkpointSizeMb: Double? = null, val trainMinutes: Double? = null, val trainCompleted: Boolean? = null, val trainOom: Boolean? = null,
    val thermalMaxStatus: Int? = null, val batteryDropPct: Double? = null,
)

data class Condition(val name: String, val status: String)

data class TrainingOption(
    val mode: String, val trainableLastLayers: Int, val trainableParamsM: Double, val ramMb: Double, val utilization: Double,
    val estTokensPerS: Double, val estMinutesForReferenceTokens: Double, val checkpointMb: Double, val storageRequiredMb: Double,
    val fits: Boolean, val measured: Boolean,
)

data class ArtifactCapabilities(
    val artifactId: String, val canDownload: Boolean, val canLoad: Boolean, val canInfer: Boolean, val canEvaluate: Boolean,
    val canSpecializeFull: Boolean, val canSpecializePartial: Boolean, val externalComputeRequired: Boolean,
    val inferContext: Int?, val estTokensPerS: Double?, val inferRamMb: Double?, val inferUtilization: Double?, val loadRamMb: Double?,
    val qualityScore: Double, val trainableLastLayers: Int?, val full: TrainingOption?, val partial: TrainingOption?,
    val specializeViaArtifactId: String?, val trainingConditions: List<Condition>, val readyToTrainNow: Boolean,
    val confidenceInfer: String, val confidenceTrain: String, val reasons: List<String>, val ignoredMeasurements: List<String>,
)

data class ChoicePick(
    val choice: String, val label: String, val artifactId: String?, val tier: String, val confidence: String? = null,
    val contextTokens: Int? = null, val trainableLastLayers: Int? = null, val licenseState: String? = null,
    val downloadable: Boolean = false, val reason: String,
)

data class CapabilityReport(val deviceId: String, val capabilities: List<ArtifactCapabilities>, val choices: List<ChoicePick>)

object Capability {
    const val ALGORITHM_VERSION = "2"
    private const val MIB = 1048576.0

    private val QUANT_TABLE: Map<String, Pair<Double, Double>> = mapOf(
        "F32" to Pair(32.0, 0.0), "F16" to Pair(16.0, 0.0), "BF16" to Pair(16.0, 0.0), "Q8_0" to Pair(8.5, 0.3), "Q6_K" to Pair(6.6, 0.6),
        "Q5_K_M" to Pair(5.7, 1.0), "Q4_K_M" to Pair(4.85, 1.8), "Q3_K_M" to Pair(3.9, 4.0),
    )
    private val UNKNOWN_QUANT = Pair(8.5, 1.0)
    private val REF: Catalog by lazy { ReferenceCatalog.load() }
    val CHOICE_LABELS: Map<String, String> = linkedMapOf(
        "fastest" to "Fastest / lowest resource use",
        "balanced" to "Balanced (Recommended)",
        "best_quality" to "Best quality this device can reasonably handle",
        "best_specialize" to "Best model you can specialize on this phone",
    )

    // ----------------------------- artifact -> v1 types ---------------------------------------------------------

    fun specParams(s: ArtifactSpec): Double = s.params ?: (s.nominalParamsB * 1e9)

    private fun quantRow(s: ArtifactSpec): Pair<Double, Double> = QUANT_TABLE[s.quantization.uppercase()] ?: UNKNOWN_QUANT

    fun specFileMb(s: ArtifactSpec): Double {
        val sb = s.sizeBytes
        if (sb != null) return sb.toDouble() / MIB
        return specParams(s) * quantRow(s).first / 8 / MIB
    }

    private class Geo(val layers: Int, val kvHeads: Int, val headDim: Int, val estimated: Boolean)

    private fun geometry(s: ArtifactSpec): Geo {
        val l = s.layers
        val k = s.kvHeads
        val h = s.headDim
        if (l != null && k != null && h != null) return Geo(l, k, h, false)
        val models = REF.models
        val ref = models.firstOrNull { it.paramsB >= s.nominalParamsB } ?: models.last()
        return Geo(ref.layers, ref.kvHeads, ref.headDim, true)
    }

    fun specCandidate(s: ArtifactSpec, ctx: Int): CandidateConfig {
        val g = geometry(s)
        val (bits, pen) = quantRow(s)
        val maxCtx = minOf(s.ctxTrain ?: 8192, 8192)
        val model = ModelSpec(s.artifactId, specParams(s) / 1e9, g.layers, g.kvHeads, g.headDim, null, maxCtx)
        val quant = QuantSpec(s.quantization, bits, pen)
        val rt = REF.runtimes[0]
        val ret = RetrievalBudget()
        return CandidateConfig(Qualifier.makeConfigId(model, quant, ctx, rt, ret), model, quant, ctx, rt, ret, specFileMb(s))
    }

    // ----------------------------- training model ---------------------------------------------------------------

    fun trainGflopsPrior(d: DeviceProfile): Double {
        val t = d.totalRamMb
        return when {
            t >= 12000 -> 40.0
            t >= 8000 -> 25.0
            t >= 6000 -> 15.0
            else -> 8.0
        }
    }

    /** (model-side training RAM in MB, trainable params). [k] = trailing layers tuned; null = full tuning. */
    fun trainingRamMb(s: ArtifactSpec, k: Int?, cap: CapabilityPolicy, safety: SafetyPolicy): Pair<Double, Double> {
        val layers = geometry(s).layers
        val p = specParams(s)
        val perLayer = p / layers
        val trainable = if (k == null) p else perLayer * k
        val tLayers = k ?: layers
        val weights = p * 4 / MIB
        val state = trainable * 12 / MIB
        val embd = s.embd
        val act: Double
        if (embd != null) {
            val ffn = s.ffn ?: (4 * embd)
            val vocab = s.vocab ?: 0
            val t = cap.trainCtxTokens.toLong()
            val num = tLayers.toLong() * 4L * t * (16L * embd + 2L * ffn) + 2L * 4L * t * vocab
            act = num.toDouble() / MIB
        } else {
            act = 0.25 * weights
        }
        val total = (weights + state) * safety.weightsInflation + (act + cap.trainOverheadMb) * safety.estimateInflation
        return Pair(total, trainable)
    }

    fun trainTps(d: DeviceProfile, s: ArtifactSpec, k: Int?, safety: SafetyPolicy): Double {
        val layers = geometry(s).layers
        val p = specParams(s)
        val perLayer = p / layers
        val flops = if (k == null) 6 * p else 2 * p + (4 * k).toDouble() * perLayer
        return trainGflopsPrior(d) * 1e9 * safety.estimateThermalDerate / flops
    }

    // ----------------------------- measurement records ----------------------------------------------------------

    class Usable(val records: List<MeasurementRecord>, val ignored: List<String>)

    fun validRecords(s: ArtifactSpec, d: DeviceProfile, records: List<MeasurementRecord>): Usable {
        val ok = ArrayList<MeasurementRecord>()
        val ignored = ArrayList<String>()
        for (r in records) {
            if (r.artifactId != s.artifactId) continue
            if (r.deviceId != d.deviceId) {
                ignored.add("${r.recordId}:other_device")
            } else if (r.artifactSha256 != null && s.sha256 != null && r.artifactSha256 != s.sha256) {
                ignored.add("${r.recordId}:artifact_hash_mismatch")
            } else {
                ok.add(r)
            }
        }
        return Usable(ok, ignored)
    }

    private class Key(val artifact: String, val device: String, val kind: String, val ctx: Int, val k: Int) : Comparable<Key> {
        override fun compareTo(other: Key): Int {
            var r = artifact.compareTo(other.artifact)
            if (r != 0) return r
            r = device.compareTo(other.device)
            if (r != 0) return r
            r = kind.compareTo(other.kind)
            if (r != 0) return r
            r = ctx.compareTo(other.ctx)
            if (r != 0) return r
            return k.compareTo(other.k)
        }
        override fun equals(other: Any?) = other is Key && compareTo(other) == 0
        override fun hashCode() = artifact.hashCode() * 31 + device.hashCode() + kind.hashCode() * 7 + ctx * 13 + k
    }

    private fun keyOf(r: MeasurementRecord): Key {
        val ctx = if (r.kind == "inference") r.contextTokens else null
        val k: Int? = if (r.kind == "training") (r.trainableLastLayers ?: -1) else null
        return Key(r.artifactId, r.deviceId, r.kind, ctx ?: -1, k ?: -2)
    }

    /** Union keyed by (artifact, device, kind, context, trainable layers); the latest recordedAt wins (ties: the later argument). */
    fun mergeRecords(existing: List<MeasurementRecord>, new: List<MeasurementRecord>): List<MeasurementRecord> {
        val best = java.util.TreeMap<Key, MeasurementRecord>()
        for (r in existing + new) {
            val key = keyOf(r)
            val cur = best[key]
            if (cur == null || r.recordedAt >= cur.recordedAt) best[key] = r
        }
        return best.values.toList()
    }

    private fun inferenceMeasurement(r: MeasurementRecord, configId: String) = Measurement(
        configId, r.ttftMs, r.tokensPerS, r.peakRamMb, r.sustainedRamMb, r.thermalThrottleRatio, r.uiJankPct, r.crashes, r.anrs,
        r.backgroundKills, r.sustainedMinutes,
    )

    private fun complete(m: Measurement) = m.ttftMs != null && m.tokensPerS != null && m.peakRamMb != null && m.sustainedRamMb != null &&
        m.thermalThrottleRatio != null && m.uiJankPct != null && m.crashes != null && m.anrs != null && m.backgroundKills != null &&
        m.sustainedMinutes != null

    // ----------------------------- classification ---------------------------------------------------------------

    fun trainingConditions(state: DeviceState?, cap: CapabilityPolicy, storageOk: Boolean): List<Condition> {
        val s = state ?: DeviceState()
        fun tri(v: Boolean?) = if (v == null) "unknown" else if (v) "met" else "unmet"
        val ts = s.thermalStatus
        val thermal: Boolean? = if (ts == null) null else ts <= cap.trainMaxThermalStatus
        val pct = s.batteryPct
        val battery: Boolean? = if (s.charging == true) true else if (pct != null) pct >= cap.trainMinBatteryPct else null
        return listOf(
            Condition("thermal", tri(thermal)),
            Condition("charging", tri(s.charging)),
            Condition("battery", tri(battery)),
            Condition("not_power_save", tri(s.powerSave?.let { !it })),
            Condition("storage_for_checkpoints", if (storageOk) "met" else "unmet"),
        )
    }

    fun classify(
        device: DeviceProfile, s: ArtifactSpec, safety: SafetyPolicy = SafetyPolicy(), cap: CapabilityPolicy = CapabilityPolicy(),
        state: DeviceState? = null, records: List<MeasurementRecord> = emptyList(),
    ): ArtifactCapabilities {
        val reasons = ArrayList<String>()
        val usable = validRecords(s, device, records)
        val geo = geometry(s)
        val layers = geo.layers
        if (geo.estimated) reasons.add("geometry_estimated")
        if (s.sizeBytes == null) reasons.add("size_estimated")
        if (s.params == null) reasons.add("params_nominal")
        if (s.runtimeSupported == "unverified") reasons.add("runtime_support_unverified")
        if (s.runtimeSupported == "no") reasons.add("runtime_unsupported")

        val reserve = Qualifier.storageReserveMb(device, safety)
        val fileMb = specFileMb(s)
        val avail = device.storageFreeMb - reserve
        val storageOk = fileMb <= avail
        val canDownload = s.downloadable && storageOk
        if (!s.downloadable) reasons.add("not_downloadable")
        if (!storageOk) reasons.add("storage_insufficient")

        val maxCtx = minOf(s.ctxTrain ?: 8192, 8192)
        val infMeas = usable.records.filter { it.kind == "inference" && it.contextTokens != null }
            .map { inferenceMeasurement(it, specCandidate(s, it.contextTokens!!).configId) }
        val dev = if (infMeas.isNotEmpty()) device.copy(measurements = device.measurements + infMeas) else device
        val loadC = specCandidate(s, minOf(cap.loadContext, maxCtx))
        val loadA = Qualifier.assess(dev, loadC, safety)
        var loadRam = loadA.ram.requiredModelSideMb
        var canLoad = loadA.ramVerdict == Verdict.FIT
        val loadRecs = usable.records.filter { it.kind == "load" }
        val peaks = loadRecs.mapNotNull { it.loadPeakRamMb }
        if (peaks.isNotEmpty()) {
            loadRam = peaks.max()
            canLoad = loadA.ram.ramBudgetMb > 0 && loadRam <= loadA.ram.ramBudgetMb
        }
        if (!canLoad) reasons.add("ram_insufficient_to_load")

        var bestOk: CandidateAssessment? = null
        var bestComfortable: CandidateAssessment? = null
        for (ctx in cap.inferContexts.sorted()) {
            if (ctx > maxCtx) continue
            val a = Qualifier.assess(dev, specCandidate(s, ctx), safety)
            val ok = a.ramVerdict == Verdict.FIT && a.sustainedVerdict != Verdict.NO_FIT &&
                a.config.quant.qualityPenalty <= safety.maxQuantPenalty && s.runtimeSupported != "no"
            if (ok) {
                bestOk = a
                if (a.ram.utilization <= safety.balancedMaxUtilization) bestComfortable = a
            }
        }
        val chosen = bestComfortable ?: bestOk
        val canInfer = chosen != null
        if (!canInfer && canLoad && s.runtimeSupported != "no") reasons.add("infer_too_slow_or_unsafe")
        var estTps: Double? = null
        var inferUtil: Double? = null
        var inferRam: Double? = null
        var ctxOut: Int? = null
        if (chosen != null) {
            ctxOut = chosen.config.contextTokens
            estTps = chosen.sustained.tokensPerS ?: chosen.sustained.estTokensPerS
            inferUtil = chosen.ram.utilization
            inferRam = chosen.ram.requiredModelSideMb
        }
        val canEval = canInfer && ctxOut != null && ctxOut >= cap.evalMinContext && estTps != null && estTps >= cap.evalMinTps
        val q = Qualifier.qualityScore(specCandidate(s, minOf(2048, maxCtx)))

        // ---- specialization (training): a SEPARATE question from inference ----
        val (budget, _) = Qualifier.ramBudgetMb(device, safety)
        val trainRecs = usable.records.filter { it.kind == "training" }
        val failedLayers = trainRecs.filter { it.trainOom == true || it.trainCompleted == false }.map { it.trainableLastLayers ?: layers }
        val minFailed: Int? = failedLayers.minOrNull()

        fun option(k: Int?): TrainingOption {
            val (estRam, trainable) = trainingRamMb(s, k, cap, safety)
            val rec = trainRecs.lastOrNull { it.trainableLastLayers == k }
            val measured = rec?.trainPeakRamMb != null
            val ram = rec?.trainPeakRamMb ?: estRam
            val tps = rec?.trainTokensPerS ?: trainTps(device, s, k, safety)
            val ckpt = rec?.checkpointSizeMb ?: (trainable * 4 / MIB)
            val storageReq = 2 * ckpt + cap.trainDatasetStorageMb
            val effK = k ?: layers
            val failed = rec != null && (rec.trainOom == true || rec.trainCompleted == false)
            val fits = budget > 0 && ram <= budget * cap.trainMaxUtilization && storageReq <= avail &&
                !(minFailed != null && effK >= minFailed) && !failed
            return TrainingOption(
                if (k == null) "full" else "partial", effK, PyFormat.round(trainable / 1e6, 2), PyFormat.round(ram, 1),
                if (budget > 0) PyFormat.round(ram / budget, 4) else 999.0, PyFormat.round(tps, 3),
                PyFormat.round(cap.trainRefTokens.toDouble() / tps / 60, 1), PyFormat.round(ckpt, 1), PyFormat.round(storageReq, 1), fits, measured,
            )
        }

        var fullOpt: TrainingOption? = null
        var partialOpt: TrainingOption? = null
        var bestK: Int? = null
        val cls = s.trainingClass
        val fullPrecisionArtifact = s.precision != "quantized"
        if (cls == "inference_only" || !fullPrecisionArtifact) {
            reasons.add("training_needs_full_precision_artifact")
        } else if (cls == "external_only") {
            reasons.add("training_class_external_only")
        } else {
            if (cls == "local_full") fullOpt = option(null)
            var k = if (cls == "local_full") layers else layers - 1
            while (k >= cap.minPartialLayers) {
                val o = option(k)
                if (o.fits) { partialOpt = o; bestK = k; break }
                k -= 1
            }
            if (partialOpt == null && layers >= cap.minPartialLayers) partialOpt = option(cap.minPartialLayers)
        }
        val canFull = fullOpt != null && fullOpt.fits
        val canPartial = partialOpt != null && partialOpt.fits
        if ((cls == "local_full" || cls == "local_partial") && fullPrecisionArtifact && !canFull && !canPartial) reasons.add("training_ram_insufficient")
        if (canPartial && partialOpt != null && partialOpt.estMinutesForReferenceTokens / 60 > cap.maxPracticalTrainHours) reasons.add("training_very_slow")
        val external = !canFull && !canPartial
        val via = if (cls == "inference_only" || !fullPrecisionArtifact) s.trainingSourceArtifactId else null
        val options = listOfNotNull(fullOpt, partialOpt)
        var conds: List<Condition> = emptyList()
        if (options.isNotEmpty()) conds = trainingConditions(state, cap, options.any { it.storageRequiredMb <= avail })
        val ready = (canFull || canPartial) && conds.all { it.status == "met" }

        var confInfer = "low"
        if (infMeas.isNotEmpty()) {
            val atCtx = infMeas.filter { chosen != null && it.configId == chosen.config.configId }
            confInfer = if (atCtx.any { complete(it) }) "high" else "medium"
        } else if (loadRecs.isNotEmpty()) {
            confInfer = "medium"
        }
        val done = trainRecs.filter { it.trainCompleted == true && it.trainOom != true && it.trainPeakRamMb != null && it.trainTokensPerS != null }
        val confTrain = if (done.isNotEmpty()) "high" else if (trainRecs.isNotEmpty()) "medium" else "low"
        return ArtifactCapabilities(
            s.artifactId, canDownload, canLoad, canInfer, canEval, canFull, canPartial, external, ctxOut,
            estTps?.let { PyFormat.round(it, 2) }, inferRam?.let { PyFormat.round(it, 1) }, inferUtil, PyFormat.round(loadRam, 1), q, bestK,
            fullOpt, partialOpt, via, conds, ready, confInfer, confTrain, reasons, usable.ignored,
        )
    }

    // ----------------------------- owner-facing choices ---------------------------------------------------------

    private fun tier(downloadable: Boolean, conf: String): String =
        if (!downloadable) "preview" else if (conf == "high") "recommended" else "provisional"

    private fun <T> firstMax(pool: List<T>, cmp: (T, T) -> Int): T? {
        var best: T? = null
        for (x in pool) if (best == null || cmp(x, best) > 0) best = x
        return best
    }

    private fun cmp(a: Double, b: Double): Int = a.compareTo(b)

    fun choose(
        device: DeviceProfile, specs: List<ArtifactSpec>, safety: SafetyPolicy = SafetyPolicy(), cap: CapabilityPolicy = CapabilityPolicy(),
        state: DeviceState? = null, records: List<MeasurementRecord> = emptyList(),
    ): CapabilityReport {
        val caps = specs.map { classify(device, it, safety, cap, state, records) }
        val pairs = specs.zip(caps).filter { (s, c) ->
            s.licenseState != "DISALLOWED" && c.canInfer && c.canLoad && "storage_insufficient" !in c.reasons
        }

        fun pools(pred: (ArtifactSpec, ArtifactCapabilities) -> Boolean): Pair<List<Pair<ArtifactSpec, ArtifactCapabilities>>, Boolean> {
            val sel = pairs.filter { pred(it.first, it.second) }
            val dl = sel.filter { it.first.downloadable }
            return if (dl.isNotEmpty()) Pair(dl, true) else Pair(sel, false)
        }
        fun util(c: ArtifactCapabilities) = c.inferUtilization ?: 999.0
        fun tps(c: ArtifactCapabilities) = c.estTokensPerS ?: 0.0

        fun pick(
            choice: String, info: Pair<List<Pair<ArtifactSpec, ArtifactCapabilities>>, Boolean>,
            cmpFn: (Pair<ArtifactSpec, ArtifactCapabilities>, Pair<ArtifactSpec, ArtifactCapabilities>) -> Int, whyOk: String, whyNone: String,
        ): ChoicePick {
            val best = firstMax(info.first, cmpFn)
                ?: return ChoicePick(choice, CHOICE_LABELS.getValue(choice), null, "none", reason = whyNone)
            val (s, c) = best
            return ChoicePick(choice, CHOICE_LABELS.getValue(choice), s.artifactId, tier(info.second, c.confidenceInfer), c.confidenceInfer,
                c.inferContext, null, s.licenseState, s.downloadable, whyOk)
        }

        fun qualityCmp(a: Pair<ArtifactSpec, ArtifactCapabilities>, b: Pair<ArtifactSpec, ArtifactCapabilities>): Int {
            var r = cmp(a.second.qualityScore, b.second.qualityScore)
            if (r != 0) return r
            r = (a.second.inferContext ?: 0).compareTo(b.second.inferContext ?: 0)
            if (r != 0) return r
            return cmp(-util(a.second), -util(b.second))
        }

        val fastest = pick("fastest", pools { _, _ -> true }, { a, b ->
            val r = cmp(tps(a.second), tps(b.second))
            if (r != 0) r else cmp(-specFileMb(a.first), -specFileMb(b.first))
        }, "fastest_decode_within_envelope", "nothing_fits")
        val balanced = pick("balanced", pools { _, c ->
            util(c) <= safety.balancedMaxUtilization && tps(c) >= safety.balancedMinTps && (c.inferContext ?: 0) >= safety.balancedMinContext
        }, ::qualityCmp, "best_quality_with_comfortable_headroom", "nothing_comfortable")
        val bestQ = pick("best_quality", pools { _, c -> util(c) <= safety.maxQualityMaxUtilization }, ::qualityCmp,
            "highest_quality_within_envelope", "nothing_fits")

        fun specMinutes(c: ArtifactCapabilities): Double {
            val o = if (c.canSpecializeFull) c.full else c.partial
            return o?.estMinutesForReferenceTokens ?: 0.0
        }
        val (spPool, spDl) = pools { _, c -> c.canSpecializeFull || c.canSpecializePartial }
        val sbest = firstMax(spPool) { a, b ->
            var r = cmp(a.second.qualityScore, b.second.qualityScore)
            if (r == 0) r = (if (a.second.canSpecializeFull) 1 else 0).compareTo(if (b.second.canSpecializeFull) 1 else 0)
            if (r == 0) r = cmp(-specMinutes(a.second), -specMinutes(b.second))
            r
        }
        val sp = if (sbest == null) {
            ChoicePick("best_specialize", CHOICE_LABELS.getValue("best_specialize"), null, "none", reason = "no_local_specialization_possible")
        } else {
            val (s, c) = sbest
            val layersOut = if (c.canSpecializeFull && c.full != null) c.full.trainableLastLayers else c.trainableLastLayers
            ChoicePick("best_specialize", CHOICE_LABELS.getValue("best_specialize"), s.artifactId, tier(spDl, c.confidenceTrain), c.confidenceTrain,
                c.inferContext, layersOut, s.licenseState, s.downloadable, if (c.canSpecializeFull) "full_tuning_fits" else "partial_tuning_fits")
        }
        return CapabilityReport(device.deviceId, caps, listOf(fastest, balanced, bestQ, sp))
    }
}

/** JSON entry points (snake_case keys as in the Python schemas and registry/artifacts). */
object CapabilityJson {
    private fun <T> req(v: T?, what: String): T = v ?: throw JsonException("missing or invalid $what")
    private fun int(o: Map<String, Any?>, k: String): Int? = o.lng(k)?.toInt()

    fun spec(o: Map<String, Any?>) = ArtifactSpec(
        req(o.str("artifact_id"), "artifact_id"), o.str("model_id") ?: "", req(o.str("quantization"), "quantization"),
        req(o.str("precision"), "precision"), o.str("sha256"), o.lng("size_bytes"), o.dbl("params"), req(o.dbl("nominal_params_b"), "nominal_params_b"),
        int(o, "layers"), int(o, "kv_heads"), int(o, "head_dim"), int(o, "embd"), int(o, "ffn"), int(o, "vocab"), int(o, "ctx_train"),
        req(o.str("training_class"), "training_class"), o.str("training_source_artifact_id"), o.bool("downloadable") ?: false,
        o.str("license_state") ?: "UNVERIFIED", o.str("runtime_supported") ?: "unverified",
    )

    fun record(o: Map<String, Any?>) = MeasurementRecord(
        req(o.str("record_id"), "record_id"), req(o.str("artifact_id"), "artifact_id"), o.str("artifact_sha256"), req(o.str("device_id"), "device_id"),
        o.str("runtime_id"), o.str("app_version"), req(o.str("recorded_at"), "recorded_at"), req(o.str("kind"), "kind"),
        o.dbl("load_time_ms"), o.dbl("load_peak_ram_mb"), int(o, "context_tokens"), o.dbl("ttft_ms"), o.dbl("tokens_per_s"), o.dbl("peak_ram_mb"),
        o.dbl("sustained_ram_mb"), o.dbl("thermal_throttle_ratio"), o.dbl("ui_jank_pct"), o.lng("crashes"), o.lng("anrs"), o.lng("background_kills"),
        o.dbl("sustained_minutes"), int(o, "trainable_last_layers"), o.dbl("train_peak_ram_mb"), o.dbl("train_tokens_per_s"), o.dbl("checkpoint_size_mb"),
        o.dbl("train_minutes"), o.bool("train_completed"), o.bool("train_oom"), int(o, "thermal_max_status"), o.dbl("battery_drop_pct"),
    )

    /** Parses one `device_measurement.v1` record (JSON text). Throws [JsonException] when malformed or of another schema. */
    fun parseRecord(json: String): MeasurementRecord {
        val o = MiniJson.parse(json).asObj() ?: throw JsonException("record must be a JSON object")
        val schema = o.str("schema_id")
        if (schema != null && schema != "device_measurement.v1") throw JsonException("unsupported record schema $schema")
        if (o.str("kind") !in setOf("load", "inference", "training")) throw JsonException("kind must be load, inference or training")
        return record(o)
    }

    fun state(o: Map<String, Any?>) = DeviceState(int(o, "thermal_status"), o.dbl("battery_pct"), o.bool("charging"), o.bool("power_save"))

    fun capPolicy(o: Map<String, Any?>): CapabilityPolicy {
        val d = CapabilityPolicy()
        return CapabilityPolicy(
            o.list("infer_contexts")?.mapNotNull { (it as? Long)?.toInt() } ?: d.inferContexts, int(o, "load_context") ?: d.loadContext,
            int(o, "eval_min_context") ?: d.evalMinContext, o.dbl("eval_min_tps") ?: d.evalMinTps, int(o, "train_ctx_tokens") ?: d.trainCtxTokens,
            o.dbl("train_overhead_mb") ?: d.trainOverheadMb, o.dbl("train_max_utilization") ?: d.trainMaxUtilization,
            int(o, "min_partial_layers") ?: d.minPartialLayers, o.dbl("train_dataset_storage_mb") ?: d.trainDatasetStorageMb,
            int(o, "train_ref_tokens") ?: d.trainRefTokens, o.dbl("max_practical_train_hours") ?: d.maxPracticalTrainHours,
            o.dbl("train_min_battery_pct") ?: d.trainMinBatteryPct, int(o, "train_max_thermal_status") ?: d.trainMaxThermalStatus,
        )
    }

    /** Serialize a record the way the Python schema does (null fields omitted), for the on-device measurement ledger. */
    fun recordToMap(r: MeasurementRecord): Map<String, Any?> {
        val m = LinkedHashMap<String, Any?>()
        m["schema_id"] = "device_measurement.v1"
        m["record_id"] = r.recordId; m["artifact_id"] = r.artifactId; m["artifact_sha256"] = r.artifactSha256; m["device_id"] = r.deviceId
        m["runtime_id"] = r.runtimeId; m["app_version"] = r.appVersion; m["recorded_at"] = r.recordedAt; m["kind"] = r.kind
        m["load_time_ms"] = r.loadTimeMs; m["load_peak_ram_mb"] = r.loadPeakRamMb; m["context_tokens"] = r.contextTokens; m["ttft_ms"] = r.ttftMs
        m["tokens_per_s"] = r.tokensPerS; m["peak_ram_mb"] = r.peakRamMb; m["sustained_ram_mb"] = r.sustainedRamMb
        m["thermal_throttle_ratio"] = r.thermalThrottleRatio; m["ui_jank_pct"] = r.uiJankPct; m["crashes"] = r.crashes; m["anrs"] = r.anrs
        m["background_kills"] = r.backgroundKills; m["sustained_minutes"] = r.sustainedMinutes; m["trainable_last_layers"] = r.trainableLastLayers
        m["train_peak_ram_mb"] = r.trainPeakRamMb; m["train_tokens_per_s"] = r.trainTokensPerS; m["checkpoint_size_mb"] = r.checkpointSizeMb
        m["train_minutes"] = r.trainMinutes; m["train_completed"] = r.trainCompleted; m["train_oom"] = r.trainOom
        m["thermal_max_status"] = r.thermalMaxStatus; m["battery_drop_pct"] = r.batteryDropPct
        return m
    }
}
