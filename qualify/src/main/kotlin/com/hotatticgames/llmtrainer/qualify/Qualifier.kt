package com.hotatticgames.llmtrainer.qualify

import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Kotlin port of factory/llmtrainer/device.py (ALGORITHM_VERSION 1). The Python module is the executable spec;
 * golden vectors (factory/tests/golden/device_qualification.v1.json) pin both. Keep operation order identical
 * to Python so floating-point results match bit for bit.
 */
object Qualifier {
    const val ALGORITHM_VERSION = "1"
    private const val MIB = 1048576.0
    private val LN2 = ln(2.0)

    val PROFILE_LABELS: Map<String, String> = linkedMapOf(
        "performance" to "Performance",
        "balanced" to "Balanced (Recommended)",
        "max_quality" to "Maximum Quality Within Safe Envelope",
    )
    private val REQUIRED_MEASUREMENTS = listOf(
        "ttft_ms", "tokens_per_s", "peak_ram_mb", "sustained_ram_mb", "thermal_throttle_ratio",
        "ui_jank_pct", "crashes", "anrs", "background_kills", "sustained_minutes",
    )

    private fun field(m: Measurement, name: String): Any? = when (name) {
        "ttft_ms" -> m.ttftMs
        "tokens_per_s" -> m.tokensPerS
        "peak_ram_mb" -> m.peakRamMb
        "sustained_ram_mb" -> m.sustainedRamMb
        "thermal_throttle_ratio" -> m.thermalThrottleRatio
        "ui_jank_pct" -> m.uiJankPct
        "crashes" -> m.crashes
        "anrs" -> m.anrs
        "background_kills" -> m.backgroundKills
        else -> m.sustainedMinutes
    }

    private fun kvBytes(dtype: String): Double = when (dtype) {
        "f16" -> 2.0
        "q8_0" -> 1.0625
        "q4_0" -> 0.5625
        else -> throw IllegalArgumentException("unknown kv dtype $dtype")
    }

    // ----------------------------- component math ------------------------------

    fun kvCacheMb(layers: Int, kvHeads: Int, headDim: Int, context: Int, kvDtype: String = "f16"): Double {
        val n = 2L * layers * kvHeads * headDim * context
        return n.toDouble() * kvBytes(kvDtype) / MIB
    }

    fun weightsMb(c: CandidateConfig): Double =
        c.weightsFileMb ?: (c.model.paramsB * 1e9 * c.quant.bitsPerWeight / 8 / MIB)

    fun runtimeMb(c: CandidateConfig): Double =
        c.runtime.overheadMb + c.runtime.scratchMbPer1kCtx * c.contextTokens / 1024

    fun qualityScore(c: CandidateConfig): Double {
        val base = c.model.qualityScore ?: (10 * (ln(1 + c.model.paramsB) / LN2))
        return PyFormat.round(base - c.quant.qualityPenalty, 4)
    }

    fun makeConfigId(m: ModelSpec, q: QuantSpec, ctx: Int, rt: RuntimeSpec, ret: RetrievalBudget): String =
        m.modelId + "." + q.name + ".c" + ctx + "." + rt.name + ".kv-" + rt.kvDtype + ".r" + ret.indexRamMb.toLong()

    fun generateCandidates(
        models: List<ModelSpec>, quants: List<QuantSpec>, contexts: List<Int>, runtimes: List<RuntimeSpec>,
        retrievals: List<RetrievalBudget> = listOf(RetrievalBudget()),
    ): List<CandidateConfig> {
        val out = ArrayList<CandidateConfig>()
        for (m in models) for (q in quants) for (ctx in contexts) for (rt in runtimes) for (ret in retrievals) {
            if (ctx > m.maxContext) continue
            out.add(CandidateConfig(makeConfigId(m, q, ctx, rt, ret), m, q, ctx, rt, ret))
        }
        return out
    }

    // ----------------------------- envelopes -----------------------------------

    fun safetyReserveMb(d: DeviceProfile, p: SafetyPolicy): Double = max(p.safetyReserveMinMb, p.safetyReserveFrac * d.totalRamMb)

    /** (model-side RAM budget, safety reserve). Budget may be negative on tiny devices. */
    fun ramBudgetMb(d: DeviceProfile, p: SafetyPolicy): Pair<Double, Double> {
        var ceiling = d.totalRamMb - d.osReserveMb - d.backgroundReserveMb
        val typ = d.typicalAvailableRamMb
        if (typ != null) ceiling = min(ceiling, typ)
        val reserve = safetyReserveMb(d, p)
        return Pair(ceiling - d.hostAppMb - reserve, reserve)
    }

    fun storageReserveMb(d: DeviceProfile, p: SafetyPolicy): Double = max(p.storageReserveMinMb, p.storageReserveFrac * d.storageTotalMb)

    fun findMeasurement(d: DeviceProfile, configId: String): Measurement? = d.measurements.lastOrNull { it.configId == configId }

    // ----------------------------- verdicts ------------------------------------

    fun assessStorage(d: DeviceProfile, c: CandidateConfig, p: SafetyPolicy): Pair<String, StorageBreakdown> {
        val modelFile = weightsMb(c)
        val required = modelFile + c.retrieval.indexStorageMb
        val reserve = storageReserveMb(d, p)
        val avail = d.storageFreeMb - reserve
        val sb = StorageBreakdown(
            PyFormat.round(modelFile, 1), c.retrieval.indexStorageMb, PyFormat.round(required, 1), d.storageFreeMb,
            PyFormat.round(reserve, 1), PyFormat.round(avail, 1), PyFormat.round(avail - required, 1),
        )
        return Pair(if (required <= avail) Verdict.FIT else Verdict.NO_FIT, sb)
    }

    fun assessRam(d: DeviceProfile, c: CandidateConfig, p: SafetyPolicy, m: Measurement?): Pair<String, RamBreakdown> {
        val w = weightsMb(c)
        val rt = runtimeMb(c)
        val kv = kvCacheMb(c.model.layers, c.model.kvHeads, c.model.headDim, c.contextTokens, c.runtime.kvDtype)
        val ret = c.retrieval.indexRamMb
        val (budget, reserve) = ramBudgetMb(d, p)
        val estimated = w * p.weightsInflation + (rt + kv + ret) * p.estimateInflation
        val measured = if (m == null) emptyList() else listOfNotNull(m.peakRamMb, m.sustainedRamMb)
        val overridden = measured.isNotEmpty()
        val required = if (overridden) measured.max() else estimated
        val util = if (budget > 0) PyFormat.round(required / budget, 4) else (if (required > 0) 999.0 else 0.0)
        val rb = RamBreakdown(
            PyFormat.round(w, 1), PyFormat.round(rt, 1), PyFormat.round(kv, 1), ret,
            if (overridden) 1.0 else p.estimateInflation, if (overridden) 1.0 else p.weightsInflation,
            PyFormat.round(required, 1), overridden, d.totalRamMb, d.osReserveMb, d.backgroundReserveMb, d.hostAppMb,
            PyFormat.round(reserve, 1), PyFormat.round(budget, 1), PyFormat.round(budget - required, 1), util,
        )
        return Pair(if (budget > 0 && required <= budget) Verdict.FIT else Verdict.NO_FIT, rb)
    }

    fun estimateDecodeTps(d: DeviceProfile, c: CandidateConfig, p: SafetyPolicy): Double {
        val bwRaw = d.memBandwidthGbps
        val bw = if (bwRaw != null && bwRaw != 0.0) bwRaw else p.fallbackBandwidthGbps
        val bytesPerToken = weightsMb(c) * MIB
        return bw * 1e9 * p.bandwidthEfficiency * p.estimateThermalDerate / bytesPerToken
    }

    fun assessSustained(d: DeviceProfile, c: CandidateConfig, p: SafetyPolicy, m: Measurement?, ramOk: Boolean): SustainedAssessment {
        val estTps = estimateDecodeTps(d, c, p)
        val estTtft = 512 / (estTps * 6) * 1000
        var missing: List<String> = REQUIRED_MEASUREMENTS.filter { m == null || field(m, it) == null }
        var reasons: List<String> = emptyList()
        val failed = ArrayList<String>()
        if (m != null) {
            val tps = m.tokensPerS
            if (tps != null && tps < p.minTps) failed.add("measured ${PyFormat.fixed(tps, 1)} tok/s < minimum ${PyFormat.repr(p.minTps)}")
            val ttft = m.ttftMs
            if (ttft != null && ttft > p.maxTtftMs) failed.add("TTFT ${PyFormat.fixed(ttft, 0)} ms > ${PyFormat.fixed(p.maxTtftMs, 0)} ms")
            val tr = m.thermalThrottleRatio
            if (tr != null && tr < p.minThermalRatio) {
                failed.add("sustained throttle ratio ${PyFormat.fixed(tr, 2)} < ${PyFormat.repr(p.minThermalRatio)}")
            }
            val jank = m.uiJankPct
            if (jank != null && jank > p.maxUiJankPct) failed.add("UI jank ${PyFormat.fixed(jank, 1)}% > ${PyFormat.repr(p.maxUiJankPct)}%")
            for ((name, v) in listOf("crashes" to m.crashes, "ANRs" to m.anrs, "background-process kills" to m.backgroundKills)) {
                if (v != null && v != 0L) failed.add("$v $name observed")
            }
            val sm = m.sustainedMinutes
            if (sm != null && sm < p.minSustainedMinutes) {
                reasons = reasons + ("run lasted ${PyFormat.fixed(sm, 1)} min < required ${PyFormat.repr(p.minSustainedMinutes)} min; inconclusive")
                missing = (missing.toSet() + "sustained_minutes").sorted()
            }
        }
        if (!ramOk) failed.add("RAM envelope exceeded")
        val basis = when {
            m == null || REQUIRED_MEASUREMENTS.all { field(m, it) == null } -> Confidence.ESTIMATED
            missing.isEmpty() -> Confidence.MEASURED
            else -> Confidence.PARTIAL
        }
        if (m == null || m.tokensPerS == null) {
            if (estTps < p.minTps) failed.add("estimated sustained ${PyFormat.fixed(estTps, 1)} tok/s < minimum ${PyFormat.repr(p.minTps)}")
        }
        val verdict: String
        if (failed.isNotEmpty()) {
            verdict = Verdict.NO_FIT
            reasons = failed + reasons
        } else if (basis == Confidence.MEASURED) {
            verdict = Verdict.FIT
        } else {
            verdict = Verdict.UNVERIFIED
            reasons = reasons + "sustained behavior not fully measured on this device; estimates only"
        }
        return SustainedAssessment(
            verdict, basis, PyFormat.round(estTps, 2), PyFormat.roundInt(estTtft), m?.tokensPerS, m?.ttftMs, reasons,
            if (basis != Confidence.MEASURED) missing else emptyList(),
        )
    }

    fun assess(d: DeviceProfile, c: CandidateConfig, p: SafetyPolicy = SafetyPolicy()): CandidateAssessment {
        val m = findMeasurement(d, c.configId)
        val (sv, sb) = assessStorage(d, c, p)
        val (rv, rb) = assessRam(d, c, p, m)
        val su = assessSustained(d, c, p, m, rv == Verdict.FIT)
        val blocking = ArrayList<String>()
        if (sv == Verdict.NO_FIT) {
            blocking.add("storage: needs ${PyFormat.fixed(sb.requiredMb, 0)} MB, ${PyFormat.fixed(sb.availableAfterReserveMb, 0)} MB available after reserve")
        }
        if (rv == Verdict.NO_FIT) {
            blocking.add("RAM: needs ${PyFormat.fixed(rb.requiredModelSideMb, 0)} MB, budget ${PyFormat.fixed(rb.ramBudgetMb, 0)} MB")
        }
        if (su.verdict == Verdict.NO_FIT) blocking.addAll(su.reasons)
        if (c.quant.qualityPenalty > p.maxQuantPenalty) {
            blocking.add("quantization ${c.quant.name} exceeds the allowed quality penalty (${PyFormat.repr(p.maxQuantPenalty)})")
        }
        return CandidateAssessment(
            c, sv, rv, su.verdict, sb, rb, su, su.basis, qualityScore(c),
            safeToDeploy = sv == Verdict.FIT && rv == Verdict.FIT && su.verdict == Verdict.FIT && blocking.isEmpty(),
            eligible = blocking.isEmpty(), blockingReasons = blocking,
        )
    }

    // ----------------------------- recommendation ------------------------------

    private fun tps(a: CandidateAssessment): Double = a.sustained.tokensPerS ?: a.sustained.estTokensPerS

    private fun cmpQuality(a: CandidateAssessment, b: CandidateAssessment): Int {
        var r = a.qualityScore.compareTo(b.qualityScore)
        if (r != 0) return r
        r = a.config.retrieval.indexRamMb.compareTo(b.config.retrieval.indexRamMb)
        if (r != 0) return r
        r = a.config.contextTokens.compareTo(b.config.contextTokens)
        if (r != 0) return r
        return (-a.ram.utilization).compareTo(-b.ram.utilization)
    }

    /** Python max(): first maximal element wins. */
    private fun firstMax(pool: List<CandidateAssessment>, cmp: (CandidateAssessment, CandidateAssessment) -> Int): CandidateAssessment? {
        var best: CandidateAssessment? = null
        for (a in pool) if (best == null || cmp(a, best) > 0) best = a
        return best
    }

    private fun pick(
        profile: String, pool: List<CandidateAssessment>, pickFn: (List<CandidateAssessment>) -> CandidateAssessment?,
        reasonOk: String, whyNone: String,
    ): ProfilePick {
        val label = PROFILE_LABELS.getValue(profile)
        val best = pickFn(pool) ?: return ProfilePick(profile, label, null, Tier.NONE, false, null, whyNone)
        val verified = pool.filter { it.safeToDeploy }
        val verifiedBest = if (verified.isNotEmpty()) pickFn(verified) else null
        val measured = best.safeToDeploy
        val warnings = ArrayList<String>()
        if (!measured) {
            warnings.add(
                "Based on conservative estimates only. Benchmark this exact configuration on the device " +
                    "before treating it as recommended.",
            )
            if (best.sustained.missingMeasurements.isNotEmpty()) {
                warnings.add("missing measurements: " + best.sustained.missingMeasurements.joinToString(", "))
            }
        }
        return ProfilePick(
            profile, label, best.config, if (measured) Tier.RECOMMENDED else Tier.PROVISIONAL, measured, best.confidence,
            reasonOk + " Quality score ${PyFormat.repr(best.qualityScore)}, RAM utilization ${PyFormat.percent0(best.ram.utilization)}, " +
                "~${PyFormat.fixed(tps(best), 1)} tok/s.",
            warnings, if (measured || verifiedBest == null) null else verifiedBest.config.configId,
        )
    }

    const val NONE_MESSAGE = "No configuration fits the safe envelope for this profile on this device; shrink the retrieval budget or " +
        "context, free storage, or choose a smaller model class."

    fun recommend(
        d: DeviceProfile, candidates: List<CandidateConfig>, policy: SafetyPolicy = SafetyPolicy(), requireRetrieval: Boolean = false,
    ): Recommendation {
        val cands = if (requireRetrieval) candidates.filter { it.retrieval.indexRamMb > 0 } else candidates
        val assessed = cands.map { assess(d, it, policy) }
        val eligible = assessed.filter { it.eligible }

        val maxQuality = { pool: List<CandidateAssessment> ->
            firstMax(pool.filter { it.ram.utilization <= policy.maxQualityMaxUtilization }, ::cmpQuality)
        }
        val balanced = { pool: List<CandidateAssessment> ->
            firstMax(
                pool.filter {
                    it.ram.utilization <= policy.balancedMaxUtilization && tps(it) >= policy.balancedMinTps &&
                        it.config.contextTokens >= policy.balancedMinContext
                },
                ::cmpQuality,
            )
        }
        val performance = { pool: List<CandidateAssessment> ->
            val fast = pool.filter { it.ram.utilization <= policy.performanceMaxUtilization && tps(it) >= policy.performanceMinTps }
            if (fast.isNotEmpty()) {
                firstMax(fast, ::cmpQuality)
            } else {
                firstMax(pool) { a, b ->
                    val r = tps(a).compareTo(tps(b))
                    if (r != 0) r else a.qualityScore.compareTo(b.qualityScore)
                }
            }
        }
        val picks = listOf(
            pick("performance", eligible, performance, "Fast and light: leaves generous headroom.", NONE_MESSAGE),
            pick("balanced", eligible, balanced, "Best quality that keeps comfortable headroom and usable speed.", NONE_MESSAGE),
            pick("max_quality", eligible, maxQuality, "Highest quality that still fits the safe envelope.", NONE_MESSAGE),
        )
        return Recommendation(policy, picks, assessed)
    }
}
