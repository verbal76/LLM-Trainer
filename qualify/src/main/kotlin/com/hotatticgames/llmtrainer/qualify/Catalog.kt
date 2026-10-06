package com.hotatticgames.llmtrainer.qualify

/**
 * Model-config catalog: model geometry, exact weights size per quantization, quant/runtime/context/retrieval options.
 * JSON shape: see ReferenceCatalog.JSON and factory/tools/gen_golden_device.py (catalog_candidates is the spec).
 */
class Catalog(
    val models: List<ModelSpec>,
    /** model_id -> (quant name -> exact weights file size in MB). */
    val weightsMb: Map<String, Map<String, Double>>,
    val quants: List<QuantSpec>,
    val runtimes: List<RuntimeSpec>,
    val contexts: List<Int>,
    val retrieval: List<RetrievalBudget>,
) {
    fun candidates(retrievalOverride: List<RetrievalBudget>? = null): List<CandidateConfig> {
        val base = Qualifier.generateCandidates(models, quants, contexts, runtimes, retrievalOverride ?: retrieval)
        return base.map { c ->
            val w = weightsMb[c.model.modelId]?.get(c.quant.name)
            if (w != null) c.copy(weightsFileMb = w) else c
        }
    }

    companion object {
        fun parse(json: String): Catalog = fromJson(MiniJson.parse(json).asObj() ?: throw JsonException("catalog must be an object"))

        fun fromJson(o: Map<String, Any?>): Catalog {
            val models = ArrayList<ModelSpec>()
            val weights = LinkedHashMap<String, Map<String, Double>>()
            for (m in o.list("models").orEmpty()) {
                val mo = m.asObj() ?: continue
                val spec = Codec.model(mo)
                models.add(spec)
                val w = mo.obj("weights_mb")
                if (w != null) {
                    val per = LinkedHashMap<String, Double>()
                    for ((k, _) in w) w.dbl(k)?.let { per[k] = it }
                    weights[spec.modelId] = per
                }
            }
            return Catalog(
                models, weights,
                o.list("quants").orEmpty().mapNotNull { it.asObj()?.let(Codec::quant) },
                o.list("runtimes").orEmpty().mapNotNull { it.asObj()?.let(Codec::runtime) },
                o.list("contexts").orEmpty().mapNotNull { (it as? Long)?.toInt() },
                o.list("retrieval").orEmpty().mapNotNull { it.asObj()?.let(Codec::retrieval) },
            )
        }
    }
}

/** JSON -> model objects (snake_case keys, as in the Python schemas). Missing required keys throw JsonException. */
object Codec {
    private fun <T> req(v: T?, what: String): T = v ?: throw JsonException("missing or invalid $what")

    fun model(o: Map<String, Any?>) = ModelSpec(
        req(o.str("model_id"), "model_id"), req(o.dbl("params_b"), "params_b"), req(o.lng("layers"), "layers").toInt(),
        req(o.lng("kv_heads"), "kv_heads").toInt(), req(o.lng("head_dim"), "head_dim").toInt(),
        o.dbl("quality_score"), (o.lng("max_context") ?: 8192L).toInt(),
    )

    fun quant(o: Map<String, Any?>) = QuantSpec(
        req(o.str("name"), "quant.name"), req(o.dbl("bits_per_weight"), "bits_per_weight"), req(o.dbl("quality_penalty"), "quality_penalty"),
    )

    fun runtime(o: Map<String, Any?>) = RuntimeSpec(
        req(o.str("name"), "runtime.name"), req(o.dbl("overhead_mb"), "overhead_mb"), o.dbl("scratch_mb_per_1k_ctx") ?: 0.0,
        o.str("kv_dtype") ?: "f16",
    )

    fun retrieval(o: Map<String, Any?>) = RetrievalBudget(o.dbl("index_ram_mb") ?: 0.0, o.dbl("index_storage_mb") ?: 0.0, (o.lng("top_k") ?: 0L).toInt())

    fun candidate(o: Map<String, Any?>) = CandidateConfig(
        req(o.str("config_id"), "config_id"), model(req(o.obj("model"), "model")), quant(req(o.obj("quant"), "quant")),
        req(o.lng("context_tokens"), "context_tokens").toInt(), runtime(req(o.obj("runtime"), "runtime")),
        o.obj("retrieval")?.let(::retrieval) ?: RetrievalBudget(), o.dbl("weights_file_mb"),
    )

    fun measurement(o: Map<String, Any?>) = Measurement(
        req(o.str("config_id"), "config_id"), o.dbl("ttft_ms"), o.dbl("tokens_per_s"), o.dbl("peak_ram_mb"), o.dbl("sustained_ram_mb"),
        o.dbl("thermal_throttle_ratio"), o.dbl("ui_jank_pct"), o.lng("crashes"), o.lng("anrs"), o.lng("background_kills"),
        o.dbl("sustained_minutes"),
    )

    fun device(o: Map<String, Any?>) = DeviceProfile(
        o.str("device_id") ?: "d", o.str("name") ?: "d", o.str("device_class") ?: "c", o.str("soc"), o.str("os") ?: "android",
        req(o.lng("total_ram_mb"), "total_ram_mb"), o.lng("typical_available_ram_mb"), req(o.lng("os_reserve_mb"), "os_reserve_mb"),
        req(o.lng("background_reserve_mb"), "background_reserve_mb"), req(o.lng("host_app_mb"), "host_app_mb"),
        req(o.lng("storage_total_mb"), "storage_total_mb"), req(o.lng("storage_free_mb"), "storage_free_mb"),
        o.dbl("mem_bandwidth_gbps"), o.list("measurements").orEmpty().mapNotNull { it.asObj()?.let(::measurement) },
    )

    /** Only keys present override the defaults. */
    fun policy(o: Map<String, Any?>): SafetyPolicy {
        val d = SafetyPolicy()
        return SafetyPolicy(
            o.dbl("safety_reserve_frac") ?: d.safetyReserveFrac, o.dbl("safety_reserve_min_mb") ?: d.safetyReserveMinMb,
            o.dbl("storage_reserve_frac") ?: d.storageReserveFrac, o.dbl("storage_reserve_min_mb") ?: d.storageReserveMinMb,
            o.dbl("estimate_inflation") ?: d.estimateInflation, o.dbl("weights_inflation") ?: d.weightsInflation,
            o.dbl("fallback_bandwidth_gbps") ?: d.fallbackBandwidthGbps, o.dbl("bandwidth_efficiency") ?: d.bandwidthEfficiency,
            o.dbl("estimate_thermal_derate") ?: d.estimateThermalDerate, o.dbl("min_tps") ?: d.minTps, o.dbl("max_ttft_ms") ?: d.maxTtftMs,
            o.dbl("min_thermal_ratio") ?: d.minThermalRatio, o.dbl("max_ui_jank_pct") ?: d.maxUiJankPct,
            o.dbl("min_sustained_minutes") ?: d.minSustainedMinutes, o.dbl("max_quant_penalty") ?: d.maxQuantPenalty,
            o.dbl("performance_max_utilization") ?: d.performanceMaxUtilization, o.dbl("performance_min_tps") ?: d.performanceMinTps,
            o.dbl("balanced_max_utilization") ?: d.balancedMaxUtilization, o.dbl("balanced_min_tps") ?: d.balancedMinTps,
            (o.lng("balanced_min_context") ?: d.balancedMinContext.toLong()).toInt(),
            o.dbl("max_quality_max_utilization") ?: d.maxQualityMaxUtilization,
        )
    }
}
