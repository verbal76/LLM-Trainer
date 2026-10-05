package com.hotatticgames.llmtrainer.qualify

/** Data types mirroring factory/llmtrainer/schemas.py (device-qualification subset). Verdict/tier values are the Python strings. */

object Verdict { const val FIT = "fit"; const val NO_FIT = "no_fit"; const val UNVERIFIED = "unverified" }
object Confidence { const val MEASURED = "measured"; const val PARTIAL = "partial"; const val ESTIMATED = "estimated" }
object Tier { const val RECOMMENDED = "recommended"; const val PROVISIONAL = "provisional"; const val NONE = "none" }

data class ModelSpec(
    val modelId: String, val paramsB: Double, val layers: Int, val kvHeads: Int, val headDim: Int,
    val qualityScore: Double? = null, val maxContext: Int = 8192,
)

data class QuantSpec(val name: String, val bitsPerWeight: Double, val qualityPenalty: Double)

data class RuntimeSpec(val name: String, val overheadMb: Double, val scratchMbPer1kCtx: Double = 0.0, val kvDtype: String = "f16")

data class RetrievalBudget(val indexRamMb: Double = 0.0, val indexStorageMb: Double = 0.0, val topK: Int = 0)

data class CandidateConfig(
    val configId: String, val model: ModelSpec, val quant: QuantSpec, val contextTokens: Int,
    val runtime: RuntimeSpec, val retrieval: RetrievalBudget = RetrievalBudget(), val weightsFileMb: Double? = null,
)

/** On-device benchmark results for ONE candidate config; null = not measured. */
data class Measurement(
    val configId: String, val ttftMs: Double? = null, val tokensPerS: Double? = null, val peakRamMb: Double? = null,
    val sustainedRamMb: Double? = null, val thermalThrottleRatio: Double? = null, val uiJankPct: Double? = null,
    val crashes: Long? = null, val anrs: Long? = null, val backgroundKills: Long? = null, val sustainedMinutes: Double? = null,
)

data class DeviceProfile(
    val deviceId: String, val name: String, val deviceClass: String, val soc: String?, val os: String,
    val totalRamMb: Long, val typicalAvailableRamMb: Long?, val osReserveMb: Long, val backgroundReserveMb: Long,
    val hostAppMb: Long, val storageTotalMb: Long, val storageFreeMb: Long, val memBandwidthGbps: Double?,
    val measurements: List<Measurement> = emptyList(),
)

data class SafetyPolicy(
    val safetyReserveFrac: Double = 0.08,
    val safetyReserveMinMb: Double = 256.0,
    val storageReserveFrac: Double = 0.10,
    val storageReserveMinMb: Double = 2048.0,
    val estimateInflation: Double = 1.25,
    val weightsInflation: Double = 1.05,
    val fallbackBandwidthGbps: Double = 25.0,
    val bandwidthEfficiency: Double = 0.4,
    val estimateThermalDerate: Double = 0.7,
    val minTps: Double = 3.0,
    val maxTtftMs: Double = 8000.0,
    val minThermalRatio: Double = 0.7,
    val maxUiJankPct: Double = 5.0,
    val minSustainedMinutes: Double = 10.0,
    val maxQuantPenalty: Double = 3.0,
    val performanceMaxUtilization: Double = 0.6,
    val performanceMinTps: Double = 10.0,
    val balancedMaxUtilization: Double = 0.92,
    val balancedMinTps: Double = 4.0,
    val balancedMinContext: Int = 2048,
    val maxQualityMaxUtilization: Double = 1.0,
)

data class RamBreakdown(
    val weightsMb: Double, val runtimeMb: Double, val kvCacheMb: Double, val retrievalMb: Double,
    val estimateInflation: Double, val weightsInflation: Double, val requiredModelSideMb: Double, val measuredOverride: Boolean,
    val totalRamMb: Long, val osReserveMb: Long, val backgroundReserveMb: Long, val hostAppMb: Long,
    val safetyReserveMb: Double, val ramBudgetMb: Double, val headroomMb: Double, val utilization: Double,
)

data class StorageBreakdown(
    val modelFileMb: Double, val retrievalMb: Double, val requiredMb: Double, val freeMb: Long,
    val reserveMb: Double, val availableAfterReserveMb: Double, val headroomMb: Double,
)

data class SustainedAssessment(
    val verdict: String, val basis: String, val estTokensPerS: Double, val estTtftMs: Long,
    val tokensPerS: Double?, val ttftMs: Double?, val reasons: List<String>, val missingMeasurements: List<String>,
)

data class CandidateAssessment(
    val config: CandidateConfig, val storageVerdict: String, val ramVerdict: String, val sustainedVerdict: String,
    val storage: StorageBreakdown, val ram: RamBreakdown, val sustained: SustainedAssessment, val confidence: String,
    val qualityScore: Double, val safeToDeploy: Boolean, val eligible: Boolean, val blockingReasons: List<String>,
)

data class ProfilePick(
    val profile: String, val label: String, val config: CandidateConfig?, val tier: String, val recommended: Boolean,
    val confidence: String?, val reason: String, val warnings: List<String> = emptyList(),
    val verifiedFallbackConfigId: String? = null,
)

data class Recommendation(val policy: SafetyPolicy, val picks: List<ProfilePick>, val assessed: List<CandidateAssessment>)
