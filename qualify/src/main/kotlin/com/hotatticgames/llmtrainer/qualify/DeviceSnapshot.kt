package com.hotatticgames.llmtrainer.qualify

/**
 * Host `deviceSnapshotJson` -> DeviceProfile. Mirrors `profile_from_snapshot` in factory/tools/gen_golden_device.py.
 * Required: totalRamBytes, freeStorageBytes, totalStorageBytes (else qualification is withheld).
 * Everything else is optional; unknown fields are ignored (the host adds fields over time).
 * Snapshot-derived profiles carry NO measurements, so they can never reach the "recommended" tier.
 */
class SnapshotProfile(val profile: DeviceProfile?, val notes: List<String>, val withheld: List<String>)

class QualificationResult(
    val profile: DeviceProfile?, val notes: List<String>, val withheld: List<String>,
    val picks: List<ProfilePick>, val assessed: List<CandidateAssessment>, val policy: SafetyPolicy,
)

object DeviceSnapshot {
    private const val MIB = 1048576L

    /** Conservative class prior (GB/s) when the snapshot carries no bandwidth estimate. Heuristic, LOW confidence. */
    private fun bandwidthPrior(totalMb: Long): Double = when {
        totalMb >= 12000 -> 50.0
        totalMb >= 8000 -> 35.0
        totalMb >= 6000 -> 25.0
        else -> 15.0
    }
    private val THERMAL_NAMES = listOf("NONE", "LIGHT", "MODERATE", "SEVERE", "CRITICAL", "EMERGENCY", "SHUTDOWN")
    val WITHHOLD_TEXT: Map<String, String> = mapOf(
        "missing_core_fields" to "Qualification withheld: the device snapshot lacks total RAM or storage figures.",
        "thermal_throttled" to "Qualification withheld: device is thermally throttled (thermal status SEVERE or worse). Retry when the device has cooled.",
    )

    // (max total RAM MB or null = unbounded, os reserve, background reserve, host app)
    private class Row(val bound: Long?, val os: Long, val bg: Long, val host: Long)
    private val TABLE = listOf(
        Row(4096, 1800, 800, 400), Row(6144, 2200, 1000, 450), Row(8192, 2600, 1200, 500), Row(12288, 3000, 1500, 600),
        Row(null, 3500, 1800, 700),
    )

    private fun num(o: Map<String, Any?>, k: String): Double? = o.dbl(k)

    private fun thermalIndex(v: Any?): Int? = when (v) {
        is Long -> if (v in 0..6) v.toInt() else null
        is Double -> if (!v.isNaN() && v.toLong() in 0..6) v.toLong().toInt() else null
        is String -> THERMAL_NAMES.indexOf(v.uppercase()).takeIf { it >= 0 }
        else -> null
    }

    fun profileFromSnapshot(json: String): SnapshotProfile {
        val o = try { MiniJson.parse(json).asObj() } catch (e: JsonException) { null }
        return if (o == null) SnapshotProfile(null, emptyList(), listOf("missing_core_fields")) else profileFromSnapshot(o)
    }

    fun profileFromSnapshot(s: Map<String, Any?>): SnapshotProfile {
        val notes = ArrayList<String>()
        val withhold = ArrayList<String>()
        val totalB = num(s, "totalRamBytes")
        val freeB = num(s, "freeStorageBytes")
        val totStB = num(s, "totalStorageBytes")
        if (totalB == null || freeB == null || totStB == null || totalB <= 0) {
            withhold.add("missing_core_fields")
            return SnapshotProfile(null, notes, withhold)
        }
        val totalMb = Math.floorDiv(totalB.toLong(), MIB)
        val row = TABLE.first { it.bound == null || totalMb <= it.bound }
        var os = row.os
        var bg = row.bg
        var host = row.host
        if (s.bool("isLowRamDevice") == true) {
            os = os * 6 / 5
            bg = bg * 6 / 5
            notes.add("low_ram_device")
        }
        val mc = num(s, "memoryClassMb")
        if (mc != null && mc > host) host = mc.toLong()
        val availB = num(s, "availRamBytes")
        val procB = num(s, "procMemAvailableBytes")
        val typical: Long?
        if (s.bool("lowMemory") == true) {
            typical = 0
            notes.add("low_memory")
        } else if (availB == null && procB == null) {
            typical = totalMb / 2
            notes.add("avail_ram_unknown")
        } else if (procB != null) {
            // Kernel MemAvailable counts reclaimable cache: the right "could a new app get this" number.
            val thr = Math.floorDiv((num(s, "lowMemoryThresholdBytes") ?: 0.0).toLong(), MIB)
            typical = maxOf(0L, Math.floorDiv(maxOf(procB.toLong(), (availB ?: 0.0).toLong()), MIB) - thr)
        } else {
            // availMem alone is transient on modern Android (cached apps are reclaimable): do not cap a capable
            // phone to its momentary free memory. The low-memory STATE is handled above.
            typical = null
            notes.add("avail_ram_transient_not_capped")
        }
        val bwRaw = num(s, "memBandwidthGBpsEstimate")
        var bw: Double? = if (bwRaw != null && bwRaw > 0) bwRaw else null
        if (bw == null) {
            bw = bandwidthPrior(totalMb)
            notes.add("bandwidth_prior_by_class")
        }
        val t = thermalIndex(s["thermalStatus"])
        var factor = 1.0
        if (t == null) {
            notes.add("thermal_unknown")
        } else if (t >= 3) {
            withhold.add("thermal_throttled")
        } else if (t == 2) {
            factor *= 0.8
            notes.add("thermal_moderate")
        }
        if (s.bool("powerSaveMode") == true) {
            factor *= 0.8
            notes.add("power_save")
        }
        if (factor < 1.0) bw = bw!! * factor
        val pct = num(s, "batteryPct")
        if (pct != null && pct <= 15) notes.add("battery_low")
        val maker = s.str("manufacturer") ?: ""
        val model = s.str("model") ?: ""
        val id = "$maker-$model".trim('-').ifEmpty { "unknown" }
        val prof = DeviceProfile(
            id, "$maker $model".trim().ifEmpty { "unknown" }, "probed", s.str("socModel"), "android", totalMb, typical, os, bg, host,
            Math.floorDiv(totStB.toLong(), MIB), Math.floorDiv(freeB.toLong(), MIB), bw,
        )
        return SnapshotProfile(prof, notes, withhold)
    }

    /** Snapshot + candidates -> picks. A withheld snapshot yields tier "none" for every profile. */
    fun qualify(
        snapshotJson: String, candidates: List<CandidateConfig>, policy: SafetyPolicy = SafetyPolicy(),
    ): QualificationResult {
        val sp = profileFromSnapshot(snapshotJson)
        val prof = sp.profile
        if (sp.withheld.isNotEmpty() || prof == null) {
            val text = WITHHOLD_TEXT.getValue(sp.withheld.first())
            val picks = Qualifier.PROFILE_LABELS.map { (k, label) -> ProfilePick(k, label, null, Tier.NONE, false, null, text) }
            return QualificationResult(prof, sp.notes, sp.withheld, picks, emptyList(), policy)
        }
        val rec = Qualifier.recommend(prof, candidates, policy)
        return QualificationResult(prof, sp.notes, sp.withheld, rec.picks, rec.assessed, policy)
    }
}
