package com.hotatticgames.llmtrainer.qualify

/** Human-readable reasoning for the UI. Presentation only: no decisions are made here. */
object Explain {
    private fun mb(x: Double): String = if (x >= 1024) PyFormat.fixed(x / 1024.0, 1) + " GB" else PyFormat.fixed(x, 0) + " MB"

    fun utilizationLimit(profile: String, p: SafetyPolicy): Double = when (profile) {
        "performance" -> p.performanceMaxUtilization
        "balanced" -> p.balancedMaxUtilization
        else -> p.maxQualityMaxUtilization
    }

    /** Memory budget breakdown of the device (before any model). */
    fun budgetLines(profile: DeviceProfile, policy: SafetyPolicy): List<String> {
        val (budget, reserve) = Qualifier.ramBudgetMb(profile, policy)
        val lines = ArrayList<String>()
        lines.add("Total RAM ${mb(profile.totalRamMb.toDouble())}")
        lines.add("- OS reserve ${mb(profile.osReserveMb.toDouble())}, background apps ${mb(profile.backgroundReserveMb.toDouble())}")
        val typ = profile.typicalAvailableRamMb
        if (typ != null) lines.add("- capped by currently available RAM above the kill threshold: ${mb(typ.toDouble())}")
        lines.add("- this app ${mb(profile.hostAppMb.toDouble())}, safety reserve ${mb(reserve)}")
        lines.add("= model-side budget ${mb(maxOf(budget, 0.0))}" + if (budget <= 0) " (nothing left for a model)" else "")
        val storageReserve = Qualifier.storageReserveMb(profile, policy)
        lines.add("Storage free ${mb(profile.storageFreeMb.toDouble())}, reserved ${mb(storageReserve)}")
        return lines
    }

    /** RAM breakdown of the chosen configuration. */
    fun pickLines(a: CandidateAssessment): List<String> {
        val r = a.ram
        val lines = ArrayList<String>()
        lines.add("weights ${mb(r.weightsMb)} + runtime ${mb(r.runtimeMb)} + KV cache ${mb(r.kvCacheMb)} + retrieval index ${mb(r.retrievalMb)}")
        if (!r.measuredOverride) {
            lines.add("estimates inflated x${PyFormat.repr(r.estimateInflation)} (weights x${PyFormat.repr(r.weightsInflation)}) = ${mb(r.requiredModelSideMb)}")
        } else {
            lines.add("measured on this device: ${mb(r.requiredModelSideMb)}")
        }
        lines.add("headroom ${mb(r.headroomMb)} of budget ${mb(r.ramBudgetMb)} (${PyFormat.percent0(r.utilization)} used)")
        return lines
    }

    /** For every model larger than the pick: why its best-fitting configuration was not chosen. */
    fun exclusionLines(assessed: List<CandidateAssessment>, pick: ProfilePick, policy: SafetyPolicy): List<String> {
        val chosen = pick.config ?: return emptyList()
        val limit = utilizationLimit(pick.profile, policy)
        val byModel = LinkedHashMap<String, MutableList<CandidateAssessment>>()
        for (a in assessed) if (a.config.model.paramsB > chosen.model.paramsB) byModel.getOrPut(a.config.model.modelId) { ArrayList() }.add(a)
        val out = ArrayList<String>()
        for ((id, list) in byModel) {
            val closest = list.minByOrNull { it.ram.requiredModelSideMb } ?: continue
            val why = when {
                !closest.eligible -> closest.blockingReasons.firstOrNull() ?: "not eligible"
                closest.ram.utilization > limit ->
                    "fits only at ${PyFormat.percent0(closest.ram.utilization)} of the RAM budget; this profile allows ${PyFormat.percent0(limit)}"
                else -> "ranks below the chosen configuration for this profile"
            }
            out.add("$id: $why")
        }
        return out
    }
}
