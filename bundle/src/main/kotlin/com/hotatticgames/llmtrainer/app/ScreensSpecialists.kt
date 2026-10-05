package com.hotatticgames.llmtrainer.app

import android.widget.LinearLayout
import com.hotatticgames.llmtrainer.studio.api.DeviceProfile
import com.hotatticgames.llmtrainer.studio.api.EngineStatus
import com.hotatticgames.llmtrainer.studio.api.ModelChoices
import com.hotatticgames.llmtrainer.studio.api.ProjectId
import com.hotatticgames.llmtrainer.studio.api.SpecialistInfo
import com.hotatticgames.llmtrainer.studio.api.TrainingMethodKind
import org.json.JSONObject
import java.util.Date
import java.util.Locale

// ---------------------------------------------------------------------------------------------------------
// Specialists on this phone: select, verify, export the patch, delete
// ---------------------------------------------------------------------------------------------------------

private fun methodText(k: TrainingMethodKind) = when (k) {
    TrainingMethodKind.LOCAL_FULL -> "all layers tuned"
    TrainingMethodKind.LOCAL_PARTIAL -> "last layers tuned"
    TrainingMethodKind.LOCAL_LORA -> "LoRA adapter"
    else -> k.name.lowercase()
}

internal fun specialistsScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    if (pid == null) { missing(c, col, "project"); return }
    val project = ProjectId(pid)
    col.addView(u.tv("A specialist is a patch on top of ONE exact base model file (its hash is recorded). The patch can be exported and verified; it never contains the base model, " +
        "which stays under its own license.", 13f, u.muted, topDp = 4))
    val list = u.col()
    val exported = u.col()
    col.addView(list); col.addView(exported)

    fun load() {
        c.bg({ c.studio.specialists(project) }) { sps ->
            list.removeAllViews()
            if (sps.isEmpty()) {
                list.addView(u.tv("No specialist on this phone yet.", 14f, u.muted, topDp = 12))
                list.addView(u.button("Train on this phone", "btn:sp-to-train") { c.go(Route(Kind.TRAIN_LOCAL, pid)) })
            }
            for (sp in sps.sortedByDescending { it.createdAt }) list.addView(specialistCard(c, project, sp, exported) { load() })
        }
    }
    c.resumeHook = { load() }
    load()
}

private fun specialistCard(c: Ctl, project: ProjectId, sp: SpecialistInfo, exported: LinearLayout, reload: () -> Unit): LinearLayout {
    val u = c.ui
    val card = u.card(10)
    card.tag = "specialist:" + sp.id
    card.addView(u.tv("${sp.name} ${sp.version}", 16f, u.ink, true))
    if (sp.selected) card.addView(u.badge("SELECTED (used for chat and A/B)", u.ok))
    card.addView(u.badge(if (sp.verified) "VERIFIED" else "NOT VERIFIED", if (sp.verified) u.ok else u.warn).also { it.tag = "sp-verified" })
    card.addView(u.tv(sp.verifyMessage, 11f, if (sp.verified) u.muted else u.warn))
    if (sp.stale) {
        card.addView(u.badge("STALE", u.bad))
        card.addView(u.tv(sp.staleReason ?: "Trained on material that has since changed.", 12f, u.bad))
    }
    card.addView(u.badge(if (sp.locallyEvaluated) "EVALUATED ON THIS PHONE" else "NOT EVALUATED YET", if (sp.locallyEvaluated) u.info else u.muted))
    card.addView(u.tv(sp.statement, 12f, u.ink, topDp = 4))
    card.addView(u.tv("${methodText(sp.method)} - ${sp.steps} steps - ${sp.examples} examples - epochs ${sp.config.epochs}, lr ${sp.config.learningRate}" +
        (if (sp.method == TrainingMethodKind.LOCAL_LORA) ", rank ${if (sp.config.loraRank > 0) sp.config.loraRank else 8}" else "") + "\n" +
        "final training loss ${sp.finalTrainLoss?.let { fmtNum(it) } ?: "-"} - validation loss ${sp.finalValLoss?.let { fmtNum(it) } ?: "-"}\n" +
        "patch ${fmtBytes(sp.patchSizeBytes)} sha256 ${shortHash(sp.patchSha256)}\nbase sha256 ${shortHash(sp.baseSha256)} - dataset ${shortHash(sp.datasetSha256)}\n" +
        "${sp.sourceIds.size} source(s) - created ${Date(sp.createdAt)}", 11f, u.muted, topDp = 4))
    if (sp.selected) card.addView(u.button("Use the base model instead", "btn:sp-deselect:" + sp.id, false) {
        c.call({ c.studio.selectSpecialist(project, null) }) { reload() }
    }) else card.addView(u.button("Select this specialist", "btn:sp-select:" + sp.id) {
        c.call({ c.studio.selectSpecialist(project, sp.id) }) { reload() }
    })
    card.addView(u.button("Verify (re-hash patch and base, reload)", "btn:sp-verify:" + sp.id, false) {
        c.long({ c.studio.verifySpecialist(sp.id) }) { r -> c.handle(r) { v -> if (v.verified) c.success("Verified.") else c.notice("Not verified: ${v.verifyMessage}"); reload() } }
    })
    card.addView(u.button("Export patch (zip, without the base model)", "btn:sp-export:" + sp.id, false) {
        c.runExport("${slug(sp.name)}-${sp.version}.patch.zip", "application/zip", { s, o -> s.exportSpecialistPatch(sp.id, o) }) { p, uri ->
            showExported(c, p, uri, exported); c.success("Specialist patch exported.")
        }
    })
    card.addView(u.tv("Export refuses when the base model's license gate fails (fail closed).", 10f, u.muted))
    card.addView(u.button("Evaluate on held-out text", "btn:sp-eval:" + sp.id, false) { c.go(Route(Kind.EVAL, project.value)) })
    card.addView(u.button("Compare with the base model", "btn:sp-ab:" + sp.id, false) { c.go(Route(Kind.AB, project.value)) })
    card.addView(u.button("Delete this specialist", "btn:sp-delete:" + sp.id, false) {
        c.confirm("Delete ${sp.name} ${sp.version}?", "Removes the patch from this phone. The base model and your sources are not touched. Export it first if you may want it again.", "Delete") {
            c.call({ c.studio.deleteSpecialist(sp.id) }, sticky = true) { reload() }
        }
    })
    return card
}

// ---------------------------------------------------------------------------------------------------------
// About: identity block + an honest "what can this phone do" summary
// ---------------------------------------------------------------------------------------------------------

private class AboutData(val diag: JSONObject?, val engine: EngineStatus, val device: DeviceProfile, val models: ModelChoices?, val snap: JSONObject?, val hostLevel: Int, val nativeVersion: String, val buildSha: String)

private fun flag(u: Ui, card: LinearLayout, ok: Boolean?, title: String, detail: String) {
    card.addView(u.badge(when (ok) { true -> "YES"; false -> "NO"; null -> "LIMITED" }, when (ok) { true -> u.ok; false -> u.bad; null -> u.warn }))
    card.addView(u.tv(title, 14f, u.ink, true))
    card.addView(u.tv(detail, 12f, u.muted))
}

internal fun aboutScreen(c: Ctl, col: LinearLayout) {
    val u = c.ui
    val box = u.col()
    col.addView(box)
    c.bg({
        val diag = try { JSONObject(c.host.diagnosticsJson()) } catch (_: Throwable) { null }
        val snap = try { JSONObject(c.host.deviceSnapshotJson()) } catch (_: Throwable) { null }
        val level = try { c.host.hostApiLevel } catch (_: Throwable) { 0 }
        val nat = try { if (level >= 2) c.host.nativeVersion else "n/a (host API level $level)" } catch (_: Throwable) { "unknown" }
        val sha = try { if (level >= 2) c.host.buildSha else "" } catch (_: Throwable) { "" }
        val models = try { c.models()?.modelChoices() } catch (_: Throwable) { null }
        AboutData(diag, c.studio.engineStatus(), c.studio.deviceProfile(), models, snap, level, nat, sha)
    }) { d ->
        val id = d.diag?.optJSONObject("identity")
        val ic = u.card(8)
        ic.tag = "identity-block"
        ic.addView(u.section("IDENTITY"))
        val block = d.diag?.optString("identityBlock", "") ?: ""
        ic.addView(u.mono(if (block.isNotEmpty()) block else "Identity unavailable: the host did not provide diagnostics.", 12f, u.ink).also { it.tag = "identity-text" })
        ic.addView(u.tv(
            "Native version: ${id?.optString("nativeVersion", "?") ?: "?"} (the APK generation)\n" +
                "App version: ${id?.optString("appVersion", "?") ?: "?"} (the OTA-updatable layer)\n" +
                "OTA sequence: ${if (id != null && id.optString("runningSource") != "none") "#" + id.optInt("otaSequence") else "-"} (engineering counter)\n" +
                "Runtime id: ${id?.optString("nativeRuntimeId", "?") ?: d.nativeVersion}\n" +
                "ABI: ${id?.optInt("nativeAbi", -1) ?: -1}   host API level: ${d.hostLevel}\n" +
                "Git sha: ${id?.optString("sourceSha", "?") ?: d.buildSha}\n" +
                "Engine status: ${d.diag?.optString("engineStatus", "") ?: ""}".trimEnd(), 12f, u.muted, topDp = 6).also { it.tag = "identity-fields" })
        box.addView(ic)

        val eng = d.engine
        val sc = u.card(10)
        sc.tag = "phone-summary"
        sc.addView(u.section("WHAT CAN THIS PHONE DO"))
        flag(u, sc, eng.inferenceAvailable, "Run a language model here", if (eng.inferenceAvailable) "The native engine is ready (${eng.runtimeId}). A model file must be installed first; the model manager shows what fits." else (eng.inferenceReason ?: "The native engine is not available."))
        val m = d.models
        if (m != null) {
            val cs = m.catalog
            flag(u, sc, if (cs.downloadableCount > 0) true else null, "Get a model", "${m.artifacts.count { it.installed }} installed, ${cs.downloadableCount} of ${cs.artifactCount} catalog files downloadable now" +
                (if (cs.unrefreshedCount > 0) " (${cs.unrefreshedCount} still waiting for a catalog refresh)" else "") + ". Nothing downloads without your confirmation; you can also import a GGUF file.")
            val picks = m.choices.filter { it.variantId != null }
            if (picks.isNotEmpty()) sc.addView(u.tv("Picks: " + picks.joinToString("; ") { "${it.choice.replace('_', ' ')} = ${it.label} [${it.tier}]" }, 11f, u.muted))
        }
        val charging = d.snap?.let { if (it.has("charging")) it.optBoolean("charging") else null }
        val battery = d.snap?.let { if (it.has("batteryPct")) it.optInt("batteryPct") else null }
        val ready = m?.artifacts?.any { it.readyToTrainNow } == true
        flag(u, sc, if (!eng.trainingAvailable) false else if (ready) true else null, "Specialize a model on this phone",
            if (!eng.trainingAvailable) (eng.trainingReason ?: "The native engine is not available.")
            else "LoRA adapters work on quantized files; weight tuning needs full-precision files. Needs the charger (${when (charging) { true -> "connected"; false -> "NOT connected"; null -> "state not reported" }}), " +
                "battery >= 20% (${battery?.let { "$it%" } ?: "unknown"}), a cool phone and enough free memory. ${if (ready) "At least one file is ready to train right now." else "No file is ready right now."} " +
                "Time, not memory, is the limit: engine measurements on a PC-class CPU suggest models up to about 0.5B parameters train in minutes to an hour, 1-1.5B takes hours, and 3B or more is not practical (phones are slower).")
        flag(u, sc, eng.inferenceAvailable, "Check a specialist against its base model", "Held-out TEST questions, identical decoding, per-metric intervals and a retention check; an improvement claim is only allowed when the evidence supports it.")
        flag(u, sc, true, "Prepare a desktop training job (optional)", "Always available as a fallback you choose; the phone never reroutes work silently.")
        val dv = d.device
        sc.addView(u.tv("Device: ${dv.deviceName}, Android API ${dv.androidApi}, ${dv.abi}. RAM ${fmtMb(dv.availableRamMb.toLong())} available of ${fmtMb(dv.totalRamMb.toLong())}, free storage ${fmtMb(dv.freeStorageMb)}, safety reserve ${(dv.safetyReserveFraction * 100).toInt()}%.", 12f, u.ink, topDp = 8))
        sc.addView(u.tv("All speeds, memory figures and durations shown in this app are ESTIMATES until a real benchmark has been recorded on this phone. " +
            String.format(Locale.US, "Nothing here is a measurement of model quality."), 11f, u.warn, topDp = 4))
        box.addView(sc)

        box.addView(u.button("Model manager", "btn:about-models") { c.go(Route(Kind.MODELS)) })
        box.addView(u.button("Updates & diagnostics", "btn:about-updates", false) { c.go(Route(Kind.UPDATES)) })
        box.addView(u.button("What this app does and does not do", "btn:about-firstrun", false) { c.go(Route(Kind.FIRSTRUN)) })
    }
}
