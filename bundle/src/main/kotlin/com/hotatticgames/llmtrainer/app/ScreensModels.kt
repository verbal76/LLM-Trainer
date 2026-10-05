package com.hotatticgames.llmtrainer.app

import android.widget.LinearLayout
import com.hotatticgames.llmtrainer.studio.api.ArtifactCapabilityView
import com.hotatticgames.llmtrainer.studio.api.ChoiceView
import com.hotatticgames.llmtrainer.studio.api.InstalledModel
import com.hotatticgames.llmtrainer.studio.api.IntendedUse
import com.hotatticgames.llmtrainer.studio.api.LicenseState
import com.hotatticgames.llmtrainer.studio.api.ModelChoices
import com.hotatticgames.llmtrainer.studio.api.Operation
import com.hotatticgames.llmtrainer.studio.api.OperationState
import com.hotatticgames.llmtrainer.studio.api.ProjectId
import com.hotatticgames.llmtrainer.studio.api.StorageAccounting
import com.hotatticgames.llmtrainer.studio.api.StudioError
import com.hotatticgames.llmtrainer.studio.api.StudioResult
import java.util.Date
import java.util.Locale

// ---------------------------------------------------------------------------------------------------------
// Model manager: capability classes, picks, download (explicit confirmation), manual GGUF import, storage, license state
// ---------------------------------------------------------------------------------------------------------

internal fun tierText(t: String) = when (t) {
    "recommended" -> "RECOMMENDED (measured on this phone)"
    "provisional" -> "PROVISIONAL (estimate, not measured)"
    "preview" -> "PREVIEW (file not downloadable yet)"
    else -> "NO PICK"
}

private fun tierColor(u: Ui, t: String) = when (t) { "recommended" -> u.ok; "provisional" -> u.warn; "preview" -> u.info; else -> u.muted }

private fun choiceTitle(id: String) = when (id) {
    "fastest" -> "Fastest"
    "balanced" -> "Balanced"
    "best_quality" -> "Best quality"
    "best_specialize" -> "Best to specialize on this phone"
    else -> id
}

private fun provenanceText(p: String) = when (p) {
    "verified_download" -> "downloaded, size and SHA-256 verified against the catalog"
    "download_unverified_checksum" -> "downloaded; the catalog has no checksum for it"
    "verified_import" -> "imported and matched to a catalog checksum"
    "unverified_provenance" -> "imported by you: origin and license are NOT verified"
    else -> p
}

private class ModelsData(val choices: ModelChoices, val storage: StorageAccounting, val installed: List<InstalledModel>, val ops: List<Operation>)

private fun activeOp(ops: List<Operation>, variantId: String) = ops.firstOrNull { !it.isTerminal && it.subject == variantId }

internal fun modelsScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    val intro = u.card(8)
    intro.addView(u.tv("Nothing downloads automatically. Every download shows its size, storage impact, device fit and license state, and needs your confirmation. " +
        "A file is only installed after its size and SHA-256 match the catalog; anything else is deleted. Downloads only progress while this app is open and can be resumed.", 13f, u.muted))
    col.addView(intro)
    val opsBox = u.col()
    val dyn = u.col()
    col.addView(opsBox)
    col.addView(dyn)
    val shownActive = HashSet<String>()

    fun load() {
        c.bg({
            val m = c.models()
            if (m == null) null else ModelsData(m.modelChoices(), m.storage(), m.installedModels(), c.studio.operations())
        }) { d -> if (d == null) dyn.addView(u.tv("This Studio implementation has no model manager.", 14f, u.bad, topDp = 12)) else fill(c, pid, d, dyn) }
    }

    fun refreshOps() {
        val ep = c.epochNow()
        c.bg({ c.studio.operations() }) { ops: List<Operation> ->
            val active = ops.filter { !it.isTerminal }
            val now = active.map { it.id }.toSet()
            val finished = shownActive.any { it !in now }
            shownActive.clear(); shownActive.addAll(now)
            opsBox.removeAllViews()
            if (active.isNotEmpty()) opsBox.addView(u.section("OPERATIONS"))
            for (op in active) opsBox.addView(opCard(c, op) { refreshOps() })
            if (finished) load()                                   // a download/import ended: installed list and cards changed
            if (active.any { it.state == OperationState.RUNNING || it.state == OperationState.QUEUED }) c.postDelayed(1500, ep) { refreshOps() }
        }
    }

    c.resumeHook = { load(); refreshOps() }
    load()
    refreshOps()
}

private fun opCard(c: Ctl, op: Operation, after: () -> Unit): LinearLayout {
    val u = c.ui
    val oc = u.card(8)
    oc.tag = "op:" + op.id
    oc.addView(u.tv("${op.kind}: ${op.subject}", 14f, u.ink, true))
    oc.addView(u.tv("${op.state} - ${op.message}", 12f, u.muted))
    val p = op.progress
    oc.addView(u.bar(p.fraction, "progress:" + op.id))
    if (p.total > 0) oc.addView(u.tv("${fmtBytes(p.done)} of ${fmtBytes(p.total)}", 11f, u.muted))
    op.error?.let { oc.addView(u.tv(c.describe(it), 12f, u.bad, topDp = 4)) }
    if (!op.isTerminal && op.state != OperationState.PAUSED) {
        oc.addView(u.button("Cancel", "btn:op-cancel:" + op.id, false) { c.call({ c.studio.cancelOperation(op.id) }) { after() } })
    }
    if (op.state == OperationState.PAUSED && op.resumable) {
        oc.addView(u.button("Resume", "btn:op-resume:" + op.id) { c.call({ c.studio.resumeOperation(op.id) }) { after() } })
    }
    return oc
}

private fun fill(c: Ctl, pid: String?, d: ModelsData, dyn: LinearLayout) {
    val u = c.ui
    dyn.removeAllViews()
    val st = d.storage

    // ---- storage -----------------------------------------------------------------------------------------
    val sc = u.card(8)
    sc.tag = "models-storage"
    sc.addView(u.section("STORAGE"))
    sc.addView(u.tv("Free ${fmtBytes(st.freeBytes)} - reserve kept ${fmtBytes(st.reserveBytes)} - usable for models ${fmtBytes(maxOf(0L, st.headroomBytes))}\n" +
        "Installed models ${fmtBytes(st.installedBytes)} - partial downloads ${fmtBytes(st.partialBytes)}", 13f, u.ink, topDp = 4))
    dyn.addView(sc)

    // ---- device + catalog state ----------------------------------------------------------------------------
    val dv = d.choices.device
    val cs = d.choices.catalog
    val dc = u.card(8)
    dc.addView(u.section("THIS PHONE"))
    dc.addView(u.tv("${dv.deviceName} - RAM ${fmtMb(dv.availableRamMb.toLong())} available of ${fmtMb(dv.totalRamMb.toLong())} - safety reserve ${(dv.safetyReserveFraction * 100).toInt()}%", 13f, u.ink, topDp = 4))
    dc.addView(u.tv("Catalog: ${cs.artifactCount} files, ${cs.downloadableCount} downloadable now, ${cs.unrefreshedCount} waiting for a catalog refresh" +
        (cs.lastRefreshedAt?.let { " (last refreshed $it)" } ?: "") + "\n" + cs.note, 12f, u.muted, topDp = 4))
    if (cs.unrefreshedCount > 0) dc.addView(u.tv("A file that is not refreshed has no exact size, hash or revision yet, so the app will not download it. You can still import a GGUF file you already have.", 12f, u.warn, topDp = 4))
    dyn.addView(dc)

    // ---- picks ---------------------------------------------------------------------------------------------
    dyn.addView(u.section("PICKS FOR THIS PHONE"))
    dyn.addView(u.tv("Estimates are labelled with their confidence and stay provisional until a real benchmark exists on this phone.", 12f, u.muted))
    for (ch in d.choices.choices) dyn.addView(pickCard(c, pid, ch, d))

    // ---- import --------------------------------------------------------------------------------------------
    val ic = u.card(10)
    ic.tag = "models-import"
    ic.addView(u.section("IMPORT YOUR OWN GGUF FILE"))
    ic.addView(u.tv("Use a model file you already have. It is copied into app storage, hashed, and recorded with UNVERIFIED provenance and license: you stay responsible for the rights to use it.", 12f, u.muted, topDp = 4))
    ic.addView(u.button("Choose a .gguf file", "btn:models-import") { importUserModel(c) })
    dyn.addView(ic)

    // ---- installed -------------------------------------------------------------------------------------------
    dyn.addView(u.section("INSTALLED ON THIS PHONE (${d.installed.size})"))
    if (d.installed.isEmpty()) dyn.addView(u.tv("No model is installed yet. Chat and training need one.", 13f, u.muted))
    for (m in d.installed) {
        val mc = u.card(8)
        mc.tag = "installed:" + m.variantId
        mc.addView(u.tv(m.displayName, 15f, u.ink, true))
        mc.addView(u.tv("${fmtBytes(m.sizeBytes)} - ${provenanceText(m.provenance)}\nsha256 ${shortHash(m.sha256)} - installed ${Date(m.installedAt)}", 12f, u.muted))
        mc.addView(u.badge(u.licenseText(m.licenseStateAtInstall).replace("LICENSE", "LICENSE AT INSTALL"), u.licenseColor(m.licenseStateAtInstall)))
        mc.addView(u.button("Remove from this phone", "btn:uninstall:" + m.variantId, false) {
            c.confirm("Remove ${m.displayName}?", "Deletes the file (${fmtBytes(m.sizeBytes)}) from this phone. Specialists trained on it stay, but cannot run until it is installed again.", "Remove") {
                c.call({ c.models()?.uninstall(m.variantId) ?: StudioResult.Err(StudioError.Invalid("NOT_SUPPORTED", "No model manager")) }, sticky = true) { c.success("Removed."); c.render() }
            }
        })
        dyn.addView(mc)
    }

    // ---- all files -------------------------------------------------------------------------------------------
    val all = d.choices.artifacts.sortedWith(compareByDescending<ArtifactCapabilityView> { it.installed }.thenBy { it.sizeBytes ?: Long.MAX_VALUE })
    dyn.addView(u.section("ALL FILES (${all.size})"))
    for (av in all) dyn.addView(artifactCard(c, pid, av, d.ops, "artifact:"))
}

private fun pickCard(c: Ctl, pid: String?, ch: ChoiceView, d: ModelsData): LinearLayout {
    val u = c.ui
    val pc = u.card(8)
    pc.tag = "pick:" + ch.choice
    pc.addView(u.tv(choiceTitle(ch.choice), 16f, u.ink, true))
    pc.addView(u.badge(tierText(ch.tier), tierColor(u, ch.tier)))
    ch.confidence?.let { pc.addView(u.tv("Confidence: $it", 11f, u.muted)) }
    pc.addView(u.tv(ch.label, 13f, u.ink, topDp = 4))
    pc.addView(u.tv(ch.reason, 12f, u.muted))
    ch.licenseState?.let { pc.addView(u.badge(u.licenseText(it), u.licenseColor(it))) }
    val av = ch.variantId?.let { vid -> d.choices.artifacts.firstOrNull { it.variantId == vid } }
    if (av != null) addActions(c, pid, av, d.ops, pc, "pick:" + ch.choice + ":")
    return pc
}

private fun artifactCard(c: Ctl, pid: String?, av: ArtifactCapabilityView, ops: List<Operation>, tagPrefix: String): LinearLayout {
    val u = c.ui
    val ac = u.card(8)
    ac.tag = tagPrefix + av.variantId
    ac.addView(u.tv(av.label, 15f, u.ink, true))
    ac.addView(u.tv("${av.quantization} (${av.precision}) - ${av.sizeBytes?.let { fmtBytes(it) } ?: "size unknown"}${if (av.sizeIsEstimate) " (estimate)" else ""}", 12f, u.muted))
    if (av.installed) ac.addView(u.badge("INSTALLED ON THIS PHONE", u.ok))
    // capability classes (a file's class is static; the phone decides the rest)
    if (av.canInfer) ac.addView(u.badge("RUNS ON THIS PHONE (estimate)", u.ok)) else ac.addView(u.badge(if (av.canLoad) "TOO SLOW TO USE" else "TOO BIG FOR THIS PHONE", u.bad))
    when (av.trainingClass) {
        "local_full" -> ac.addView(u.badge("CAN BE FULLY TUNED ON THIS PHONE", u.info))
        "local_partial" -> ac.addView(u.badge("CAN BE PARTIALLY TUNED ON THIS PHONE (last layers)", u.info))
        "inference_only" -> ac.addView(u.badge("CHAT / RUN ONLY FOR WEIGHT TUNING (quantized)", u.muted))
        else -> ac.addView(u.badge("DESKTOP TRAINING ONLY", u.muted))
    }
    if (av.trainingClass == "inference_only") {
        ac.addView(u.tv("Quantized weights are not tuned directly" + (av.specializeViaVariantId?.let { " (full-precision file: $it)" } ?: "") +
            ". A LoRA adapter can still be trained on top of this file: the engine checks that when you plan training.", 11f, u.muted))
    }
    if (av.trainingClass == "local_full" || av.trainingClass == "local_partial") {
        if (av.readyToTrainNow) ac.addView(u.badge("READY TO TRAIN NOW", u.ok)) else ac.addView(u.badge("NOT READY TO TRAIN NOW", u.warn))
        for (cv in av.conditions) ac.addView(u.tv("${u.conditionText(cv.status)}: ${cv.text}", 11f, u.conditionColor(cv.status)))
    }
    ac.addView(u.badge(u.licenseText(av.licenseState), u.licenseColor(av.licenseState)))
    ac.addView(u.tv(
        "Context up to ${av.inferContext ?: "?"} tokens - est. peak RAM ${av.estPeakRamMb?.let { fmtMb(it.toLong()) } ?: "?"} - est. " +
            (av.estTokensPerS?.let { String.format(Locale.US, "%.1f tok/s", it) } ?: "speed unknown") + " (confidence: ${av.confidenceInfer})", 11f, u.muted, topDp = 2))
    for (opt in listOfNotNull(av.full, av.partial)) {
        ac.addView(u.tv("Tuning ${if (opt.mode == "full") "all layers" else "last ${opt.trainableLastLayers} layers"}: est. RAM ${fmtMb(opt.ramMb.toLong())}, " +
            "checkpoint ${fmtMb(opt.checkpointMb.toLong())}, ${if (opt.fits) "fits the safe envelope" else "does NOT fit the safe envelope"}${if (opt.measured) "" else " (estimate)"}", 11f, if (opt.fits) u.muted else u.warn))
    }
    for (r in av.reasons.take(4)) ac.addView(u.tv("- $r", 11f, u.muted))
    addActions(c, pid, av, ops, ac, tagPrefix)
    return ac
}

private fun addActions(c: Ctl, pid: String?, av: ArtifactCapabilityView, ops: List<Operation>, card: LinearLayout, tagPrefix: String) {
    val u = c.ui
    val op = activeOp(ops, av.variantId)
    if (av.installed) {
        if (pid != null) card.addView(u.button("Use as this specialist's base model", "btn:use-base:" + av.variantId) {
            c.call({ c.studio.selectBaseModel(ProjectId(pid), av.modelId, av.variantId) }, sticky = true) { _ -> c.success("Base model selected."); c.render() }
        })
    } else if (op != null) {
        card.addView(u.tv("Download in progress (${op.state}); see OPERATIONS above.", 12f, u.info, true, 4))
    } else if (av.downloadable) {
        card.addView(u.button("Download (${av.sizeBytes?.let { fmtBytes(it) } ?: "size unknown"})...", "btn:download:" + av.variantId) { downloadFlow(c, av) })
    } else {
        card.addView(u.badge("NOT DOWNLOADABLE", u.bad))
        card.addView(u.tv(av.downloadBlocker ?: (if (av.catalogState == "unrefreshed") "The catalog entry has not been refreshed yet: no exact size, hash or revision." else "Not available."), 12f, u.bad))
        card.addView(u.tv("You can import this file yourself with 'Choose a .gguf file'.", 11f, u.muted))
    }
    card.addView(u.button("License evidence", "btn:license:" + av.variantId, false) { c.go(Route(Kind.LICENSE, pid, av.modelId)) })
}

private fun downloadFlow(c: Ctl, av: ArtifactCapabilityView) {
    val use = IntendedUse(fineTune = true, adapter = true)
    c.call({ c.studio.planAcquisition(av.variantId, use) }) { plan ->
        if (!plan.allowed) {
            c.error("This download is blocked:\n" + plan.blocking.joinToString("\n") { "- ${it.message}" })
            return@call
        }
        val msg = "Size: ${fmtBytes(plan.sizeBytes)}\nFrom: ${plan.url ?: "(import a file instead)"}\n" +
            "Free storage now: ${fmtBytes(plan.freeStorageBytes)}; after: ${fmtBytes(plan.storageAfterBytes)} (reserve kept: ${fmtBytes(plan.storageReserveBytes)})\n" +
            "Device fit: ${plan.deviceCompat.verdict.name.lowercase().replace('_', ' ')}\n${c.ui.licenseText(plan.licenseState)}\n\n" +
            "The file is verified (size and SHA-256) before it is installed. Downloads continue while this app is open and can be paused and resumed."
        c.confirm("Download ${plan.modelName}?", msg, "Download ${fmtBytes(plan.sizeBytes)}") {
            c.call({ c.studio.startDownload(plan.variantId, use, true) }, sticky = true) { _ ->
                c.success("Download started.")
                c.render()
            }
        }
    }
}

private fun importUserModel(c: Ctl) {
    val hook = StudioTestHooks.modelFile()
    fun run(input: () -> com.hotatticgames.llmtrainer.studio.api.SourceInput) {
        c.call({
            val si = input()
            c.models()?.importUserModel(si) ?: StudioResult.Err(StudioError.Invalid("NOT_SUPPORTED", "No model manager"))
        }, sticky = true) { _ -> c.success("Import started: the file is copied and hashed in the background."); c.render() }
    }
    if (hook != null) { run { hook }; return }
    c.pick(Pickers.openOne()) { data ->
        val uri = Pickers.uris(data).firstOrNull()
        if (uri != null) { val cr = c.contentResolver(); run { Pickers.inputFrom(cr, uri) } }
    }
}
