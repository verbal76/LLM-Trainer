package com.hotatticgames.llmtrainer.app

import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import com.hotatticgames.llmtrainer.studio.api.AcquisitionPlan
import com.hotatticgames.llmtrainer.studio.api.DatasetOptions
import com.hotatticgames.llmtrainer.studio.api.DatasetPreview
import com.hotatticgames.llmtrainer.studio.api.DatasetStatus
import com.hotatticgames.llmtrainer.studio.api.EvaluationView
import com.hotatticgames.llmtrainer.studio.api.ExportedPackage
import com.hotatticgames.llmtrainer.studio.api.IngestReport
import com.hotatticgames.llmtrainer.studio.api.IngestStatus
import com.hotatticgames.llmtrainer.studio.api.IntendedUse
import com.hotatticgames.llmtrainer.studio.api.IssueCode
import com.hotatticgames.llmtrainer.studio.api.Operation
import com.hotatticgames.llmtrainer.studio.api.OperationState
import com.hotatticgames.llmtrainer.studio.api.ProjectId
import com.hotatticgames.llmtrainer.studio.api.Progress
import com.hotatticgames.llmtrainer.studio.api.ReviewFilter
import com.hotatticgames.llmtrainer.studio.api.RightsStatus
import com.hotatticgames.llmtrainer.studio.api.RunLocation
import com.hotatticgames.llmtrainer.studio.api.SourceInput
import com.hotatticgames.llmtrainer.studio.api.SourceRecord
import com.hotatticgames.llmtrainer.studio.api.SpecialistPackageView
import org.json.JSONObject
import java.util.Date

// ---------------------------------------------------------------------------------------------------------
// Acquisition
// ---------------------------------------------------------------------------------------------------------
internal fun acquireScreen(c: Ctl, pid: String?, mid: String?, vid: String?, col: LinearLayout) {
    val u = c.ui
    col.addView(u.tv("Downloads are never automatic. You see the size, storage impact, device fit and license state, then confirm. " +
        "Downloads only progress while this app is open and resume next time you open it.", 13f, u.muted, topDp = 4))
    if (vid == null) {
        if (mid == null) { missing(c, col, "model"); return }
        c.call({ c.studio.model(mid) }) { m ->
            col.addView(u.tv("Choose a variant of ${m.name}", 15f, u.ink, true, 10))
            for (v in m.variants) col.addView(u.button("${v.quant} - ${fmtBytes(v.sizeBytes)}", "btn:variant:" + v.id, false) {
                c.replaceTop(Route(Kind.ACQUIRE, pid, m.id, v.id))
            })
        }
        return
    }
    val fineTune = u.check("Intended use: fine-tune", "use:finetune", true)
    val adapter = u.check("Intended use: train adapters (LoRA)", "use:adapter", true)
    val commercial = u.check("Intended use: commercial", "use:commercial", false)
    val redistribute = u.check("Intended use: redistribute the specialist", "use:redistribute", false)
    val useCard = u.card(8)
    useCard.addView(u.tv("INTENDED USE (checked against the license)", 12f, u.accent, true))
    for (cb in listOf(fineTune, adapter, commercial, redistribute)) useCard.addView(cb)
    col.addView(useCard)
    fun use() = IntendedUse(fineTune.isChecked, adapter.isChecked, commercial.isChecked, redistribute.isChecked)

    val planBox = u.col()
    val opsBox = u.col()
    col.addView(planBox)
    col.addView(u.button("Import this model from a file", "btn:import-model", false) {
        val hook = StudioTestHooks.modelFile()
        if (hook != null) c.call({ c.studio.importModelFile(vid, hook) }, sticky = true) { _ ->
            c.success("Import started."); refreshOps(c, opsBox)
        } else c.pick(Pickers.openOne()) { data ->
            val uri = Pickers.uris(data).firstOrNull()
            if (uri != null) {
                val cr = c.contentResolver()
                c.call({ c.studio.importModelFile(vid, Pickers.inputFrom(cr, uri)) }, sticky = true) { _ ->
                    c.success("Import started."); refreshOps(c, opsBox)
                }
            }
        }
    })
    col.addView(opsBox)

    fun replan() {
        c.call({ c.studio.planAcquisition(vid, use()) }) { plan -> showPlan(c, plan, use(), planBox, opsBox) }
    }
    for (cb in listOf(fineTune, adapter, commercial, redistribute)) cb.setOnCheckedChangeListener { _, _ -> replan() }
    c.resumeHook = { refreshOps(c, opsBox) }
    replan()
    refreshOps(c, opsBox)
}

private fun showPlan(c: Ctl, plan: AcquisitionPlan, use: IntendedUse, box: LinearLayout, opsBox: LinearLayout) {
    val u = c.ui
    box.removeAllViews()
    val pc = u.card(10)
    pc.tag = "acquisition-plan"
    pc.addView(u.tv("DOWNLOAD PLAN", 12f, u.accent, true))
    pc.addView(u.tv("${plan.modelName}\nFrom: ${plan.url ?: "(no download URL; import a file)"}\nSize: ${fmtBytes(plan.sizeBytes)}\n" +
        "Free storage now: ${fmtBytes(plan.freeStorageBytes)}\nFree after: ${fmtBytes(plan.storageAfterBytes)} (reserve kept: ${fmtBytes(plan.storageReserveBytes)})", 13f, u.ink, topDp = 4))
    pc.addView(u.badge("PHONE: " + u.verdictText(plan.deviceCompat.verdict), u.verdictColor(plan.deviceCompat.verdict)))
    for (r in plan.deviceCompat.reasons) pc.addView(u.tv("- $r", 12f, u.muted))
    pc.addView(u.badge(u.licenseText(plan.licenseState), u.licenseColor(plan.licenseState)))
    if (plan.blocking.isNotEmpty()) {
        pc.addView(u.tv("Blocked:", 13f, u.bad, true, 6))
        for (b in plan.blocking) pc.addView(u.tv("- ${b.message}", 13f, u.bad))
    } else {
        pc.addView(u.tv("Allowed for the selected use.", 13f, u.ok, true, 6))
    }
    val confirm = u.check("I confirm: download ${fmtBytes(plan.sizeBytes)} over HTTPS to this device", "confirm:download")
    pc.addView(confirm)
    pc.addView(u.button("Download", "btn:download") {
        if (!plan.allowed) c.error("This download is blocked: " + plan.blocking.joinToString("; ") { it.message })
        else if (!confirm.isChecked) c.error("Tick the confirmation first. Downloads are never automatic.")
        else c.call({ c.studio.startDownload(plan.variantId, use, true) }, sticky = true) { _ ->
            c.success("Download started."); refreshOps(c, opsBox)
        }
    })
    box.addView(pc)
}

private fun refreshOps(c: Ctl, box: LinearLayout) {
    val ep = c.epochNow()
    c.bg({ c.studio.operations() }) { ops: List<Operation> ->
        val u = c.ui
        box.removeAllViews()
        if (ops.isNotEmpty()) box.addView(u.tv("OPERATIONS", 12f, u.accent, true, 12))
        for (op in ops.sortedByDescending { it.updatedAt }.take(6)) {
            val oc = u.card(8)
            oc.tag = "op:" + op.id
            oc.addView(u.tv("${op.kind}: ${op.subject}", 14f, u.ink, true))
            oc.addView(u.tv("${op.state} - ${op.message}", 12f, u.muted))
            oc.addView(u.bar(op.progress.fraction, "progress:" + op.id))
            op.error?.let { oc.addView(u.tv(c.describe(it), 12f, u.bad, topDp = 4)) }
            if (!op.isTerminal && op.state != OperationState.PAUSED) {
                oc.addView(u.button("Cancel", "btn:op-cancel:" + op.id, false) { c.call({ c.studio.cancelOperation(op.id) }) { refreshOps(c, box) } })
            }
            if (op.state == OperationState.PAUSED && op.resumable) {
                oc.addView(u.button("Resume", "btn:op-resume:" + op.id) { c.call({ c.studio.resumeOperation(op.id) }) { refreshOps(c, box) } })
            }
            box.addView(oc)
        }
        if (ops.any { it.state == OperationState.RUNNING || it.state == OperationState.QUEUED }) {
            c.postDelayed(1500, ep) { refreshOps(c, box) }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Sources & ingestion
// ---------------------------------------------------------------------------------------------------------
private fun rightsLabel(r: RightsStatus) = when (r) {
    RightsStatus.UNSET -> "Not set yet (dataset build will be blocked)"
    RightsStatus.OWNER_AUTHORED -> "I authored this material"
    RightsStatus.LICENSED_FOR_TRAINING -> "Licensed for training"
    RightsStatus.PERMISSION_GRANTED -> "Permission granted by the rights holder"
    RightsStatus.REFERENCE_ONLY -> "Reference only (retrieval, never training)"
}

private fun itemLabel(status: IngestStatus, issues: List<com.hotatticgames.llmtrainer.studio.api.IngestIssue>): String =
    if (issues.any { it.code == IssueCode.NEEDS_OCR }) "needs_ocr" else status.name.lowercase()

internal fun sourcesScreen(c: Ctl, pid: String?, @Suppress("UNUSED_PARAMETER") reportFirst: Boolean, col: LinearLayout) {
    val u = c.ui
    if (pid == null) { missing(c, col, "project"); return }
    val project = ProjectId(pid)
    col.addView(u.tv("Add only material you are authorized to use. Rights are recorded per source and the dataset cannot be built while they are unset. Removing a source removes everything derived from it.", 13f, u.muted, topDp = 4))
    var rights = RightsStatus.UNSET
    val rc = u.card(8)
    rc.addView(u.tv("RIGHTS FOR THE FILES YOU ADD NEXT", 12f, u.accent, true))
    val group = RadioGroup(c.context)
    for (r in RightsStatus.values()) {
        val rb = RadioButton(c.context)
        rb.text = rightsLabel(r); rb.setTextColor(u.ink); rb.textSize = 14f; rb.minimumHeight = u.px(48); rb.tag = "rights:" + r.name
        rb.id = View.generateViewId()
        rb.setOnClickListener { rights = r }
        group.addView(rb)
        if (r == RightsStatus.UNSET) rb.isChecked = true
    }
    rc.addView(group)
    col.addView(rc)

    val progressBox = u.col()
    val reportBox = u.col()
    val sourcesBox = u.col()

    fun runIngest(inputs: () -> List<SourceInput>) {
        progressBox.removeAllViews()
        val pb: ProgressBar = u.bar(-1.0, "progress:ingest")
        val pt: TextView = u.tv("Starting...", 12f, u.muted)
        progressBox.addView(pt); progressBox.addView(pb)
        val chosen = rights
        c.bg({
            val list = inputs()
            c.studio.ingest(project, list, chosen) { p: Progress ->
                c.post {
                    pt.text = "Ingesting ${p.done} of ${p.total} ${p.unit}"
                    val f = p.fraction
                    if (f >= 0) { pb.isIndeterminate = false; pb.progress = (f * 1000).toInt() }
                }
            }
        }, sticky = true) { r ->
            progressBox.removeAllViews()
            c.handle(r) { rep -> showReport(c, rep, reportBox); refreshSources(c, project, sourcesBox, reportBox) }
        }
    }
    col.addView(u.button("Add documents (PDF, TXT, MD, DOCX, CSV, JSON)", "btn:add-sources") {
        val hook = StudioTestHooks.sources()
        if (hook != null) runIngest { hook }
        else c.pick(Pickers.openMany()) { data ->
            val uris = Pickers.uris(data)
            if (uris.isEmpty()) c.notice("Nothing selected.")
            else {
                val cr = c.contentResolver()
                runIngest { uris.map { Pickers.inputFrom(cr, it) } }
            }
        }
    })
    col.addView(progressBox)
    col.addView(reportBox)
    col.addView(u.button("Next: build dataset", "btn:to-dataset", false) { c.go(Route(Kind.DATASET, pid)) })
    col.addView(sourcesBox)
    c.call({ c.studio.lastIngestReport(project) }) { rep -> if (rep != null) showReport(c, rep, reportBox) }
    refreshSources(c, project, sourcesBox, reportBox)
}

private fun showReport(c: Ctl, rep: IngestReport, box: LinearLayout) {
    val u = c.ui
    box.removeAllViews()
    val h = u.card(12)
    h.tag = "ingest-report"
    h.addView(u.tv("INGESTION REPORT", 12f, u.accent, true))
    h.addView(u.tv("${rep.ingested.size} ingested, ${rep.duplicates.size} duplicate, ${rep.failed.size} failed - ${Date(rep.at)}", 14f, u.ink, true, 4))
    box.addView(h)
    for (it in rep.items) {
        val ic = u.card(6)
        ic.tag = "ingest-item:" + it.name
        ic.addView(u.tv(it.name, 14f, u.ink, true))
        val label = itemLabel(it.status, it.issues)
        val color = when (it.status) { IngestStatus.INGESTED -> u.ok; IngestStatus.DUPLICATE -> u.warn; IngestStatus.FAILED -> u.bad }
        ic.addView(u.badge(label, color))
        for (i in it.issues) ic.addView(u.tv("${i.code.name.lowercase()}: ${i.message}", 12f, u.warn))
        if (it.duplicateOfSourceId != null) ic.addView(u.tv("Duplicate of source ${it.duplicateOfSourceId}", 12f, u.muted))
        it.provenance?.let { p ->
            ic.addView(u.tv("sha256 ${shortHash(p.sha256)} - ${p.mime} - ${fmtBytes(p.sizeBytes)} - ${p.pages?.let { n -> "$n pages - " } ?: ""}" +
                "${p.chunks} chunks - extractor ${p.extractor} - rights ${p.rights.name.lowercase()}", 11f, u.muted, topDp = 4))
        }
        box.addView(ic)
    }
}

private fun refreshSources(c: Ctl, project: ProjectId, box: LinearLayout, reportBox: LinearLayout) {
    c.call({ c.studio.listSources(project) }) { list: List<SourceRecord> ->
        val u = c.ui
        box.removeAllViews()
        box.addView(u.tv("SOURCES IN THIS SPECIALIST (${list.size})", 12f, u.accent, true, 14))
        for (s in list) {
            val sc = u.card(6)
            sc.tag = "source:" + s.sourceId
            sc.addView(u.tv(s.name, 14f, u.ink, true))
            val p = s.provenance
            sc.addView(u.tv("sha256 ${shortHash(p.sha256)} - ${fmtBytes(p.sizeBytes)} - ${p.chunks} chunks - ingested ${Date(p.ingestedAt)}", 11f, u.muted))
            sc.addView(u.button("Rights: ${p.rights.name.lowercase()} (tap to change)", "btn:rights:" + s.sourceId, false) {
                val all = RightsStatus.values()
                val next = all[(p.rights.ordinal + 1) % all.size]
                c.call({ c.studio.setSourceRights(project, s.sourceId, next) }) { refreshSources(c, project, box, reportBox) }
            })
            sc.addView(u.button("Remove source and derived data", "btn:remove-source:" + s.sourceId, false) {
                c.confirm("Remove source?", "Removes ${s.name} and everything derived from it. The dataset will need a rebuild.", "Remove") {
                    c.call({ c.studio.removeSource(project, s.sourceId) }, sticky = true) { refreshSources(c, project, box, reportBox) }
                }
            })
            box.addView(sc)
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------------------------------------
internal fun datasetScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    if (pid == null) { missing(c, col, "project"); return }
    val project = ProjectId(pid)
    col.addView(u.tv("Chunks are split into training, validation and held-out evaluation sets BEFORE anything is generated, so evaluation material is never trained on. Review what goes in, then approve.", 13f, u.muted, topDp = 4))
    val d = DatasetOptions()
    val oc = u.card(8)
    oc.addView(u.tv("BUILD OPTIONS", 12f, u.accent, true))
    val seed = u.edit("Seed", "opt:seed", number = true); seed.setText(d.seed.toString())
    val chunk = u.edit("Chunk target tokens", "opt:chunk", number = true); chunk.setText(d.chunkTargetTokens.toString())
    val held = u.edit("Held-out fraction (0-0.5)", "opt:heldout", number = true, decimal = true); held.setText(d.heldOutFraction.toString())
    val valf = u.edit("Validation fraction (0-0.5)", "opt:validation", number = true, decimal = true); valf.setText(d.validationFraction.toString())
    val synth = u.check("Include synthetic (generated) examples", "opt:synthetic", d.includeSynthetic)
    val exRef = u.check("Exclude reference-only sources from training", "opt:exclude-ref", d.excludeReferenceOnlySources)
    for (v in listOf<View>(seed, chunk, held, valf, synth, exRef)) oc.addView(v)
    col.addView(oc)
    val progressBox = u.col()
    val statsBox = u.col()
    col.addView(u.button("Build dataset", "btn:build-dataset") {
        val s = seed.text.toString().toLongOrNull()
        val ch = chunk.text.toString().toIntOrNull()
        val h = held.text.toString().toDoubleOrNull()
        val v = valf.text.toString().toDoubleOrNull()
        if (s == null || ch == null || h == null || v == null || ch < 50 || h < 0 || v < 0 || h + v >= 0.9) {
            c.error("Check the options: seed and chunk size must be numbers (chunk at least 50 tokens); fractions must be small and sum below 0.9.")
        } else {
            val opts = DatasetOptions(s, ch, h, v, synth.isChecked, exRef.isChecked)
            progressBox.removeAllViews()
            val pt = u.tv("Starting...", 12f, u.muted); val pb = u.bar(-1.0, "progress:dataset")
            progressBox.addView(pt); progressBox.addView(pb)
            c.bg({
                c.studio.buildDataset(project, opts) { p: Progress ->
                    c.post {
                        pt.text = "Building: ${p.done} of ${p.total} ${p.unit}"
                        val f = p.fraction
                        if (f >= 0) { pb.isIndeterminate = false; pb.progress = (f * 1000).toInt() }
                    }
                }
            }, sticky = true) { r -> progressBox.removeAllViews(); c.handle(r) { p -> showDataset(c, project, p, statsBox) } }
        }
    })
    col.addView(progressBox)
    col.addView(statsBox)
    c.call({ c.studio.datasetPreview(project) }) { p -> if (p != null) showDataset(c, project, p, statsBox) else statsBox.addView(u.tv("No dataset built yet.", 13f, u.muted, topDp = 10)) }
}

private fun showDataset(c: Ctl, project: ProjectId, p: DatasetPreview, box: LinearLayout) {
    val u = c.ui
    box.removeAllViews()
    val sc = u.card(12)
    sc.tag = "dataset-stats"
    sc.addView(u.tv("DATASET v${p.version}", 12f, u.accent, true))
    val color = when (p.status) {
        DatasetStatus.APPROVED -> u.ok; DatasetStatus.STALE -> u.bad; DatasetStatus.NONE -> u.muted; else -> u.warn
    }
    sc.addView(u.badge(p.status.name, color))
    val s = p.stats
    sc.addView(u.tv("sha256 ${shortHash(p.datasetSha256)} - built ${Date(p.builtAt)}\n${s.totalChunks} chunks: ${s.included} included, ${s.excluded} excluded, ${s.flagged} flagged\n" +
        "Roles: " + s.byRole.entries.joinToString { "${it.key.name.lowercase()} ${it.value}" } + "\n" +
        "Origin: " + s.byOrigin.entries.joinToString { "${it.key.name.lowercase()} ${it.value}" } + "\n" +
        "Flags: " + (if (s.byFlag.isEmpty()) "none" else s.byFlag.entries.joinToString { "${it.key.name.lowercase()} ${it.value}" }) + "\n" +
        "Leakage suspects: ${s.leakageSuspects}", 13f, u.ink, topDp = 4))
    for (w in p.warnings) sc.addView(u.tv("! $w", 12f, u.warn))
    sc.addView(u.button("Review items", "btn:open-review", false) { c.go(Route(Kind.REVIEW, project.value)) })
    sc.addView(u.button("Approve dataset", "btn:approve-dataset") {
        c.call({ c.studio.approveDataset(project) }, sticky = true) { np -> c.success("Dataset approved."); showDataset(c, project, np, box) }
    })
    box.addView(sc)
}

// ---------------------------------------------------------------------------------------------------------
internal fun reviewScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    if (pid == null) { missing(c, col, "project"); return }
    val project = ProjectId(pid)
    var offset = 0
    var onlyFlagged = false
    val pageSize = 25
    val list = u.col()
    val only = u.check("Only flagged items", "filter:flagged", false) { v -> onlyFlagged = v; offset = 0; loadReview(c, project, onlyFlagged, offset, pageSize, list) }
    col.addView(only)
    col.addView(list)
    val nav = u.row()
    nav.addView(u.button("Previous", "btn:review-prev", false) {
        if (offset > 0) { offset = (offset - pageSize).coerceAtLeast(0); loadReview(c, project, onlyFlagged, offset, pageSize, list) }
    }, LinearLayout.LayoutParams(0, -2, 1f))
    nav.addView(u.button("Next", "btn:review-next", false) {
        offset += pageSize; loadReview(c, project, onlyFlagged, offset, pageSize, list)
    }, LinearLayout.LayoutParams(0, -2, 1f))
    col.addView(nav)
    col.addView(u.button("Approve dataset", "btn:review-approve") {
        c.call({ c.studio.approveDataset(project) }, sticky = true) { c.success("Dataset approved."); c.back() }
    })
    loadReview(c, project, false, 0, pageSize, list)
}

private fun loadReview(c: Ctl, project: ProjectId, flagged: Boolean, offset: Int, limit: Int, box: LinearLayout) {
    c.call({ c.studio.reviewItems(project, ReviewFilter(onlyFlagged = flagged), offset, limit) }) { page ->
        val u = c.ui
        box.removeAllViews()
        box.addView(u.tv("Showing ${if (page.items.isEmpty()) 0 else page.offset + 1}-${page.offset + page.items.size} of ${page.total}", 12f, u.muted, topDp = 6))
        for (it in page.items) {
            val ic = u.card(8)
            ic.tag = "review:" + it.id
            ic.addView(u.tv("${it.sourceName}" + (it.page?.let { p -> " p.$p" } ?: "") + (it.section?.let { s -> " - $s" } ?: ""), 13f, u.ink, true))
            ic.addView(u.tv("${it.role.name.lowercase()} - ${it.origin.name.lowercase()}", 11f, u.muted))
            if (it.flags.isNotEmpty()) ic.addView(u.badge("FLAGS: " + it.flags.joinToString { f -> f.name.lowercase() }, u.warn))
            val ex = u.tv(it.excerpt, 12f, u.ink, topDp = 4)
            ex.maxLines = 6
            ic.addView(ex)
            val id = it.id
            ic.addView(u.check("Include in the dataset", "item:" + id, it.included) { inc ->
                c.call({ c.studio.setIncluded(project, listOf(id), inc) }, sticky = true) { }
            })
            box.addView(ic)
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Method
// ---------------------------------------------------------------------------------------------------------
internal fun methodScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    if (pid == null) { missing(c, col, "project"); return }
    val warn = u.card(8)
    warn.addView(u.tv("What counts as training", 15f, u.warn, true))
    val warnText = u.tv("A reference package (retrieval over your sources) and prompt instructions are NOT training and are labelled that way. Only a run that changes the model's parameters is training.", 13f, u.ink, topDp = 4)
    warnText.tag = "method-warning"
    warn.addView(warnText)
    col.addView(warn)
    c.bg({ c.studio.engineStatus() }) { es ->
        warn.addView(u.tv(if (es.trainingAvailable) "This phone can train when a base model is installed, the dataset is approved and the phone is charging and cool (see 'Train on this phone'). A desktop job is an optional fallback you choose."
        else "Training on this phone is unavailable on this install: ${es.trainingReason ?: "no native engine"}. The desktop job is the way to train.", 12f, if (es.trainingAvailable) u.muted else u.bad, topDp = 6))
    }
    c.call({ c.studio.methodOptions(ProjectId(pid)) }) { opts ->
        for (o in opts) {
            val oc = u.card(10)
            oc.tag = "method:" + o.id
            oc.addView(u.tv(o.label, 16f, u.ink, true))
            oc.addView(u.badge(if (o.isTraining) "TRAINING" else "NOT TRAINING", if (o.isTraining) u.info else u.muted))
            val where = when (o.whereItRuns) { RunLocation.DEVICE -> "Runs on this phone"; RunLocation.DESKTOP -> "Runs on a desktop/GPU"; RunLocation.NONE -> "Not applicable" }
            oc.addView(u.tv(where, 13f, u.ink, topDp = 4))
            oc.addView(u.tv(o.honestyNote, 12f, u.muted, topDp = 2))
            if (!o.available) {
                oc.addView(u.badge("NOT AVAILABLE", u.bad))
                oc.addView(u.tv(o.whyNotAvailable ?: "Not available in this version.", 12f, u.bad))
            } else {
                oc.addView(u.button("Choose this method", "btn:method:" + o.id) {
                    c.call({ c.studio.selectMethod(ProjectId(pid), o.id) }, sticky = true) { _ -> c.success("Method chosen: ${o.label}"); c.back() }
                })
            }
            col.addView(oc)
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Export / packages
// ---------------------------------------------------------------------------------------------------------
internal fun showExported(c: Ctl, p: ExportedPackage, uri: android.net.Uri?, box: LinearLayout) {
    val u = c.ui
    box.removeAllViews()
    val pc = u.card(12)
    pc.tag = "export-result"
    pc.addView(u.tv("EXPORTED: ${p.kind}", 12f, u.accent, true))
    pc.addView(u.badge("EXPORT SUCCEEDED", u.ok))
    pc.addView(u.tv("${p.suggestedFileName}\n${fmtBytes(p.sizeBytes)}\nsha256 ${p.sha256}", 12f, u.ink, topDp = 4).also { it.setTextIsSelectable(true) })
    pc.addView(u.tv("Contents: " + p.files.joinToString(", "), 11f, u.muted, topDp = 4))
    for (w in p.warnings) pc.addView(u.tv("! $w", 12f, u.warn))
    if (uri != null) pc.addView(u.button("Share", "btn:share", false) { c.share(uri, "application/zip") })
    box.addView(pc)
}

internal fun trainingScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    if (pid == null) { missing(c, col, "project"); return }
    val project = ProjectId(pid)
    val result = u.col()
    val tc = u.card(8)
    tc.addView(u.tv("Training job package (for desktop/GPU training)", 15f, u.ink, true))
    tc.addView(u.badge("TRAINING JOB - RUNS ON DESKTOP", u.info))
    tc.addView(u.tv("Contains the approved dataset (JSONL), training config, manifest and license evidence. Run it with `llmtrainer import-job` on a desktop. Nothing is trained on this phone.", 12f, u.muted, topDp = 4))
    tc.addView(u.button("Export training job package", "btn:export-job") {
        c.runExport("${project.value}-training-job.zip", "application/zip", { s, o -> s.exportTrainingJobPackage(project, o) }) { p, uri -> showExported(c, p, uri, result); c.success("Training job package exported.") }
    })
    col.addView(tc)
    val rc = u.card(10)
    rc.addView(u.tv("Reference package (exact-fact retrieval)", 15f, u.ink, true))
    rc.addView(u.badge("NOT TRAINING", u.muted))
    rc.addView(u.tv("Source chunks with citations so other apps can look up exact values (torque specs, clearances, standards text). It does not change any model.", 12f, u.muted, topDp = 4))
    rc.addView(u.button("Export reference package", "btn:export-reference") {
        c.runExport("${project.value}-reference.zip", "application/zip", { s, o -> s.exportReferencePackage(project, o) }) { p, uri -> showExported(c, p, uri, result); c.success("Reference package exported.") }
    })
    col.addView(rc)
    col.addView(result)
}

// ---------------------------------------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------------------------------------
internal fun evalScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    if (pid == null) { missing(c, col, "project"); return }
    val project = ProjectId(pid)
    val result = u.col()
    val view = u.col()
    val localBox = u.col()
    col.addView(localBox)
    localEvalSection(c, project, localBox)
    val ec = u.card(8)
    ec.addView(u.tv("Evaluate on a desktop (optional)", 15f, u.ink, true))
    ec.addView(u.tv("Export the held-out questions (never trained on), run `llmtrainer evaluate` on the desktop, then import the results package here.", 12f, u.ink, topDp = 4))
    ec.addView(u.button("Export held-out evaluation set", "btn:export-heldout") {
        c.runExport("${project.value}-heldout-eval.zip", "application/zip", { s, o -> s.exportHeldOutEvalSet(project, o) }) { p, uri -> showExported(c, p, uri, result); c.success("Held-out set exported.") }
    })
    ec.addView(u.button("Import results package", "btn:import-eval") {
        c.pick(Pickers.openOne()) { data ->
            val uri = Pickers.uris(data).firstOrNull()
            if (uri != null) {
                val cr = c.contentResolver()
                c.call({
                    val si = Pickers.inputFrom(cr, uri)
                    si.open().use { c.studio.importEvaluation(project, it) }
                }, sticky = true) { ev -> showEvaluation(c, ev, view); c.success("Results imported.") }
            }
        }
    })
    col.addView(ec)
    col.addView(result)
    col.addView(view)
    c.call({ c.studio.evaluation(project) }) { ev -> if (ev != null) showEvaluation(c, ev, view) else view.addView(u.tv("No evaluation results imported yet.", 13f, u.muted, topDp = 10)) }
}

internal fun showEvaluation(c: Ctl, ev: EvaluationView, box: LinearLayout, local: Boolean = false) {
    val u = c.ui
    box.removeAllViews()
    if (ev.isStub) {
        val sb = u.card(10)
        sb.addView(u.badge("STUB RESULTS - NOT A REAL EVALUATION", u.bad))
        sb.addView(u.tv("These numbers are placeholders and say nothing about model quality.", 13f, u.bad, topDp = 4))
        box.addView(sb)
    }
    val cl = u.card(10)
    cl.tag = "claim-status"
    cl.addView(u.tv("RESULT: ${ev.specialistLabel} vs ${ev.baseModelLabel} (run ${ev.runId})", 12f, u.accent, true))
    if (ev.improvementClaimAllowed) cl.addView(u.badge("IMPROVEMENT CLAIM SUPPORTED", u.ok)) else cl.addView(u.badge("NO IMPROVEMENT CLAIM", u.warn))
    cl.addView(u.tv(ev.claimReason, 13f, u.ink, topDp = 4))
    box.addView(cl)
    for (m in ev.metrics) {
        val mc = u.card(6)
        mc.tag = "metric:" + m.id
        mc.addView(u.tv(m.label, 14f, u.ink, true))
        mc.addView(u.tv("base ${fmtNum(m.base)}   specialist ${fmtNum(m.specialist)}   delta ${fmtNum(m.delta)}", 13f, u.ink))
        val lo = m.ciLow; val hi = m.ciHigh
        val ci = if (lo != null && hi != null) "CI [${fmtNum(lo)}, ${fmtNum(hi)}]" else "no confidence interval"
        mc.addView(u.tv("n = ${m.n} - $ci - ${if (m.higherIsBetter) "higher is better" else "lower is better"}", 11f, u.muted))
        box.addView(mc)
    }
    if (ev.caveats.isNotEmpty()) {
        val cv = u.card(8)
        cv.addView(u.tv("CAVEATS", 12f, u.accent, true))
        for (x in ev.caveats) cv.addView(u.tv("- $x", 12f, u.warn))
        box.addView(cv)
    }
    box.addView(u.tv(if (local) "Measured on this phone ${Date(ev.importedAt)} on held-out TEST text. Heuristic lexical and numeric metrics, not human judgement."
    else "Imported ${Date(ev.importedAt)}. Shown exactly as supplied by the evaluation run; this app does not recompute them.", 11f, u.muted, topDp = 6))
}

// ---------------------------------------------------------------------------------------------------------
// Specialist package
// ---------------------------------------------------------------------------------------------------------
internal fun specialistScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    val result = u.col()
    val inspect = u.col()
    if (pid != null) {
        val project = ProjectId(pid)
        val ec = u.card(8)
        ec.addView(u.tv("Export specialist package", 15f, u.ink, true))
        ec.addView(u.tv("Adapter/model references with hashes, base model identity and version, license state, dataset and evaluation metadata. Usable by other apps without LLM Trainer. Requires imported evaluation results.", 12f, u.muted, topDp = 4))
        ec.addView(u.button("Export specialist package", "btn:export-specialist") {
            c.runExport("${project.value}-specialist.zip", "application/zip", { s, o -> s.exportSpecialistPackage(project, o) }) { p, uri -> showExported(c, p, uri, result); c.success("Specialist package exported.") }
        })
        col.addView(ec)
    }
    col.addView(result)
    val ic = u.card(12)
    ic.addView(u.tv("Inspect a specialist package", 15f, u.ink, true))
    ic.addView(u.tv("Validate and describe an exported package without installing it.", 12f, u.muted, topDp = 4))
    ic.addView(u.button("Choose package file", "btn:import-package", false) {
        c.pick(Pickers.openOne()) { data ->
            val uri = Pickers.uris(data).firstOrNull()
            if (uri != null) {
                val cr = c.contentResolver()
                c.call({
                    val si = Pickers.inputFrom(cr, uri)
                    si.open().use { c.studio.importSpecialistPackage(it) }
                }, sticky = true) { v -> showPackageView(c, v, inspect) }
            }
        }
    })
    col.addView(ic)
    col.addView(inspect)
}

private fun showPackageView(c: Ctl, v: SpecialistPackageView, box: LinearLayout) {
    val u = c.ui
    box.removeAllViews()
    val pc = u.card(10)
    pc.tag = "package-view"
    pc.addView(u.tv("${v.name} ${v.version}", 16f, u.ink, true))
    pc.addView(u.badge(if (v.valid) "PACKAGE VALID" else "PACKAGE INVALID", if (v.valid) u.ok else u.bad))
    pc.addView(u.tv("Base model ${v.baseModelId} ${v.baseModelVersion}\nMethod: ${v.method}\nDataset sha256 ${shortHash(v.datasetSha256)}", 12f, u.ink, topDp = 4))
    pc.addView(u.badge(u.licenseText(v.licenseState), u.licenseColor(v.licenseState)))
    pc.addView(u.tv(v.licenseSummary, 12f, u.muted))
    pc.addView(u.tv(v.evaluationSummary ?: "No evaluation summary.", 12f, u.ink, topDp = 4))
    pc.addView(u.tv(if (v.improvementClaimAllowed) "Improvement claim supported by the recorded evidence." else "No improvement claim is supported.", 12f, if (v.improvementClaimAllowed) u.ok else u.warn))
    for (ch in v.validation) pc.addView(u.tv("${if (ch.passed) "PASS" else "FAIL"} ${ch.name}: ${ch.detail}", 12f, if (ch.passed) u.ok else u.bad))
    for (l in v.knownLimitations) pc.addView(u.tv("Limitation: $l", 12f, u.warn))
    for (cp in v.compatibility) pc.addView(u.tv("Compatibility: $cp", 11f, u.muted))
    box.addView(pc)
}

// ---------------------------------------------------------------------------------------------------------
// Updates & diagnostics
// ---------------------------------------------------------------------------------------------------------
internal fun interpretUpdate(kind: String, message: String): Pair<String, Int> = when {
    kind == "UP_TO_DATE" -> "Up to date. $message" to 0
    kind == "STAGED" -> "Update downloaded and verified: $message. Restart the app to apply." to 1
    kind == "NEEDS_NEW_APK" -> "A newer version needs a new app install (native/runtime change): $message" to 2
    kind == "CHECKING" -> "Checking..." to 0
    message.contains("404") || message.contains("not published", ignoreCase = true) ->
        "No update channel published yet - nothing to install." to 0
    else -> "Update check failed: $message" to 3
}

internal fun updatesScreen(c: Ctl, col: LinearLayout) {
    val u = c.ui
    val status = u.tv("Not checked yet.", 14f, u.ink)
    status.tag = "update-status"
    val uc = u.card(8)
    uc.addView(u.tv("UPDATES", 12f, u.accent, true))
    uc.addView(status)
    val restart = u.button("Restart to apply update", "btn:restart") { c.host.restartApp() }
    restart.visibility = View.GONE
    uc.addView(u.button("Check for updates", "btn:check-updates") {
        status.text = "Checking..."
        try {
            c.host.checkForUpdates { s ->
                if (c.alive) {
                    val (msg, sev) = interpretUpdate(s.kind, s.message)
                    status.text = msg
                    status.setTextColor(if (sev == 3) u.bad else if (sev == 2) u.warn else u.ink)
                    restart.visibility = if (s.kind == "STAGED") View.VISIBLE else View.GONE
                }
            }
        } catch (t: Throwable) { c.error("Update check failed: ${t.message}") }
    })
    uc.addView(restart)
    col.addView(uc)
    val idc = u.card(10)
    idc.addView(u.tv("IDENTITY", 12f, u.accent, true))
    val idText = u.mono("Loading...", 12f, u.ink)
    idText.tag = "identity-text"
    idc.addView(idText)
    col.addView(idc)
    val dc = u.card(10)
    dc.addView(u.tv("DIAGNOSTICS (long-press to copy)", 12f, u.accent, true))
    val diag = u.mono("Loading...", 11f, u.muted)
    diag.tag = "diagnostics-json"
    dc.addView(diag)
    col.addView(dc)
    fun loadDiag() {
        // Lazy: the v1 host reports runningSource "none" while createContentView runs, so read after attach.
        c.bg({
            try {
                val o = JSONObject(c.host.diagnosticsJson())
                Pair(o.optString("identityBlock", ""), o.toString(2))
            } catch (t: Throwable) { Pair("", "Diagnostics unavailable: ${t.message}") }
        }) { (block, json) -> idText.text = if (block.isNotEmpty()) block else "Identity unavailable."; diag.text = json }
    }
    c.resumeHook = { loadDiag() }
    col.post { loadDiag() }
}
