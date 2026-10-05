package com.hotatticgames.llmtrainer.app

import android.view.View
import android.widget.LinearLayout
import com.hotatticgames.llmtrainer.studio.api.CatalogModel
import com.hotatticgames.llmtrainer.studio.api.DeviceProfile
import com.hotatticgames.llmtrainer.studio.api.LicenseAttestation
import com.hotatticgames.llmtrainer.studio.api.LicenseState
import com.hotatticgames.llmtrainer.studio.api.LicenseTextFetch
import com.hotatticgames.llmtrainer.studio.api.NewProject
import com.hotatticgames.llmtrainer.studio.api.Operation
import com.hotatticgames.llmtrainer.studio.api.OperationState
import com.hotatticgames.llmtrainer.studio.api.Permission
import com.hotatticgames.llmtrainer.studio.api.ProjectId
import com.hotatticgames.llmtrainer.studio.api.ProjectSummary
import com.hotatticgames.llmtrainer.studio.api.Recommendations
import com.hotatticgames.llmtrainer.studio.api.StageId
import com.hotatticgames.llmtrainer.studio.api.StageStatus
import com.hotatticgames.llmtrainer.studio.api.Tri
import java.util.Date

/** Dispatches a route to its screen builder. Runs on the UI thread; data loads go through [Ctl.bg]/[Ctl.call]. */
internal fun buildScreen(c: Ctl, r: Route, col: LinearLayout) {
    val pid = r.projectId
    when (r.kind) {
        Kind.FIRSTRUN -> firstRunScreen(c, col)
        Kind.DASHBOARD -> dashboardScreen(c, col)
        Kind.CREATE -> createScreen(c, col)
        Kind.HUB -> hubScreen(c, pid, col)
        Kind.DEVICE -> deviceScreen(c, pid, col)
        Kind.CATALOG -> catalogScreen(c, pid, col)
        Kind.MODEL -> modelScreen(c, pid, r.modelId, col)
        Kind.LICENSE -> licenseScreen(c, r.modelId, col)
        Kind.ACQUIRE -> acquireScreen(c, pid, r.modelId, r.variantId, col)
        Kind.SOURCES -> sourcesScreen(c, pid, false, col)
        Kind.INGEST -> sourcesScreen(c, pid, true, col)
        Kind.DATASET -> datasetScreen(c, pid, col)
        Kind.REVIEW -> reviewScreen(c, pid, col)
        Kind.METHOD -> methodScreen(c, pid, col)
        Kind.TRAINING -> trainingScreen(c, pid, col)
        Kind.EVAL -> evalScreen(c, pid, col)
        Kind.SPECIALIST -> specialistScreen(c, pid, col)
        Kind.UPDATES -> updatesScreen(c, col)
        Kind.MODELS -> modelsScreen(c, pid, col)
        Kind.CHAT -> chatScreen(c, pid, col)
        Kind.TRAIN_LOCAL -> trainLocalScreen(c, pid, col)
        Kind.AB -> abScreen(c, pid, col)
        Kind.SPECIALISTS -> specialistsScreen(c, pid, col)
        Kind.ABOUT -> aboutScreen(c, col)
    }
}

internal fun missing(c: Ctl, col: LinearLayout, what: String) {
    col.addView(c.ui.tv("Missing $what. Go back and choose again.", 14f, c.ui.bad, topDp = 12))
}

// ---------------------------------------------------------------------------------------------------------
internal fun firstRunScreen(c: Ctl, col: LinearLayout) {
    val u = c.ui
    val card = u.card()
    card.addView(u.tv("What LLM Trainer does", 15f, u.accent, true))
    card.addView(u.tv(
        "You bring manuals, books and documents you are authorized to use. This app ingests them, keeps provenance and " +
            "usage rights, builds a reviewable dataset, helps you choose a base model, checks licenses, and packages " +
            "everything for training and reuse.", 14f, u.ink, topDp = 6))
    col.addView(card)
    val no = u.card()
    no.addView(u.tv("What it does NOT do, or only under conditions", 15f, u.warn, true))
    no.addView(u.tv(
        "- A model runs and is specialized on this phone only if the native engine works on it, a base model file is installed, " +
            "your dataset is approved and the phone is charging, cool and has enough free memory. Small models (up to roughly 0.5B parameters) " +
            "are practical; larger ones take hours or are not practical. If something is missing, the app tells you what instead of pretending.\n" +
            "- A desktop job package is an optional fallback that you choose. The app never sends work to a desktop on its own.\n" +
            "- A reference package (retrieval over your sources) is NOT training. Prompt instructions are NOT training. Source excerpts in a chat prompt are labelled as retrieval.\n" +
            "- Fine-tuning can make a model forget general knowledge, so a specialist is only called better when the held-out evaluation, including a retention check, supports it.\n" +
            "- Device recommendations are estimates until a real on-device benchmark exists.\n" +
            "- Models are never downloaded without your confirmation. Your source documents stay on this device unless you choose to export or share a package.", 14f, u.ink, topDp = 6))
    col.addView(no)
    col.addView(u.button("Got it, continue", "btn:firstrun-ok") { c.markFirstRunDone(); c.resetTo(Route(Kind.DASHBOARD)) })
}

// ---------------------------------------------------------------------------------------------------------
private class DashData(val projects: List<ProjectSummary>, val device: DeviceProfile, val ops: List<Operation>)

internal fun dashboardScreen(c: Ctl, col: LinearLayout) {
    val u = c.ui
    col.addView(u.tv("Turn authorized domain material into a portable specialist model. On a capable phone the specialist is trained and checked on the phone itself; " +
        "a desktop job is an optional fallback you choose.", 13f, u.muted, topDp = 4))
    col.addView(u.button("Create specialist", "btn:create") { c.go(Route(Kind.CREATE)) })
    val dyn = u.col()
    col.addView(dyn)
    col.addView(u.button("Model manager", "btn:models", false) { c.go(Route(Kind.MODELS)) })
    col.addView(u.button("Device & recommendations", "btn:device", false) { c.go(Route(Kind.DEVICE)) })
    col.addView(u.button("Model catalog", "btn:catalog", false) { c.go(Route(Kind.CATALOG)) })
    col.addView(u.button("Updates & diagnostics", "btn:updates", false) { c.go(Route(Kind.UPDATES)) })
    col.addView(u.button("About & what this phone can do", "btn:about-phone", false) { c.go(Route(Kind.ABOUT)) })
    col.addView(u.button("What this app does and does not do", "btn:about", false) { c.go(Route(Kind.FIRSTRUN)) })

    fun fill(d: DashData) {
        dyn.removeAllViews()
        for (op in d.ops.filter { !it.isTerminal }) {
            val oc = u.card()
            oc.tag = "op:" + op.id
            oc.addView(u.tv("Operation: ${op.subject}", 14f, u.accent, true))
            oc.addView(u.tv("${op.state}  ${(op.progress.fraction.coerceAtLeast(0.0) * 100).toInt()}%  ${op.message}", 12f, u.muted))
            if (op.state == OperationState.PAUSED && op.resumable) {
                oc.addView(u.button("Resume", "btn:op-resume:" + op.id) {
                    c.call({ c.studio.resumeOperation(op.id) }) { c.render() }
                })
            }
            dyn.addView(oc)
        }
        if (d.projects.isEmpty()) {
            dyn.addView(u.tv("No specialists yet. Create one to start.", 14f, u.muted, topDp = 14))
        }
        for (p in d.projects) {
            val pc = u.card(14)
            pc.tag = "project:" + p.id.value
            pc.addView(u.tv(p.name, 18f, u.ink, true))
            pc.addView(u.tv(p.domain, 13f, u.muted))
            val strip = u.row().apply { setPadding(0, u.px(8), 0, u.px(4)) }
            for (s in p.stages) {
                val seg = View(c.context)
                seg.setBackgroundColor(u.stageColor(s.status))
                strip.addView(seg, LinearLayout.LayoutParams(0, u.px(8), 1f).apply { rightMargin = u.px(2) })
            }
            pc.addView(strip)
            pc.addView(u.tv("${p.stages.count { it.status == StageStatus.DONE }} of ${p.stages.size} stages done", 12f, u.muted))
            val na = p.nextAction
            if (na != null) {
                pc.addView(u.button("Next: " + na.label, "btn:next:" + p.id.value) { c.go(c.routeFor(na.screen, p)) })
            } else {
                pc.addView(u.tv("All stages done.", 13f, u.ok, true, 6))
            }
            pc.setOnClickListener { c.go(Route(Kind.HUB, p.id.value)) }
            dyn.addView(pc)
        }
        val dc = u.card(14)
        val dv = d.device
        dc.addView(u.tv("THIS DEVICE", 12f, u.accent, true))
        dc.addView(u.tv("${dv.deviceName} - Android API ${dv.androidApi} (${dv.abi})\nRAM ${fmtMb(dv.availableRamMb.toLong())} available of " +
            "${fmtMb(dv.totalRamMb.toLong())}; free storage ${fmtMb(dv.freeStorageMb)}\nOn-device runtime: ${dv.nativeRuntimeId} " +
            (if (dv.nativeRuntimeId.startsWith("hag-engine")) "(native engine present)" else "(no native engine on this install)"), 13f, u.ink, topDp = 4))
        dyn.addView(dc)
        if (d.ops.any { it.state == OperationState.RUNNING || it.state == OperationState.QUEUED }) {
            c.postDelayed(2000) { load(c, dyn) { x -> fill(x) } }
        }
    }
    c.resumeHook = { load(c, dyn) { x -> fill(x) } }
    load(c, dyn) { x -> fill(x) }
}

private fun load(c: Ctl, @Suppress("UNUSED_PARAMETER") dyn: LinearLayout, show: (DashData) -> Unit) {
    c.bg({ DashData(c.studio.listProjects(), c.studio.deviceProfile(), c.studio.operations()) }) { show(it) }
}

// ---------------------------------------------------------------------------------------------------------
internal fun createScreen(c: Ctl, col: LinearLayout) {
    val u = c.ui
    col.addView(u.tv("A specialist is one domain (for example Motorcycle Mechanic or HVAC Technician) built from sources you are authorized to use.", 13f, u.muted, topDp = 4))
    val name = u.edit("Name", "field:name")
    val domain = u.edit("Domain (e.g. motorcycle repair)", "field:domain")
    val purpose = u.edit("Purpose (what should the specialist help with?)", "field:purpose", multiline = true)
    col.addView(name); col.addView(domain); col.addView(purpose)
    col.addView(u.button("Create specialist", "btn:create-submit") {
        val n = name.text.toString().trim()
        if (n.isEmpty()) c.error("Give the specialist a name.")
        else if (!c.isBusy()) {
            val np = NewProject(n, domain.text.toString().trim(), purpose.text.toString().trim())
            c.call({ c.studio.createProject(np) }, sticky = true) { p -> c.replaceTop(Route(Kind.HUB, p.id.value)) }
        }
    })
}

// ---------------------------------------------------------------------------------------------------------
private fun stageTitle(id: StageId) = when (id) {
    StageId.BASE_MODEL -> "Choose base model"
    StageId.SOURCES -> "Add sources and review corpus"
    StageId.DATASET -> "Build and review dataset"
    StageId.METHOD -> "Choose method"
    StageId.TRAINING_PACKAGE -> "Training job / reference package"
    StageId.EVALUATION -> "Evaluate against the base model"
    StageId.EXPORT -> "Export specialist package"
}

internal fun hubScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    if (pid == null) { missing(c, col, "project"); return }
    c.call({ c.studio.getProject(ProjectId(pid)) }) { p ->
        col.addView(u.tv(p.name, 20f, u.ink, true, 6))
        col.addView(u.tv("${p.domain}\n${p.purpose}", 13f, u.muted))
        col.addView(u.tv("Base model: ${p.baseModelId ?: "not chosen"}", 13f, u.ink, topDp = 4))
        val first = u.card(10)
        first.addView(u.tv("Create project", 15f, u.ink, true))
        first.addView(u.badge("DONE", u.ok))
        col.addView(first)
        for (s in p.stages) {
            val sc = u.card(8)
            sc.tag = "stage:" + s.id.name
            sc.addView(u.tv(stageTitle(s.id), 15f, u.ink, true))
            sc.addView(u.badge(u.stageText(s.status), u.stageColor(s.status)))
            sc.addView(u.tv(s.summary, 13f, if (s.status == StageStatus.BLOCKED) u.bad else u.muted, topDp = 4))
            val na = s.nextAction
            if (na != null && s.status != StageStatus.DONE) {
                sc.addView(u.button(na.label, "btn:stage:" + s.id.name, s.status != StageStatus.BLOCKED) { c.go(c.routeFor(na.screen, p)) })
            }
            col.addView(sc)
        }
        val opt = u.card(8)
        opt.addView(u.tv("Optimize and benchmark on device", 15f, u.ink, true))
        opt.addView(u.badge("NOT AVAILABLE YET", u.muted))
        opt.addView(u.tv("Device benchmarks (time to first token, sustained speed, heat, memory pressure) are not recorded yet. Every speed and memory figure in the app is an estimate; nothing is claimed here.", 13f, u.muted, topDp = 4))
        col.addView(opt)
        val phone = u.card(10)
        phone.tag = "hub-phone"
        phone.addView(u.tv("On this phone", 15f, u.ink, true))
        val phoneStatus = u.tv("Checking what this phone can do...", 12f, u.muted, topDp = 4)
        phoneStatus.tag = "hub-phone-status"
        phone.addView(phoneStatus)
        phone.addView(u.button("Model manager", "btn:hub-model-manager", false) { c.go(Route(Kind.MODELS, pid)) })
        phone.addView(u.button("Chat (base model or specialist)", "btn:hub-chat", false) { c.go(Route(Kind.CHAT, pid)) })
        phone.addView(u.button("Train on this phone", "btn:hub-train", false) { c.go(Route(Kind.TRAIN_LOCAL, pid)) })
        phone.addView(u.button("Compare base and specialist (A/B)", "btn:hub-ab", false) { c.go(Route(Kind.AB, pid)) })
        phone.addView(u.button("Specialists on this phone", "btn:hub-specialists-local", false) { c.go(Route(Kind.SPECIALISTS, pid)) })
        phone.addView(u.button("About & what this phone can do", "btn:hub-about", false) { c.go(Route(Kind.ABOUT, pid)) })
        col.addView(phone)
        c.call({ c.studio.projectModelState(ProjectId(pid)) }) { ms ->
            val chatB = if (ms.canChatBase.ok) "Chat with the base model: ready." else "Chat with the base model: ${ms.canChatBase.reason}"
            val chatS = if (ms.canChatSpecialist.ok) "Chat with the specialist: ready." else "Specialist: ${ms.canChatSpecialist.reason}"
            val tr = if (ms.canTrainLocally.ok) "Training on this phone: possible now." else "Training on this phone: ${ms.canTrainLocally.reason}"
            phoneStatus.text = chatB + "\n" + chatS + "\n" + tr
        }
        if (p.baseModelId != null) col.addView(u.button("Base model license evidence", "btn:hub-license", false) { c.go(Route(Kind.LICENSE, pid, p.baseModelId)) })
        col.addView(u.button("Packages and exports (training job, reference)", "btn:hub-packages", false) { c.go(Route(Kind.TRAINING, pid)) })
        col.addView(u.button("Evaluation", "btn:hub-eval", false) { c.go(Route(Kind.EVAL, pid)) })
        col.addView(u.button("Specialist package", "btn:hub-specialist", false) { c.go(Route(Kind.SPECIALIST, pid)) })
        col.addView(u.button("Sources", "btn:hub-sources", false) { c.go(Route(Kind.SOURCES, pid)) })
        col.addView(u.button("Dataset", "btn:hub-dataset", false) { c.go(Route(Kind.DATASET, pid)) })
        col.addView(u.button("Model catalog and recommendations", "btn:hub-models", false) { c.go(Route(Kind.DEVICE, pid)) })
        col.addView(u.button("Delete this specialist", "btn:delete-project", false) {
            c.confirm("Delete specialist?", "This removes ${p.name} and everything derived from it on this device.", "Delete") {
                c.call({ c.studio.deleteProject(ProjectId(pid)) }, sticky = true) { c.resetTo(Route(Kind.DASHBOARD)) }
            }
        })
    }
}

// ---------------------------------------------------------------------------------------------------------
internal fun deviceScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    c.bg({ c.studio.recommendations() }) { rec: Recommendations ->
        val d = rec.device
        val dc = u.card(8)
        dc.addView(u.tv("THIS DEVICE", 12f, u.accent, true))
        dc.addView(u.tv(
            "${d.deviceName}\nAndroid API ${d.androidApi}, ${d.abi}\nRAM: ${fmtMb(d.totalRamMb.toLong())} total, ${fmtMb(d.availableRamMb.toLong())} available now\n" +
                "Free storage: ${fmtMb(d.freeStorageMb)}\nSafety reserve: ${(d.safetyReserveFraction * 100).toInt()}% (the model must leave this free so the phone keeps working normally)\n" +
                "Inference runtime: ${d.nativeRuntimeId}", 13f, u.ink, topDp = 4))
        dc.addView(u.tv("Nothing below has been measured on this phone yet. " +
            "Profiles are ESTIMATES from device numbers until a benchmark exists.", 12f, u.warn, topDp = 6))
        col.addView(dc)
        for (p in rec.profiles) {
            val pc = u.card(10)
            pc.tag = "profile:" + p.kind.name
            val label = when (p.kind) {
                com.hotatticgames.llmtrainer.studio.api.ProfileKind.PERFORMANCE -> "Performance"
                com.hotatticgames.llmtrainer.studio.api.ProfileKind.BALANCED -> "Balanced"
                com.hotatticgames.llmtrainer.studio.api.ProfileKind.MAX_QUALITY -> "Maximum quality within safe envelope"
            }
            pc.addView(u.tv(label, 16f, u.ink, true))
            pc.addView(u.badge(if (p.confidence.benchmarked) "BENCHMARKED" else "ESTIMATE - NOT BENCHMARKED", if (p.confidence.benchmarked) u.ok else u.warn))
            if (p.modelId == null) {
                pc.addView(u.tv("No model qualifies for this profile on this device.", 13f, u.bad, true, 4))
            } else {
                pc.addView(u.tv("${p.modelId} - ${p.quant ?: "?"} - context ${p.contextTokens ?: "?"} tokens - est. peak RAM " +
                    (p.estimatedPeakRamMb?.let { fmtMb(it.toLong()) } ?: "?"), 13f, u.ink, topDp = 4))
                val ls = p.licenseState
                if (ls != null) pc.addView(u.badge(u.licenseText(ls), u.licenseColor(ls)))
            }
            pc.addView(u.tv("Confidence ${p.confidence.level}: ${p.confidence.basis}", 12f, u.muted, topDp = 4))
            for (rs in p.reasons) pc.addView(u.tv("- $rs", 12f, u.muted))
            for (w in p.warnings) pc.addView(u.tv("! $w", 12f, u.warn))
            val mid = p.modelId
            if (mid != null) pc.addView(u.button("View model", "btn:profile-model:" + p.kind.name, false) { c.go(Route(Kind.MODEL, pid, mid)) })
            col.addView(pc)
        }
        col.addView(u.button("Browse model catalog", "btn:open-catalog") { c.go(Route(Kind.CATALOG, pid)) })
    }
}

// ---------------------------------------------------------------------------------------------------------
internal fun catalogScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    col.addView(u.tv("Open-weight base models. License state is shown with colour AND text; a model is only usable once its license text has been reviewed and attested by you.", 13f, u.muted, topDp = 4))
    c.bg({ c.studio.catalog() }) { models: List<CatalogModel> ->
        if (models.isEmpty()) col.addView(u.tv("The catalog is empty.", 14f, u.muted, topDp = 12))
        for (m in models) {
            val mc = u.card(10)
            mc.tag = "model:" + m.id
            mc.addView(u.tv("${m.name} (${m.version})", 16f, u.ink, true))
            mc.addView(u.tv("${m.family} - ${m.paramsBillions}B params - ${m.architecture}", 12f, u.muted))
            mc.addView(u.badge(u.licenseText(m.license.state), u.licenseColor(m.license.state)))
            val sizes = m.variants.joinToString(", ") { "${it.quant} ${fmtBytes(it.sizeBytes)}" }
            mc.addView(u.tv("Variants: $sizes", 12f, u.ink, topDp = 4))
            mc.setOnClickListener { c.go(Route(Kind.MODEL, pid, m.id)) }
            col.addView(mc)
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
internal fun modelScreen(c: Ctl, pid: String?, mid: String?, col: LinearLayout) {
    val u = c.ui
    if (mid == null) { missing(c, col, "model"); return }
    c.call({ c.studio.model(mid) }) { m ->
        col.addView(u.tv("${m.name} (${m.version})", 20f, u.ink, true, 6))
        col.addView(u.tv("${m.family} - ${m.paramsBillions}B params - ${m.architecture}", 13f, u.muted))
        col.addView(u.tv("Source: ${m.provenance.sourceUrl}\nEvidence: ${m.provenance.evidenceLevel}" +
            (m.provenance.retrievedOn?.let { " (retrieved $it)" } ?: "") + (m.provenance.note?.let { "\n$it" } ?: ""), 12f, u.muted, topDp = 4))
        val lic = m.license
        val lc = u.card(10)
        lc.tag = "license-card"
        lc.addView(u.tv("LICENSE", 12f, u.accent, true))
        lc.addView(u.badge(u.licenseText(lic.state), u.licenseColor(lic.state)))
        lc.addView(u.tv("${lic.licenseName}\nEvidence level: ${lic.evidenceLevel}\nAuthoritative text: ${lic.authoritativeUrl}", 13f, u.ink, topDp = 4))
        lic.scopeText?.let { lc.addView(u.tv(it, 12f, u.muted, topDp = 4)) }
        lic.disallowedReason?.let { lc.addView(u.tv("Restriction: $it", 13f, u.bad, true, 4)) }
        val perms = Permission.values().joinToString("\n") { p ->
            val t = lic.permissions[p] ?: Tri.UNVERIFIED
            "${p.name.lowercase().replace('_', ' ')}: $t"
        }
        lc.addView(u.tv(perms + "\nAttribution required: ${lic.attributionRequired ?: "unknown"}", 12f, u.ink, topDp = 4))
        if (lic.state != LicenseState.VERIFIED) {
            lc.addView(u.tv("These permissions are claims until you review the license text and attest. The gate ignores them until the license is VERIFIED.", 12f, u.warn, topDp = 4))
        }
        lc.addView(u.button("Review license evidence", "btn:license-evidence") { c.go(Route(Kind.LICENSE, pid, m.id)) })
        col.addView(lc)
        // Base-model selection does not depend on quantized on-device files: heavy adaptation runs on the desktop from the
        // model's source repository, so the owner can always pick the model itself. Files are only needed for on-device use.
        val sel = u.card(10)
        sel.tag = "select-card"
        sel.addView(u.tv("USE AS BASE MODEL", 12f, u.accent, true))
        sel.addView(u.tv(if (m.variants.isEmpty())
            "No quantized on-device files are registered for this model yet. You can still select it: training runs on the desktop from the source repository (${m.provenance.sourceUrl}). Selecting does not download anything and does not change the license state (training stays blocked until the license is VERIFIED)."
        else "Select the model itself (desktop training uses the source repository), or pick a specific file below for on-device use.", 12f, u.muted, topDp = 4))
        if (pid != null) {
            sel.addView(u.button("Select as base model", "btn:select-base-model") {
                c.call({ c.studio.selectBaseModel(ProjectId(pid), m.id, null) }, sticky = true) { _ ->
                    c.success("Base model selected.")
                    c.resetTo(Route(Kind.HUB, pid))
                }
            })
        } else {
            sel.addView(u.tv("Open this model from a specialist to select it as that specialist's base model.", 12f, u.muted, topDp = 6))
        }
        col.addView(sel)
        for (v in m.variants) {
            val vc = u.card(10)
            vc.tag = "variant:" + v.id
            vc.addView(u.tv("${v.quant} (${v.format}) - ${fmtBytes(v.sizeBytes)}", 15f, u.ink, true))
            vc.addView(u.tv("${v.paramsBillions}B - ${v.architecture} - max context ${v.contextTokensMax} tokens" + (if (v.acquired) " - ON DEVICE" else " - not downloaded"), 12f, u.muted))
            vc.addView(u.badge("PHONE: " + u.verdictText(v.android.verdict), u.verdictColor(v.android.verdict)))
            v.android.estimatedPeakRamMb?.let { vc.addView(u.tv("Estimated peak RAM ${fmtMb(it.toLong())}", 12f, u.muted)) }
            for (r in v.android.reasons) vc.addView(u.tv("- $r", 12f, u.muted))
            vc.addView(u.tv("Training: ${v.training.where} ${v.training.method ?: ""}", 12f, u.ink, topDp = 4))
            for (r in v.training.reasons) vc.addView(u.tv("- $r", 12f, u.muted))
            vc.addView(u.tv("Provenance: ${v.provenance.sourceUrl} (${v.provenance.evidenceLevel})", 11f, u.muted, topDp = 4))
            vc.addView(u.button("Acquire or import this file", "btn:acquire:" + v.id, false) { c.go(Route(Kind.ACQUIRE, pid, m.id, v.id)) })
            if (pid != null) {
                vc.addView(u.button("Select as base model", "btn:select-base:" + v.id) {
                    c.call({ c.studio.selectBaseModel(ProjectId(pid), m.id, v.id) }, sticky = true) { _ ->
                        c.success("Base model selected.")
                        c.resetTo(Route(Kind.HUB, pid))
                    }
                })
            }
            col.addView(vc)
        }
        if (pid == null) col.addView(u.tv("Open this model from a specialist to select it as that specialist's base model.", 12f, u.muted, topDp = 10))
    }
}

// ---------------------------------------------------------------------------------------------------------
private val fetchCache = HashMap<String, LicenseTextFetch>()

internal fun licenseScreen(c: Ctl, mid: String?, col: LinearLayout) {
    val u = c.ui
    if (mid == null) { missing(c, col, "model"); return }
    val dis = u.card(8)
    dis.addView(u.tv("Not legal advice", 15f, u.warn, true))
    dis.addView(u.tv("This app does not interpret legal text. It fetches the license text, shows it to you, and records its hash. " +
        "YOU read it and attest which uses it permits. Your attestation is stored with the model and exported with packages.", 13f, u.ink, topDp = 4))
    col.addView(dis)
    val box = u.col()
    c.call({ c.studio.model(mid) }) { m ->
        col.addView(u.tv("${m.name}: ${m.license.licenseName}", 15f, u.ink, true, 10))
        col.addView(u.badge(u.licenseText(m.license.state), u.licenseColor(m.license.state)))
        col.addView(u.tv("Authoritative URL: ${m.license.authoritativeUrl}", 12f, u.muted, topDp = 4))
        col.addView(u.button("Fetch license text from the authoritative URL", "btn:license-fetch") {
            c.call({ c.studio.fetchLicenseText(mid) }) { f -> fetchCache[mid] = f; showFetch(c, mid, f, box) }
        })
        col.addView(u.button("Import a license file instead", "btn:license-import", false) {
            val hook = StudioTestHooks.licenseFile()
            if (hook != null) {
                c.call({ c.studio.importLicenseText(mid, hook.first, java.io.ByteArrayInputStream(hook.second)) }, sticky = true) { f -> fetchCache[mid] = f; showFetch(c, mid, f, box) }
            } else c.pick(Pickers.openOne()) { data ->
                val uri = Pickers.uris(data).firstOrNull()
                if (uri != null) {
                    val cr = c.contentResolver()
                    c.call({
                        val si = Pickers.inputFrom(cr, uri)
                        si.open().use { c.studio.importLicenseText(mid, si.name, it) }
                    }, sticky = true) { f -> fetchCache[mid] = f; showFetch(c, mid, f, box) }
                }
            }
        })
        col.addView(box)
        fetchCache[mid]?.let { showFetch(c, mid, it, box) }
    }
}

private fun showFetch(c: Ctl, mid: String, f: LicenseTextFetch, box: LinearLayout) {
    val u = c.ui
    box.removeAllViews()
    val fc = u.card(12)
    fc.addView(u.tv("LICENSE TEXT", 12f, u.accent, true))
    fc.addView(u.tv("From: ${f.url}\nFetched: ${Date(f.fetchedAt)}\nSHA-256: ${f.sha256}", 12f, u.ink, topDp = 4).also { it.setTextIsSelectable(true) })
    if (f.truncated) fc.addView(u.tv("The text was truncated (too long to store in full). Read the full text at the URL before attesting.", 12f, u.warn, true, 4))
    val body = u.mono(f.text, 11f, u.ink)
    body.maxLines = 30
    body.tag = "license-text"
    fc.addView(body)
    var expanded = false
    fc.addView(u.button("Show all / fewer lines", "btn:license-expand", false) {
        expanded = !expanded; body.maxLines = if (expanded) Int.MAX_VALUE else 30
    })
    box.addView(fc)

    val ac = u.card(12)
    ac.addView(u.tv("YOUR ATTESTATION", 12f, u.accent, true))
    ac.addView(u.tv("Tick only what the text above allows. Unticked means not permitted. If fine-tuning or adapters are not permitted the model becomes DISALLOWED for this app.", 12f, u.muted, topDp = 4))
    val perms = Permission.values().map { p ->
        p to u.check("The license permits: ${p.name.lowercase().replace('_', ' ')}", "attest:" + p.name)
    }
    for ((_, cb) in perms) ac.addView(cb)
    val attr = u.check("Attribution is required", "attest:attribution")
    ac.addView(attr)
    val note = u.edit("Note (optional)", "field:attest-note")
    ac.addView(note)
    val read = u.check("I have read this license text (or the full text at the URL) myself", "attest:read")
    ac.addView(read)
    ac.addView(u.button("Record my attestation", "btn:attest") {
        if (!read.isChecked) c.error("Confirm that you read the license text first.")
        else {
            val map = perms.associate { (p, cb) -> p to (if (cb.isChecked) Tri.YES else Tri.NO) }
            val att = LicenseAttestation(f.sha256, map, attr.isChecked, note.text.toString().trim().ifEmpty { null })
            c.call({ c.studio.attestLicense(mid, att) }, sticky = true) { li ->
                c.success("License is now ${li.state}.")
                c.render()
            }
        }
    })
    box.addView(ac)
}
