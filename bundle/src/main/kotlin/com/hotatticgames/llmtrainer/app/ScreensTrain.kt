package com.hotatticgames.llmtrainer.app

import android.view.View
import android.widget.EditText
import android.widget.LinearLayout
import com.hotatticgames.llmtrainer.studio.api.CheckpointState
import com.hotatticgames.llmtrainer.studio.api.EngineStatus
import com.hotatticgames.llmtrainer.studio.api.LocalEvalOptions
import com.hotatticgames.llmtrainer.studio.api.LocalEvalState
import com.hotatticgames.llmtrainer.studio.api.LocalEvaluation
import com.hotatticgames.llmtrainer.studio.api.LocalTrainingPlan
import com.hotatticgames.llmtrainer.studio.api.ProjectId
import com.hotatticgames.llmtrainer.studio.api.RunLocation
import com.hotatticgames.llmtrainer.studio.api.SpecialistInfo
import com.hotatticgames.llmtrainer.studio.api.TrainingMethodKind
import com.hotatticgames.llmtrainer.studio.api.TrainingOption
import com.hotatticgames.llmtrainer.studio.api.TrainingRun
import com.hotatticgames.llmtrainer.studio.api.TrainingRunState
import com.hotatticgames.llmtrainer.studio.api.TrainingSettings
import java.util.Locale

// ---------------------------------------------------------------------------------------------------------
// Train on this phone: plan (LoRA / partial / full / desktop / not-training), safety gates, run control, checkpoints
// ---------------------------------------------------------------------------------------------------------

internal fun fmtDur(ms: Long): String {
    val s = ms / 1000
    return if (s >= 3600) String.format(Locale.US, "%dh %02dm", s / 3600, (s % 3600) / 60) else if (s >= 60) String.format(Locale.US, "%dm %02ds", s / 60, s % 60) else "${s}s"
}

private fun kindName(k: TrainingMethodKind) = when (k) {
    TrainingMethodKind.LOCAL_FULL -> "Fine-tune all layers"
    TrainingMethodKind.LOCAL_PARTIAL -> "Fine-tune the last layers"
    TrainingMethodKind.LOCAL_LORA -> "LoRA adapter"
    TrainingMethodKind.EXTERNAL_COMPUTE -> "Desktop job"
    TrainingMethodKind.RAG_ONLY -> "Reference package"
    TrainingMethodKind.PROMPT_ONLY -> "Prompt specialization"
}

private fun runColor(u: Ui, s: TrainingRunState) = when (s) {
    TrainingRunState.RUNNING, TrainingRunState.QUEUED -> u.info
    TrainingRunState.PAUSED -> u.warn
    TrainingRunState.SUCCEEDED -> u.ok
    TrainingRunState.FAILED -> u.bad
    TrainingRunState.CANCELLED -> u.muted
}

private class PlanData(val plan: LocalTrainingPlan, val engine: EngineStatus)

internal fun trainLocalScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    if (pid == null) { missing(c, col, "project"); return }
    val project = ProjectId(pid)

    val honest = u.card(8)
    honest.addView(u.tv("What counts as training", 15f, u.accent, true))
    honest.addView(u.tv(
        "Only a run that changes the model's parameters is training. Source excerpts in the prompt (retrieval), prompt instructions and building a dataset are NOT training and are labelled that way.\n\n" +
            "A LoRA adapter trains small extra matrices and leaves the base weights frozen: it works on quantized files and needs the least memory. " +
            "Fine-tuning can make a model forget general knowledge (the engine measured +4 nats on unrelated prose with aggressive full tuning), so the Evaluation screen measures retention and only then allows an improvement claim. " +
            "A falling training loss alone proves nothing.", 12f, u.ink, topDp = 4))
    col.addView(honest)
    val runBox = u.col()
    val planBox = u.col()
    col.addView(runBox); col.addView(planBox)

    var selected: TrainingMethodKind? = null

    fun refreshRun() {
        val ep = c.epochNow()
        c.bg({ c.studio.trainingRuns(project) }) { runs: List<TrainingRun> ->
            runBox.removeAllViews()
            val r = runs.maxByOrNull { it.createdAt }
            if (r != null) {
                runBox.addView(runCard(c, project, r, runs.size) { c.postDelayed(300, ep) { refreshRun() } })
                if (r.state == TrainingRunState.RUNNING || r.state == TrainingRunState.QUEUED) c.postDelayed(2000, ep) { refreshRun() }
            }
        }
    }
    var reloadPlan: () -> Unit = {}

    fun showPlan(d: PlanData) {
        planBox.removeAllViews()
        val plan = d.plan
        val dev = plan.device
        val dc = u.card(8)
        dc.tag = "train-conditions"
        dc.addView(u.section("THIS PHONE NOW"))
        val chg = when (dev.charging) { true -> "charger connected" ; false -> "NOT charging (blocks training)"; null -> "charger state not reported by this phone" }
        dc.addView(u.tv("Free memory ${fmtMb(dev.availableRamMb.toLong())} - free storage ${fmtMb(dev.freeStorageMb)}\n" +
            "Battery ${dev.batteryPercent?.let { "$it%" } ?: "unknown"} - $chg\nTemperature ${dev.thermal ?: "unknown"} - battery saver ${dev.powerSave?.let { if (it) "ON (blocks training)" else "off" } ?: "unknown"}", 13f, u.ink, topDp = 4))
        dc.addView(u.tv("Dataset: " + (if (plan.datasetApproved) "approved, ${plan.trainSequences} training sequences (TRAIN split only; validation and TEST text never enter training)" else "NOT approved yet"),
            12f, if (plan.datasetApproved) u.muted else u.bad, topDp = 4))
        dc.addView(u.badge(if (d.engine.trainingAvailable) "TRAINING ENGINE READY" else "TRAINING ENGINE UNAVAILABLE", if (d.engine.trainingAvailable) u.ok else u.bad))
        if (!d.engine.trainingAvailable) dc.addView(u.tv(d.engine.trainingReason ?: "The native engine is not available on this install.", 12f, u.bad))
        planBox.addView(dc)

        val plugged = u.check("I have plugged in the charger (this phone does not report charger state)", "confirm:plugged")
        val fEpochs = u.edit("Epochs", "field:train-epochs", number = true)
        val fLr = u.edit("Learning rate", "field:train-lr", number = true, decimal = true)
        val fCtx = u.edit("Context tokens", "field:train-ctx", number = true)
        val fLayers = u.edit("Layers to train (last N; 0 = all)", "field:train-layers", number = true)
        val fRank = u.edit("LoRA rank", "field:train-rank", number = true)
        val fMaxSeq = u.edit("Max training sequences", "field:train-maxseq", number = true)
        val settingsBox = u.col()

        fun fillDefaults(kind: TrainingMethodKind, opt: TrainingOption?) {
            val base = plan.defaultSettings?.takeIf { it.kind == kind } ?: TrainingSettings(kind = kind,
                trainableLastLayers = opt?.trainableLastLayers ?: 0,
                learningRate = if (kind == TrainingMethodKind.LOCAL_LORA) 2e-3f else 2e-5f, loraRank = if (kind == TrainingMethodKind.LOCAL_LORA) 8 else 0)
            fEpochs.setText(base.epochs.toString()); fLr.setText(base.learningRate.toString()); fCtx.setText(base.contextTokens.toString())
            fLayers.setText((opt?.trainableLastLayers ?: base.trainableLastLayers).toString()); fRank.setText((if (base.loraRank > 0) base.loraRank else 8).toString())
            fMaxSeq.setText(base.maxSequences.toString())
        }

        fun showSettings(opt: TrainingOption) {
            settingsBox.removeAllViews()
            val sc = u.card(8)
            sc.tag = "train-settings"
            sc.addView(u.section("SETTINGS: ${opt.label.uppercase()}"))
            for (v in listOf<View>(fEpochs, fLr, fCtx, fMaxSeq)) { (v.parent as? LinearLayout)?.removeView(v); sc.addView(v) }
            if (opt.kind == TrainingMethodKind.LOCAL_PARTIAL || opt.kind == TrainingMethodKind.LOCAL_LORA) { (fLayers.parent as? LinearLayout)?.removeView(fLayers); sc.addView(fLayers) }
            if (opt.kind == TrainingMethodKind.LOCAL_LORA) { (fRank.parent as? LinearLayout)?.removeView(fRank); sc.addView(fRank) }
            if (plan.device.charging == null) { (plugged.parent as? LinearLayout)?.removeView(plugged); sc.addView(plugged) }
            sc.addView(u.tv("Start refuses unless the engine, base model, approved dataset, license gate, memory, storage, charger, battery (>= 20%) and temperature all pass. " +
                "During a run, overheating, unplugging or low battery pauses it at its checkpoint.", 11f, u.muted, topDp = 4))
            sc.addView(u.button("Start training on this phone", "btn:train-start") {
                val epochs = fEpochs.text.toString().toIntOrNull(); val lr = fLr.text.toString().toFloatOrNull(); val ctx = fCtx.text.toString().toIntOrNull()
                val layers = fLayers.text.toString().toIntOrNull(); val rank = fRank.text.toString().toIntOrNull(); val maxSeq = fMaxSeq.text.toString().toIntOrNull()
                if (epochs == null || lr == null || ctx == null || maxSeq == null || (opt.kind != TrainingMethodKind.LOCAL_FULL && layers == null) ||
                    (opt.kind == TrainingMethodKind.LOCAL_LORA && rank == null)) {
                    c.error("Check the settings: every field must be a number.")
                } else {
                    val s = TrainingSettings(kind = opt.kind, trainableLastLayers = if (opt.kind == TrainingMethodKind.LOCAL_FULL) 0 else (layers ?: 0), epochs = epochs, learningRate = lr,
                        contextTokens = ctx, maxSequences = maxSeq, ownerConfirmsPluggedIn = plugged.isChecked,
                        loraRank = if (opt.kind == TrainingMethodKind.LOCAL_LORA) (rank ?: 8) else 0)
                    val est = opt.estimate
                    val time = if (est?.minMinutes != null && est.maxMinutes != null) "about ${est.minMinutes}-${est.maxMinutes} minutes (estimate)" else "an unknown time (not estimated until the first steps are timed)"
                    c.confirm("Train on this phone?", "${opt.label}\nThis runs for $time, keeps the phone busy, uses battery and warms the device. Keep the charger connected and the app open.\n" +
                        "Progress is checkpointed; you can pause and resume.", "Start training") {
                        c.call({ c.studio.startLocalTraining(project, s, true) }, sticky = true) { _ -> c.success("Training started."); refreshRun(); reloadPlan() }
                    }
                }
            })
            settingsBox.addView(sc)
        }

        val recommended = plan.options.firstOrNull { it.recommended }
        if (selected == null) selected = recommended?.kind
        for (opt in plan.options) {
            val oc = u.card(8)
            oc.tag = "option:" + opt.kind.name
            oc.addView(u.tv(opt.label, 16f, u.ink, true))
            oc.addView(u.badge(if (opt.isTraining) "TRAINING (changes parameters)" else "NOT TRAINING", if (opt.isTraining) u.info else u.muted))
            oc.addView(u.badge(when (opt.whereItRuns) { RunLocation.DEVICE -> "RUNS ON THIS PHONE"; RunLocation.DESKTOP -> "RUNS ON A DESKTOP/GPU (optional, you choose)"; RunLocation.NONE -> "NOTHING TO RUN" }, u.muted))
            if (opt.recommended) oc.addView(u.badge("RECOMMENDED FOR THIS PHONE", u.ok))
            oc.addView(u.badge(if (opt.available) "AVAILABLE NOW" else "BLOCKED", if (opt.available) u.ok else u.bad))
            for (r in opt.reasons) oc.addView(u.tv("- $r", 12f, u.muted))
            for (b in opt.blockers) oc.addView(u.tv("! ${b.message}", 12f, u.bad).also { it.tag = "blocker:" + b.code })
            for (r in opt.requirements) oc.addView(u.tv("Needs: $r", 11f, u.muted))
            opt.estimate?.let { e ->
                oc.addView(u.tv("Estimate: peak memory ${e.peakRamMb?.let { fmtMb(it.toLong()) } ?: "?"}, " +
                    (if (e.minMinutes != null && e.maxMinutes != null) "${e.minMinutes}-${e.maxMinutes} min" else "duration unknown") + ", working storage ${e.workingStorageMb?.let { fmtMb(it) } ?: "?"}, thermal risk ${e.thermalRisk.name.lowercase()}\n${e.basis}", 11f, u.muted, topDp = 2))
            }
            oc.addView(u.tv(opt.honestyNote, 12f, u.warn, topDp = 4))
            when (opt.kind) {
                TrainingMethodKind.LOCAL_FULL, TrainingMethodKind.LOCAL_PARTIAL, TrainingMethodKind.LOCAL_LORA -> if (opt.available) {
                    oc.addView(u.button(if (selected == opt.kind) "Selected: ${kindName(opt.kind)}" else "Choose ${kindName(opt.kind)}", "btn:train-select:" + opt.kind.name, selected == opt.kind) {
                        selected = opt.kind; showPlan(d)
                    })
                }
                TrainingMethodKind.EXTERNAL_COMPUTE -> {
                    val b = u.button("Prepare a desktop job package (optional fallback)", "btn:train-external", false) { c.go(Route(Kind.TRAINING, pid)) }
                    oc.addView(b)
                    oc.addView(u.tv("Never chosen for you: the phone only prepares a job; nothing trains on this device.", 11f, u.muted))
                }
                TrainingMethodKind.RAG_ONLY -> oc.addView(u.button("Open the reference package export", "btn:train-rag", false) { c.go(Route(Kind.TRAINING, pid)) })
                TrainingMethodKind.PROMPT_ONLY -> {}
            }
            planBox.addView(oc)
        }
        planBox.addView(settingsBox)
        val sel = plan.options.firstOrNull { it.kind == selected && it.available }
        if (sel != null) { fillDefaults(sel.kind, sel); showSettings(sel) }
        else if (plan.options.none { it.kind in setOf(TrainingMethodKind.LOCAL_FULL, TrainingMethodKind.LOCAL_PARTIAL, TrainingMethodKind.LOCAL_LORA) && it.available })
            planBox.addView(u.tv("No on-phone configuration is available right now. Fix the blockers above, or use the optional desktop job.", 13f, u.warn, true, 8))
    }

    fun loadPlan() {
        c.bg({
            val p = c.studio.localTrainingPlan(project)
            Pair(p, c.studio.engineStatus())
        }) { (pr, eng) -> c.handle(pr) { plan -> showPlan(PlanData(plan, eng)) } }
    }
    reloadPlan = { loadPlan() }
    c.resumeHook = { loadPlan(); refreshRun() }
    refreshRun()
    loadPlan()
}

private fun runCard(c: Ctl, project: ProjectId, r: TrainingRun, total: Int, after: () -> Unit): LinearLayout {
    val u = c.ui
    val rc = u.card(10)
    rc.tag = "train-run"
    rc.addView(u.section("TRAINING RUN ${r.id}" + if (total > 1) "  (latest of $total)" else ""))
    rc.addView(u.badge(r.state.name, runColor(u, r.state)).also { it.tag = "train-state" })
    rc.addView(u.tv("${kindName(r.settings.kind)} - stage ${r.stage.name.lowercase()}", 13f, u.ink, true, 4))
    val frac = if (r.steps > 0) (r.step.toDouble() / r.steps).coerceIn(0.0, 1.0) else -1.0
    rc.addView(u.bar(frac, "progress:train"))
    rc.addView(u.tv("Step ${r.step} of ${r.steps} - epoch ${r.epoch} of ${r.epochs} - ${r.examplesDone} examples - elapsed ${fmtDur(r.elapsedMs)}", 12f, u.ink))
    val first = r.lossTrend.firstOrNull { it.trainLoss != null }?.trainLoss
    rc.addView(u.tv("Training loss ${r.latestTrainLoss?.let { fmtNum(it) } ?: "-"}" + (first?.let { " (first ${fmtNum(it)})" } ?: "") + " - validation loss ${r.latestValLoss?.let { fmtNum(it) } ?: "-"}\n" +
        "Phone: temperature ${r.thermal ?: "?"}, battery ${r.batteryPercent?.let { "$it%" } ?: "?"}, process memory ${r.rssMb?.let { fmtMb(it.toLong()) } ?: "?"}", 11f, u.muted, topDp = 2))
    val ck = when (r.checkpoint) {
        CheckpointState.NONE -> "Checkpoint: none yet."
        CheckpointState.PRESENT -> "Checkpoint: saved. Pausing, an interruption or a phone restart keeps your progress; Resume continues from it."
        CheckpointState.RECOVERED_FROM_SCRATCH -> "Checkpoint: a damaged checkpoint was set aside (kept for inspection) and the run restarted from the beginning."
    }
    rc.addView(u.tv(ck + if (r.resumedFromStep > 0) " Resumed from step ${r.resumedFromStep}." else "", 12f, u.info).also { it.tag = "train-checkpoint" })
    rc.addView(u.tv(r.message, 12f, if (r.state == TrainingRunState.PAUSED) u.warn else u.muted, topDp = 2).also { it.tag = "train-message" })
    r.error?.let { rc.addView(u.tv(c.describe(it), 12f, u.bad, topDp = 4)) }
    if (r.state == TrainingRunState.RUNNING || r.state == TrainingRunState.QUEUED)
        rc.addView(u.button("Pause (keeps the checkpoint)", "btn:train-pause", false) { c.call({ c.studio.pauseTraining(r.id) }) { after() } })
    if (r.state == TrainingRunState.PAUSED && r.resumable)
        rc.addView(u.button("Resume", "btn:train-resume") { c.call({ c.studio.resumeTraining(r.id) }) { after() } })
    if (!r.isTerminal)
        rc.addView(u.button("Cancel and discard the checkpoint", "btn:train-cancel", false) {
            c.confirm("Cancel this run?", "Stops training and deletes its checkpoints. A cancelled run cannot be resumed.", "Cancel run") {
                c.call({ c.studio.cancelTraining(r.id) }) { after() }
            }
        })
    if (r.state == TrainingRunState.SUCCEEDED && r.specialistId != null) {
        val sc = u.card(8)
        sc.tag = "train-done"
        sc.addView(u.tv("Specialist ready", 15f, u.ok, true))
        sc.addView(u.tv("Parameters were changed on this phone from your approved dataset. The base model file is untouched. A lower training loss does not prove the specialist is better: evaluate it on held-out material before you rely on it.", 12f, u.ink, topDp = 4))
        sc.addView(u.button("Evaluate on held-out text", "btn:train-to-eval") { c.go(Route(Kind.EVAL, project.value)) })
        sc.addView(u.button("Compare with the base model", "btn:train-to-ab", false) { c.go(Route(Kind.AB, project.value)) })
        sc.addView(u.button("Chat with the specialist", "btn:train-to-chat", false) { c.go(Route(Kind.CHAT, project.value)) })
        sc.addView(u.button("Specialists on this phone", "btn:train-to-specialists", false) { c.go(Route(Kind.SPECIALISTS, project.value)) })
        rc.addView(sc)
    }
    return rc
}

// ---------------------------------------------------------------------------------------------------------
// Local evaluation (embedded in the Evaluation screen)
// ---------------------------------------------------------------------------------------------------------

private class EvalData(val engine: EngineStatus, val specialists: List<SpecialistInfo>, val evals: List<LocalEvaluation>)

internal fun localEvalSection(c: Ctl, project: ProjectId, box: LinearLayout) {
    val u = c.ui
    val ep = c.epochNow()
    val resultBox = u.col()

    fun showLatest(evals: List<LocalEvaluation>) {
        resultBox.removeAllViews()
        val e = evals.maxByOrNull { it.startedAt } ?: return
        val ec = u.card(10)
        ec.tag = "local-eval"
        ec.addView(u.section("LATEST EVALUATION ON THIS PHONE"))
        val color = when (e.state) { LocalEvalState.SUCCEEDED -> u.ok; LocalEvalState.RUNNING, LocalEvalState.QUEUED -> u.info; LocalEvalState.FAILED -> u.bad; else -> u.warn }
        ec.addView(u.badge(e.state.name, color).also { it.tag = "eval-state" })
        ec.addView(u.tv("${e.message}\nHeld-out TEST chunks ${e.testChunks} - ${e.itemCount} questions - dataset ${shortHash(e.datasetSha256)}", 12f, u.ink, topDp = 4))
        if (e.state == LocalEvalState.RUNNING || e.state == LocalEvalState.QUEUED) {
            ec.addView(u.bar(e.progress.fraction, "progress:eval"))
            ec.addView(u.button("Cancel evaluation", "btn:eval-cancel", false) { c.call({ c.studio.cancelLocalEvaluation(e.id) }) { c.render() } })
        }
        e.error?.let { ec.addView(u.tv(c.describe(it), 12f, u.bad, topDp = 4)) }
        if (e.state == LocalEvalState.INTERRUPTED) ec.addView(u.tv("The app was closed during the run. Nothing partial is reported; start it again.", 12f, u.warn))
        resultBox.addView(ec)
        val v = e.view
        if (e.state == LocalEvalState.SUCCEEDED && v != null) {
            val rt = u.card(8)
            rt.addView(u.section("WHAT THIS CHECKS"))
            rt.addView(u.tv("The same held-out questions (TEST split, never trained on) go to the base model and the specialist with identical greedy decoding. " +
                "Every metric is reported on its own with its sample size and a bootstrap interval; there is no single score. " +
                "Retention (fixed general probes) guards against forgetting: full tuning can raise unrelated-text loss sharply, so a gain on domain questions only counts if retention holds.", 12f, u.ink, topDp = 4))
            resultBox.addView(rt)
            showEvaluation(c, v, resultBox, local = true)
            if (e.performance.isNotEmpty()) {
                val pc = u.card(8)
                pc.tag = "eval-performance"
                pc.addView(u.section("SPEED AND MEMORY (measured on this phone)"))
                for (p in e.performance) pc.addView(u.tv("${p.subject}: ${String.format(Locale.US, "%.1f", p.tokensPerSecond)} tok/s - answer latency mean ${fmtSecs(p.latencyMsMean)}, p95 ${fmtSecs(p.latencyMsP95)} - load ${fmtSecs(p.loadMs.toDouble())}" +
                    (p.peakRssMb?.let { " - process peak $it MB" } ?: ""), 12f, u.ink))
                resultBox.addView(pc)
            }
            if (e.items.isNotEmpty()) {
                val ic = u.card(8)
                ic.tag = "eval-items"
                ic.addView(u.section("SAMPLE ANSWERS (first ${minOf(5, e.items.size)} of ${e.items.size})"))
                for (ri in e.items.take(5)) {
                    ic.addView(u.tv(ri.question, 13f, u.ink, true, 6))
                    ic.addView(u.tv("Base: ${ri.baseAnswer.take(160)} ${mark(ri.baseOk)}", 12f, u.muted))
                    ic.addView(u.tv("Specialist: ${ri.specialistAnswer.take(160)} ${mark(ri.specialistOk)}", 12f, u.muted))
                    ic.addView(u.tv("from ${ri.sourceRef}", 10f, u.muted))
                }
                resultBox.addView(ic)
            }
        }
    }

    fun load() {
        c.bg({ EvalData(c.studio.engineStatus(), c.studio.specialists(project), c.studio.localEvaluations(project)) }) { d ->
            box.removeAllViews()
            val hc = u.card(8)
            hc.tag = "local-eval-card"
            hc.addView(u.tv("Evaluate on this phone", 15f, u.ink, true))
            if (!d.engine.inferenceAvailable) {
                hc.addView(u.badge("ENGINE UNAVAILABLE", u.bad))
                hc.addView(u.tv(d.engine.inferenceReason ?: "The native engine is not available on this install.", 12f, u.bad, topDp = 4))
                hc.addView(u.tv("You can still evaluate on a desktop (below).", 12f, u.muted))
                box.addView(hc)
            } else if (d.specialists.isEmpty()) {
                hc.addView(u.tv("Train a specialist first (Train on this phone). Evaluation compares it with the base model on held-out text.", 12f, u.muted, topDp = 4))
                hc.addView(u.button("Train on this phone", "btn:eval-to-train", false) { c.go(Route(Kind.TRAIN_LOCAL, project.value)) })
                box.addView(hc)
            } else {
                val sp = d.specialists.firstOrNull { it.selected } ?: d.specialists.maxByOrNull { it.createdAt }!!
                hc.addView(u.tv("Specialist: ${sp.name} ${sp.version} (${sp.method.name.lowercase().replace('_', ' ')})" + if (sp.locallyEvaluated) " - evaluated before" else " - never evaluated", 13f, u.ink, topDp = 4))
                if (sp.stale) hc.addView(u.tv("STALE: ${sp.staleReason}", 12f, u.warn))
                hc.addView(u.tv("Uses the TEST split only. If the dataset was rebuilt after training, the app refuses when held-out chunks overlap with what the specialist saw. Takes several minutes; one model is in memory at a time and nothing else runs meanwhile.", 12f, u.muted, topDp = 4))
                val items = u.edit("Number of questions (at least 50 are needed before any improvement claim)", "field:eval-items", number = true)
                items.setText(LocalEvalOptions().maxItems.toString())
                hc.addView(items)
                hc.addView(u.button("Run evaluation on this phone", "btn:eval-local-start") {
                    val n = items.text.toString().toIntOrNull()
                    if (n == null || n < 1 || n > 1000) c.error("Enter a number of questions between 1 and 1000.")
                    else c.long({ c.studio.startLocalEvaluation(project, sp.id, LocalEvalOptions(maxItems = n)) }) { r -> c.handle(r) { c.success("Evaluation started."); load() } }
                })
                box.addView(hc)
            }
            box.addView(resultBox)
            showLatest(d.evals)
            if (d.evals.any { it.state == LocalEvalState.RUNNING || it.state == LocalEvalState.QUEUED }) c.postDelayed(2000, ep) { load() }
        }
    }
    load()
}

private fun mark(ok: Boolean?) = when (ok) { true -> "[correct]"; false -> "[wrong]"; null -> "" }
