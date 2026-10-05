package com.hotatticgames.llmtrainer.studio.api

import java.io.OutputStream
import java.security.MessageDigest

/**
 * Deterministic scripted implementation of the v2 (phone-first) surface for UI development and as an oracle for studio-core's behaviour.
 * It does NOT run a model: replies are scripted text, labelled by the model that "answered". Real runs come from studio-core + the native engine.
 *
 * Time/progress: background work (training, evaluation) advances only when [tick] is called (FakeStudio.tick delegates here).
 * Knobs for UI states: [engineAvailable], [charging], [batteryPercent], [thermal], [freeStorageMb], [simulateProcessDeath].
 */
class FakeStudioV2(private val env: Env) {
    /** What the v2 fake needs from the surrounding [FakeStudio]. */
    interface Env {
        fun now(): Long
        fun nextId(prefix: String): String
        fun projectExists(id: ProjectId): Boolean
        fun baseModel(id: ProjectId): Pair<String, String?>?          // (modelId, variantId)
        fun baseInstalled(id: ProjectId): Boolean
        fun datasetApproved(id: ProjectId): Boolean
        fun baseName(modelId: String): String
        fun projectsUsing(variantId: String): List<ProjectId>
        fun installedVariants(): List<Triple<String, String, Long>>     // (modelId, variantId, sizeBytes)
        fun variantMeta(variantId: String): Pair<String, String>?        // (display name, quant)
        fun licenseOk(modelId: String): Boolean
    }

    var engineAvailable = true
    var charging: Boolean? = true
    var batteryPercent: Int? = 88
    var thermal: String? = "NONE"
    var freeStorageMb = 6000L
    var availableRamMb = 5200

    private class Chat(val info0: ChatSessionInfo, val msgs: ArrayList<ChatMessageRecord> = ArrayList()) { var info = info0 }
    private val chats = LinkedHashMap<String, Chat>()
    private val runs = LinkedHashMap<String, TrainingRun>()
    private val specs = LinkedHashMap<String, SpecialistInfo>()
    private val evals = LinkedHashMap<String, LocalEvaluation>()
    private val abs = LinkedHashMap<String, ABComparison>()
    private val selected = HashMap<String, String>()                 // project -> specialist
    private val TOTAL_STEPS = 12

    private fun <T> ok(v: T): StudioResult<T> = StudioResult.Ok(v)
    private fun <T> err(e: StudioError): StudioResult<T> = StudioResult.Err(e)
    private fun notFound(what: String) = StudioError.NotFound(what)
    private fun sha(s: String) = MessageDigest.getInstance("SHA-256").digest(s.toByteArray()).joinToString("") { "%02x".format(it) }
    private val engineReason get() = if (engineAvailable) null else "The native engine is not available on this install (scripted state)."

    fun engineStatus() = EngineStatus(if (engineAvailable) "fake-engine" else "none-v1", engineAvailable, engineReason, engineAvailable, engineReason)

    fun installedModels() = env.installedVariants().map { (m, v, size) ->
        val (name, quant) = env.variantMeta(v) ?: (m to "")
        InstalledModelSummary(m, v, name, quant, size, if (env.licenseOk(m)) LicenseState.VERIFIED else LicenseState.UNVERIFIED, env.projectsUsing(v))
    }

    fun projectModelState(id: ProjectId): StudioResult<ProjectModelState> {
        if (!env.projectExists(id)) return err(notFound("project ${id.value}"))
        val b = env.baseModel(id); val inst = env.baseInstalled(id)
        val sp = specialists(id)
        val chatBase = when { !engineAvailable -> Capability(false, engineReason); b == null -> Capability(false, "Choose a base model first")
            !inst -> Capability(false, "Download or import the base model first"); else -> Capability(true, null) }
        val chatSp = if (!chatBase.ok) chatBase else if (selected[id.value] == null) Capability(false, "No specialist trained yet") else Capability(true, null)
        val train = trainBlockers(id).let { if (it.isEmpty()) Capability(true, null) else Capability(false, it.first().message) }
        val ev = if (sp.isEmpty()) Capability(false, "Train a specialist first") else chatBase
        return ok(ProjectModelState(id, b?.first, b?.second, inst, null, selected[id.value], sp.size, chatBase, chatSp, train, ev))
    }

    // ---------------------------------------------------------------- chat
    fun createChat(pid: ProjectId, target: ChatTarget, specialistId: String?, options: ChatOptions): StudioResult<ChatSessionInfo> {
        val st = projectModelState(pid).let { it.getOrNull() ?: return err(it.errorOrNull()!!) }
        val cap = if (target == ChatTarget.BASE) st.canChatBase else st.canChatSpecialist
        if (!cap.ok) return err(StudioError.Blocked("CHAT_UNAVAILABLE", cap.reason ?: "Unavailable", listOf(Blocker("CHAT_UNAVAILABLE", cap.reason ?: ""))))
        var spId = specialistId
        if (target == ChatTarget.SPECIALIST) {
            spId = spId ?: selected[pid.value]
            if (spId == null || specs[spId]?.projectId != pid) return err(notFound("specialist ${spId ?: "(none selected)"}"))
        }
        val label = if (target == ChatTarget.BASE) "Base: ${env.baseName(st.baseModelId!!)}" else "Specialist ${specs[spId]!!.version} (parameters changed on this phone)"
        val t = env.now()
        val info = ChatSessionInfo(env.nextId("chat"), pid, target, spId, if (target == ChatTarget.BASE) "Chat with base" else "Chat with specialist", t, t, 0, options, label)
        chats[info.id] = Chat(info)
        return ok(info)
    }

    fun listChats(pid: ProjectId) = chats.values.map { it.info }.filter { it.projectId == pid }
    fun chatHistory(id: String) = chats[id]?.let { ok(it.msgs.toList()) } ?: err(notFound("chat $id"))
    fun deleteChat(id: String) = if (chats.remove(id) != null) ok(Unit) else err(notFound("chat $id"))

    private fun scripted(target: ChatTarget, prompt: String, ctx: List<ContextChunk>): String {
        val who = if (target == ChatTarget.BASE) "Base model" else "Specialist"
        val grounded = if (ctx.isNotEmpty()) " Source excerpt used: ${ctx[0].excerpt.take(40)}." else ""
        return if (target == ChatTarget.BASE) "$who (scripted): general answer to \"${prompt.take(60)}\".$grounded"
        else "$who (scripted): domain-aware answer to \"${prompt.take(60)}\" using the specialist vocabulary.$grounded"
    }

    private fun contextFor(pid: ProjectId, use: Boolean, max: Int): List<ContextChunk> =
        if (!use) emptyList() else listOf(ContextChunk("src-1/c-0001", "shop-manual.pdf", 12, "Torque values", "Rear axle nut torque is 85 N·m (scripted sample excerpt)")).take(max)

    private fun statsFor(text: String, target: ChatTarget) = GenerationStats(
        promptTokens = 40, generatedTokens = text.split(' ').size, timeToFirstTokenMs = if (target == ChatTarget.BASE) 420.0 else 450.0,
        tokensPerSecond = if (target == ChatTarget.BASE) 6.5 else 6.3, totalMs = 3000.0, modelLoadMs = 1800, peakRssMb = if (target == ChatTarget.BASE) 3400 else 3450, stopReason = StopReason.END)

    private fun stream(text: String, cancel: CancelToken, sink: (String) -> Unit): Pair<String, Boolean> {
        val sb = StringBuilder()
        for ((i, w) in text.split(' ').withIndex()) {
            if (cancel.isCancelled) return sb.toString() to true
            val piece = if (i == 0) w else " $w"
            sb.append(piece); sink(piece)
        }
        return sb.toString() to false
    }

    fun sendMessage(chatId: String, text: String, cancel: CancelToken, onToken: (String) -> Unit): StudioResult<ChatMessageRecord> {
        val c = chats[chatId] ?: return err(notFound("chat $chatId"))
        if (text.isBlank()) return err(StudioError.Invalid("EMPTY_MESSAGE", "Type a message"))
        if (!engineAvailable) return err(StudioError.Blocked("ENGINE_UNAVAILABLE", engineReason!!, listOf(Blocker("ENGINE_UNAVAILABLE", engineReason!!))))
        if (runs.values.any { it.state == TrainingRunState.RUNNING }) return err(StudioError.Conflict("Training is running; chat is unavailable until it pauses or finishes (one engine, limited RAM)."))
        val ctx = contextFor(c.info.projectId, c.info.options.useSourceContext, c.info.options.maxContextChunks)
        c.msgs.add(ChatMessageRecord(env.nextId("msg"), ChatRole.USER, text, env.now(), emptyList(), null, false))
        val (out, interrupted) = stream(scripted(c.info.target, text, ctx), cancel, onToken)
        val rec = ChatMessageRecord(env.nextId("msg"), ChatRole.ASSISTANT, out, env.now(), ctx, statsFor(out, c.info.target).copy(stopReason = if (interrupted) StopReason.CANCELLED else StopReason.END), interrupted)
        c.msgs.add(rec); c.info = c.info.copy(updatedAt = env.now(), messageCount = c.msgs.size)
        return ok(rec)
    }

    // ---------------------------------------------------------------- training plan
    private fun conditions() = DeviceConditions(availableRamMb, freeStorageMb, batteryPercent, charging, thermal, false)

    private fun trainBlockers(pid: ProjectId): List<Blocker> = buildList {
        if (!engineAvailable) add(Blocker("ENGINE_UNAVAILABLE", engineReason!!))
        if (env.baseModel(pid) == null) add(Blocker("NO_BASE_MODEL", "Choose a base model first"))
        else if (!env.baseInstalled(pid)) add(Blocker("BASE_NOT_INSTALLED", "Download or import the base model first"))
        if (!env.datasetApproved(pid)) add(Blocker("DATASET_NOT_APPROVED", "Approve the dataset first"))
        if (charging != true) add(Blocker("NOT_CHARGING", "Plug in the charger before training; training is heavy and drains the battery"))
        if ((batteryPercent ?: 0) < 30) add(Blocker("BATTERY_LOW", "Battery is below 30%"))
        if (thermal != null && thermal != "NONE" && thermal != "LIGHT") add(Blocker("THERMAL", "Device is warm ($thermal); let it cool first"))
        if (freeStorageMb < 3000) add(Blocker("STORAGE", "Not enough free storage for checkpoints"))
    }

    fun localTrainingPlan(pid: ProjectId): StudioResult<LocalTrainingPlan> {
        if (!env.projectExists(id = pid)) return err(notFound("project ${pid.value}"))
        val base = env.baseModel(pid)
        val blockers = trainBlockers(pid)
        val partialEst = ResourceEstimate(3100, 25, 70, Risk.MEDIUM, 12, 1500, "ESTIMATE from parameter count, last 4 layers, context 512; not benchmarked")
        val fullEst = ResourceEstimate(9800, 180, 500, Risk.HIGH, 45, 5200, "ESTIMATE from parameter count, all layers; not benchmarked")
        val reqs = listOf("Plug in the charger", "Keep the app open on screen; the phone may get warm", "Free storage for checkpoints")
        val fullBlock = blockers + Blocker("TOO_LARGE", "Estimated peak ${fullEst.peakRamMb} MB exceeds the safe budget (${(availableRamMb * 0.75).toInt()} MB); it would degrade the phone")
        val opts = listOf(
            TrainingOption(TrainingMethodKind.LOCAL_FULL, "Fine-tune all layers on this phone", true, true, RunLocation.DEVICE, false, false, null,
                listOf("Highest quality potential"), fullBlock, reqs, fullEst, "Changes model parameters on this phone. Too large for this device in the safe envelope."),
            TrainingOption(TrainingMethodKind.LOCAL_PARTIAL, "Fine-tune the last 4 layers on this phone", true, true, RunLocation.DEVICE, blockers.isEmpty(), blockers.isEmpty(), 4,
                listOf("Fits the safe memory envelope", "Updates only the last transformer blocks, so gains are bounded"), blockers, reqs, partialEst,
                "Changes model parameters (a patch file, base model untouched). Partial fine-tune, not full."),
            TrainingOption(TrainingMethodKind.EXTERNAL_COMPUTE, "Train on a desktop/GPU (export a job)", true, true, RunLocation.DESKTOP, base != null && env.datasetApproved(pid), false, null,
                listOf("For jobs too big for the phone"), if (env.datasetApproved(pid)) emptyList() else listOf(Blocker("DATASET_NOT_APPROVED", "Approve the dataset first")), listOf("A desktop with llmtrainer installed"), null,
                "Optional fallback: the phone only prepares a job package; nothing trains here."),
            TrainingOption(TrainingMethodKind.RAG_ONLY, "Reference package (retrieval)", false, false, RunLocation.DEVICE, true, false, null,
                listOf("Exact facts grounded in your sources"), emptyList(), emptyList(), null, "NOT training: the model is unchanged."),
            TrainingOption(TrainingMethodKind.PROMPT_ONLY, "Prompt specialization", false, false, RunLocation.NONE, true, false, null,
                listOf("Instant, no resources"), emptyList(), emptyList(), null, "NOT training: instructions only."))
        return ok(LocalTrainingPlan(pid, base?.first, null, env.datasetApproved(pid), 480, conditions(), opts,
            if (blockers.isEmpty()) TrainingMethodKind.LOCAL_PARTIAL else null, if (blockers.isEmpty()) TrainingSettings() else null, env.now()))
    }

    // ---------------------------------------------------------------- training runs
    private fun update(id: String, f: (TrainingRun) -> TrainingRun): TrainingRun { val r = f(runs.getValue(id)).let { it.copy(updatedAt = env.now()) }; runs[id] = r; return r }

    fun startLocalTraining(pid: ProjectId, s: TrainingSettings, confirmed: Boolean): StudioResult<TrainingRun> {
        if (!env.projectExists(pid)) return err(notFound("project ${pid.value}"))
        if (s.kind != TrainingMethodKind.LOCAL_FULL && s.kind != TrainingMethodKind.LOCAL_PARTIAL)
            return err(StudioError.Invalid("NOT_LOCAL_TRAINING", "${s.kind} is not a local training method"))
        val bl = trainBlockers(pid).toMutableList()
        if (s.kind == TrainingMethodKind.LOCAL_FULL) bl.add(Blocker("TOO_LARGE", "All-layer fine-tuning does not fit this phone's safe envelope; use the partial option or a desktop job"))
        if (bl.isNotEmpty()) return err(StudioError.Blocked("TRAINING_BLOCKED", bl.first().message, bl))
        if (!confirmed) return err(StudioError.Blocked("CONFIRMATION_REQUIRED", "Confirm that you want to start training on this phone", listOf(Blocker("CONFIRMATION_REQUIRED", "Confirmation required"))))
        if (runs.values.any { it.projectId == pid && !it.isTerminal && it.state != TrainingRunState.PAUSED }) return err(StudioError.Conflict("A training run is already active"))
        val t = env.now()
        val r = TrainingRun(env.nextId("run"), pid, TrainingRunState.RUNNING, TrainingStage.PREPARE, s, env.baseModel(pid)!!.first, sha("dataset-$pid"), 480, 1, s.epochs, 0, TOTAL_STEPS, 0,
            emptyList(), null, null, 0, thermal, batteryPercent, 3100, CheckpointState.NONE, 0, "Preparing", null, true, null, t, t)
        runs[r.id] = r
        return ok(r)
    }

    fun trainingRuns(pid: ProjectId) = runs.values.filter { it.projectId == pid }
    fun trainingRun(id: String) = runs[id]?.let { ok(it) } ?: err(notFound("training run $id"))
    fun pauseTraining(id: String): StudioResult<TrainingRun> {
        val r = runs[id] ?: return err(notFound("training run $id"))
        if (r.state != TrainingRunState.RUNNING) return err(StudioError.Conflict("Run is ${r.state}"))
        return ok(update(id) { it.copy(state = TrainingRunState.PAUSED, message = "Paused at step ${it.step}; checkpoint kept", checkpoint = if (it.step > 0) CheckpointState.PRESENT else CheckpointState.NONE) })
    }
    fun cancelTraining(id: String): StudioResult<TrainingRun> {
        val r = runs[id] ?: return err(notFound("training run $id"))
        if (r.isTerminal) return err(StudioError.Conflict("Run already ${r.state}"))
        return ok(update(id) { it.copy(state = TrainingRunState.CANCELLED, resumable = false, checkpoint = CheckpointState.NONE, message = "Cancelled; checkpoints discarded") })
    }
    fun resumeTraining(id: String): StudioResult<TrainingRun> {
        val r = runs[id] ?: return err(notFound("training run $id"))
        if (r.state != TrainingRunState.PAUSED) return err(StudioError.Conflict("Only a paused run can be resumed (run is ${r.state})"))
        val bl = trainBlockers(r.projectId)
        if (bl.isNotEmpty()) return err(StudioError.Blocked("TRAINING_BLOCKED", bl.first().message, bl))
        return ok(update(id) { it.copy(state = TrainingRunState.RUNNING, resumedFromStep = it.step, message = "Resumed from step ${it.step}") })
    }

    // ---------------------------------------------------------------- specialists
    fun specialists(pid: ProjectId) = specs.values.filter { it.projectId == pid }.map { it.copy(selected = selected[pid.value] == it.id) }
    fun verifySpecialist(id: String): StudioResult<SpecialistInfo> {
        val s = specs[id] ?: return err(notFound("specialist $id"))
        val v = s.copy(verified = true, verifyMessage = "Patch hash and base hash match; reloaded by the (scripted) engine")
        specs[id] = v; return ok(v.copy(selected = selected[s.projectId.value] == id))
    }
    fun selectSpecialist(pid: ProjectId, id: String?): StudioResult<List<SpecialistInfo>> {
        if (!env.projectExists(pid)) return err(notFound("project ${pid.value}"))
        if (id == null) selected.remove(pid.value)
        else { if (specs[id]?.projectId != pid) return err(notFound("specialist $id")); selected[pid.value] = id }
        return ok(specialists(pid))
    }
    fun deleteSpecialist(id: String): StudioResult<Unit> {
        val s = specs.remove(id) ?: return err(notFound("specialist $id"))
        if (selected[s.projectId.value] == id) selected.remove(s.projectId.value)
        return ok(Unit)
    }
    fun exportSpecialistPatch(id: String, out: OutputStream): StudioResult<ExportedPackage> {
        val s = specs[id] ?: return err(notFound("specialist $id"))
        val bytes = "FAKE-SPECIALIST-PATCH\nid=${s.id}\nbaseSha=${s.baseSha256}\npatchSha=${s.patchSha256}\n".toByteArray()
        return try { out.write(bytes); out.flush(); ok(ExportedPackage("specialist-patch", "${s.name}-${s.version}.patch.zip", bytes.size.toLong(), sha(String(bytes)), listOf("manifest.json", "patch.hagpatch", "checksums.json"), emptyList())) }
        catch (e: Exception) { err(StudioError.Io("Could not write the package: ${e.message}")) }
    }

    // ---------------------------------------------------------------- local evaluation
    fun startLocalEvaluation(pid: ProjectId, specialistId: String, o: LocalEvalOptions): StudioResult<LocalEvaluation> {
        val s = specs[specialistId]?.takeIf { it.projectId == pid } ?: return err(notFound("specialist $specialistId"))
        if (!engineAvailable) return err(StudioError.Blocked("ENGINE_UNAVAILABLE", engineReason!!, listOf(Blocker("ENGINE_UNAVAILABLE", engineReason!!))))
        if (runs.values.any { it.state == TrainingRunState.RUNNING }) return err(StudioError.Conflict("Training is running"))
        val t = env.now()
        val e = LocalEvaluation(env.nextId("leval"), pid, s.id, LocalEvalState.RUNNING, Progress(0, o.maxItems.toLong(), "items"), "Running", null, null, emptyList(), emptyList(),
            s.datasetSha256, 14, o.maxItems, t, null)
        evals[e.id] = e
        evalOptions[e.id] = o
        return ok(e)
    }
    private val evalOptions = HashMap<String, LocalEvalOptions>()
    fun localEvaluations(pid: ProjectId) = evals.values.filter { it.projectId == pid }
    fun localEvaluation(id: String) = evals[id]?.let { ok(it) } ?: err(notFound("evaluation $id"))
    fun cancelLocalEvaluation(id: String): StudioResult<LocalEvaluation> {
        val e = evals[id] ?: return err(notFound("evaluation $id"))
        if (e.state != LocalEvalState.RUNNING) return err(StudioError.Conflict("Evaluation is ${e.state}"))
        val c = e.copy(state = LocalEvalState.CANCELLED, message = "Cancelled; no result", finishedAt = env.now()); evals[id] = c; return ok(c)
    }

    private fun finishEval(e: LocalEvaluation): LocalEvaluation {
        val n = e.itemCount
        val big = n >= 50
        val half = if (n >= 100) 0.03 else 0.14
        fun row(id: String, label: String, b: Double, s: Double, hib: Boolean = true) = MetricRow(id, label, hib, b, s, s - b, s - b - half, s - b + half, n)
        val metrics = listOf(
            row("terminology_coverage", "Domain terminology coverage", 0.41, 0.58), row("exact_fact_accuracy", "Exact-fact accuracy (closed book)", 0.05, 0.08),
            row("rag_unsupported_claim_rate", "Unsupported claims (with source excerpts)", 0.22, 0.15, false),
            row("heldout_nll", "Held-out negative log-likelihood", 2.9, 2.6, false),
            row("general_probe_pass_rate", "General-capability retention", 0.92, 0.9).copy(n = 12))
        val significant = n >= 100
        val allowed = big && significant
        val reason = when { !big -> "Only n=$n held-out items (< 50): no improvement claim"; !significant -> "Intervals include zero: not significant"; else -> "n=$n, every domain interval excludes zero, retention within tolerance" }
        val view = EvaluationView(e.projectId, e.id, "Base model", "Specialist ${specs[e.specialistId]?.version}", metrics,
            listOf("Heuristic lexical/numeric metrics (scripted fake data); not human judgement", "Evaluated on this phone"), allowed, reason, false, env.now())
        val perf = listOf(PerfRow("base", 2100.0, 4200.0, 6.5, 1800, 3400), PerfRow("specialist", 2150.0, 4300.0, 6.3, 1900, 3450))
        specs[e.specialistId]?.let { specs[it.id] = it.copy(locallyEvaluated = true) }
        return e.copy(state = LocalEvalState.SUCCEEDED, progress = Progress(n.toLong(), n.toLong(), "items"), message = "Done", view = view, performance = perf,
            items = listOf(LocalEvalItemResult("it-1", "fact", "Rear axle nut torque? ____", "src-1/c-0001", "about 60 N·m", "85 N·m", false, true)), finishedAt = env.now())
    }

    // ---------------------------------------------------------------- A/B
    fun compareAB(pid: ProjectId, specialistId: String, prompt: String, o: ABOptions, cancel: CancelToken, onToken: (ChatTarget, String) -> Unit): StudioResult<ABComparison> {
        val s = specs[specialistId]?.takeIf { it.projectId == pid } ?: return err(notFound("specialist $specialistId"))
        if (prompt.isBlank()) return err(StudioError.Invalid("EMPTY_MESSAGE", "Type a question"))
        if (!engineAvailable) return err(StudioError.Blocked("ENGINE_UNAVAILABLE", engineReason!!, listOf(Blocker("ENGINE_UNAVAILABLE", engineReason!!))))
        if (runs.values.any { it.state == TrainingRunState.RUNNING }) return err(StudioError.Conflict("Training is running"))
        val ctx = contextFor(pid, o.useSourceContext, o.maxContextChunks)
        fun side(t: ChatTarget, label: String): ABSide {
            val (txt, _) = stream(scripted(t, prompt, ctx), cancel) { onToken(t, it) }
            return ABSide(label, t, txt, statsFor(txt, t), ctx)
        }
        val b = side(ChatTarget.BASE, "Base: ${env.baseName(env.baseModel(pid)!!.first)}")
        if (cancel.isCancelled) return err(StudioError.Cancelled())
        val sp = side(ChatTarget.SPECIALIST, "Specialist ${s.version}")
        val c = ABComparison(env.nextId("ab"), pid, specialistId, prompt, b, sp, env.now(), null, false)
        abs[c.id] = c
        return ok(c)
    }
    fun abComparisons(pid: ProjectId) = abs.values.filter { it.projectId == pid }
    fun saveABNote(pid: ProjectId, id: String, note: String): StudioResult<ABComparison> {
        val c = abs[id]?.takeIf { it.projectId == pid } ?: return err(notFound("comparison $id"))
        val n = c.copy(note = note, savedAsNote = true); abs[id] = n; return ok(n)
    }
    fun deleteAB(pid: ProjectId, id: String): StudioResult<Unit> = if (abs[id]?.projectId == pid) { abs.remove(id); ok(Unit) } else err(notFound("comparison $id"))

    // ---------------------------------------------------------------- scripted time
    /** Advances RUNNING training runs by 3 steps and RUNNING evaluations to completion. */
    fun tick() {
        for (r in runs.values.toList().filter { it.state == TrainingRunState.RUNNING }) {
            val step = minOf(TOTAL_STEPS, r.step + 3)
            val trend = r.lossTrend + (r.step + 1..step).map { LossPoint(it, 2.8 - it * 0.12, if (it % 3 == 0) 2.9 - it * 0.1 else null) }
            val last = trend.last()
            if (step >= TOTAL_STEPS) {
                val spId = env.nextId("spec")
                val n = specs.values.count { it.projectId == r.projectId } + 1
                specs[spId] = SpecialistInfo(spId, r.projectId, "Specialist", "1.0.$n", env.now(), r.baseModelId, env.baseModel(r.projectId)?.second, sha("base-${r.baseModelId}"), sha("patch-$spId"),
                    48_000_000, r.datasetSha256, r.id, r.settings.kind, r.settings, last.trainLoss, trend.lastOrNull { it.valLoss != null }?.valLoss, TOTAL_STEPS, 480L * r.epochs,
                    true, "Patch hash and base hash verified", false, false,
                    "Parameters changed on this device by fine-tuning on your approved dataset. The base model file is untouched.")
                selected.putIfAbsent(r.projectId.value, spId)
                update(r.id) { it.copy(state = TrainingRunState.SUCCEEDED, stage = TrainingStage.DONE, step = step, examplesDone = 480L * it.epochs, lossTrend = trend, latestTrainLoss = last.trainLoss,
                    latestValLoss = trend.lastOrNull { p -> p.valLoss != null }?.valLoss, specialistId = spId, resumable = false, checkpoint = CheckpointState.PRESENT, message = "Specialist ready") }
            } else update(r.id) { it.copy(stage = TrainingStage.TRAIN, step = step, examplesDone = step * 40L, lossTrend = trend, latestTrainLoss = last.trainLoss,
                latestValLoss = trend.lastOrNull { p -> p.valLoss != null }?.valLoss ?: it.latestValLoss, elapsedMs = it.elapsedMs + 60_000, message = "Training step $step of $TOTAL_STEPS",
                checkpoint = CheckpointState.PRESENT) }
        }
        for (e in evals.values.toList().filter { it.state == LocalEvalState.RUNNING }) evals[e.id] = finishEval(e)
    }

    /** RUNNING training -> PAUSED (resumable, checkpoint kept); RUNNING evaluation -> INTERRUPTED (rerun; nothing partial is reported). */
    fun simulateProcessDeath() {
        for (r in runs.values.toList().filter { it.state == TrainingRunState.RUNNING })
            update(r.id) { it.copy(state = TrainingRunState.PAUSED, message = "Interrupted (app was closed); resume to continue from step ${it.step}", checkpoint = if (it.step > 0) CheckpointState.PRESENT else CheckpointState.NONE) }
        for (e in evals.values.toList().filter { it.state == LocalEvalState.RUNNING }) evals[e.id] = e.copy(state = LocalEvalState.INTERRUPTED, message = "Interrupted; start the evaluation again", finishedAt = env.now())
    }
}
