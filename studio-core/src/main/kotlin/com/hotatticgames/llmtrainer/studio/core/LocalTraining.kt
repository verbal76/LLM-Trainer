package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.qualify.SafetyPolicy
import com.hotatticgames.llmtrainer.studio.api.*
import java.io.File
import java.io.IOException
import java.util.Random
import org.json.JSONObject

// ===== device conditions & safety gates =====================================================================================

object Conditions {
    private val THERMAL = listOf("NONE", "LIGHT", "MODERATE", "SEVERE", "CRITICAL", "EMERGENCY", "SHUTDOWN")
    const val MIB = 1024.0 * 1024.0

    fun thermalIndex(o: JSONObject): Int? = o.opt("thermalStatus").let { v ->
        when (v) { is Number -> v.toInt().takeIf { it in 0..6 }; is String -> THERMAL.indexOf(v.uppercase()).takeIf { it >= 0 }; else -> null }
    }

    fun parse(snapshotJson: String): DeviceConditions {
        val o = J.parseOrNull(snapshotJson) ?: JSONObject()
        val avail = o.dbl("availRamBytes"); val proc = o.dbl("procMemAvailableBytes")
        val thr = o.dbl("lowMemoryThresholdBytes") ?: 0.0
        val ramMb = when {
            o.bool("lowMemory") == true -> 0
            avail == null && proc == null -> 0                  // unknown: treated as nothing available (fail closed)
            else -> maxOf(0.0, (maxOf(avail ?: 0.0, proc ?: 0.0) - thr) / MIB).toInt()
        }
        val charging = listOf("isCharging", "charging", "batteryCharging").firstNotNullOfOrNull { o.bool(it) }
            ?: o.int("plugged")?.let { it > 0 }
        return DeviceConditions(ramMb, ((o.dbl("freeStorageBytes") ?: 0.0) / MIB).toLong(), o.dbl("batteryPct")?.toInt(), charging,
            thermalIndex(o)?.let { THERMAL[it] }, o.bool("powerSaveMode"))
    }

    fun thermalLevel(c: DeviceConditions): Int? = c.thermal?.let { THERMAL.indexOf(it).takeIf { i -> i >= 0 } }
}

object TrainGates {
    const val MIN_BATTERY = 20
    const val ABORT_BATTERY = 15
    const val STORAGE_RESERVE_MB = 1024L

    /** Conditions that must hold to START or RESUME. Fails closed on unknowns that matter (RAM, charging unless the owner confirms). */
    fun device(c: DeviceConditions, s: TrainingSettings): List<Blocker> = buildList {
        when (c.charging) {
            true -> {}
            false -> add(Blocker("NOT_CHARGING", "Plug in the charger first: training is heavy, drains the battery and heats the phone."))
            null -> if (!s.ownerConfirmsPluggedIn) add(Blocker("CHARGER_UNKNOWN", "This device does not report charger state. Plug in the charger and confirm it, then try again."))
        }
        val bat = c.batteryPercent
        if (bat != null && bat < MIN_BATTERY) add(Blocker("BATTERY_LOW", "Battery is at $bat%; training needs at least $MIN_BATTERY% (and the charger)."))
        val t = Conditions.thermalLevel(c)
        if (t != null && t >= 2) add(Blocker("THERMAL", "The device is warm (${c.thermal}). Let it cool down before training."))
        if (c.powerSave == true) add(Blocker("POWER_SAVE", "Battery saver is on and throttles the CPU. Turn it off for training."))
        if (c.availableRamMb <= 0) add(Blocker("RAM_UNKNOWN_OR_LOW", "Available memory is unknown or the system reports low memory; close other apps."))
    }

    fun reserveMb(totalHintMb: Int, p: SafetyPolicy) = maxOf(p.safetyReserveMinMb, p.safetyReserveFrac * totalHintMb)

    /** Memory/storage fit against the engine's own estimate, with the configured safety reserve. */
    fun fit(c: DeviceConditions, est: TrainEstimate, requiredStorageMb: Long, p: SafetyPolicy): List<Blocker> = buildList {
        if (!est.trainable) add(Blocker("NOT_TRAINABLE", est.reason ?: "The engine reports that this model cannot be trained with these settings."))
        val peakMb = est.peakBytes / Conditions.MIB
        val budget = c.availableRamMb - reserveMb(c.availableRamMb, p)
        if (est.trainable && peakMb > budget)
            add(Blocker("TOO_LARGE", "Estimated peak memory ${peakMb.toInt()} MB exceeds the safe budget ${maxOf(0.0, budget).toInt()} MB (available ${c.availableRamMb} MB minus the safety reserve). " +
                "Running it would degrade the phone; choose a smaller configuration${"" } or export a desktop job."))
        if (c.freeStorageMb < requiredStorageMb + STORAGE_RESERVE_MB)
            add(Blocker("STORAGE", "Needs about ${requiredStorageMb} MB for checkpoints plus a ${STORAGE_RESERVE_MB} MB reserve; only ${c.freeStorageMb} MB is free."))
    }

    /** During a run: returns a reason to pause now, or null. */
    fun runtimeGuard(c: DeviceConditions, s: TrainingSettings): String? {
        val t = Conditions.thermalLevel(c)
        if (t != null && t >= 3) return "Paused: the device is overheating (${c.thermal}). Let it cool, then resume."
        if (c.charging == false) return "Paused: the charger was disconnected. Plug in to resume."
        val bat = c.batteryPercent
        if (bat != null && bat < ABORT_BATTERY && c.charging != true) return "Paused: battery is at $bat%. Plug in to resume."
        if (c.availableRamMb in 1..255 || c.availableRamMb == 0) return "Paused: the system is low on memory. Close other apps, then resume."
        return null
    }
}

// ===== training sequences (provenance-carrying, deterministic, memory-bounded) ==================================================

class SequenceSet(val texts: List<String>, val rows: List<Map<String, Any?>>, val sha256: String, val exampleCount: Int, val droppedExamples: Int, val sourceIds: Set<String>)

object SequenceBuilder {
    const val MIN_SEQUENCES = 8
    private const val CHARS_PER_TOKEN = 3

    /** Text the model trains on: the source sentence itself (source-derived) or the template Q/A (synthetic, train-only). */
    fun textOf(e: Example): String = if (e.origin == "source_derived") ExampleGen.leakText(e.prompt, e.response) else e.prompt + "\n" + e.response

    /**
     * Only the TRAIN split of the user-approved dataset; the validation and TEST splits never enter. Leakage is re-verified on the final rows.
     * Deterministic for (dataset, seed, contextTokens, maxSequences).
     */
    fun build(view: DataView, s: TrainingSettings): SequenceSet {
        val a = view.assembled
        val problems = DatasetEngine.verify(a)
        if (problems.isNotEmpty()) throw blockedEx("LEAKAGE_DETECTED", "The dataset has train/evaluation leakage; fix it before training: ${problems.first()}")
        val train = a.bySplit["train"].orEmpty()
        if (train.any { it.split != "train" }) throw StudioException(StudioError.Invalid("SPLIT_VIOLATION", "Internal error: a non-train example reached the training set"))
        val ordered = train.sortedBy { it.id }.toMutableList()
        ordered.shuffle(Random(s.seed))
        val maxChars = maxOf(64, s.contextTokens * CHARS_PER_TOKEN)
        val texts = ArrayList<String>(); val rows = ArrayList<Map<String, Any?>>()
        val cur = StringBuilder(); val curIds = ArrayList<String>(); val curRefs = LinkedHashSet<String>()
        val srcs = LinkedHashSet<String>()
        var used = 0; var dropped = 0
        fun flush() {
            if (cur.isEmpty()) return
            rows.add(linkedMapOf("i" to texts.size, "text" to cur.toString(), "examples" to ArrayList(curIds), "chunks" to ArrayList(curRefs)))
            texts.add(cur.toString()); cur.setLength(0); curIds.clear(); curRefs.clear()
        }
        for (e in ordered) {
            if (texts.size >= s.maxSequences) { dropped++; continue }
            var t = textOf(e).trim()
            if (t.isEmpty()) continue
            if (t.length > maxChars) t = t.take(maxChars)
            if (cur.isNotEmpty() && cur.length + 2 + t.length > maxChars) { flush(); if (texts.size >= s.maxSequences) { dropped++; continue } }
            if (cur.isNotEmpty()) cur.append("\n\n")
            cur.append(t); curIds.add(e.id); curRefs.add(e.sourceId + "/" + e.chunkId); srcs.add(e.sourceId); used++
        }
        flush()
        return SequenceSet(texts, rows, Hashing.sha256(J.canonical(texts)), used, dropped, srcs)
    }
}

// ===== the service ==========================================================================================================

class TrainingService(
    private val host: LocalHost, private val eng: EngineCore, private val specs: SpecialistRegistry, private val hashes: FileHashes,
    private val clock: Clock, private val ids: IdSource, private val runner: TaskRunner, private val policy: SafetyPolicy, private val problems: MutableList<String>,
) {
    private enum class Intent { NONE, PAUSE, CANCEL }

    private class RunRec(var run: TrainingRun, val dir: File, val basePath: String, val sources: Map<String, String>, val sequencesSha: String, var baseSha: String?) {
        @Volatile var intent = Intent.NONE
        @Volatile var cancel: CancelToken? = null
        @Volatile var live = false
        @Volatile var guardMessage: String? = null
        var lastPersist = 0L
        var segmentBaseElapsed = 0L
    }

    private val lock = Any()
    private val runs = LinkedHashMap<String, RunRec>()

    private fun trainDir(pid: ProjectId): File? = host.projectDir(pid)?.let { File(it, "training") }

    // ---- persistence -------------------------------------------------------------------------------------------------------

    private fun persist(r: RunRec) {
        val m = LinkedHashMap(LJ.run(r.run))
        m["internal"] = linkedMapOf("base_path" to r.basePath, "sources" to r.sources, "sequences_sha" to r.sequencesSha, "base_sha" to r.baseSha)
        Fs.writeJson(File(r.dir, "run.json"), m)
        r.lastPersist = clock.nowMs()
    }

    private fun update(r: RunRec, force: Boolean = true, f: (TrainingRun) -> TrainingRun) {
        synchronized(lock) { r.run = f(r.run).let { it.copy(updatedAt = clock.nowMs()) } }
        if (force || clock.nowMs() - r.lastPersist >= 1500) try { persist(r) } catch (e: IOException) { problems.add("run ${r.run.id}: could not persist state: ${e.message}") }
    }

    /** Called once at startup per project: reloads runs; anything that was RUNNING/QUEUED when the process died becomes PAUSED (resumable). */
    fun load(pid: ProjectId) {
        val root = trainDir(pid) ?: return
        synchronized(lock) { runs.values.removeAll { it.run.projectId == pid } }
        root.listFiles { f -> f.isDirectory && !f.name.startsWith(".") }?.sortedBy { it.name }?.forEach { d ->
            val o = Fs.readJson(File(d, "run.json")) { problems.add("training run ${d.name}: $it") } ?: return@forEach
            try {
                val internal = o.obj("internal")
                var run = LJ.run(o)
                val hasSeq = File(d, "sequences.jsonl").isFile
                val rec = RunRec(run, d, internal?.str("base_path") ?: "", internal?.obj("sources")?.let { s -> s.keyList().associateWith { s.str(it) ?: "" } } ?: emptyMap(), internal?.str("sequences_sha") ?: "", internal?.str("base_sha"))
                if (run.state == TrainingRunState.RUNNING || run.state == TrainingRunState.QUEUED) {
                    val ck = if (hasCheckpoint(rec)) CheckpointState.PRESENT else CheckpointState.NONE
                    run = run.copy(state = TrainingRunState.PAUSED, resumable = hasSeq, checkpoint = if (run.checkpoint == CheckpointState.RECOVERED_FROM_SCRATCH && ck != CheckpointState.NONE) run.checkpoint else ck,
                        message = "Interrupted (the app was closed or the phone restarted). " + if (ck == CheckpointState.PRESENT) "Resume to continue from the last checkpoint." else "Resume to start again from the beginning.")
                    rec.run = run; persist(rec)
                }
                synchronized(lock) { runs[run.id] = rec }
            } catch (e: Exception) { problems.add("training run ${d.name} unreadable: ${e.message}") }
        }
    }

    fun forget(pid: ProjectId) { synchronized(lock) { runs.values.filter { it.run.projectId == pid }.forEach { it.cancel?.cancel() }; runs.values.removeAll { it.run.projectId == pid } } }

    private fun hasCheckpoint(r: RunRec) = File(r.dir, "work").let { w -> w.isDirectory && (w.list()?.isNotEmpty() == true) }

    // ---- plan --------------------------------------------------------------------------------------------------------------

    private fun paramsOf(s: TrainingSettings, kind: TrainingMethodKind, budgetBytes: Long) = TrainParams(
        contextTokens = s.contextTokens, batchTokens = s.contextTokens, epochs = s.epochs, learningRate = s.learningRate, valFraction = 0.1f, seed = s.seed,
        trainableLastLayers = if (kind == TrainingMethodKind.LOCAL_FULL) 0 else s.trainableLastLayers.coerceAtLeast(1), trainEmbeddings = s.trainEmbeddings,
        checkpointEverySteps = s.checkpointEverySteps, maxMemoryBytes = budgetBytes)

    private fun storageMb(est: TrainEstimate?): Long {
        val p = est?.trainableParams ?: return 2048
        return maxOf(256L, (p * 4 * 3 / Conditions.MIB).toLong())       // fp32 weights + optimizer state worst case + patch
    }

    private fun resource(est: TrainEstimate, seqs: Int, s: TrainingSettings, basis: String): ResourceEstimate {
        val steps = seqs.toLong() * s.epochs
        val mins = est.secondsPerStep?.let { sps -> Math.ceil(steps * sps / 60.0).toInt() }
        val risk = when { mins == null -> Risk.UNKNOWN; mins > 45 -> Risk.HIGH; mins > 15 -> Risk.MEDIUM; else -> Risk.LOW }
        return ResourceEstimate(Math.ceil(est.peakBytes / Conditions.MIB).toInt(), mins?.let { maxOf(1, (it * 0.7).toInt()) }, mins?.let { (it * 1.5).toInt() + 1 }, risk, null, storageMb(est),
            "$basis; battery use is not estimated until measured on this device" + if (mins == null) "; duration unknown until the first steps are timed" else "")
    }

    fun plan(pid: ProjectId): LocalTrainingPlan {
        if (host.projectName(pid) == null) throw StudioException(StudioError.NotFound("project ${pid.value}"))
        val c = Conditions.parse(host.snapshotJson())
        val base = host.baseRef(pid)
        val dv = host.dataView(pid)
        val approved = dv?.status == DatasetStatus.APPROVED
        val trStatus = eng.trainingStatus()
        val common = ArrayList<Blocker>()
        if (!trStatus.available) common.add(Blocker("ENGINE_UNAVAILABLE", trStatus.reason ?: "The native training engine is not available on this install."))
        if (base == null) common.add(Blocker("NO_BASE_MODEL", "Choose a base model first"))
        else {
            common.addAll(base.gate)
            if (base.file?.isFile != true) common.add(Blocker("BASE_NOT_INSTALLED", "The base model is not on this phone yet. Download or import it first."))
        }
        if (dv == null || dv.status == DatasetStatus.NONE) common.add(Blocker("NO_DATASET", "Build a dataset first"))
        else if (!approved) common.add(Blocker("DATASET_NOT_APPROVED", "Review and approve the dataset first (status ${dv.status})"))
        else if (!dv.splitsAvailable) common.add(Blocker("NO_SPLITS", "Fewer than 3 independent document/section groups: no honest train/held-out split exists. Add more documents."))
        var seqs = 0
        var datasetSha: String? = null
        if (dv != null && approved && dv.splitsAvailable) {
            datasetSha = dv.sha
            try { seqs = SequenceBuilder.build(dv, TrainingSettings()).texts.size } catch (e: StudioException) { common.add((e.error as? StudioError.Blocked)?.reasons?.firstOrNull() ?: Blocker(e.error.code, e.error.message)) }
            if (seqs in 1 until SequenceBuilder.MIN_SEQUENCES) common.add(Blocker("TOO_LITTLE_DATA", "Only $seqs training sequences; at least ${SequenceBuilder.MIN_SEQUENCES} are needed for a meaningful fine-tune."))
            if (seqs == 0 && common.none { it.code == "LEAKAGE_DETECTED" }) common.add(Blocker("NO_TRAINING_DATA", "The approved dataset has no training examples."))
        }
        val dev = TrainGates.device(c, TrainingSettings())
        val desktopOk = base != null && base.gate.isEmpty() && approved && dv?.splitsAvailable == true
        val desktopBlockers = if (desktopOk) emptyList() else common.filter { it.code in setOf("NO_BASE_MODEL", "NO_DATASET", "DATASET_NOT_APPROVED", "NO_SPLITS", "LEAKAGE_DETECTED") || it.code.startsWith("LICENSE") }

        val reqs = listOf("Plug in the charger and keep the phone cool", "Keep the app open (a long run can take a while)", "Free storage for checkpoints")
        val canEstimate = trStatus.available && base?.file?.isFile == true
        val defaults = TrainingSettings()
        val budgetBytes = ((c.availableRamMb - TrainGates.reserveMb(c.availableRamMb, policy)).coerceAtLeast(0.0) * Conditions.MIB).toLong()

        fun localOption(kind: TrainingMethodKind, layers: Int?): TrainingOption {
            val s = defaults.copy(kind = kind, trainableLastLayers = layers ?: 0)
            val blockers = ArrayList(common)
            var estimate: ResourceEstimate? = null
            if (canEstimate) {
                val est = try { eng.trainer!!.estimate(base!!.file!!.absolutePath, paramsOf(s, kind, budgetBytes)) } catch (e: BackendException) { TrainEstimate(false, 0, null, e.message) }
                estimate = if (est.peakBytes > 0) resource(est, seqs, s, "ENGINE ESTIMATE for ${if (kind == TrainingMethodKind.LOCAL_FULL) "all layers" else "last $layers layers"}, context ${s.contextTokens}") else null
                blockers.addAll(TrainGates.fit(c, est, storageMb(est), policy))
            }
            blockers.addAll(dev)
            val reasons = ArrayList<String>()
            if (kind == TrainingMethodKind.LOCAL_FULL) reasons.add("Updates every layer: highest quality potential, largest memory need")
            else reasons.add("Updates only the last $layers transformer block(s): smaller memory need, bounded gains")
            if (blockers.any { it.code == "TOO_LARGE" || it.code == "NOT_TRAINABLE" }) reasons.add("A desktop job is available as an optional alternative for configurations too big for the phone")
            return TrainingOption(kind, if (kind == TrainingMethodKind.LOCAL_FULL) "Fine-tune all layers on this phone" else "Fine-tune the last $layers layers on this phone", true, true, RunLocation.DEVICE,
                blockers.isEmpty(), false, layers, reasons, blockers.distinctBy { it.code }, reqs, estimate,
                "Changes model parameters on this phone and produces a patch file; the base model file is untouched." + if (kind == TrainingMethodKind.LOCAL_PARTIAL) " Partial fine-tune, not full." else "")
        }

        val full = localOption(TrainingMethodKind.LOCAL_FULL, null)
        // largest partial configuration that fits; if none fits, show the smallest with its blockers
        val partials = listOf(8, 4, 2, 1).map { localOption(TrainingMethodKind.LOCAL_PARTIAL, it) }
        val partial = partials.firstOrNull { it.available } ?: partials.last()
        val recommendedKind = when { full.available -> TrainingMethodKind.LOCAL_FULL; partial.available -> TrainingMethodKind.LOCAL_PARTIAL; else -> null }
        val options = listOf(
            full.copy(recommended = recommendedKind == TrainingMethodKind.LOCAL_FULL),
            partial.copy(recommended = recommendedKind == TrainingMethodKind.LOCAL_PARTIAL),
            TrainingOption(TrainingMethodKind.EXTERNAL_COMPUTE, "Train on a desktop/GPU (export a job)", true, true, RunLocation.DESKTOP, desktopBlockers.isEmpty() && desktopOk, false, null,
                listOf("Optional fallback for jobs too big for the phone"), desktopBlockers, listOf("A desktop with llmtrainer installed"), null,
                "The phone only prepares a job package; nothing trains on this device. You choose whether to use it."),
            TrainingOption(TrainingMethodKind.RAG_ONLY, "Reference package (retrieval)", false, false, RunLocation.DEVICE, true, false, null,
                listOf("Grounds answers in your exact source text"), emptyList(), emptyList(), null, "NOT training: the model's parameters are unchanged."),
            TrainingOption(TrainingMethodKind.PROMPT_ONLY, "Prompt specialization", false, false, RunLocation.NONE, true, false, null,
                listOf("Instant; no resources"), emptyList(), emptyList(), null, "NOT training: instructions only, no new knowledge."))
        val rec = options.firstOrNull { it.recommended }
        val defaultSettings = rec?.let { defaults.copy(kind = it.kind, trainableLastLayers = it.trainableLastLayers ?: 0) }
        return LocalTrainingPlan(pid, base?.modelId, datasetSha, approved, seqs, c, options, recommendedKind, defaultSettings, clock.nowMs())
    }

    // ---- start / pause / cancel / resume -----------------------------------------------------------------------------------

    private fun runDir(pid: ProjectId, id: String) = File(trainDir(pid) ?: throw StudioException(StudioError.NotFound("project ${pid.value}")), id)

    private fun gateOrThrow(pid: ProjectId, s: TrainingSettings): Pair<BaseRef, DataView> {
        val plan = plan(pid)
        val base = host.baseRef(pid)!!
        val dv = host.dataView(pid)!!
        val opt = plan.options.firstOrNull { it.kind == s.kind }
        // recompute with the caller's exact settings: the plan's option may use other layer counts
        val c = Conditions.parse(host.snapshotJson())
        val bl = ArrayList<Blocker>()
        opt?.blockers?.filter { it.code != "TOO_LARGE" && it.code != "STORAGE" && it.code != "NOT_TRAINABLE" && !it.code.startsWith("CHARGER") && it.code != "NOT_CHARGING" && it.code != "BATTERY_LOW" && it.code != "THERMAL" && it.code != "POWER_SAVE" && it.code != "RAM_UNKNOWN_OR_LOW" }?.let { bl.addAll(it) }
        bl.addAll(TrainGates.device(c, s))
        if (bl.none { it.code == "ENGINE_UNAVAILABLE" || it.code == "BASE_NOT_INSTALLED" }) {
            val budget = ((c.availableRamMb - TrainGates.reserveMb(c.availableRamMb, policy)).coerceAtLeast(0.0) * Conditions.MIB).toLong()
            val est = try { eng.trainer!!.estimate(base.file!!.absolutePath, paramsOf(s, s.kind, budget)) } catch (e: BackendException) { TrainEstimate(false, 0, null, e.message) }
            bl.addAll(TrainGates.fit(c, est, storageMb(est), policy))
        }
        val distinct = bl.distinctBy { it.code }
        if (distinct.isNotEmpty()) {
            val extra = if (distinct.any { it.code == "TOO_LARGE" || it.code == "NOT_TRAINABLE" }) " A desktop job is available as an optional alternative." else ""
            throw StudioException(StudioError.Blocked("TRAINING_BLOCKED", distinct.first().message + extra, distinct))
        }
        return base to dv
    }

    fun start(pid: ProjectId, s: TrainingSettings, confirmed: Boolean): TrainingRun {
        if (host.projectDir(pid) == null) throw StudioException(StudioError.NotFound("project ${pid.value}"))
        if (s.kind != TrainingMethodKind.LOCAL_FULL && s.kind != TrainingMethodKind.LOCAL_PARTIAL)
            throw StudioException(StudioError.Invalid("NOT_LOCAL_TRAINING", "${s.kind} does not train on this phone. Retrieval and prompting are not training; desktop jobs use the export flow."))
        if (s.epochs < 1 || s.epochs > 20 || s.contextTokens !in 64..8192 || s.maxSequences !in 1..100_000 || s.learningRate <= 0f || s.learningRate > 1f || s.trainableLastLayers < 0)
            throw StudioException(StudioError.Invalid("BAD_SETTINGS", "Training settings are out of range"))
        synchronized(lock) {
            runs.values.firstOrNull { it.run.projectId == pid && !it.run.isTerminal }?.let {
                throw StudioException(StudioError.Conflict("Run ${it.run.id} is ${it.run.state.name.lowercase()}; resume or cancel it before starting another."))
            }
        }
        val (base, dv) = gateOrThrow(pid, s)
        if (!confirmed) throw blockedEx("CONFIRMATION_REQUIRED", "Confirm that you want to train on this phone: it runs for a while, uses the battery and warms the device.")
        val seq = SequenceBuilder.build(dv, s)
        if (seq.texts.size < SequenceBuilder.MIN_SEQUENCES) throw blockedEx("TOO_LITTLE_DATA", "Only ${seq.texts.size} training sequences; at least ${SequenceBuilder.MIN_SEQUENCES} are needed.")
        val facts = host.sourceFacts(pid)
        val used = seq.sourceIds.associateWith { facts[it]?.sha256 ?: "" }
        val id = "run-" + ids.hex(10)
        val dir = runDir(pid, id).also { it.mkdirs() }
        Fs.writeAtomic(File(dir, "sequences.jsonl")) { out -> seq.rows.forEach { out.write(J.bytes(it)); out.write('\n'.code) } }
        val now = clock.nowMs()
        val epochs = s.epochs
        val run = TrainingRun(id, pid, TrainingRunState.QUEUED, TrainingStage.PREFLIGHT, s, base.modelId, dv.sha, seq.texts.size, 1, epochs, 0, seq.texts.size * epochs, 0, emptyList(), null, null, 0,
            Conditions.parse(host.snapshotJson()).thermal, Conditions.parse(host.snapshotJson()).batteryPercent, null, CheckpointState.NONE, 0,
            "Queued" + if (seq.droppedExamples > 0) " (${seq.droppedExamples} examples left out by the sequence cap)" else "", null, true, null, now, now)
        val rec = RunRec(run, dir, base.file!!.absolutePath, used, seq.sha256, null)
        synchronized(lock) { runs[id] = rec }
        persist(rec)
        runner.submit(Runnable { execute(rec) })
        return rec.run
    }

    fun list(pid: ProjectId): List<TrainingRun> = synchronized(lock) { runs.values.filter { it.run.projectId == pid }.map { it.run } }
    fun get(id: String): TrainingRun? = synchronized(lock) { runs[id]?.run }

    fun pause(id: String): TrainingRun {
        val r = synchronized(lock) { runs[id] } ?: throw StudioException(StudioError.NotFound("training run $id"))
        if (r.run.isTerminal) throw StudioException(StudioError.Conflict("Run already ${r.run.state.name.lowercase()}"))
        if (r.run.state == TrainingRunState.PAUSED) return r.run
        r.intent = Intent.PAUSE
        r.cancel?.cancel()
        if (!r.live) update(r) { it.copy(state = TrainingRunState.PAUSED, message = "Paused", checkpoint = if (hasCheckpoint(r)) CheckpointState.PRESENT else CheckpointState.NONE) }
        return r.run
    }

    fun cancel(id: String): TrainingRun {
        val r = synchronized(lock) { runs[id] } ?: throw StudioException(StudioError.NotFound("training run $id"))
        if (r.run.isTerminal) throw StudioException(StudioError.Conflict("Run already ${r.run.state.name.lowercase()}"))
        r.intent = Intent.CANCEL
        r.cancel?.cancel()
        if (!r.live) { File(r.dir, "work").deleteRecursively(); update(r) { it.copy(state = TrainingRunState.CANCELLED, resumable = false, checkpoint = CheckpointState.NONE, message = "Cancelled; checkpoints discarded") } }
        return r.run
    }

    fun resume(id: String): TrainingRun {
        val r = synchronized(lock) { runs[id] } ?: throw StudioException(StudioError.NotFound("training run $id"))
        if (r.run.state != TrainingRunState.PAUSED || !r.run.resumable) throw StudioException(StudioError.Conflict("Only a paused, resumable run can be resumed (this one is ${r.run.state.name.lowercase()})"))
        val pid = r.run.projectId
        // rights: never continue training on material that was removed or is no longer cleared
        val facts = host.sourceFacts(pid)
        val bad = r.sources.filter { (sid, sha) -> facts[sid]?.let { it.sha256 != sha || !it.trainable } != false }.keys
        val gone = r.sources.keys.filter { it !in facts }
        if (gone.isNotEmpty() || r.sources.any { (sid, sha) -> facts[sid]?.let { it.sha256 != sha || !it.trainable } == true }) {
            File(r.dir, "work").deleteRecursively()
            update(r) { it.copy(state = TrainingRunState.CANCELLED, resumable = false, checkpoint = CheckpointState.NONE, message = "Cancelled: a source this run was using was removed, changed or is no longer cleared for training. Start a new run.") }
            throw blockedEx("SOURCE_CHANGED", "A source used by this run was removed or changed (${(gone + bad).distinct().size}). The run was cancelled and its checkpoints discarded; start a new run from the current dataset.")
        }
        if (!File(r.dir, "sequences.jsonl").isFile) throw StudioException(StudioError.Invalid("INPUTS_MISSING", "The saved training inputs are missing; start a new run."))
        val base = host.baseRef(pid)
        if (base?.file?.absolutePath != r.basePath || base.file?.isFile != true) throw blockedEx("BASE_CHANGED", "The base model is not the one this run started with; start a new run.")
        val c = Conditions.parse(host.snapshotJson())
        val bl = TrainGates.device(c, r.run.settings).toMutableList()
        if (eng.trainingStatus().available.not()) bl.add(Blocker("ENGINE_UNAVAILABLE", eng.trainingStatus().reason ?: "Training engine unavailable"))
        else {
            val budget = ((c.availableRamMb - TrainGates.reserveMb(c.availableRamMb, policy)).coerceAtLeast(0.0) * Conditions.MIB).toLong()
            val est = try { eng.trainer!!.estimate(r.basePath, paramsOf(r.run.settings, r.run.settings.kind, budget)) } catch (e: BackendException) { TrainEstimate(false, 0, null, e.message) }
            bl.addAll(TrainGates.fit(c, est, 0, policy).filter { it.code != "STORAGE" })
        }
        if (bl.isNotEmpty()) throw StudioException(StudioError.Blocked("TRAINING_BLOCKED", bl.first().message, bl))
        r.intent = Intent.NONE; r.guardMessage = null
        update(r) { it.copy(state = TrainingRunState.QUEUED, message = "Queued to resume", error = null) }
        runner.submit(Runnable { execute(r) })
        return r.run
    }

    // ---- execution ---------------------------------------------------------------------------------------------------------

    private fun execute(r: RunRec) {
        if (r.run.state != TrainingRunState.QUEUED) return                     // paused/cancelled before it started
        if (!eng.beginTraining()) {
            update(r) { it.copy(state = TrainingRunState.PAUSED, message = "Could not start: the model is busy with another operation. Try again in a moment.") }
            return
        }
        r.live = true
        val token = CancelToken().also { r.cancel = it }
        if (r.intent != Intent.NONE) token.cancel()
        try { doTrain(r, token) }
        catch (e: StudioException) { fail(r, e.error) }
        catch (e: BackendException) { fail(r, e.toStudioError()) }
        catch (e: Exception) { fail(r, StudioError.Io("Internal error (${e.javaClass.simpleName}): ${e.message}")) }
        finally { r.live = false; r.cancel = null; eng.endTraining() }
    }

    private fun fail(r: RunRec, e: StudioError) {
        update(r) { it.copy(state = TrainingRunState.FAILED, resumable = false, error = e, message = e.message) }
    }

    private fun doTrain(r: RunRec, token: CancelToken) {
        val tr = eng.trainer ?: throw blockedEx("ENGINE_UNAVAILABLE", "No training engine is installed in this app build.")
        val st = tr.status()
        if (!st.available) throw blockedEx("ENGINE_UNAVAILABLE", st.reason ?: "The native training engine is not available.")
        val s = r.run.settings
        val base = File(r.basePath)
        update(r) { it.copy(state = TrainingRunState.RUNNING, stage = TrainingStage.PREPARE, message = "Checking the base model") }
        if (!base.isFile) throw StudioException(StudioError.Invalid("BASE_MISSING", "The base model file is missing"))
        val baseSha = r.baseSha ?: hashes.sha256(base, host.baseRef(r.run.projectId)?.takeIf { it.file?.absolutePath == r.basePath }?.sha256).also { r.baseSha = it }
        val seqFile = File(r.dir, "sequences.jsonl")
        val texts = Fs.readLines(seqFile).map { JSONObject(it).getString("text") }
        if (Hashing.sha256(J.canonical(texts)) != r.sequencesSha) throw StudioException(StudioError.Invalid("INPUTS_CHANGED", "The saved training inputs no longer match their recorded hash; start a new run."))
        if (token.isCancelled) { stopped(r); return }
        val c0 = Conditions.parse(host.snapshotJson())
        val budget = ((c0.availableRamMb - TrainGates.reserveMb(c0.availableRamMb, policy)).coerceAtLeast(0.0) * Conditions.MIB).toLong()
        val params = paramsOf(s, s.kind, budget)
        val work = File(r.dir, "work").also { it.mkdirs() }
        val out = File(r.dir, "out.hagpatch")
        out.delete()
        synchronized(lock) { r.segmentBaseElapsed = r.run.elapsedMs }
        var lastTrain: Double? = r.run.latestTrainLoss
        var lastVal: Double? = r.run.latestValLoss
        var attempt = 0
        while (true) {
            try {
                tr.train(r.basePath, texts, params, work.absolutePath, out.absolutePath, token) { ev ->
                    val trl = ev.trainLoss?.takeIf { it.isFinite() }; val vl = ev.valLoss?.takeIf { it.isFinite() }
                    if (trl != null) lastTrain = trl
                    if (vl != null) lastVal = vl
                    val cond = Conditions.parse(host.snapshotJson())
                    val stage = when (ev.phase) { TrainPhase.PREPARE -> TrainingStage.PREPARE; TrainPhase.TRAIN -> TrainingStage.TRAIN; TrainPhase.EVAL -> TrainingStage.EVAL; TrainPhase.SAVE -> TrainingStage.SAVE; TrainPhase.DONE -> TrainingStage.SAVE }
                    update(r, force = ev.phase != TrainPhase.TRAIN) { cur ->
                        var trend = cur.lossTrend
                        if ((trl != null || vl != null) && (trend.lastOrNull()?.step != ev.step || trl != trend.lastOrNull()?.trainLoss)) {
                            trend = trend + LossPoint(ev.step, trl, vl)
                            if (trend.size > 400) trend = trend.filterIndexed { i, _ -> i % 2 == 1 || i == trend.size - 1 }
                        }
                        cur.copy(stage = stage, epoch = ev.epoch.coerceAtLeast(1), epochs = maxOf(ev.epochs, 1), step = ev.step, steps = maxOf(ev.steps, cur.steps.takeIf { ev.steps <= 0 } ?: 0), examplesDone = ev.examplesDone,
                            lossTrend = trend, latestTrainLoss = lastTrain, latestValLoss = lastVal, elapsedMs = r.segmentBaseElapsed + (ev.elapsedS * 1000).toLong(), thermal = cond.thermal, batteryPercent = cond.batteryPercent,
                            rssMb = (ev.rssBytes / Conditions.MIB).toInt().takeIf { it > 0 }, resumedFromStep = maxOf(cur.resumedFromStep, ev.resumedFromStep),
                            checkpoint = if (cur.checkpoint == CheckpointState.RECOVERED_FROM_SCRATCH) cur.checkpoint else if (hasCheckpoint(r)) CheckpointState.PRESENT else CheckpointState.NONE,
                            message = when (stage) { TrainingStage.TRAIN -> "Training: step ${ev.step} of ${maxOf(ev.steps, 1)}"; TrainingStage.EVAL -> "Measuring loss on validation text"; TrainingStage.SAVE -> "Saving the specialist patch"; else -> "Preparing" })
                    }
                    if (!token.isCancelled) {
                        val g = TrainGates.runtimeGuard(cond, s)
                        if (g != null) { r.guardMessage = g; r.intent = Intent.PAUSE; token.cancel() }
                    }
                }
                break
            } catch (e: BackendException) {
                when {
                    e.code == BackendError.CANCELLED -> { stopped(r); return }
                    e.code == BackendError.CORRUPT && attempt == 0 -> {
                        attempt++
                        val aside = File(r.dir, "work.corrupt-" + clock.nowMs())
                        work.renameTo(aside)
                        work.mkdirs()
                        problems.add("training run ${r.run.id}: checkpoint rejected by the engine (${e.message}); set aside as ${aside.name}")
                        synchronized(lock) { r.segmentBaseElapsed = r.run.elapsedMs }
                        update(r) { it.copy(checkpoint = CheckpointState.RECOVERED_FROM_SCRATCH, resumedFromStep = 0, step = 0, examplesDone = 0, lossTrend = emptyList(), latestTrainLoss = null, latestValLoss = null,
                            message = "A saved checkpoint was damaged and was set aside. Restarting from the beginning.") }
                        lastTrain = null; lastVal = null
                    }
                    else -> throw e
                }
            }
        }
        finalizeRun(r, tr, baseSha, out, lastTrain, lastVal)
    }

    private fun stopped(r: RunRec) {
        val intent = r.intent
        if (intent == Intent.CANCEL) {
            File(r.dir, "work").deleteRecursively()
            update(r) { it.copy(state = TrainingRunState.CANCELLED, resumable = false, checkpoint = CheckpointState.NONE, message = "Cancelled; checkpoints discarded") }
        } else {
            val ck = if (hasCheckpoint(r)) CheckpointState.PRESENT else CheckpointState.NONE
            update(r) { it.copy(state = TrainingRunState.PAUSED, resumable = true, checkpoint = if (it.checkpoint == CheckpointState.RECOVERED_FROM_SCRATCH && ck == CheckpointState.PRESENT) it.checkpoint else ck,
                message = r.guardMessage ?: ("Paused at step ${it.step}." + if (ck == CheckpointState.PRESENT) " Checkpoint kept; resume continues from it." else " No checkpoint yet; resume starts again from the beginning.")) }
        }
    }

    private fun finalizeRun(r: RunRec, tr: TrainingBackend, baseSha: String, out: File, lastTrain: Double?, lastVal: Double?) {
        update(r) { it.copy(stage = TrainingStage.VERIFY, message = "Verifying the patch") }
        if (!out.isFile || out.length() == 0L) throw StudioException(StudioError.Invalid("NO_PATCH", "Training finished but the engine produced no patch file."))
        val info = tr.patchInfo(out.absolutePath)
        if (info.baseSha256.isNotEmpty() && !info.baseSha256.equals(baseSha, ignoreCase = true)) {
            out.delete()
            throw StudioException(StudioError.Invalid("PATCH_BASE_MISMATCH", "The patch names a different base model than the one trained. It was discarded."))
        }
        val pid = r.run.projectId
        val spId = "spec-" + ids.hex(10)
        val dir = specs.newDir(pid, spId)
        val target = File(dir, SpecialistRec.PATCH_NAME)
        if (!out.renameTo(target)) { out.copyTo(target, overwrite = true); out.delete() }
        val patchSha = Hashing.sha256File(target)
        val projectName = host.projectName(pid) ?: "Specialist"
        val base = host.baseRef(pid)
        val rec = SpecialistRec(spId, pid, "$projectName specialist", specs.nextVersion(pid), clock.nowMs(), r.run.baseModelId, base?.variantId, baseSha, patchSha, target.length(), r.run.datasetSha256, r.run.id,
            r.run.settings.kind, r.run.settings, lastTrain, lastVal, maxOf(info.steps, r.run.step), r.run.examplesDone, r.sources, dir)
        specs.add(rec, select = true)
        eng.endTraining()                                     // the verification reload needs the engine
        val verified = try { specs.verify(spId) } catch (e: StudioException) { null }
        File(r.dir, "work").deleteRecursively()
        update(r) { it.copy(state = TrainingRunState.SUCCEEDED, stage = TrainingStage.DONE, resumable = false, specialistId = spId, latestTrainLoss = lastTrain, latestValLoss = lastVal,
            message = "Specialist ${rec.version} ready." + if (verified?.verified == true) "" else " Not yet verified by reload: ${verified?.verifyMessage ?: "unknown"}") }
    }

    /** A source was removed/changed: stop any live run that used it and discard its checkpoints (rights: never train on removed material). */
    fun onSourceChanged(pid: ProjectId, sourceId: String) {
        val affected = synchronized(lock) { runs.values.filter { it.run.projectId == pid && !it.run.isTerminal && sourceId in it.sources } }
        for (r in affected) {
            r.intent = Intent.CANCEL
            r.cancel?.cancel()
            if (!r.live) { File(r.dir, "work").deleteRecursively(); update(r) { it.copy(state = TrainingRunState.CANCELLED, resumable = false, checkpoint = CheckpointState.NONE, message = "Cancelled: a source this run used was removed or changed. Start a new run.") } }
        }
    }
}
