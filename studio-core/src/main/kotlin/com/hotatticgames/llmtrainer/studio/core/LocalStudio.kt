package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.qualify.SafetyPolicy
import com.hotatticgames.llmtrainer.studio.api.*
import java.io.File
import java.io.IOException
import java.io.OutputStream

/**
 * The phone-first (v2) half of the Studio: chat, local training, specialist registry, local evaluation and A/B.
 * Talks to the engine ONLY through [InferenceBackend] / [TrainingBackend]; knows the rest of the studio only through [LocalHost].
 * Pure JVM. Every public method returns a [StudioResult]; nothing here fabricates an answer when the engine cannot provide one.
 */
class LocalStudio(
    private val host: LocalHost, inference: InferenceBackend?, trainer: TrainingBackend?, cacheDir: File,
    private val clock: Clock, private val ids: IdSource, runner: TaskRunner, policy: SafetyPolicy, private val problems: MutableList<String>,
) {
    val engine = EngineCore(inference, trainer)
    private val hashes = FileHashes(File(cacheDir, "file_hashes.json"))
    private val specs = SpecialistRegistry(host, engine, hashes, clock, problems)
    private val training = TrainingService(host, engine, specs, hashes, clock, ids, runner, policy, problems)
    private val evals = EvalService(host, engine, specs, clock, ids, runner, problems)

    private class ChatRec(var info: ChatSessionInfo, val msgs: ArrayList<ChatMessageRecord>)
    private val lock = Any()
    private val chats = LinkedHashMap<String, ChatRec>()
    private val abs = LinkedHashMap<String, ABComparison>()
    private val verifiedThisSession = HashSet<String>()

    init { for (pid in host.projectIds()) loadProject(pid) }

    private fun loadProject(pid: ProjectId) {
        val dir = host.projectDir(pid) ?: return
        specs.load(pid); training.load(pid); evals.load(pid)
        File(dir, "chats").listFiles { f -> f.isFile && f.name.endsWith(".json") }?.sortedBy { it.name }?.forEach { f ->
            val o = Fs.readJson(f) { problems.add("chat ${f.name}: $it") } ?: return@forEach
            try {
                val info = LJ.chatInfo(o.obj("info") ?: return@forEach)
                synchronized(lock) { chats[info.id] = ChatRec(info, ArrayList(o.objList("messages").map { LJ.msg(it) })) }
            } catch (e: Exception) { problems.add("chat ${f.name} unreadable: ${e.message}") }
        }
        File(dir, "ab").listFiles { f -> f.isFile && f.name.endsWith(".json") }?.sortedBy { it.name }?.forEach { f ->
            val o = Fs.readJson(f) { problems.add("comparison ${f.name}: $it") } ?: return@forEach
            try { val c = LJ.ab(o); synchronized(lock) { abs[c.id] = c } } catch (e: Exception) { problems.add("comparison ${f.name} unreadable: ${e.message}") }
        }
    }

    /** Called by StudioCore after a project was deleted from disk. */
    fun projectDeleted(pid: ProjectId) {
        training.forget(pid); evals.forget(pid); specs.forget(pid)
        synchronized(lock) { chats.values.removeAll { it.info.projectId == pid }; abs.values.removeAll { it.projectId == pid } }
    }

    /** A source was removed or changed: stop and discard any training run that used it. */
    fun sourceChanged(pid: ProjectId, sourceId: String) = training.onSourceChanged(pid, sourceId)

    fun releaseModels() = engine.release()

    private fun need(pid: ProjectId) { if (host.projectDir(pid) == null) throw StudioException(StudioError.NotFound("project ${pid.value}")) }

    // ---- engine & models -----------------------------------------------------------------------------------------------------

    fun engineStatus(): EngineStatus {
        val i = engine.inferenceStatus(); val t = engine.trainingStatus()
        return EngineStatus(listOf(i.runtimeId, t.runtimeId).firstOrNull { it != "none" } ?: "none", i.available, i.reason, t.available, t.reason)
    }

    fun installedModels(): List<InstalledModelSummary> = host.installed()

    fun projectModelState(pid: ProjectId): StudioResult<ProjectModelState> = guarded {
        need(pid)
        val base = host.baseRef(pid)
        val installed = base?.file?.isFile == true
        val inf = engine.inferenceStatus()
        val sp = specs.list(pid)
        val selected = specs.selectedId(pid)
        val chatBase = when {
            !inf.available -> Capability(false, inf.reason ?: "The native engine is not available on this install.")
            base == null -> Capability(false, "Choose a base model first")
            !installed -> Capability(false, "Download or import the base model first")
            else -> Capability(true, null)
        }
        val chatSp = if (!chatBase.ok) chatBase else if (selected == null) Capability(false, "No specialist has been trained for this project yet") else Capability(true, null)
        val plan = try { training.plan(pid) } catch (e: StudioException) { null }
        val trainOpt = plan?.options?.firstOrNull { it.recommended } ?: plan?.options?.firstOrNull { it.kind == TrainingMethodKind.LOCAL_PARTIAL }
        val train = when { plan == null -> Capability(false, "Unavailable"); trainOpt != null && trainOpt.available -> Capability(true, null); else -> Capability(false, trainOpt?.blockers?.firstOrNull()?.message ?: "Unavailable") }
        val ev = if (!chatBase.ok) chatBase else if (sp.isEmpty()) Capability(false, "Train a specialist first") else Capability(true, null)
        ProjectModelState(pid, base?.modelId, base?.variantId, installed, base?.file?.takeIf { it.isFile }?.length(), selected, sp.size, chatBase, chatSp, train, ev)
    }

    // ---- chat --------------------------------------------------------------------------------------------------------------------

    private fun saveChat(r: ChatRec) {
        val dir = host.projectDir(r.info.projectId) ?: return
        Fs.writeJson(File(dir, "chats/${r.info.id}.json"), linkedMapOf("schema" to 1, "info" to LJ.chatInfo(r.info), "messages" to r.msgs.map { LJ.msg(it) }))
    }

    private fun validOptions(o: ChatOptions) {
        if (o.contextTokens !in 256..16384 || o.maxContextChunks !in 0..10 || o.sampling.maxNewTokens !in 1..4096 || (o.systemPrompt?.length ?: 0) > 4000)
            throw StudioException(StudioError.Invalid("BAD_OPTIONS", "Chat options are out of range"))
    }

    fun createChat(pid: ProjectId, target: ChatTarget, specialistId: String?, options: ChatOptions): StudioResult<ChatSessionInfo> = guarded {
        need(pid); validOptions(options)
        val st = projectModelState(pid).getOrNull()!!
        val cap = if (target == ChatTarget.BASE) st.canChatBase else st.canChatSpecialist
        if (!cap.ok) throw blockedEx("CHAT_UNAVAILABLE", cap.reason ?: "Unavailable")
        val base = host.baseRef(pid)!!
        var spId: String? = null
        var label = "Base: ${base.name}"
        if (target == ChatTarget.SPECIALIST) {
            spId = specialistId ?: specs.selectedId(pid)
            val rec = spId?.let { specs.get(it) }?.takeIf { it.projectId == pid } ?: throw StudioException(StudioError.NotFound("specialist ${spId ?: "(none selected)"}"))
            label = "Specialist ${rec.version} (parameters changed on this phone) on ${base.name}"
        }
        val now = clock.nowMs()
        val info = ChatSessionInfo("chat-" + ids.hex(10), pid, target, spId, if (target == ChatTarget.BASE) "Chat with the base model" else "Chat with the specialist", now, now, 0, options, label)
        val r = ChatRec(info, ArrayList())
        synchronized(lock) { chats[info.id] = r }
        saveChat(r)
        info
    }

    fun listChats(pid: ProjectId): List<ChatSessionInfo> = synchronized(lock) { chats.values.filter { it.info.projectId == pid }.map { it.info } }

    fun chatHistory(id: String): StudioResult<List<ChatMessageRecord>> = guarded { synchronized(lock) { chats[id]?.msgs?.toList() } ?: throw StudioException(StudioError.NotFound("chat $id")) }

    fun deleteChat(id: String): StudioResult<Unit> = guarded {
        val r = synchronized(lock) { chats.remove(id) } ?: throw StudioException(StudioError.NotFound("chat $id"))
        host.projectDir(r.info.projectId)?.let { File(it, "chats/$id.json").delete(); File(it, "chats/$id.json.bak").delete() }
    }

    /** Source excerpts for the PROMPT (retrieval). Never part of training. */
    private fun retrieve(pid: ProjectId, query: String, k: Int): List<ContextChunk> {
        if (k <= 0) return emptyList()
        val dv = host.dataView(pid) ?: throw blockedEx("NO_SOURCES_INDEXED", "Source context needs ingested sources: add sources and build the dataset first.")
        val pool = dv.chunks.filter { it.chunk.role != ChunkRoles.EXCLUDED }
        if (pool.isEmpty()) throw blockedEx("NO_SOURCES_INDEXED", "There is no source text to retrieve from.")
        val byRef = pool.associateBy { it.ref }
        return LexicalRetriever(pool.map { it.ref to it.chunk.text }).top(query, k).map { (ref, _) ->
            val c = byRef.getValue(ref)
            ContextChunk(ref, c.sourceName, c.chunk.page, c.chunk.section, c.chunk.text.let { t -> if (t.length <= 1200) t else t.take(1200) + "…" })
        }
    }

    private fun systemText(base: String?, ctx: List<ContextChunk>): String? {
        val parts = ArrayList<String>()
        base?.takeIf { it.isNotBlank() }?.let { parts.add(it) }
        if (ctx.isNotEmpty()) parts.add("Use these excerpts from the owner's source material when they are relevant, and say so when they do not answer the question.\n" +
            ctx.joinToString("\n") { "[${it.ref}] ${it.excerpt}" })
        return parts.takeIf { it.isNotEmpty() }?.joinToString("\n\n")
    }

    private fun modelFiles(pid: ProjectId, target: ChatTarget, specialistId: String?): Pair<File, File?> {
        val base = host.baseRef(pid)?.file?.takeIf { it.isFile } ?: throw blockedEx("BASE_NOT_INSTALLED", "The base model is not on this phone. Download or import it first.")
        if (target == ChatTarget.BASE) return base to null
        val rec = specialistId?.let { specs.get(it) }?.takeIf { it.projectId == pid } ?: throw StudioException(StudioError.NotFound("specialist ${specialistId ?: ""}"))
        if (synchronized(lock) { rec.id !in verifiedThisSession }) {
            val v = specs.verify(rec.id)
            if (!v.verified) throw blockedEx("SPECIALIST_UNVERIFIED", "This specialist cannot be used: ${v.verifyMessage}")
            synchronized(lock) { verifiedThisSession.add(rec.id) }
        }
        return base to rec.patchFile
    }

    private fun genStats(st: GenStats, info: ModelInfo, ttfbMs: Double?): GenerationStats =
        GenerationStats(st.promptTokens, st.generatedTokens, ttfbMs ?: st.promptMs, st.tokensPerSecond, st.promptMs + st.genMs, info.loadMs, (st.peakRssBytes / Conditions.MIB).toInt().takeIf { it > 0 }, st.stopReason)

    /** One generation on a loaded model: the full formatted conversation is processed each turn (reset + prompt). */
    private fun generate(inf: InferenceBackend, L: EngineCore.Loaded, msgs: List<ChatMessage>, sampling: SamplingParams, cancel: CancelToken, onToken: (String) -> Unit): Pair<String, GenStats?> {
        inf.resetChat(L.chat)
        val prompt = inf.chatFormat(L.model, msgs, true)
        val sb = StringBuilder()
        return try {
            val st = inf.generate(L.chat, prompt, sampling, cancel) { piece -> sb.append(piece); onToken(piece); cancel.isCancelled }
            sb.toString() to st
        } catch (e: BackendException) {
            if (e.code == BackendError.CANCELLED) sb.toString() to null else throw e
        }
    }

    fun sendMessage(chatId: String, text: String, cancel: CancelToken, onToken: (String) -> Unit): StudioResult<ChatMessageRecord> = guarded {
        val rec = synchronized(lock) { chats[chatId] } ?: throw StudioException(StudioError.NotFound("chat $chatId"))
        val info = rec.info
        val o = info.options
        if (text.isBlank()) throw StudioException(StudioError.Invalid("EMPTY_MESSAGE", "Type a message first"))
        if (text.length > 8000) throw StudioException(StudioError.Invalid("MESSAGE_TOO_LONG", "Messages are limited to 8000 characters"))
        val inf = engine.requireInference()
        val (baseFile, patch) = modelFiles(info.projectId, info.target, info.specialistId)
        val ctx = if (o.useSourceContext) retrieve(info.projectId, text, o.maxContextChunks) else emptyList()
        // history within the context budget (newest kept); the system part and the new message always stay
        val sys = systemText(o.systemPrompt, ctx)
        val budgetChars = maxOf(256, (o.contextTokens - o.sampling.maxNewTokens) * 3) - (sys?.length ?: 0) - text.length
        if (budgetChars < 0) throw StudioException(StudioError.Invalid("MESSAGE_TOO_LONG", "The message plus source excerpts do not fit this chat's context window"))
        val history = ArrayList<ChatMessage>()
        var used = 0
        for (m in synchronized(lock) { rec.msgs.toList() }.reversed()) {
            if (m.role == ChatRole.SYSTEM) continue
            if (m.text.isEmpty()) continue
            if (used + m.text.length > budgetChars) break
            used += m.text.length
            history.add(ChatMessage(if (m.role == ChatRole.USER) "user" else "assistant", m.text))
        }
        history.reverse()
        val msgs = ArrayList<ChatMessage>().apply { sys?.let { add(ChatMessage("system", it)) }; addAll(history); add(ChatMessage("user", text)) }
        val t0 = System.nanoTime()
        var first: Double? = null
        val (reply, st) = engine.withModel(baseFile, patch, o.contextTokens, keepLoaded = true) { L ->
            val (txt, stats) = generate(inf, L, msgs, o.sampling, cancel) { if (first == null) first = (System.nanoTime() - t0) / 1e6; onToken(it) }
            txt to (stats?.let { genStats(it, L.info, first) })
        }
        val now = clock.nowMs()
        val interrupted = cancel.isCancelled || st == null || st.stopReason == StopReason.CANCELLED
        val user = ChatMessageRecord("msg-" + ids.hex(10), ChatRole.USER, text, now, emptyList(), null, false)
        val ans = ChatMessageRecord("msg-" + ids.hex(10), ChatRole.ASSISTANT, reply, clock.nowMs(), ctx, st, interrupted)
        synchronized(lock) { rec.msgs.add(user); rec.msgs.add(ans); rec.info = rec.info.copy(updatedAt = ans.at, messageCount = rec.msgs.size) }
        saveChat(rec)
        ans
    }

    // ---- training / specialists --------------------------------------------------------------------------------------------

    fun localTrainingPlan(pid: ProjectId): StudioResult<LocalTrainingPlan> = guarded { training.plan(pid) }
    fun startLocalTraining(pid: ProjectId, s: TrainingSettings, confirmed: Boolean): StudioResult<TrainingRun> = guarded { training.start(pid, s, confirmed) }
    fun trainingRuns(pid: ProjectId): List<TrainingRun> = training.list(pid)
    fun trainingRun(id: String): StudioResult<TrainingRun> = guarded { training.get(id) ?: throw StudioException(StudioError.NotFound("training run $id")) }
    fun pauseTraining(id: String): StudioResult<TrainingRun> = guarded { training.pause(id) }
    fun cancelTraining(id: String): StudioResult<TrainingRun> = guarded { training.cancel(id) }
    fun resumeTraining(id: String): StudioResult<TrainingRun> = guarded { training.resume(id) }

    fun specialists(pid: ProjectId): List<SpecialistInfo> = specs.list(pid)
    fun verifySpecialist(id: String): StudioResult<SpecialistInfo> = guarded { specs.verify(id).also { if (it.verified) synchronized(lock) { verifiedThisSession.add(id) } else synchronized(lock) { verifiedThisSession.remove(id) } } }
    fun selectSpecialist(pid: ProjectId, id: String?): StudioResult<List<SpecialistInfo>> = guarded { specs.select(pid, id) }
    fun deleteSpecialist(id: String): StudioResult<Unit> = guarded {
        engine.release()          // the file may be mapped by a loaded model
        specs.delete(id); synchronized(lock) { verifiedThisSession.remove(id) }
    }
    fun exportSpecialistPatch(id: String, out: OutputStream): StudioResult<ExportedPackage> = guarded {
        val rec = specs.get(id) ?: throw StudioException(StudioError.NotFound("specialist $id"))
        val base = host.baseRef(rec.projectId)
        val lic = base?.gate ?: listOf(Blocker("NO_BASE_MODEL", "Base model unknown"))
        if (lic.isNotEmpty()) throw StudioException(StudioError.Blocked("LICENSE_GATE", "The base model's license does not permit exporting a derivative: ${lic.first().message}", lic))
        val ev = evals.list(rec.projectId).filter { it.specialistId == id && it.state == LocalEvalState.SUCCEEDED }.lastOrNull()
        specs.export(id, out, mapOf("base_model_name" to base?.name, "local_evaluation" to ev?.view?.let { LJ.view(it) }))
    }

    // ---- evaluation -----------------------------------------------------------------------------------------------------------

    fun startLocalEvaluation(pid: ProjectId, spId: String, o: LocalEvalOptions): StudioResult<LocalEvaluation> = guarded { evals.start(pid, spId, o) }
    fun localEvaluations(pid: ProjectId): List<LocalEvaluation> = evals.list(pid)
    fun localEvaluation(id: String): StudioResult<LocalEvaluation> = guarded { evals.get(id) ?: throw StudioException(StudioError.NotFound("evaluation $id")) }
    fun cancelLocalEvaluation(id: String): StudioResult<LocalEvaluation> = guarded { evals.cancel(id) }

    // ---- A/B ------------------------------------------------------------------------------------------------------------------------

    private fun saveAB(c: ABComparison) { host.projectDir(c.projectId)?.let { Fs.writeJson(File(it, "ab/${c.id}.json"), LJ.ab(c)) } }

    fun compareAB(pid: ProjectId, spId: String, prompt: String, o: ABOptions, cancel: CancelToken, onToken: (ChatTarget, String) -> Unit): StudioResult<ABComparison> = guarded {
        need(pid)
        if (prompt.isBlank()) throw StudioException(StudioError.Invalid("EMPTY_MESSAGE", "Type a question first"))
        if (prompt.length > 8000) throw StudioException(StudioError.Invalid("MESSAGE_TOO_LONG", "Questions are limited to 8000 characters"))
        if (o.contextTokens !in 256..16384 || o.maxContextChunks !in 0..10 || o.sampling.maxNewTokens !in 1..4096) throw StudioException(StudioError.Invalid("BAD_OPTIONS", "Options are out of range"))
        val inf = engine.requireInference()
        val sp = specs.get(spId)?.takeIf { it.projectId == pid } ?: throw StudioException(StudioError.NotFound("specialist $spId"))
        val base = host.baseRef(pid)
        val (baseFile, patch) = modelFiles(pid, ChatTarget.SPECIALIST, spId)
        val ctx = if (o.useSourceContext) retrieve(pid, prompt, o.maxContextChunks) else emptyList()      // identical context for both sides
        val sys = systemText(o.systemPrompt, ctx)
        val msgs = listOf(ChatMessage("system", sys ?: "")).filter { it.content.isNotEmpty() } + ChatMessage("user", prompt)
        fun side(t: ChatTarget, label: String, p: File?): ABSide {
            val t0 = System.nanoTime(); var first: Double? = null
            return engine.withModel(baseFile, p, o.contextTokens, keepLoaded = false) { L ->
                val (txt, st) = generate(inf, L, msgs, o.sampling, cancel) { if (first == null) first = (System.nanoTime() - t0) / 1e6; onToken(t, it) }
                ABSide(label, t, txt, st?.let { genStats(it, L.info, first) }, ctx)
            }
        }
        val b = side(ChatTarget.BASE, "Base: ${base?.name ?: sp.baseModelId}", null)
        if (cancel.isCancelled) throw StudioException(StudioError.Cancelled())
        val s = side(ChatTarget.SPECIALIST, "Specialist ${sp.version} (parameters changed on this phone)", patch)
        if (cancel.isCancelled) throw StudioException(StudioError.Cancelled())
        val c = ABComparison("ab-" + ids.hex(10), pid, spId, prompt, b, s, clock.nowMs(), null, false)
        synchronized(lock) { abs[c.id] = c }
        saveAB(c)
        c
    }

    fun abComparisons(pid: ProjectId): List<ABComparison> = synchronized(lock) { abs.values.filter { it.projectId == pid } }

    fun saveABNote(pid: ProjectId, id: String, note: String): StudioResult<ABComparison> = guarded {
        if (note.length > 4000) throw StudioException(StudioError.Invalid("NOTE_TOO_LONG", "Notes are limited to 4000 characters"))
        val c = synchronized(lock) { abs[id] }?.takeIf { it.projectId == pid } ?: throw StudioException(StudioError.NotFound("comparison $id"))
        val n = c.copy(note = note, savedAsNote = true)
        synchronized(lock) { abs[id] = n }; saveAB(n); n
    }

    fun deleteAB(pid: ProjectId, id: String): StudioResult<Unit> = guarded {
        val c = synchronized(lock) { abs[id] }?.takeIf { it.projectId == pid } ?: throw StudioException(StudioError.NotFound("comparison $id"))
        synchronized(lock) { abs.remove(id) }
        host.projectDir(c.projectId)?.let { File(it, "ab/$id.json").delete(); File(it, "ab/$id.json.bak").delete() }
    }
}
