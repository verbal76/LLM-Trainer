package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import java.io.File
import java.io.IOException
import java.util.Random

/** One metric's definition (mirrors evalsuite METRIC_SPECS). */
private class MSpec(val id: String, val label: String, val category: String, val higherIsBetter: Boolean)

private val SPECS = listOf(
    MSpec("terminology_coverage", "Domain terminology coverage", "domain", true),
    MSpec("exact_fact_accuracy", "Exact-fact accuracy (closed book)", "grounding", true),
    MSpec("grounded_exact_fact_accuracy", "Exact-fact accuracy (with source excerpts in the prompt)", "domain", true),
    MSpec("grounded_unsupported_claim_rate", "Unsupported claims (with source excerpts in the prompt)", "safety", false),
    MSpec("heldout_nll", "Held-out negative log-likelihood (nats/token)", "domain", false),
    MSpec("general_probe_pass_rate", "General-capability retention (fixed probes)", "general_regression", true),
)

/**
 * Local (on-phone) evaluation of BASE vs SPECIALIST on the TEST split only. Port of the evalsuite semantics:
 * same items, same greedy decoding, one model resident at a time, per-metric deltas with paired-bootstrap CIs and sample sizes,
 * no aggregate score, and `improvementClaimAllowed` that fails closed.
 */
class EvalService(
    private val host: LocalHost, private val eng: EngineCore, private val specs: SpecialistRegistry, private val clock: Clock, private val ids: IdSource,
    private val runner: TaskRunner, private val problems: MutableList<String>,
) {
    companion object {
        const val MIN_CLAIM_ITEMS = 50
        const val SMALL_N = 30
        const val TOLERANCE = 0.02
        /** Largest tolerated drop of the general-probe pass rate before no improvement claim is allowed. */
        const val RETENTION_DROP = 0.10
    }

    private class Rec(var ev: LocalEvaluation) { @Volatile var cancel = CancelToken() }
    private val lock = Any()
    private val recs = LinkedHashMap<String, Rec>()
    private fun dir(pid: ProjectId) = host.projectDir(pid)?.let { File(it, "local_eval") }

    private fun persist(r: Rec) { dir(r.ev.projectId)?.let { try { Fs.writeJson(File(it, r.ev.id + ".json"), LJ.eval(r.ev)) } catch (e: IOException) { problems.add("evaluation ${r.ev.id}: could not persist: ${e.message}") } } }
    private fun update(r: Rec, persist: Boolean = true, f: (LocalEvaluation) -> LocalEvaluation) { synchronized(lock) { r.ev = f(r.ev) }; if (persist) persist(r) }

    fun load(pid: ProjectId) {
        synchronized(lock) { recs.values.removeAll { it.ev.projectId == pid } }
        dir(pid)?.listFiles { f -> f.isFile && f.name.endsWith(".json") }?.sortedBy { it.name }?.forEach { f ->
            val o = Fs.readJson(f) { problems.add("evaluation ${f.name}: $it") } ?: return@forEach
            try {
                var e = LJ.eval(o)
                if (e.state == LocalEvalState.RUNNING || e.state == LocalEvalState.QUEUED)
                    e = e.copy(state = LocalEvalState.INTERRUPTED, message = "Interrupted (the app was closed). Nothing partial is reported; start the evaluation again.", finishedAt = clock.nowMs())
                val r = Rec(e); synchronized(lock) { recs[e.id] = r }; if (e !== null && o.str("state") != e.state.name) persist(r)
            } catch (ex: Exception) { problems.add("evaluation ${f.name} unreadable: ${ex.message}") }
        }
    }

    fun forget(pid: ProjectId) { synchronized(lock) { recs.values.filter { it.ev.projectId == pid }.forEach { it.cancel.cancel() }; recs.values.removeAll { it.ev.projectId == pid } } }
    fun list(pid: ProjectId) = synchronized(lock) { recs.values.filter { it.ev.projectId == pid }.map { it.ev } }
    fun get(id: String) = synchronized(lock) { recs[id]?.ev }

    fun cancel(id: String): LocalEvaluation {
        val r = synchronized(lock) { recs[id] } ?: throw StudioException(StudioError.NotFound("evaluation $id"))
        if (r.ev.state != LocalEvalState.RUNNING && r.ev.state != LocalEvalState.QUEUED) throw StudioException(StudioError.Conflict("Evaluation is ${r.ev.state.name.lowercase()}"))
        r.cancel.cancel()
        return r.ev
    }

    // ---- start ---------------------------------------------------------------------------------------------------------------

    private class Prepared(val base: File, val patch: File, val items: List<DatasetEngine.EvalItem>, val testChunks: List<DChunk>, val synthetic: Boolean, val dv: DataView, val spec: SpecialistRec)

    fun start(pid: ProjectId, spId: String, o: LocalEvalOptions): LocalEvaluation {
        if (host.projectDir(pid) == null) throw StudioException(StudioError.NotFound("project ${pid.value}"))
        if (o.maxItems < 1 || o.maxItems > 1000 || o.bootstrapSamples !in 100..20000 || o.maxNewTokens !in 8..1024) throw StudioException(StudioError.Invalid("BAD_OPTIONS", "Evaluation options are out of range"))
        val spec = specs.get(spId)?.takeIf { it.projectId == pid } ?: throw StudioException(StudioError.NotFound("specialist $spId"))
        eng.requireInference()
        if (eng.trainingActive) throw StudioException(StudioError.Conflict("Training is running; evaluation needs the same engine and memory."))
        synchronized(lock) { recs.values.firstOrNull { it.ev.projectId == pid && (it.ev.state == LocalEvalState.RUNNING || it.ev.state == LocalEvalState.QUEUED) }?.let { throw StudioException(StudioError.Conflict("Evaluation ${it.ev.id} is still running")) } }
        val base = host.baseRef(pid)?.file?.takeIf { it.isFile } ?: throw blockedEx("BASE_NOT_INSTALLED", "The base model is not on this phone.")
        val v = specs.verify(spId)
        if (!v.verified) throw blockedEx("SPECIALIST_UNVERIFIED", "The specialist could not be verified: ${v.verifyMessage}")
        val dv = host.dataView(pid) ?: throw blockedEx("NO_DATASET", "Build and approve a dataset first")
        if (dv.status != DatasetStatus.APPROVED && dv.status != DatasetStatus.STALE) throw blockedEx("DATASET_NOT_APPROVED", "The dataset must be approved so its held-out split is defined (status ${dv.status}).")
        val test = dv.testChunks()
        if (test.isEmpty() || !dv.splitsAvailable) throw blockedEx("NO_HELDOUT", "There is no held-out (test) material. Add more documents so a genuine unseen split exists.")
        var items = DatasetEngine.buildEvalItems(test.map { it.ref to (it.chunk.section to it.chunk.text) }, dv.domain)
        if (items.isEmpty()) throw blockedEx("NO_HELDOUT_ITEMS", "No evaluation questions could be derived from the held-out text.")
        if (items.size > o.maxItems) { val sh = items.toMutableList(); sh.shuffle(Random(o.seed)); items = sh.take(o.maxItems).sortedBy { it.itemId } }
        val synthetic = dv.assembled.bySplit["test"].orEmpty().any { it.origin == "synthetic" }
        val prep = Prepared(base, File(spec.patchFile.path), items, test, synthetic, dv, spec)
        val steps = 2L * (items.size * (1 + if (o.includeGroundedMetrics) 1 else 0) + (if (o.includeRetention) EvalMetrics.PROBES.size else 0) + test.size)
        val ev = LocalEvaluation("leval-" + ids.hex(10), pid, spId, LocalEvalState.QUEUED, Progress(0, steps, "steps"), "Queued", null, null, emptyList(), emptyList(), dv.sha, test.size, items.size, clock.nowMs(), null)
        val rec = Rec(ev)
        synchronized(lock) { recs[ev.id] = rec }
        persist(rec)
        runner.submit(Runnable { run(rec, prep, o) })
        return ev
    }

    // ---- execution -----------------------------------------------------------------------------------------------------------

    private class Subject(val role: String) {
        val pairs = HashMap<String, EvalMetrics.Pairs>()
        val closed = HashMap<String, String>()
        val ok = HashMap<String, Boolean?>()
        val latencies = ArrayList<Double>()
        var genTokens = 0L; var genMs = 0.0
        var loadMs = 0L; var peakRss = 0L
        fun p(id: String) = pairs.getOrPut(id) { EvalMetrics.Pairs() }
    }

    private fun run(rec: Rec, prep: Prepared, o: LocalEvalOptions) {
        if (rec.ev.state != LocalEvalState.QUEUED) return
        update(rec) { it.copy(state = LocalEvalState.RUNNING, message = "Evaluating the base model") }
        try {
            val base = runSubject(rec, prep, o, "base", null)
            val spec = runSubject(rec, prep, o, "specialist", prep.patch)
            finish(rec, prep, o, base, spec)
        } catch (e: StudioException) {
            if (e.error is StudioError.Cancelled) update(rec) { it.copy(state = LocalEvalState.CANCELLED, message = "Cancelled; no result is reported.", finishedAt = clock.nowMs()) }
            else update(rec) { it.copy(state = LocalEvalState.FAILED, error = e.error, message = e.error.message, finishedAt = clock.nowMs()) }
        } catch (e: BackendException) {
            if (e.code == BackendError.CANCELLED) update(rec) { it.copy(state = LocalEvalState.CANCELLED, message = "Cancelled; no result is reported.", finishedAt = clock.nowMs()) }
            else e.toStudioError().let { err -> update(rec) { it.copy(state = LocalEvalState.FAILED, error = err, message = err.message, finishedAt = clock.nowMs()) } }
        } catch (e: Exception) {
            val err = StudioError.Io("Internal error (${e.javaClass.simpleName}): ${e.message}")
            update(rec) { it.copy(state = LocalEvalState.FAILED, error = err, message = err.message, finishedAt = clock.nowMs()) }
        }
    }

    private fun runSubject(rec: Rec, prep: Prepared, o: LocalEvalOptions, role: String, patch: File?): Subject {
        val inf = eng.requireInference()
        val s = Subject(role)
        val cancel = rec.cancel
        val retriever = LexicalRetriever(prep.testChunks.map { it.ref to it.chunk.text })
        val sampling = SamplingParams.greedy(o.maxNewTokens)
        fun tick(msg: String) { update(rec, persist = false) { it.copy(progress = Progress(it.progress.done + 1, it.progress.total, "steps"), message = msg) } }
        eng.withModel(prep.base, patch, 2048, keepLoaded = false) { L ->
            s.loadMs = L.info.loadMs
            fun ask(prompt: String): String {
                if (cancel.isCancelled) throw StudioException(StudioError.Cancelled())
                inf.resetChat(L.chat)
                val formatted = inf.chatFormat(L.model, listOf(ChatMessage("user", prompt)), true)
                val sb = StringBuilder()
                val st = inf.generate(L.chat, formatted, sampling, cancel) { sb.append(it); false }
                if (st.stopReason == StopReason.CANCELLED || cancel.isCancelled) throw StudioException(StudioError.Cancelled())
                s.latencies.add(st.promptMs + st.genMs); s.genTokens += st.generatedTokens; s.genMs += st.genMs; s.peakRss = maxOf(s.peakRss, st.peakRssBytes)
                return sb.toString().trim()
            }
            for (it in prep.items) {
                tick("Evaluating the $role model: closed-book questions")
                val ans = ask(it.question)
                s.closed[it.itemId] = ans
                if (it.kind == "concept") {
                    val (n, d) = EvalMetrics.terminologyUse(ans, it.requiredTerms)
                    s.p("terminology_coverage").add(n.toDouble(), d.toDouble()); s.ok[it.itemId] = d > 0 && n.toDouble() / d >= 0.5
                } else {
                    val exp = it.expectedValue?.let { v -> it.expectedUnit?.let { u -> EvalMetrics.qty(v, u) } }
                    val good = exp != null && EvalMetrics.factCorrect(ans, exp)
                    if (exp != null) s.p("exact_fact_accuracy").add(if (good) 1.0 else 0.0, 1.0)
                    s.ok[it.itemId] = if (exp != null) good else null
                }
            }
            if (o.includeGroundedMetrics) for (it in prep.items) {
                tick("Evaluating the $role model: with source excerpts in the prompt (retrieval)")
                val got = retriever.top(it.question, 3)
                val src = got.joinToString("\n") { (ref, text) -> "[$ref] $text" }
                val ans = ask("Answer using only the sources below. Cite the sources you use in square brackets, e.g. [source/chunk].\n\nSources:\n$src\n\nQuestion: ${it.question}")
                val (bad, tot) = EvalMetrics.unsupportedClaims(ans, got.joinToString(" ") { c -> c.second })
                s.p("grounded_unsupported_claim_rate").add(bad.toDouble(), tot.toDouble())
                if (it.kind == "fact") {
                    val exp = it.expectedValue?.let { v -> it.expectedUnit?.let { u -> EvalMetrics.qty(v, u) } }
                    if (exp != null) s.p("grounded_exact_fact_accuracy").add(if (EvalMetrics.factCorrect(ans, exp)) 1.0 else 0.0, 1.0)
                }
            }
            if (o.includeRetention) for ((q, pat) in EvalMetrics.PROBES) {
                tick("Evaluating the $role model: general-capability probes")
                s.p("general_probe_pass_rate").add(if (EvalMetrics.probePass(ask(q), pat)) 1.0 else 0.0, 1.0)
            }
            for (c in prep.testChunks) {
                tick("Evaluating the $role model: held-out likelihood")
                if (cancel.isCancelled) throw StudioException(StudioError.Cancelled())
                inf.resetChat(L.chat)
                val sc = inf.score(L.chat, c.chunk.text.take(2000), cancel)
                s.p("heldout_nll").add(sc.meanNll * sc.nTokens, sc.nTokens.toDouble())
            }
        }
        update(rec) { it.copy(message = if (role == "base") "Evaluating the specialist" else "Computing results") }
        return s
    }

    private fun finish(rec: Rec, prep: Prepared, o: LocalEvalOptions, b: Subject, sp: Subject) {
        val rows = ArrayList<MetricRow>()
        val cat = HashMap<String, String>()
        for (m in SPECS) {
            val pb = b.pairs[m.id] ?: continue
            val ps = sp.pairs[m.id] ?: continue
            if (pb.n == 0 && ps.n == 0) continue
            val (lo, hi) = EvalMetrics.pairedDeltaCi(pb, ps, o.seed, o.bootstrapSamples)
            val bv = pb.ratio(); val sv = ps.ratio()
            rows.add(MetricRow(m.id, m.label, m.higherIsBetter, bv, sv, sv - bv, lo, hi, ps.n)); cat[m.id] = m.category
        }
        val nItems = prep.items.size
        // ---- improvement_claim_allowed (fails closed) ----
        val why = ArrayList<String>()
        if (prep.synthetic) why.add("evaluation data includes synthetic material")
        if (nItems < MIN_CLAIM_ITEMS) why.add("only $nItems held-out items (< $MIN_CLAIM_ITEMS)")
        val domain = rows.filter { cat[it.id] == "domain" }
        if (domain.isEmpty()) why.add("no domain metric could be measured")
        val notSig = domain.filter { r -> !(r.delta != 0.0 && (if (r.higherIsBetter) r.ciLow!! > 0 else r.ciHigh!! < 0)) }
        if (notSig.isNotEmpty()) why.add("no significant improvement on: " + notSig.joinToString { it.label })
        val regress = rows.filter { r -> (if (r.higherIsBetter) 1 else -1) * r.delta < -TOLERANCE }
        if (regress.isNotEmpty()) why.add("regression beyond ±$TOLERANCE on: " + regress.joinToString { it.label })
        val probe = rows.firstOrNull { it.id == "general_probe_pass_rate" }
        if (probe == null) why.add("general-capability retention was not measured")
        else if (probe.base - probe.specialist > RETENTION_DROP) why.add("general-capability retention dropped by ${"%.2f".format(probe.base - probe.specialist)} (limit $RETENTION_DROP)")
        val allowed = why.isEmpty()
        val reason = if (allowed) "$nItems held-out items; every domain metric improved with a 95% interval excluding zero; no regression; retention within tolerance." else "No improvement claim: " + why.joinToString("; ") + "."
        val caveats = arrayListOf(
            "Metrics are deterministic lexical/numeric heuristics, not human or model judgements. Exact-fact answers from unseen documents are expected to be near zero closed-book (exact facts belong to retrieval).",
            "Evaluated on this phone with identical greedy decoding for both models, on ${prep.testChunks.size} held-out (test-split) chunks / $nItems items; ${if (o.includeRetention) "${EvalMetrics.PROBES.size} general probes (a canary, not a benchmark)." else "retention probes were skipped."}",
            "95% intervals are seeded item-level paired bootstrap; intervals across metrics are not corrected for multiple comparisons.",
            "Metrics labelled \"with source excerpts in the prompt\" measure prompting with retrieved text, not what the weights learned.")
        if (rows.any { it.n < SMALL_N }) caveats.add("Some metrics have n < $SMALL_N: statistically weak: " + rows.filter { it.n < SMALL_N }.joinToString { "${it.id} (n=${it.n})" })
        if (o.includeGroundedMetrics.not()) caveats.add("Grounded metrics were not measured.")
        val spInfo = specs.info(prep.spec.id)
        val view = EvaluationView(prep.spec.projectId, rec.ev.id, "Base: ${host.baseRef(prep.spec.projectId)?.name ?: prep.spec.baseModelId}", "Specialist ${spInfo?.version ?: prep.spec.version}",
            rows, caveats, allowed, reason, false, clock.nowMs())
        fun perf(s: Subject): PerfRow {
            val l = s.latencies.sorted()
            return PerfRow(s.role, if (l.isEmpty()) 0.0 else l.sum() / l.size, if (l.isEmpty()) 0.0 else l[minOf(l.size - 1, (0.95 * l.size).toInt())],
                if (s.genMs > 0) s.genTokens * 1000.0 / s.genMs else 0.0, s.loadMs, (s.peakRss / Conditions.MIB).toInt().takeIf { it > 0 })
        }
        val items = prep.items.map { it0 ->
            LocalEvalItemResult(it0.itemId, it0.kind, it0.question, it0.goldRefs.firstOrNull() ?: "", (b.closed[it0.itemId] ?: "").take(600), (sp.closed[it0.itemId] ?: "").take(600), b.ok[it0.itemId], sp.ok[it0.itemId])
        }
        specs.markEvaluated(prep.spec.id)
        update(rec) { it.copy(state = LocalEvalState.SUCCEEDED, progress = Progress(it.progress.total, it.progress.total, "steps"), message = "Done", view = view, performance = listOf(perf(b), perf(sp)), items = items, finishedAt = clock.nowMs()) }
    }
}
