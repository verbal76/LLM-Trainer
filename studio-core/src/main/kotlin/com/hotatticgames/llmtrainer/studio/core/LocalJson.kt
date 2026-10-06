package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import org.json.JSONObject

/** (De)serialization of the v2 records persisted under `projects/<id>/`. Tolerant readers: unknown keys ignored, missing keys defaulted. */
object LJ {
    private inline fun <reified E : Enum<E>> enumOf(s: String?, d: E): E = s?.let { v -> enumValues<E>().firstOrNull { it.name == v } } ?: d
    private inline fun <reified E : Enum<E>> enumOrNull(s: String?): E? = s?.let { v -> enumValues<E>().firstOrNull { it.name == v } }

    fun d(o: JSONObject, k: String): Double? = o.dbl(k)?.takeIf { !it.isNaN() }

    fun sampling(s: SamplingParams): Map<String, Any?> = linkedMapOf("temperature" to s.temperature.toDouble(), "top_k" to s.topK, "top_p" to s.topP.toDouble(), "min_p" to s.minP.toDouble(),
        "repeat_penalty" to s.repeatPenalty.toDouble(), "seed" to s.seed, "max_new_tokens" to s.maxNewTokens)
    fun sampling(o: JSONObject?): SamplingParams {
        val d = SamplingParams()
        if (o == null) return d
        return SamplingParams((o.dbl("temperature") ?: d.temperature.toDouble()).toFloat(), o.int("top_k") ?: d.topK, (o.dbl("top_p") ?: d.topP.toDouble()).toFloat(),
            (o.dbl("min_p") ?: d.minP.toDouble()).toFloat(), (o.dbl("repeat_penalty") ?: d.repeatPenalty.toDouble()).toFloat(), o.lng("seed") ?: d.seed, o.int("max_new_tokens") ?: d.maxNewTokens)
    }

    fun chatOptions(c: ChatOptions): Map<String, Any?> = linkedMapOf("use_source_context" to c.useSourceContext, "max_context_chunks" to c.maxContextChunks, "system_prompt" to c.systemPrompt,
        "context_tokens" to c.contextTokens, "sampling" to sampling(c.sampling))
    fun chatOptions(o: JSONObject?): ChatOptions {
        val d = ChatOptions()
        if (o == null) return d
        return ChatOptions(o.bool("use_source_context") ?: d.useSourceContext, o.int("max_context_chunks") ?: d.maxContextChunks, o.str("system_prompt"),
            o.int("context_tokens") ?: d.contextTokens, sampling(o.obj("sampling")))
    }

    fun settings(s: TrainingSettings): Map<String, Any?> = linkedMapOf("kind" to s.kind.name, "trainable_last_layers" to s.trainableLastLayers, "epochs" to s.epochs,
        "learning_rate" to s.learningRate.toDouble(), "context_tokens" to s.contextTokens, "seed" to s.seed, "train_embeddings" to s.trainEmbeddings,
        "checkpoint_every_steps" to s.checkpointEverySteps, "max_sequences" to s.maxSequences, "owner_confirms_plugged_in" to s.ownerConfirmsPluggedIn,
        "lora_rank" to s.loraRank, "lora_alpha" to s.loraAlpha.toDouble())
    fun settings(o: JSONObject?): TrainingSettings {
        val d = TrainingSettings()
        if (o == null) return d
        return TrainingSettings(enumOf(o.str("kind"), d.kind), o.int("trainable_last_layers") ?: d.trainableLastLayers, o.int("epochs") ?: d.epochs,
            (o.dbl("learning_rate") ?: d.learningRate.toDouble()).toFloat(), o.int("context_tokens") ?: d.contextTokens, o.lng("seed") ?: d.seed,
            o.bool("train_embeddings") ?: d.trainEmbeddings, o.int("checkpoint_every_steps") ?: d.checkpointEverySteps, o.int("max_sequences") ?: d.maxSequences,
            o.bool("owner_confirms_plugged_in") ?: false, o.int("lora_rank") ?: 0, (o.dbl("lora_alpha") ?: 0.0).toFloat())
    }

    fun err(e: StudioError?): Map<String, Any?>? = e?.let { linkedMapOf("code" to it.code, "message" to it.message) }
    fun err(o: JSONObject?): StudioError? = o?.let { StudioError.Invalid(it.str("code") ?: "ERROR", it.str("message") ?: "") }

    fun stats(s: GenerationStats?): Map<String, Any?>? = s?.let {
        linkedMapOf("prompt_tokens" to it.promptTokens, "generated_tokens" to it.generatedTokens, "ttft_ms" to it.timeToFirstTokenMs, "tokens_per_s" to it.tokensPerSecond,
            "total_ms" to it.totalMs, "load_ms" to it.modelLoadMs, "peak_rss_mb" to it.peakRssMb, "stop" to it.stopReason.name)
    }
    fun stats(o: JSONObject?): GenerationStats? = o?.let {
        GenerationStats(it.int("prompt_tokens") ?: 0, it.int("generated_tokens") ?: 0, d(it, "ttft_ms"), it.dbl("tokens_per_s") ?: 0.0, it.dbl("total_ms") ?: 0.0, it.lng("load_ms"),
            it.int("peak_rss_mb"), enumOf(it.str("stop"), StopReason.END))
    }

    fun ctx(c: ContextChunk): Map<String, Any?> = linkedMapOf("ref" to c.ref, "source" to c.sourceName, "page" to c.page, "section" to c.section, "excerpt" to c.excerpt)
    fun ctx(o: JSONObject) = ContextChunk(o.str("ref") ?: "", o.str("source") ?: "", o.int("page"), o.str("section"), o.str("excerpt") ?: "")

    fun msg(m: ChatMessageRecord): Map<String, Any?> = linkedMapOf("id" to m.id, "role" to m.role.name, "text" to m.text, "at" to m.at, "context" to m.contextUsed.map { ctx(it) },
        "stats" to stats(m.stats), "interrupted" to m.interrupted)
    fun msg(o: JSONObject) = ChatMessageRecord(o.str("id") ?: "", enumOf(o.str("role"), ChatRole.USER), o.str("text") ?: "", o.lng("at") ?: 0L,
        o.objList("context").map { ctx(it) }, stats(o.obj("stats")), o.bool("interrupted") ?: false)

    fun chatInfo(i: ChatSessionInfo): Map<String, Any?> = linkedMapOf("id" to i.id, "project" to i.projectId.value, "target" to i.target.name, "specialist_id" to i.specialistId, "title" to i.title,
        "created_at" to i.createdAt, "updated_at" to i.updatedAt, "message_count" to i.messageCount, "options" to chatOptions(i.options), "model_label" to i.modelLabel)
    fun chatInfo(o: JSONObject) = ChatSessionInfo(o.str("id") ?: "", ProjectId(o.str("project") ?: ""), enumOf(o.str("target"), ChatTarget.BASE), o.str("specialist_id"), o.str("title") ?: "",
        o.lng("created_at") ?: 0L, o.lng("updated_at") ?: 0L, o.int("message_count") ?: 0, chatOptions(o.obj("options")), o.str("model_label") ?: "")

    fun loss(p: LossPoint): Map<String, Any?> = linkedMapOf("step" to p.step, "train" to p.trainLoss, "val" to p.valLoss)
    fun loss(o: JSONObject) = LossPoint(o.int("step") ?: 0, d(o, "train"), d(o, "val"))

    fun run(r: TrainingRun): Map<String, Any?> = linkedMapOf("id" to r.id, "project" to r.projectId.value, "state" to r.state.name, "stage" to r.stage.name, "settings" to settings(r.settings),
        "base_model_id" to r.baseModelId, "dataset_sha256" to r.datasetSha256, "sequences" to r.sequences, "epoch" to r.epoch, "epochs" to r.epochs, "step" to r.step, "steps" to r.steps,
        "examples_done" to r.examplesDone, "loss" to r.lossTrend.map { loss(it) }, "train_loss" to r.latestTrainLoss, "val_loss" to r.latestValLoss, "elapsed_ms" to r.elapsedMs,
        "thermal" to r.thermal, "battery" to r.batteryPercent, "rss_mb" to r.rssMb, "checkpoint" to r.checkpoint.name, "resumed_from_step" to r.resumedFromStep, "message" to r.message,
        "error" to err(r.error), "resumable" to r.resumable, "specialist_id" to r.specialistId, "created_at" to r.createdAt, "updated_at" to r.updatedAt)
    fun run(o: JSONObject) = TrainingRun(o.str("id") ?: "", ProjectId(o.str("project") ?: ""), enumOf(o.str("state"), TrainingRunState.FAILED), enumOf(o.str("stage"), TrainingStage.PREFLIGHT),
        settings(o.obj("settings")), o.str("base_model_id") ?: "", o.str("dataset_sha256") ?: "", o.int("sequences") ?: 0, o.int("epoch") ?: 0, o.int("epochs") ?: 1, o.int("step") ?: 0,
        o.int("steps") ?: 0, o.lng("examples_done") ?: 0L, o.objList("loss").map { loss(it) }, d(o, "train_loss"), d(o, "val_loss"), o.lng("elapsed_ms") ?: 0L, o.str("thermal"),
        o.int("battery"), o.int("rss_mb"), enumOf(o.str("checkpoint"), CheckpointState.NONE), o.int("resumed_from_step") ?: 0, o.str("message") ?: "", err(o.obj("error")),
        o.bool("resumable") ?: false, o.str("specialist_id"), o.lng("created_at") ?: 0L, o.lng("updated_at") ?: 0L)

    fun metric(m: MetricRow): Map<String, Any?> = linkedMapOf("id" to m.id, "label" to m.label, "hib" to m.higherIsBetter, "base" to m.base, "specialist" to m.specialist, "delta" to m.delta,
        "ci_low" to m.ciLow, "ci_high" to m.ciHigh, "n" to m.n)
    fun metric(o: JSONObject) = MetricRow(o.str("id") ?: "", o.str("label") ?: "", o.bool("hib") ?: true, o.dbl("base") ?: 0.0, o.dbl("specialist") ?: 0.0, o.dbl("delta") ?: 0.0,
        d(o, "ci_low"), d(o, "ci_high"), o.int("n") ?: 0)

    fun view(v: EvaluationView?): Map<String, Any?>? = v?.let {
        linkedMapOf("project" to it.projectId.value, "run_id" to it.runId, "base" to it.baseModelLabel, "specialist" to it.specialistLabel, "metrics" to it.metrics.map { m -> metric(m) },
            "caveats" to it.caveats, "claim" to it.improvementClaimAllowed, "claim_reason" to it.claimReason, "stub" to it.isStub, "at" to it.importedAt)
    }
    fun view(o: JSONObject?): EvaluationView? = o?.let {
        EvaluationView(ProjectId(it.str("project") ?: ""), it.str("run_id") ?: "", it.str("base") ?: "", it.str("specialist") ?: "", it.objList("metrics").map { m -> metric(m) },
            it.strList("caveats"), it.bool("claim") ?: false, it.str("claim_reason") ?: "", it.bool("stub") ?: false, it.lng("at") ?: 0L)
    }

    fun eval(e: LocalEvaluation): Map<String, Any?> = linkedMapOf("id" to e.id, "project" to e.projectId.value, "specialist_id" to e.specialistId, "state" to e.state.name,
        "done" to e.progress.done, "total" to e.progress.total, "unit" to e.progress.unit, "message" to e.message, "error" to err(e.error), "view" to view(e.view),
        "performance" to e.performance.map { p -> linkedMapOf("subject" to p.subject, "lat_mean" to p.latencyMsMean, "lat_p95" to p.latencyMsP95, "tps" to p.tokensPerSecond, "load_ms" to p.loadMs, "rss_mb" to p.peakRssMb) },
        "items" to e.items.map { i -> linkedMapOf("id" to i.itemId, "kind" to i.kind, "q" to i.question, "ref" to i.sourceRef, "base" to i.baseAnswer, "spec" to i.specialistAnswer, "base_ok" to i.baseOk, "spec_ok" to i.specialistOk) },
        "dataset_sha256" to e.datasetSha256, "test_chunks" to e.testChunks, "item_count" to e.itemCount, "started_at" to e.startedAt, "finished_at" to e.finishedAt)
    fun eval(o: JSONObject) = LocalEvaluation(o.str("id") ?: "", ProjectId(o.str("project") ?: ""), o.str("specialist_id") ?: "", enumOf(o.str("state"), LocalEvalState.FAILED),
        Progress(o.lng("done") ?: 0L, o.lng("total") ?: 0L, o.str("unit") ?: "items"), o.str("message") ?: "", err(o.obj("error")), view(o.obj("view")),
        o.objList("performance").map { p -> PerfRow(p.str("subject") ?: "", p.dbl("lat_mean") ?: 0.0, p.dbl("lat_p95") ?: 0.0, p.dbl("tps") ?: 0.0, p.lng("load_ms") ?: 0L, p.int("rss_mb")) },
        o.objList("items").map { i -> LocalEvalItemResult(i.str("id") ?: "", i.str("kind") ?: "", i.str("q") ?: "", i.str("ref") ?: "", i.str("base") ?: "", i.str("spec") ?: "", i.bool("base_ok"), i.bool("spec_ok")) },
        o.str("dataset_sha256") ?: "", o.int("test_chunks") ?: 0, o.int("item_count") ?: 0, o.lng("started_at") ?: 0L, o.lng("finished_at"))

    fun side(s: ABSide): Map<String, Any?> = linkedMapOf("label" to s.label, "target" to s.target.name, "text" to s.text, "stats" to stats(s.stats), "context" to s.contextUsed.map { ctx(it) })
    fun side(o: JSONObject?): ABSide = ABSide(o?.str("label") ?: "", enumOf(o?.str("target"), ChatTarget.BASE), o?.str("text") ?: "", stats(o?.obj("stats")), (o?.objList("context") ?: emptyList()).map { ctx(it) })
    fun ab(c: ABComparison): Map<String, Any?> = linkedMapOf("id" to c.id, "project" to c.projectId.value, "specialist_id" to c.specialistId, "prompt" to c.prompt, "base" to side(c.base),
        "specialist" to side(c.specialist), "created_at" to c.createdAt, "note" to c.note, "saved" to c.savedAsNote)
    fun ab(o: JSONObject) = ABComparison(o.str("id") ?: "", ProjectId(o.str("project") ?: ""), o.str("specialist_id") ?: "", o.str("prompt") ?: "", side(o.obj("base")), side(o.obj("specialist")),
        o.lng("created_at") ?: 0L, o.str("note"), o.bool("saved") ?: false)
}
