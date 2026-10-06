package com.hotatticgames.llmtrainer.app

import android.view.View
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import com.hotatticgames.llmtrainer.studio.api.ABComparison
import com.hotatticgames.llmtrainer.studio.api.ABOptions
import com.hotatticgames.llmtrainer.studio.api.CancelToken
import com.hotatticgames.llmtrainer.studio.api.ChatMessageRecord
import com.hotatticgames.llmtrainer.studio.api.ChatOptions
import com.hotatticgames.llmtrainer.studio.api.ChatRole
import com.hotatticgames.llmtrainer.studio.api.ChatSessionInfo
import com.hotatticgames.llmtrainer.studio.api.ChatTarget
import com.hotatticgames.llmtrainer.studio.api.ContextChunk
import com.hotatticgames.llmtrainer.studio.api.EngineStatus
import com.hotatticgames.llmtrainer.studio.api.ProjectId
import com.hotatticgames.llmtrainer.studio.api.ProjectModelState
import com.hotatticgames.llmtrainer.studio.api.SpecialistInfo
import com.hotatticgames.llmtrainer.studio.api.StudioError
import com.hotatticgames.llmtrainer.studio.api.StudioResult

// ---------------------------------------------------------------------------------------------------------
// Chat: base model or specialist, streaming, cancel, measured speed, retrieval clearly labelled "not training"
// ---------------------------------------------------------------------------------------------------------

private class ChatInit(val state: StudioResult<ProjectModelState>, val engine: EngineStatus, val specialists: List<SpecialistInfo>)

private fun contextLines(chunks: List<ContextChunk>): String = chunks.joinToString("\n") { k ->
    "Excerpt: ${k.sourceName}${k.page?.let { " p.$it" } ?: ""}${k.section?.let { " - $it" } ?: ""}: ${k.excerpt.take(140)}"
}

/** Engine-unavailable / blocked card shared by chat, A/B and evaluation. */
internal fun blockedCard(c: Ctl, title: String, reason: String?, pid: String?, col: LinearLayout) {
    val u = c.ui
    val bc = u.card(10)
    bc.tag = "blocked-card"
    bc.addView(u.tv(title, 15f, u.bad, true))
    bc.addView(u.tv(reason ?: "Unavailable.", 13f, u.ink, topDp = 4))
    bc.addView(u.button("Open the model manager", "btn:to-models", false) { c.go(Route(Kind.MODELS, pid)) })
    col.addView(bc)
}

internal fun chatScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    if (pid == null) { missing(c, col, "project"); return }
    val project = ProjectId(pid)
    col.addView(u.tv("Ask the model installed on this phone. Everything runs on the device; nothing is sent anywhere.", 13f, u.muted, topDp = 4))
    val top = u.col()
    val msgs = u.col()
    val inputBox = u.col()
    col.addView(top); col.addView(msgs); col.addView(inputBox)

    var st: ProjectModelState? = null
    var target = ChatTarget.BASE
    var useCtx = false
    var chat: ChatSessionInfo? = null
    var generating = false
    var token: CancelToken? = null
    var live: TextView? = null
    var input: EditText? = null
    var sendBtn: View? = null
    var stopBtn: View? = null
    var baseBtn: android.widget.Button? = null
    var specBtn: android.widget.Button? = null

    fun bubble(rec: ChatMessageRecord, info: ChatSessionInfo?): LinearLayout {
        val card = u.card(8)
        card.tag = "msg:" + rec.id
        if (rec.role == ChatRole.USER) {
            card.addView(u.tv("YOU", 11f, u.accent, true))
            card.addView(u.tv(rec.text, 14f, u.ink).also { it.setTextIsSelectable(true) })
        } else {
            card.addView(u.tv("MODEL" + (info?.let { " - ${it.modelLabel}" } ?: ""), 11f, u.info, true))
            val t = u.tv(rec.text, 14f, u.ink); t.setTextIsSelectable(true); t.tag = "chat-answer"
            card.addView(t)
            if (rec.interrupted) card.addView(u.badge("STOPPED BY YOU - partial answer", u.warn))
            rec.contextLabel?.let { label ->
                val l = u.tv(label, 12f, u.warn, true, 4); l.tag = "chat-context-label"
                card.addView(l)
                card.addView(u.tv(contextLines(rec.contextUsed), 11f, u.muted))
            }
            rec.stats?.let { g -> val s = u.tv(fmtGenStats(g), 11f, u.muted, topDp = 4); s.tag = "chat-stats"; card.addView(s) }
        }
        return card
    }

    fun showMessages(info: ChatSessionInfo, hist: List<ChatMessageRecord>) {
        msgs.removeAllViews()
        val lab = u.tv("Talking to: ${info.modelLabel}", 13f, u.ink, true, 10)
        lab.tag = "chat-model-label"
        msgs.addView(lab)
        if (info.options.useSourceContext) msgs.addView(u.tv("Retrieval is ON: excerpts of your sources are added to the prompt. That is not training and does not change the model.", 12f, u.warn))
        if (hist.isEmpty()) msgs.addView(u.tv("No messages yet. The first answer includes the time to load the model.", 12f, u.muted, topDp = 6))
        for (m in hist) msgs.addView(bubble(m, info))
    }

    fun startChat(forceNew: Boolean) {
        val s = st ?: return
        baseBtn?.let { u.setPrimary(it, target == ChatTarget.BASE) }
        specBtn?.let { u.setPrimary(it, target == ChatTarget.SPECIALIST) }
        val tg = target
        val uc = useCtx
        c.bg({
            val existing = if (forceNew) null else c.studio.listChats(project).lastOrNull {
                it.target == tg && it.options.useSourceContext == uc && (tg == ChatTarget.BASE || it.specialistId == s.selectedSpecialistId)
            }
            val info: StudioResult<ChatSessionInfo> = if (existing != null) StudioResult.Ok(existing)
            else c.studio.createChat(project, tg, if (tg == ChatTarget.SPECIALIST) s.selectedSpecialistId else null, ChatOptions(useSourceContext = uc))
            when (info) {
                is StudioResult.Ok -> StudioResult.Ok(Pair(info.value, c.studio.chatHistory(info.value.id).getOrNull() ?: emptyList<ChatMessageRecord>()))
                is StudioResult.Err -> StudioResult.Err(info.error)
            }
        }) { r -> c.handle(r) { pair -> chat = pair.first; showMessages(pair.first, pair.second) } }
    }

    fun doSend() {
        val ch = chat
        val inp = input
        if (ch == null || inp == null || generating) return
        val text = inp.text.toString().trim()
        if (text.isEmpty()) return
        generating = true
        inp.setText("")
        val now = System.currentTimeMillis()
        msgs.addView(bubble(ChatMessageRecord("pending-user", ChatRole.USER, text, now, emptyList(), null, false), ch))
        val liveCard = u.card(8)
        liveCard.tag = "msg:live"
        liveCard.addView(u.tv("MODEL - ${ch.modelLabel} (answering...)", 11f, u.info, true))
        val lt = u.tv("", 14f, u.ink); lt.tag = "chat-live"
        liveCard.addView(lt)
        msgs.addView(liveCard)
        live = lt
        val tok = CancelToken(); token = tok
        sendBtn?.visibility = View.GONE; stopBtn?.visibility = View.VISIBLE
        c.long({
            c.studio.sendMessage(ch.id, text, tok) { piece -> c.post { lt.append(piece) } }
        }) { r ->
            generating = false; token = null; live = null
            sendBtn?.visibility = View.VISIBLE; stopBtn?.visibility = View.GONE
            msgs.removeView(liveCard)
            when (r) {
                is StudioResult.Ok -> msgs.addView(bubble(r.value, ch))
                is StudioResult.Err -> { c.error(c.describe(r.error)); inp.setText(text) }
            }
        }
    }

    fun build(init: ChatInit) {
        top.removeAllViews(); inputBox.removeAllViews()
        val ms = when (val r = init.state) { is StudioResult.Ok -> r.value; is StudioResult.Err -> { c.error(c.describe(r.error)); return } }
        st = ms
        val badge = if (init.engine.inferenceAvailable) u.badge("ENGINE READY (${init.engine.runtimeId})", u.ok) else u.badge("ENGINE UNAVAILABLE", u.bad)
        top.addView(badge)
        if (!init.engine.inferenceAvailable) top.addView(u.tv(init.engine.inferenceReason ?: "The native engine is not available on this install.", 12f, u.bad))
        if (!ms.canChatBase.ok) { blockedCard(c, "Chat is not available yet", ms.canChatBase.reason, pid, top); return }

        val tb = u.row()
        val bb = u.button("Base model", "btn:chat-target-base", target == ChatTarget.BASE) { target = ChatTarget.BASE; startChat(false) }
        val sb = u.button("Specialist", "btn:chat-target-specialist", target == ChatTarget.SPECIALIST) {
            if (!ms.canChatSpecialist.ok) c.notice(ms.canChatSpecialist.reason ?: "No specialist available.") else { target = ChatTarget.SPECIALIST; startChat(false) }
        }
        baseBtn = bb; specBtn = sb
        tb.addView(bb, LinearLayout.LayoutParams(0, -2, 1f))
        tb.addView(sb, LinearLayout.LayoutParams(0, -2, 1f))
        top.addView(tb)
        if (!ms.canChatSpecialist.ok) top.addView(u.tv("Specialist chat: ${ms.canChatSpecialist.reason}", 11f, u.muted))
        else init.specialists.firstOrNull { it.selected }?.let { sp ->
            top.addView(u.tv("Selected specialist: ${sp.name} ${sp.version} - parameters changed on this phone" + (if (sp.stale) " - STALE: ${sp.staleReason}" else ""), 11f, if (sp.stale) u.warn else u.muted))
        }
        top.addView(u.check("Add excerpts from my sources to the prompt (retrieval - NOT training; the model is unchanged)", "chat:use-context", useCtx) { v ->
            useCtx = v; startChat(false)
        })
        top.addView(u.button("New conversation", "btn:chat-new", false) { startChat(true) })

        val inp = u.edit("Type a message", "field:chat-input", multiline = true)
        input = inp
        inputBox.addView(inp)
        val row = u.row()
        val sendB = u.button("Send", "btn:chat-send") { doSend() }
        val stop = u.button("Stop", "btn:chat-stop", false) { token?.cancel() }
        stop.visibility = View.GONE
        sendBtn = sendB; stopBtn = stop
        row.addView(sendB, LinearLayout.LayoutParams(0, -2, 1f)); row.addView(stop, LinearLayout.LayoutParams(0, -2, 1f))
        inputBox.addView(row)
        startChat(false)
    }

    c.bg({ ChatInit(c.studio.projectModelState(project), c.studio.engineStatus(), c.studio.specialists(project)) }) { init -> build(init) }
}

// ---------------------------------------------------------------------------------------------------------
// A/B: same question to the base model and the specialist, side by side
// ---------------------------------------------------------------------------------------------------------

internal fun abScreen(c: Ctl, pid: String?, col: LinearLayout) {
    val u = c.ui
    if (pid == null) { missing(c, col, "project"); return }
    val project = ProjectId(pid)
    col.addView(u.tv("The same question goes to the base model and then to the specialist with identical greedy decoding (one model in memory at a time). " +
        "Compare the answers yourself; a single answer is an anecdote, not an evaluation.", 13f, u.muted, topDp = 4))
    val top = u.col()
    val result = u.col()
    val history = u.col()
    col.addView(top); col.addView(result); col.addView(history)

    fun cmpView(cmp: ABComparison) {
        history.removeAllViews()
        val b = u.card(8); b.tag = "ab-result"
        b.addView(u.tv("Question: ${cmp.prompt}", 13f, u.ink, true))
        cmp.base.contextUsed.takeIf { it.isNotEmpty() }?.let { b.addView(u.tv(ChatMessageRecord.CONTEXT_LABEL, 11f, u.warn)) }
        val note = u.edit("Your note about this comparison", "field:ab-note", multiline = true)
        cmp.note?.let { note.setText(it) }
        b.addView(note)
        b.addView(u.button("Save note", "btn:ab-save-note", false) {
            val txt = note.text.toString().trim()
            c.call({ c.studio.saveABNote(project, cmp.id, txt) }) { c.success("Note saved with this comparison.") }
        })
        history.addView(b)
    }

    fun savedList() {
        c.bg({ c.studio.abComparisons(project) }) { list ->
            val box = u.col()
            if (list.isNotEmpty()) box.addView(u.section("SAVED COMPARISONS (${list.size})"))
            for (x in list.sortedByDescending { it.createdAt }.take(6)) {
                val cc = u.card(6)
                cc.tag = "ab-saved:" + x.id
                cc.addView(u.tv(x.prompt, 13f, u.ink, true))
                cc.addView(u.tv("Base: ${x.base.text.take(160)}", 11f, u.muted))
                cc.addView(u.tv("Specialist: ${x.specialist.text.take(160)}", 11f, u.muted))
                x.note?.let { cc.addView(u.tv("Note: $it", 11f, u.ink)) }
                cc.addView(u.button("Delete", "btn:ab-delete:" + x.id, false) { c.call({ c.studio.deleteAB(project, x.id) }) { c.render() } })
                box.addView(cc)
            }
            result.removeAllViews(); result.addView(box)
        }
    }

    c.bg({ Triple(c.studio.projectModelState(project), c.studio.engineStatus(), c.studio.specialists(project)) }) { (msR, eng, specs) ->
        val ms = when (msR) { is StudioResult.Ok -> msR.value; is StudioResult.Err -> { c.error(c.describe(msR.error)); return@bg } }
        if (!eng.inferenceAvailable) { blockedCard(c, "Engine unavailable", eng.inferenceReason, pid, top); return@bg }
        if (!ms.canChatSpecialist.ok) { blockedCard(c, "Nothing to compare yet", ms.canChatSpecialist.reason, pid, top); return@bg }
        val spId = ms.selectedSpecialistId
        if (spId == null) { blockedCard(c, "No specialist selected", "Train or select a specialist first.", pid, top); return@bg }
        val sp = specs.firstOrNull { it.id == spId }
        top.addView(u.tv("Specialist: ${sp?.name ?: spId} ${sp?.version ?: ""} (parameters changed on this phone)", 13f, u.ink, true, 6))
        if (sp != null && sp.stale) top.addView(u.tv("STALE: ${sp.staleReason}", 12f, u.warn))
        var useCtx = false
        top.addView(u.check("Add excerpts from my sources to both prompts (retrieval - NOT training)", "ab:use-context", false) { useCtx = it })
        val prompt = u.edit("Ask the same question of both", "field:ab-prompt", multiline = true)
        top.addView(prompt)

        val cols = u.row()
        cols.tag = "ab-columns"
        fun colCard(label: String, tag: String): Triple<LinearLayout, TextView, TextView> {
            val cc = u.card(8)
            cc.addView(u.tv(label, 11f, u.accent, true).also { it.tag = "$tag-label" })
            val ans = u.tv("", 13f, u.ink, topDp = 4); ans.tag = tag; ans.setTextIsSelectable(true)
            val stats = u.tv("", 10f, u.muted, topDp = 4); stats.tag = "$tag-stats"
            cc.addView(ans); cc.addView(stats)
            return Triple(cc, ans, stats)
        }
        val left = colCard("BASE MODEL", "ab:base")
        val right = colCard("SPECIALIST", "ab:specialist")
        cols.addView(left.first, LinearLayout.LayoutParams(0, -2, 1f).apply { rightMargin = u.px(4); topMargin = u.px(8) })
        cols.addView(right.first, LinearLayout.LayoutParams(0, -2, 1f).apply { leftMargin = u.px(4); topMargin = u.px(8) })

        var token: CancelToken? = null
        var runBtn: View? = null
        var stopBtn: View? = null
        val run = u.button("Ask both", "btn:ab-run") {
            val q = prompt.text.toString().trim()
            if (q.isEmpty()) { c.error("Type a question first."); return@button }
            if (token != null) return@button
            left.second.text = ""; right.second.text = ""; left.third.text = ""; right.third.text = ""
            val tok = CancelToken(); token = tok
            runBtn?.visibility = View.GONE; stopBtn?.visibility = View.VISIBLE
            c.long({
                c.studio.compareAB(project, spId, q, ABOptions(useSourceContext = useCtx), tok) { side, piece ->
                    c.post { (if (side == ChatTarget.BASE) left.second else right.second).append(piece) }
                }
            }) { r ->
                token = null
                runBtn?.visibility = View.VISIBLE; stopBtn?.visibility = View.GONE
                when (r) {
                    is StudioResult.Ok -> {
                        val x = r.value
                        left.first.findViewWithTag<TextView>("ab:base-label")?.text = x.base.label
                        right.first.findViewWithTag<TextView>("ab:specialist-label")?.text = x.specialist.label
                        left.second.text = x.base.text; right.second.text = x.specialist.text
                        x.base.stats?.let { left.third.text = fmtGenStats(it) }
                        x.specialist.stats?.let { right.third.text = fmtGenStats(it) }
                        cmpView(x)
                        savedList()
                    }
                    is StudioResult.Err -> if (r.error is StudioError.Cancelled) c.notice("Stopped. Nothing was saved.") else c.error(c.describe(r.error))
                }
            }
        }
        val stop = u.button("Stop", "btn:ab-stop", false) { token?.cancel() }
        stop.visibility = View.GONE
        runBtn = run; stopBtn = stop
        top.addView(run); top.addView(stop)
        top.addView(cols)
        savedList()
    }
}
