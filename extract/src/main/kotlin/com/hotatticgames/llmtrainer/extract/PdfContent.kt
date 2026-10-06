package com.hotatticgames.llmtrainer.extract

import kotlin.math.abs
import kotlin.math.hypot

/** Accumulates one page's text as lines ("" entries mark paragraph gaps). */
internal class PageText {
    private val lines = ArrayList<String>()
    private var cur = StringBuilder()
    fun append(s: String) { cur.append(s) }
    fun endsWithSpaceOrEmpty() = cur.isEmpty() || cur.last() == ' '
    fun newLine(paragraph: Boolean) { lines.add(cur.toString()); cur = StringBuilder(); if (paragraph) lines.add("") }
    fun finish(): List<String> {
        lines.add(cur.toString())
        val out = ArrayList<String>()
        for (l in lines) {
            val t = normalize(l)
            if (t.isEmpty() && (out.isEmpty() || out.last().isEmpty())) continue
            out.add(t)
        }
        while (out.isNotEmpty() && out.last().isEmpty()) out.removeAt(out.size - 1)
        return out
    }

    private fun normalize(s: String): String {
        val sb = StringBuilder(s.length)
        for (c in s) when {
            c == 'ﬀ' -> sb.append("ff"); c == 'ﬁ' -> sb.append("fi"); c == 'ﬂ' -> sb.append("fl")
            c == 'ﬃ' -> sb.append("ffi"); c == 'ﬄ' -> sb.append("ffl"); c == 'ﬅ' || c == 'ﬆ' -> sb.append("st")
            c == '­' || c == '\u0000' || c == '￾' || c == '﻿' -> {}
            c == '\t' || c == ' ' -> sb.append(' ')
            c.code < 32 -> {}
            else -> sb.append(c)
        }
        return sb.toString().trim()
    }
}

/**
 * Content-stream interpreter limited to text: tracks CTM/text matrices only to decide line breaks, never glyph widths.
 * Heuristics (documented limits): a new line starts when the baseline moves by more than half the font size or after
 * T*, ' and "; a gap of more than 1.6 font sizes downwards starts a paragraph; a repositioning on the same baseline
 * and a TJ kern below -[KERN_SPACE] thousandths of an em insert a single space. Rotated text and multi-column layout are
 * not analysed, so column text interleaves in content-stream order.
 */
internal class ContentInterpreter(private val doc: PdfDocument, private val fonts: FontFactory, val out: PageText, val stats: DecodeStats) {
    private class Gs(var ctm: DoubleArray, var font: FontDecoder?, var size: Double)

    var opsLeft = 3_000_000
    private var formsLeft = 1000
    private var bytesLeft = 128L * 1024 * 1024
    var sawTextOps = false

    private var tm = identity()
    private var tlm = identity()
    private var gs = Gs(identity(), null, 1.0)
    private val stack = ArrayList<Gs>()
    private var haveLast = false
    private var lastY = 0.0
    private var lastFs = 10.0
    private var moved = false
    private var forceBreak = false

    private fun identity() = doubleArrayOf(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    private fun mul(m: DoubleArray, n: DoubleArray) = doubleArrayOf(
        m[0] * n[0] + m[1] * n[2], m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2], m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4], m[4] * n[1] + m[5] * n[3] + n[5])

    fun run(content: ByteArray, resources: PDict?, depth: Int, seen: MutableSet<PStream>) {
        bytesLeft -= content.size
        if (bytesLeft < 0) throw PdfException("page content exceeds decoded size budget")
        val lx = PdfLexer(content, 0, content.size, refs = false)
        val ops = ArrayList<PObj>()
        val fontRes = doc.dict(doc.get(resources, "Font"))
        while (true) {
            if (--opsLeft < 0) throw PdfException("operator budget exceeded")
            val o = lx.readObject() ?: break
            if (o !is PKeyword) { ops.add(o); if (ops.size > 600) ops.removeAt(0); continue }
            try {
                operator(o.v, ops, lx, fontRes, resources, depth, seen)
            } catch (e: IndexOutOfBoundsException) { /* malformed operand count: ignore the operator */ }
            ops.clear()
        }
    }

    private fun n(ops: List<PObj>, i: Int): Double = (ops[i] as? PNum)?.v ?: 0.0

    private fun operator(op: String, ops: List<PObj>, lx: PdfLexer, fontRes: PDict?, res: PDict?, depth: Int, seen: MutableSet<PStream>) {
        when (op) {
            "q" -> if (stack.size < 256) stack.add(Gs(gs.ctm, gs.font, gs.size))
            "Q" -> if (stack.isNotEmpty()) gs = stack.removeAt(stack.size - 1)
            "cm" -> if (ops.size >= 6) gs.ctm = mul(doubleArrayOf(n(ops, 0), n(ops, 1), n(ops, 2), n(ops, 3), n(ops, 4), n(ops, 5)), gs.ctm)
            "BT" -> { tm = identity(); tlm = identity(); moved = true }
            "ET" -> {}
            "Tf" -> if (ops.size >= 2) {
                val fname = (ops[ops.size - 2] as? PName)?.v
                val fd = if (fname != null) doc.dict(fontRes?.map?.get(fname)) else null
                gs.font = if (fd != null) fonts.forFont(fd) else fonts.default
                gs.size = (ops.last() as? PNum)?.v ?: 1.0
            }
            "Td" -> if (ops.size >= 2) { tlm = mul(doubleArrayOf(1.0, 0.0, 0.0, 1.0, n(ops, 0), n(ops, 1)), tlm); tm = tlm; moved = true }
            "TD" -> if (ops.size >= 2) { tlm = mul(doubleArrayOf(1.0, 0.0, 0.0, 1.0, n(ops, 0), n(ops, 1)), tlm); tm = tlm; moved = true }
            "Tm" -> if (ops.size >= 6) { tlm = doubleArrayOf(n(ops, 0), n(ops, 1), n(ops, 2), n(ops, 3), n(ops, 4), n(ops, 5)); tm = tlm; moved = true }
            "T*" -> { forceBreak = true; moved = true }
            "Tj" -> (ops.lastOrNull() as? PStr)?.let { beginShow(); showBytes(it.bytes) }
            "'", "\"" -> { forceBreak = true; (ops.lastOrNull() as? PStr)?.let { beginShow(); showBytes(it.bytes) } }
            "TJ" -> (ops.lastOrNull() as? PArr)?.let { arr ->
                var begun = false
                for (e in arr.items) {
                    if (e is PStr) { if (!begun) { beginShow(); begun = true }; showBytes(e.bytes) }
                    else if (e is PNum && begun && e.v < -KERN_SPACE && !out.endsWithSpaceOrEmpty()) out.append(" ")
                }
            }
            "Do" -> if (depth < 4 && formsLeft-- > 0) {
                val name = (ops.lastOrNull() as? PName)?.v ?: return
                val xo = doc.get(doc.dict(doc.get(res, "XObject")), name) as? PStream ?: return
                if (doc.name(xo.dict.map["Subtype"]) != "Form" || !seen.add(xo)) return
                val data = doc.decode(xo) ?: return
                val saved = gs
                val savedStack = ArrayList(stack)
                val m = doc.arr(xo.dict.map["Matrix"])?.map { doc.num(it) ?: 0.0 }
                gs = Gs(if (m != null && m.size == 6) mul(m.toDoubleArray(), gs.ctm) else gs.ctm, gs.font, gs.size)
                val savedTm = tm; val savedTlm = tlm
                try { run(data, doc.dict(xo.dict.map["Resources"]) ?: res, depth + 1, seen) }
                finally { gs = saved; stack.clear(); stack.addAll(savedStack); tm = savedTm; tlm = savedTlm; seen.remove(xo) }
            }
            "BI" -> skipInlineImage(lx)
        }
    }

    private fun skipInlineImage(lx: PdfLexer) {
        // dictionary-ish key/values up to ID, then binary data up to a standalone EI
        var guard = 0
        while (guard++ < 10000) {
            val o = lx.readObject() ?: return
            if (o is PKeyword && o.v == "ID") break
        }
        var p = lx.pos + 1
        while (true) {
            val e = indexOfBytes(lx.b, "EI", p, lx.end)
            if (e < 0) { lx.pos = lx.end; return }
            val before = if (e == 0) 32 else lx.b[e - 1].toInt() and 0xFF
            val after = if (e + 2 >= lx.end) 32 else lx.b[e + 2].toInt() and 0xFF
            if (isPdfWs(before) && (isPdfWs(after) || isPdfDelim(after))) { lx.pos = e + 2; return }
            p = e + 2
        }
    }

    /** Decide, from the text position, whether this show continues the line, starts a new one, or a new paragraph. */
    private fun beginShow() {
        sawTextOps = true
        val m = mul(tm, gs.ctm)
        val y = m[5]
        var fs = abs(gs.size) * hypot(m[2], m[3])
        if (fs < 1e-3) fs = abs(gs.size).coerceAtLeast(1.0)
        if (!haveLast) { haveLast = true; lastY = y; lastFs = fs }
        else {
            val down = lastY - y
            val big = maxOf(fs, lastFs)
            if (forceBreak || abs(down) > 0.5 * big) {
                out.newLine(down > PARAGRAPH_GAP * big)
                lastY = y; lastFs = fs
            } else if (moved && !out.endsWithSpaceOrEmpty()) out.append(" ")
        }
        forceBreak = false
        moved = false
    }

    private fun showBytes(b: ByteArray) {
        val f = gs.font ?: fonts.default
        out.append(f.decode(b, stats))
    }

    companion object {
        const val KERN_SPACE = 180.0
        const val PARAGRAPH_GAP = 1.6
    }
}
