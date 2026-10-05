package com.hotatticgames.llmtrainer.extract

/** Where an object lives: a file offset (type 1), inside an object stream (type 2) or free (0). */
private class XEntry(val type: Int, val a: Long, val b: Int)

internal class PdfPage(val dict: PDict, val resources: PDict?)

/**
 * Low-level PDF reader: cross-reference tables, xref streams (incl. hybrid /XRefStm and /Prev chains), object streams,
 * stream filters, and a full-file rebuild scan when the xref data is missing or wrong. Everything is bounded; malformed
 * input raises [PdfException] or yields [PNull], never an endless loop.
 */
internal class PdfDocument(private val buf: ByteArray) {
    private val xref = HashMap<Int, XEntry>()
    private val cache = HashMap<Int, PObj>()
    private val inProgress = HashSet<Int>()
    private val objStmCache = HashMap<Int, Map<Int, PObj>>()
    private var rebuilt = false
    var trailer: PDict = PDict(emptyMap())
        private set
    private val base: Int = run {
        val i = indexOfBytes(buf, "%PDF-", 0, minOf(buf.size, 1024))
        if (i > 0) i else 0
    }
    val notes = ArrayList<String>()

    val isEncrypted: Boolean get() = trailer.map.containsKey("Encrypt")

    // ---- typed access helpers -------------------------------------------------------------------------------------

    fun resolve(o: PObj?): PObj {
        var cur = o ?: return PNull
        var n = 0
        while (cur is PRef) {
            if (++n > 32) return PNull
            cur = getObject(cur.num)
        }
        return cur
    }
    fun dict(o: PObj?): PDict? = when (val r = resolve(o)) { is PDict -> r; is PStream -> r.dict; else -> null }
    fun get(d: PDict?, key: String): PObj = if (d == null) PNull else resolve(d.map[key])
    fun name(o: PObj?): String? = (resolve(o) as? PName)?.v
    fun int(o: PObj?): Int? = (resolve(o) as? PNum)?.int()
    fun num(o: PObj?): Double? = (resolve(o) as? PNum)?.v
    fun arr(o: PObj?): List<PObj>? = (resolve(o) as? PArr)?.items

    // ---- loading --------------------------------------------------------------------------------------------------

    /** Returns false only when nothing usable could be found (no objects at all). */
    fun load(): Boolean {
        var ok = false
        try { ok = loadXrefChain() && isUsableRoot() } catch (e: Exception) { ok = false } catch (e: StackOverflowError) { ok = false }
        if (!ok) { rebuild() }
        return xref.isNotEmpty() && (isUsableRoot() || trailer.map.containsKey("Encrypt") || findPagesFallback())
    }

    private fun isUsableRoot(): Boolean {
        if (trailer.map.containsKey("Encrypt")) return true
        val root = dict(trailer.map["Root"]) ?: return false
        return dict(root.map["Pages"]) != null
    }

    private fun findPagesFallback(): Boolean = xref.keys.take(50000).any { (dict(PRef(it, 0))?.map?.get("Type") as? PName)?.v == "Page" }

    private fun loadXrefChain(): Boolean {
        val sx = lastIndexOfBytes(buf, "startxref")
        if (sx < 0) return false
        val lx = PdfLexer(buf, sx + 9)
        val first = (lx.readObject() as? PNum)?.v?.toLong() ?: return false
        val visited = HashSet<Long>()
        val queue = ArrayDeque<Long>()
        queue.add(first)
        var merged: MutableMap<String, PObj>? = null
        while (queue.isNotEmpty()) {
            val off = queue.removeFirst()
            if (!visited.add(off) || visited.size > 512) continue
            val t = readXrefSection(off) ?: return false
            if (merged == null) merged = LinkedHashMap()
            for ((k, v) in t.map) merged.putIfAbsent(k, v)
            (t.map["XRefStm"] as? PNum)?.let { queue.addFirst(it.v.toLong()) }
            (t.map["Prev"] as? PNum)?.let { queue.addLast(it.v.toLong()) }
        }
        trailer = PDict(merged ?: return false)
        return xref.isNotEmpty()
    }

    private fun fixOffset(off: Long): Int {
        val o = off.coerceIn(0, Int.MAX_VALUE.toLong()).toInt()
        return o
    }

    private fun readXrefSection(offL: Long): PDict? {
        var off = fixOffset(offL)
        var lx = PdfLexer(buf, off)
        lx.skipWs()
        if (!lx.startsWith("xref") && base > 0 && off + base < buf.size) { off += base; lx = PdfLexer(buf, off); lx.skipWs() }
        if (lx.startsWith("xref")) {
            lx.pos += 4
            while (true) {
                lx.skipWs()
                if (lx.startsWith("trailer")) { lx.pos += 7; break }
                val firstNum = (lx.readObject() as? PNum) ?: return null
                val count = (lx.readObject() as? PNum) ?: return null
                if (count.v < 0 || count.v > 50_000_000) return null
                for (i in 0 until count.v.toInt()) {
                    val o = lx.readObject() as? PNum ?: return null
                    val g = lx.readObject() as? PNum ?: return null
                    val k = lx.readObject() as? PKeyword ?: return null
                    val num = firstNum.int() + i
                    xref.putIfAbsent(num, if (k.v == "n") XEntry(1, o.v.toLong(), g.int()) else XEntry(0, 0, 0))
                }
            }
            return lx.readObject() as? PDict
        }
        // xref stream
        val (_, obj) = parseIndirectAt(off, null) ?: return null
        val st = obj as? PStream ?: return null
        if (name(st.dict.map["Type"]) != "XRef") return null
        val data = decode(st) ?: return null
        val w = arr(st.dict.map["W"])?.map { int(it) ?: 0 } ?: return null
        if (w.size < 3 || w.any { it < 0 || it > 8 }) return null
        val size = int(st.dict.map["Size"]) ?: 0
        val index = arr(st.dict.map["Index"])?.map { int(it) ?: 0 } ?: listOf(0, size)
        val rowLen = w.sum()
        if (rowLen == 0) return null
        var p = 0
        fun field(n: Int, def: Long): Long {
            if (n == 0) return def
            var v = 0L
            repeat(n) { v = (v shl 8) or (data[p++].toLong() and 0xFF) }
            return v
        }
        var i = 0
        while (i + 1 < index.size) {
            for (k in 0 until index[i + 1]) {
                if (p + rowLen > data.size) break
                val type = field(w[0], 1).toInt()
                val f2 = field(w[1], 0)
                val f3 = field(w[2], 0).toInt()
                xref.putIfAbsent(index[i] + k, when (type) { 1 -> XEntry(1, f2, f3); 2 -> XEntry(2, f2, f3); else -> XEntry(0, 0, 0) })
            }
            i += 2
        }
        return st.dict
    }

    /** Scan the whole file for "N G obj" and trailer/catalog information. */
    private fun rebuild() {
        rebuilt = true
        xref.clear(); cache.clear(); objStmCache.clear()
        notes.add("xref_rebuilt")
        var i = 0
        val n = buf.size
        while (true) {
            val k = indexOfBytes(buf, "obj", i)
            if (k < 0) break
            i = k + 3
            if (k >= 3 && buf[k - 1] == 'd'.code.toByte() && buf[k - 2] == 'n'.code.toByte() && buf[k - 3] == 'e'.code.toByte()) continue
            if (k + 3 < n && !isPdfWs(buf[k + 3].toInt() and 0xFF) && !isPdfDelim(buf[k + 3].toInt() and 0xFF)) continue
            var j = k - 1
            while (j >= 0 && isPdfWs(buf[j].toInt() and 0xFF)) j--
            val genEnd = j
            while (j >= 0 && buf[j] in '0'.code.toByte()..'9'.code.toByte()) j--
            if (j == genEnd) continue
            val sp = j
            while (j >= 0 && isPdfWs(buf[j].toInt() and 0xFF)) j--
            if (j == sp) continue
            val numEnd = j
            while (j >= 0 && buf[j] in '0'.code.toByte()..'9'.code.toByte()) j--
            if (j == numEnd) continue
            if (numEnd - j > 9) continue
            val num = String(buf, j + 1, numEnd - j, Charsets.ISO_8859_1).toInt()
            xref[num] = XEntry(1, (j + 1).toLong(), 0)
            if (xref.size > 5_000_000) break
        }
        // trailers
        val merged = LinkedHashMap<String, PObj>()
        var t = 0
        while (true) {
            val k = indexOfBytes(buf, "trailer", t)
            if (k < 0) break
            t = k + 7
            try {
                val d = PdfLexer(buf, t).readObject() as? PDict ?: continue
                for ((kk, v) in d.map) merged[kk] = v // later trailers win
            } catch (e: Exception) { }
        }
        // xref streams, object streams, catalog
        var catalog: PRef? = null
        for (num in xref.keys.sorted().take(300_000)) {
            val o = try { getObject(num) } catch (e: Exception) { PNull }
            val d = when (o) { is PDict -> o; is PStream -> o.dict; else -> continue }
            when ((d.map["Type"] as? PName)?.v) {
                "XRef" -> for (key in listOf("Root", "Encrypt", "Info", "ID")) d.map[key]?.let { merged[key] = it }
                "Catalog" -> catalog = PRef(num, 0)
                "ObjStm" -> if (o is PStream) {
                    try { for (inner in objStmEntries(num).keys) xref.putIfAbsent(inner, XEntry(2, num.toLong(), 0)) } catch (e: Exception) { }
                }
            }
        }
        if (!merged.containsKey("Root") && catalog != null) merged["Root"] = catalog
        trailer = PDict(merged)
    }

    // ---- objects --------------------------------------------------------------------------------------------------

    fun getObject(num: Int): PObj {
        cache[num]?.let { return it }
        val e = xref[num] ?: return PNull
        if (!inProgress.add(num)) return PNull
        try {
            val o: PObj = when (e.type) {
                1 -> {
                    val off = fixOffset(e.a)
                    var parsed = parseIndirectAt(off, num)
                    if (parsed == null && base > 0) parsed = parseIndirectAt(off + base, num)
                    if (parsed == null) {
                        if (!rebuilt) { inProgress.remove(num); rebuild(); return getObject(num) }
                        PNull
                    } else parsed.second
                }
                2 -> objStmEntries(e.a.toInt())[num] ?: PNull
                else -> PNull
            }
            cache[num] = o
            return o
        } finally { inProgress.remove(num) }
    }

    /** Parse "N G obj <value> [stream ...]" at [off]. Returns null when the header is not there (or numbers differ from [expect]). */
    private fun parseIndirectAt(off: Int, expect: Int?): Pair<Int, PObj>? {
        if (off < 0 || off >= buf.size) return null
        val hdr = PdfLexer(buf, off, refs = false)
        val num = (hdr.readObject() as? PNum)?.takeIf { it.isInt }?.int() ?: return null
        if (expect != null && num != expect) return null
        (hdr.readObject() as? PNum) ?: return null
        val kw = hdr.readObject() as? PKeyword ?: return null
        if (kw.v != "obj") return null
        val lx = PdfLexer(buf, hdr.pos)
        val body = lx.readObject() ?: return num to PNull
        if (body is PKeyword) return num to PNull
        if (body !is PDict) return num to body
        lx.skipWs()
        if (!lx.startsWith("stream")) return num to body
        var p = lx.pos + 6
        if (p < buf.size && buf[p] == 13.toByte()) { p++; if (p < buf.size && buf[p] == 10.toByte()) p++ } else if (p < buf.size && buf[p] == 10.toByte()) p++
        var len = -1L
        val lo = body.map["Length"]
        if (lo is PNum) len = lo.v.toLong() else if (lo is PRef) (resolve(lo) as? PNum)?.let { len = it.v.toLong() }
        var endPos = -1
        if (len >= 0 && p + len <= buf.size) {
            val chk = PdfLexer(buf, (p + len).toInt())
            chk.skipWs()
            if (chk.startsWith("endstream") || chk.startsWith("endobj")) endPos = (p + len).toInt()
        }
        if (endPos < 0) {
            var e = indexOfBytes(buf, "endstream", p)
            if (e < 0) e = indexOfBytes(buf, "endobj", p)
            if (e < 0) e = buf.size
            if (e > p && buf[e - 1] == 10.toByte()) e--
            if (e > p && buf[e - 1] == 13.toByte()) e--
            endPos = e
        }
        return num to PStream(body, buf.copyOfRange(p, maxOf(p, endPos)))
    }

    private fun objStmEntries(stmNum: Int): Map<Int, PObj> {
        objStmCache[stmNum]?.let { return it }
        val st = getObject(stmNum) as? PStream ?: throw PdfException("object stream $stmNum missing")
        val data = decode(st) ?: throw PdfException("object stream $stmNum undecodable")
        val n = int(st.dict.map["N"]) ?: 0
        val first = int(st.dict.map["First"]) ?: 0
        if (n < 0 || n > 1_000_000 || first < 0 || first > data.size) throw PdfException("bad object stream header")
        val hdr = PdfLexer(data, 0, first, refs = false)
        val nums = ArrayList<Pair<Int, Int>>()
        for (i in 0 until n) {
            val a = (hdr.readObject() as? PNum) ?: break
            val b = (hdr.readObject() as? PNum) ?: break
            nums.add(a.int() to b.int())
        }
        val out = HashMap<Int, PObj>()
        objStmCache[stmNum] = out // guard against self-reference
        for ((num, rel) in nums) {
            val s = first + rel
            if (rel < 0 || s >= data.size) continue
            try { out[num] = PdfLexer(data, s).readObject() ?: PNull } catch (e: PdfException) { }
        }
        return out
    }

    // ---- stream decoding ------------------------------------------------------------------------------------------

    /** Decoded bytes, or null when the stream uses a filter we do not implement (image filters, Crypt). */
    fun decode(st: PStream): ByteArray? {
        val f = resolve(st.dict.map["Filter"] ?: st.dict.map["F"])
        val filters = when (f) { is PName -> listOf(f.v); is PArr -> f.items.mapNotNull { name(it) }; else -> emptyList() }
        val parmsO = resolve(st.dict.map["DecodeParms"] ?: st.dict.map["DP"])
        val parms: List<PDict?> = when (parmsO) {
            is PDict -> listOf(parmsO)
            is PArr -> parmsO.items.map { dict(it) }
            else -> emptyList()
        }
        var data = st.raw
        for ((i, fn) in filters.withIndex()) {
            val p = parms.getOrNull(i)
            data = when (fn) {
                "FlateDecode", "Fl" -> applyPredictor(PdfFilters.flate(data), p)
                "LZWDecode", "LZW" -> applyPredictor(PdfFilters.lzw(data, int(p?.map?.get("EarlyChange")) ?: 1), p)
                "ASCIIHexDecode", "AHx" -> PdfFilters.asciiHex(data)
                "ASCII85Decode", "A85" -> PdfFilters.ascii85(data)
                "RunLengthDecode", "RL" -> PdfFilters.runLength(data)
                else -> return null
            }
        }
        return data
    }

    private fun applyPredictor(d: ByteArray, p: PDict?): ByteArray {
        val pred = int(p?.map?.get("Predictor")) ?: 1
        if (pred < 2) return d
        return PdfFilters.predictor(d, pred, int(p?.map?.get("Colors")) ?: 1, int(p?.map?.get("BitsPerComponent")) ?: 8, int(p?.map?.get("Columns")) ?: 1)
    }

    // ---- page tree ------------------------------------------------------------------------------------------------

    /** Leaf pages in document order, with inherited /Resources. Falls back to scanning /Type /Page objects. */
    fun pages(): List<PdfPage> {
        val out = ArrayList<PdfPage>()
        val rootPages = dict(dict(trailer.map["Root"])?.map?.get("Pages"))
        if (rootPages != null) {
            val seen = HashSet<PDict>()
            fun walk(node: PDict, res: PDict?, depth: Int) {
                if (depth > 64 || out.size >= MAX_PAGES || !seen.add(node)) return
                val r = dict(node.map["Resources"]) ?: res
                val kids = arr(node.map["Kids"])
                val type = name(node.map["Type"])
                if (kids != null && type != "Page") {
                    for (k in kids) { val kd = dict(k) ?: continue; walk(kd, r, depth + 1) }
                } else out.add(PdfPage(node, r))
            }
            walk(rootPages, null, 0)
        }
        if (out.isEmpty()) {
            for (num in xref.keys.sorted().take(300_000)) {
                val d = (getObject(num) as? PDict) ?: continue
                if (name(d.map["Type"]) == "Page") out.add(PdfPage(d, dict(d.map["Resources"])))
                if (out.size >= MAX_PAGES) break
            }
            if (out.isNotEmpty()) notes.add("page_tree_rebuilt")
        }
        return out
    }

    companion object { const val MAX_PAGES = 50_000 }
}
