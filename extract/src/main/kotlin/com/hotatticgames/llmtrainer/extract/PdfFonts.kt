package com.hotatticgames.llmtrainer.extract

import java.util.IdentityHashMap

internal object Encodings {
    private val ASCII_NAMES = ("space exclam quotedbl numbersign dollar percent ampersand quotesingle parenleft parenright asterisk plus comma " +
        "hyphen period slash zero one two three four five six seven eight nine colon semicolon less equal greater question at " +
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z bracketleft backslash bracketright asciicircum underscore grave " +
        "a b c d e f g h i j k l m n o p q r s t u v w x y z braceleft bar braceright asciitilde").split(" ")
    private val LATIN1_NAMES = ("space exclamdown cent sterling currency yen brokenbar section dieresis copyright ordfeminine guillemotleft " +
        "logicalnot hyphen registered macron degree plusminus twosuperior threesuperior acute mu paragraph periodcentered cedilla " +
        "onesuperior ordmasculine guillemotright onequarter onehalf threequarters questiondown Agrave Aacute Acircumflex Atilde " +
        "Adieresis Aring AE Ccedilla Egrave Eacute Ecircumflex Edieresis Igrave Iacute Icircumflex Idieresis Eth Ntilde Ograve " +
        "Oacute Ocircumflex Otilde Odieresis multiply Oslash Ugrave Uacute Ucircumflex Udieresis Yacute Thorn germandbls agrave " +
        "aacute acircumflex atilde adieresis aring ae ccedilla egrave eacute ecircumflex edieresis igrave iacute icircumflex " +
        "idieresis eth ntilde ograve oacute ocircumflex otilde odieresis divide oslash ugrave uacute ucircumflex udieresis yacute " +
        "thorn ydieresis").split(" ")

    private const val WIN_80_9F = "€•‚ƒ„…†‡ˆ‰Š‹Œ•Ž•" +
        "•‘’“”•–—˜™š›œ•žŸ"
    private const val MAC_80_FF = "ÄÅÇÉÑÖÜáàâäãåçéè" +
        "êëíìîïñóòôöõúùûü" +
        "†°¢£§•¶ß®©™´¨≠ÆØ" +
        "∞±≤≥¥µ∂∑∏π∫ªºΩæø" +
        "¿¡¬√ƒ≈∆«»… ÀÃÕŒœ" +
        "–—“”‘’÷◊ÿŸ⁄€‹›ﬁﬂ" +
        "‡·‚„‰ÂÊÁËÈÍÎÏÌÓÔ" +
        " ÒÚÛÙıˆ˜¯˘˙˚¸˝˛ˇ"

    /** 256-entry code -> char tables. */
    val winAnsi: Array<String?> = Array(256) { c ->
        when {
            c < 32 -> null
            c < 127 -> c.toChar().toString()
            c == 127 -> null
            c in 0x80..0x9F -> WIN_80_9F[c - 0x80].toString()
            c == 0xA0 -> " "
            c == 0xAD -> "-"
            else -> c.toChar().toString()
        }
    }
    val macRoman: Array<String?> = Array(256) { c -> if (c in 32..126) c.toChar().toString() else if (c >= 128) MAC_80_FF[c - 128].toString() else null }
    /** StandardEncoding approximated as ASCII + WinAnsi upper half (documented limit). */
    val standard: Array<String?> = winAnsi

    private val extras: Map<String, String> = mapOf(
        "endash" to "–", "emdash" to "—", "quoteleft" to "‘", "quoteright" to "’", "quotedblleft" to "“",
        "quotedblright" to "”", "quotesinglbase" to "‚", "quotedblbase" to "„", "bullet" to "•",
        "ellipsis" to "…", "dagger" to "†", "daggerdbl" to "‡", "fi" to "fi", "fl" to "fl", "ff" to "ff",
        "ffi" to "ffi", "ffl" to "ffl", "Euro" to "€", "trademark" to "™", "OE" to "Œ", "oe" to "œ",
        "Scaron" to "Š", "scaron" to "š", "Zcaron" to "Ž", "zcaron" to "ž", "Ydieresis" to "Ÿ",
        "florin" to "ƒ", "circumflex" to "ˆ", "tilde" to "˜", "perthousand" to "‰",
        "guilsinglleft" to "‹", "guilsinglright" to "›", "fraction" to "⁄", "minus" to "−",
        "lslash" to "ł", "Lslash" to "Ł", "dotlessi" to "ı", "nbspace" to " ", "nonbreakingspace" to " ",
        "sfthyphen" to "-", "softhyphen" to "-", "hyphenminus" to "-", "middot" to "·", "mu1" to "µ",
        "Omega" to "Ω", "Delta" to "∆", "pi" to "π", "alpha" to "α", "beta" to "β", "gamma" to "γ",
        "delta" to "δ", "lambda" to "λ", "omega" to "ω", "sigma" to "σ", "theta" to "θ", "Sigma" to "Σ",
        "summation" to "∑", "infinity" to "∞", "notequal" to "≠", "lessequal" to "≤", "greaterequal" to "≥",
        "arrowright" to "→", "arrowleft" to "←", "checkmark" to "✓", "dotaccent" to "˙", "ring" to "˚",
        "hungarumlaut" to "˝", "ogonek" to "˛", "caron" to "ˇ", "breve" to "˘", "Lcaron" to "Ľ",
        "Cacute" to "Ć", "cacute" to "ć", "Ccaron" to "Č", "ccaron" to "č", "Dcroat" to "Đ", "dcroat" to "đ",
        "Eogonek" to "Ę", "eogonek" to "ę", "Nacute" to "Ń", "nacute" to "ń", "Sacute" to "Ś", "sacute" to "ś",
        "Zacute" to "Ź", "zacute" to "ź", "Zdotaccent" to "Ż", "zdotaccent" to "ż", "Aogonek" to "Ą", "aogonek" to "ą",
    )
    private val glyphMap: Map<String, String> = HashMap<String, String>().also { m ->
        ASCII_NAMES.forEachIndexed { i, n -> m[n] = (32 + i).toChar().toString() }
        LATIN1_NAMES.forEachIndexed { i, n -> m.putIfAbsent(n, (0xA0 + i).toChar().toString()) }
        m.putAll(extras)
    }

    /** Adobe-glyph-list subset: standard Latin names, uniXXXX/uXXXX, f_i style composites; null when unknown (gNN, cidNN, ...). */
    fun glyphToUnicode(raw: String): String? {
        val n = raw.substringBefore('.')
        if (n.isEmpty()) return null
        glyphMap[n]?.let { return it }
        if (n.startsWith("uni") && n.length >= 7 && (n.length - 3) % 4 == 0) {
            val sb = StringBuilder()
            var i = 3
            while (i + 4 <= n.length) { sb.append((n.substring(i, i + 4).toIntOrNull(16) ?: return null).toChar()); i += 4 }
            return sb.toString()
        }
        if (n.startsWith("u") && n.length in 5..7) n.substring(1).toIntOrNull(16)?.let { if (it in 0..0x10FFFF) return String(Character.toChars(it)) }
        if ('_' in n) { val parts = n.split('_').map { glyphMap[it] ?: return null }; return parts.joinToString("") }
        return null
    }
}

/** Parsed ToUnicode (or embedded) CMap: code -> text, plus the code lengths declared in codespace ranges. */
internal class CMap {
    val map = HashMap<Int, String>()
    val codeLengths = java.util.TreeSet<Int>()
    val ranges = ArrayList<IntArray>() // len, lo, hi

    companion object {
        private const val MAX_ENTRIES = 300_000
        private fun code(b: ByteArray): Int { var v = 0; for (x in b.take(4)) v = (v shl 8) or (x.toInt() and 0xFF); return v }
        fun utf16(b: ByteArray): String {
            if (b.size == 1) return (b[0].toInt() and 0xFF).toChar().toString()
            return try { String(b, Charsets.UTF_16BE) } catch (e: Exception) { "" }
        }

        fun parse(data: ByteArray): CMap {
            val cm = CMap()
            val lx = PdfLexer(data, 0, data.size, refs = false)
            val operands = ArrayList<PObj>()
            var steps = 0
            while (steps++ < 2_000_000) {
                val o = try { lx.readObject() } catch (e: PdfException) { break } ?: break
                if (o !is PKeyword) { operands.add(o); if (operands.size > 4) operands.removeAt(0); continue }
                when (o.v) {
                    "begincodespacerange" -> while (true) {
                        val a = lx.readObject() ?: break
                        if (a is PKeyword) break
                        val b = lx.readObject() ?: break
                        if (a is PStr && b is PStr && a.bytes.isNotEmpty() && a.bytes.size <= 4) {
                            cm.codeLengths.add(a.bytes.size); cm.ranges.add(intArrayOf(a.bytes.size, code(a.bytes), code(b.bytes)))
                        }
                    }
                    "beginbfchar" -> while (true) {
                        val a = lx.readObject() ?: break
                        if (a is PKeyword) break
                        val b = lx.readObject() ?: break
                        if (a is PStr && b is PStr && cm.map.size < MAX_ENTRIES) cm.map[code(a.bytes)] = utf16(b.bytes)
                    }
                    "beginbfrange" -> while (true) {
                        val a = lx.readObject() ?: break
                        if (a is PKeyword) break
                        val b = lx.readObject() ?: break
                        val c = lx.readObject() ?: break
                        if (a !is PStr || b !is PStr) continue
                        val lo = code(a.bytes)
                        val hi = minOf(code(b.bytes), lo + 65535)
                        if (c is PStr) {
                            val base = utf16(c.bytes)
                            for (k in lo..hi) {
                                if (cm.map.size >= MAX_ENTRIES) break
                                cm.map[k] = if (base.isEmpty()) "" else base.dropLast(1) + (base.last() + (k - lo))
                            }
                        } else if (c is PArr) {
                            for ((i, e) in c.items.withIndex()) if (e is PStr && lo + i <= hi && cm.map.size < MAX_ENTRIES) cm.map[lo + i] = utf16(e.bytes)
                        }
                    }
                }
                operands.clear()
            }
            return cm
        }
    }
}

internal class DecodeStats { var total = 0; var unmapped = 0 }

/** Turns the bytes of a show-text string into Unicode for one font resource. */
internal class FontDecoder(
    private val toUni: CMap?,
    private val encoding: Array<String?>?,
    private val twoByte: Boolean,
    private val utf16Direct: Boolean,
) {
    fun decode(b: ByteArray, stats: DecodeStats): String {
        val sb = StringBuilder()
        var i = 0
        if (!twoByte) {
            while (i < b.size) {
                val c = b[i++].toInt() and 0xFF
                stats.total++
                val s = toUni?.map?.get(c) ?: encoding?.get(c)
                if (s == null) { if (c > 32) stats.unmapped++ else if (c == 32) sb.append(' ') } else sb.append(s)
            }
        } else {
            val cm = toUni
            while (i < b.size) {
                var len = 2
                var code: Int
                if (cm != null && cm.codeLengths.isNotEmpty() && !(cm.codeLengths.size == 1 && cm.codeLengths.first() == 2)) {
                    len = matchLen(cm, b, i)
                }
                if (i + len > b.size) len = b.size - i
                code = 0
                for (k in 0 until len) code = (code shl 8) or (b[i + k].toInt() and 0xFF)
                i += len
                stats.total++
                val s = cm?.map?.get(code) ?: if (utf16Direct && len == 2) code.toChar().toString() else null
                if (s == null) stats.unmapped++ else sb.append(s)
            }
        }
        return sb.toString()
    }

    private fun matchLen(cm: CMap, b: ByteArray, i: Int): Int {
        for (len in cm.codeLengths) {
            if (i + len > b.size) continue
            var code = 0
            for (k in 0 until len) code = (code shl 8) or (b[i + k].toInt() and 0xFF)
            if (cm.ranges.any { it[0] == len && code >= it[1] && code <= it[2] }) return len
        }
        return cm.codeLengths.first()
    }
}

internal class FontFactory(private val doc: PdfDocument) {
    private val cache = IdentityHashMap<PDict, FontDecoder>()
    val default = FontDecoder(null, Encodings.winAnsi, false, false)

    fun forFont(d: PDict): FontDecoder = cache.getOrPut(d) { try { build(d) } catch (e: PdfException) { default } }

    private fun readCMap(o: PObj?): CMap? {
        val st = doc.resolve(o) as? PStream ?: return null
        val data = doc.decode(st) ?: return null
        return CMap.parse(data)
    }

    private fun build(d: PDict): FontDecoder {
        val subtype = doc.name(d.map["Subtype"])
        val toUni = readCMap(d.map["ToUnicode"])
        if (subtype == "Type0") {
            val enc = doc.resolve(d.map["Encoding"])
            val encName = (enc as? PName)?.v ?: ""
            val utf16 = Regex("""Uni.*-(UCS2|UTF16)-[HV]""").matches(encName)
            return FontDecoder(toUni, null, true, utf16)
        }
        val encObj = doc.resolve(d.map["Encoding"])
        val table: Array<String?> = Array(256) { null }
        var baseName: String? = null
        var diffs: List<PObj>? = null
        when (encObj) {
            is PName -> baseName = encObj.v
            is PDict -> { baseName = doc.name(encObj.map["BaseEncoding"]); diffs = doc.arr(encObj.map["Differences"]) }
            else -> {}
        }
        val base = when (baseName) {
            "WinAnsiEncoding" -> Encodings.winAnsi
            "MacRomanEncoding" -> Encodings.macRoman
            else -> Encodings.standard
        }
        for (i in 0 until 256) table[i] = base[i]
        if (diffs != null) {
            var code = 0
            for (item in diffs.take(2000)) {
                val r = doc.resolve(item)
                if (r is PNum) code = r.int()
                else if (r is PName) { if (code in 0..255) table[code] = Encodings.glyphToUnicode(r.v); code++ }
            }
        }
        return FontDecoder(toUni, table, false, false)
    }
}
