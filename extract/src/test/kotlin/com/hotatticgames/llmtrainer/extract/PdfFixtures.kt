package com.hotatticgames.llmtrainer.extract

import java.io.ByteArrayOutputStream
import java.util.zip.Deflater

internal fun latin1(s: String) = s.toByteArray(Charsets.ISO_8859_1)

internal fun deflate(data: ByteArray): ByteArray {
    val d = Deflater()
    d.setInput(data); d.finish()
    val out = ByteArrayOutputStream()
    val buf = ByteArray(4096)
    while (!d.finished()) out.write(buf, 0, d.deflate(buf))
    d.end()
    return out.toByteArray()
}

/** Hand-rolled PDF writer for fixtures: numbered objects, classic xref table (or none), optional trailer extras. */
internal class PdfBuilder {
    val objs = LinkedHashMap<Int, ByteArray>()

    fun obj(num: Int, body: String): PdfBuilder { objs[num] = latin1(body); return this }

    fun stream(num: Int, dict: String, data: ByteArray): PdfBuilder {
        val bos = ByteArrayOutputStream()
        bos.write(latin1("<< $dict /Length ${data.size} >>\nstream\n")); bos.write(data); bos.write(latin1("\nendstream"))
        objs[num] = bos.toByteArray()
        return this
    }

    fun flateStream(num: Int, dict: String, data: ByteArray) = stream(num, "$dict /Filter /FlateDecode", deflate(data))

    /** Serialises objects; returns bytes. With [withXref] false the file has no xref/trailer at all (forces the rebuild path). */
    fun build(root: Int = 1, trailerExtra: String = "", withXref: Boolean = true, header: String = "%PDF-1.4\n"): ByteArray {
        val bos = ByteArrayOutputStream()
        bos.write(latin1(header))
        val offsets = HashMap<Int, Int>()
        for ((n, body) in objs) {
            offsets[n] = bos.size()
            bos.write(latin1("$n 0 obj\n")); bos.write(body); bos.write(latin1("\nendobj\n"))
        }
        if (!withXref) return bos.toByteArray()
        val xrefPos = bos.size()
        val max = objs.keys.max()
        val sb = StringBuilder("xref\n0 ${max + 1}\n0000000000 65535 f \n")
        for (i in 1..max) sb.append(offsets[i]?.let { "%010d 00000 n \n".format(it) } ?: "0000000000 65535 f \n")
        sb.append("trailer\n<< /Size ${max + 1} /Root $root 0 R $trailerExtra >>\nstartxref\n$xrefPos\n%%EOF\n")
        bos.write(latin1(sb.toString()))
        return bos.toByteArray()
    }
}

internal object Pdfs {
    const val HELV = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"

    /**
     * One page per content string. Object layout: 1 catalog, 2 pages, 3 font F1, 10+i page, 100+i content stream.
     * [fontObjs] adds more fonts (name -> object body) as F2, F3... objects 4, 5...
     */
    fun pages(contents: List<String>, compress: Boolean = false, extraFonts: List<String> = emptyList(), b: PdfBuilder = PdfBuilder()): PdfBuilder {
        b.obj(1, "<< /Type /Catalog /Pages 2 0 R >>")
        val kids = contents.indices.joinToString(" ") { "${10 + it} 0 R" }
        b.obj(2, "<< /Type /Pages /Kids [$kids] /Count ${contents.size} >>")
        b.obj(3, HELV)
        extraFonts.forEachIndexed { i, f -> b.obj(4 + i, f) }
        val fonts = "/F1 3 0 R" + extraFonts.indices.joinToString("") { " /F${i2(it)} ${4 + it} 0 R" }
        contents.forEachIndexed { i, c ->
            b.obj(10 + i, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << $fonts >> >> /Contents ${100 + i} 0 R >>")
            if (compress) b.flateStream(100 + i, "", latin1(c)) else b.stream(100 + i, "", latin1(c))
        }
        return b
    }
    private fun i2(i: Int) = i + 2

    fun text(vararg lines: String) = pages(listOf("BT /F1 12 Tf 72 720 Td " + lines.joinToString(" 0 -14 Td ") { "($it) Tj" } + " ET")).build()
}
