package com.hotatticgames.llmtrainer.studio.core

import java.util.Random

/**
 * Deterministic, transparent metric primitives for local (on-phone) evaluation. Port of `factory/llmtrainer/evalsuite/metrics.py`.
 *
 * LIMITS (read before trusting a number): everything is lexical/numeric matching; it does not understand meaning.
 * Only the fixed unit table is scored. The unsupported-claim heuristic calls a sentence supported when >= 60% of its content
 * words occur in the provided source text and every number in it occurs there too: it MISSES wrong-but-lexically-similar claims
 * and FLAGS correct paraphrases. Terminology coverage rewards using the domain vocabulary, not using it correctly.
 */
object EvalMetrics {
    val STOPWORDS: Set<String> = ("the a an of to in and or is are was were be been it its for on with as by at from that this these those which not can may " +
        "should must will when where how what why do does did if then than also each all any more most other use used using before " +
        "after during between into over under about one two three first second would could has have had their there they them you " +
        "your we our i he she his her but so such only same both either neither per via being").split(" ").toSet()

    private class U(val dim: String, val factor: Double)
    private val UNITS = LinkedHashMap<String, U>()
    private fun reg(dim: String, factor: Double, vararg aliases: String) { for (a in aliases) UNITS[a] = U(dim, factor) }

    init {
        reg("length", 1.0, "mm", "millimeter", "millimeters", "millimetre", "millimetres")
        reg("length", 10.0, "cm", "centimeter", "centimeters")
        reg("length", 1000.0, "m", "meter", "meters", "metre", "metres")
        reg("length", 25.4, "in", "inch", "inches")
        reg("torque", 1.0, "n·m", "n-m", "nm", "n.m", "n m", "newton-meter", "newton-meters")
        reg("torque", 1.3558179, "lb-ft", "lb·ft", "ft-lb", "ft·lb", "ft-lbs", "lb-ft.", "lbf-ft", "ft-lbf")
        reg("torque", 0.1129848, "in-lb", "in·lb", "lb-in", "in-lbs", "lbf-in")
        reg("pressure", 1.0, "kpa"); reg("pressure", 100.0, "bar"); reg("pressure", 6.894757, "psi"); reg("pressure", 1000.0, "mpa")
        reg("voltage", 1.0, "v", "volt", "volts")
        reg("current", 1.0, "a", "amp", "amps", "ampere", "amperes"); reg("current", 0.001, "ma", "milliamp", "milliamps")
        reg("resistance", 1.0, "ohm", "ohms", "ω"); reg("resistance", 1000.0, "kohm", "kω")
        reg("frequency", 1.0, "hz"); reg("rpm", 1.0, "rpm")
        reg("mass", 1.0, "g", "gram", "grams"); reg("mass", 1000.0, "kg", "kilogram", "kilograms"); reg("mass", 453.59237, "lb", "lbs", "pound", "pounds")
        reg("volume", 1.0, "ml", "milliliter", "milliliters"); reg("volume", 1000.0, "l", "liter", "liters", "litre", "litres"); reg("volume", 946.353, "qt", "quart", "quarts")
        reg("time", 1.0, "s", "sec", "second", "seconds"); reg("time", 60.0, "min", "mins", "minute", "minutes")
        reg("time", 3600.0, "h", "hr", "hrs", "hour", "hours"); reg("time", 86400.0, "day", "days")
        reg("percent", 1.0, "%", "percent")
        reg("temp_c", 1.0, "°c", "degc", "celsius"); reg("temp_f", 1.0, "°f", "degf", "fahrenheit")
    }

    private val UNIT_RE = UNITS.keys.sortedByDescending { it.length }.joinToString("|") { Regex.escape(it) }
    private val QTY = Regex("(?<![\\w.])([-−]?\\d+(?:[.,]\\d+)?|\\.\\d+)\\s*($UNIT_RE)(?![a-zA-Z])", RegexOption.IGNORE_CASE)
    private val NUM = Regex("[-−]?\\d+(?:\\.\\d+)?")

    class Qty(val value: Double, val unit: String, val dim: String, val base: Double) {
        fun matches(o: Qty): Boolean {
            if (dim != o.dim) return false
            if (unit == o.unit) return close(value, o.value, 1e-6, 1e-9)
            return close(base, o.base, 0.02, 1e-9)           // manuals round conversions
        }
        private fun close(a: Double, b: Double, rel: Double, abs: Double) = Math.abs(a - b) <= maxOf(rel * maxOf(Math.abs(a), Math.abs(b)), abs)
    }

    /** Value + unit alias -> quantity, or null when the unit is not in the fixed table (never guessed). */
    fun qty(value: Double, unit: String): Qty? {
        val u = UNITS[unit.lowercase()] ?: return null
        return if (u.dim == "temp_f") Qty(value, unit.lowercase(), "temp_c", (value - 32) * 5 / 9) else Qty(value, unit.lowercase(), u.dim, value * u.factor)
    }

    fun extractQuantities(text: String): List<Qty> = QTY.findAll(text).mapNotNull { m ->
        val v = m.groupValues[1].replace('−', '-').replace(',', '.').toDoubleOrNull() ?: return@mapNotNull null
        qty(v, m.groupValues[2])
    }.toList()

    /** Right iff the expected quantity appears and no *different* quantity of the same dimension is also asserted. */
    fun factCorrect(answer: String, expected: Qty): Boolean {
        val same = extractQuantities(answer).filter { it.dim == expected.dim }
        if (same.none { it.matches(expected) }) return false
        return same.all { it.matches(expected) }
    }

    fun contentWords(text: String): List<String> = Regex("[a-z][a-z0-9]+").findAll(text.lowercase()).map { it.value }.filter { it !in STOPWORDS && it.length >= 3 }.toList()

    private fun numbers(text: String): Set<String> = NUM.findAll(text).map { m ->
        val n = m.value.replace('−', '-')
        if ('.' in n) n.trimEnd('0').trimEnd('.') else n
    }.toSet()

    private val HEDGES = Regex("\\b(i do not know|i don't know|not specified|no information|cannot determine|unable to find|not mentioned)\\b", RegexOption.IGNORE_CASE)
    private val CITE = Regex("\\[[^\\[\\]]*\\]")

    fun splitClaims(answer: String): List<String> {
        val out = ArrayList<String>()
        for (s0 in answer.trim().split(Regex("(?<=[.!?])\\s+|\\n+"))) {
            val s = CITE.replace(s0, "").trim()
            if (s.isEmpty() || s.endsWith("?") || HEDGES.containsMatchIn(s) || contentWords(s).size < 3) continue
            out.add(s)
        }
        return out
    }

    /** (unsupported, total) claims in [answer] relative to [support]. */
    fun unsupportedClaims(answer: String, support: String, minSupport: Double = 0.6): Pair<Int, Int> {
        val words = contentWords(support).toSet()
        val nums = numbers(support)
        val claims = splitClaims(answer)
        var bad = 0
        for (c in claims) {
            val w = contentWords(c)
            val ratio = w.count { it in words }.toDouble() / w.size
            if (ratio < minSupport || !nums.containsAll(numbers(c))) bad++
        }
        return bad to claims.size
    }

    fun terminologyUse(answer: String, terms: List<String>): Pair<Int, Int> {
        val words = Regex("[a-z][a-z0-9]+").findAll(answer.lowercase()).map { it.value }.toSet()
        return terms.count { it in words } to terms.size
    }

    // ---- fixed general-capability retention probes (tiny on purpose: a canary, not a benchmark) ----------------------
    val PROBES: List<Pair<String, String>> = listOf(
        "What is 12 plus 15? Answer with just the number." to "\\b27\\b",
        "What is 9 times 8? Answer with just the number." to "\\b72\\b",
        "What is the capital of France? One word." to "\\bparis\\b",
        "What is the chemical symbol for water? Answer with the formula." to "\\bh2o\\b|h₂o",
        "Which planet is known as the Red Planet? One word." to "\\bmars\\b",
        "What is the opposite of 'hot'? One word." to "\\bcold\\b",
        "Spell the plural of the word 'mouse'." to "\\bmice\\b",
        "How many days are in a week? Answer with just the number." to "\\b7\\b|\\bseven\\b",
        "Complete the sentence: The sun rises in the ..." to "\\beast\\b",
        "What is 100 divided by 4? Answer with just the number." to "\\b25\\b",
        "Is the number 17 prime? Answer yes or no." to "\\byes\\b",
        "Write the word 'banana' in all capital letters." to "\\bBANANA\\b",
    )

    fun probePass(answer: String, pattern: String): Boolean =
        (if (pattern == "\\bBANANA\\b") Regex(pattern) else Regex(pattern, RegexOption.IGNORE_CASE)).containsMatchIn(answer)

    // ---- uncertainty: deterministic bootstrap over items of ratio metrics -----------------------------------------------
    /** Per item (numerator, denominator); the metric is sum(num)/sum(den). */
    class Pairs {
        val num = ArrayList<Double>(); val den = ArrayList<Double>()
        fun add(n: Double, d: Double) { num.add(n); den.add(d) }
        val size get() = num.size
        /** Items that contributed to the denominator. */
        val n: Int get() = den.count { it > 0 }
        fun ratio(idx: IntArray? = null): Double {
            var a = 0.0; var b = 0.0
            if (idx == null) { for (i in num.indices) { a += num[i]; b += den[i] } } else for (i in idx) { a += num[i]; b += den[i] }
            return if (b > 0) a / b else 0.0
        }
    }

    /** 95% bootstrap CI of ratio(b) - ratio(a) with items resampled jointly (same items for both subjects). */
    fun pairedDeltaCi(a: Pairs, b: Pairs, seed: Long, nBoot: Int): Pair<Double, Double> {
        if (a.size == 0 || a.size != b.size) return 0.0 to 0.0
        val rng = Random(seed)
        val k = a.size
        val ds = DoubleArray(nBoot)
        val idx = IntArray(k)
        for (t in 0 until nBoot) {
            for (i in 0 until k) idx[i] = rng.nextInt(k)
            ds[t] = b.ratio(idx) - a.ratio(idx)
        }
        ds.sort()
        return ds[(0.025 * nBoot).toInt()] to ds[(0.975 * nBoot).toInt() - 1]
    }
}

/** Tiny deterministic lexical retriever over {ref, text} chunks (idf-weighted token overlap). */
class LexicalRetriever(private val chunks: List<Pair<String, String>>) {
    private fun tok(t: String) = Regex("[a-z0-9][a-z0-9-]+").findAll(t.lowercase()).map { it.value }.filter { it !in EvalMetrics.STOPWORDS && it.length >= 3 }.toSet()
    private val sets = chunks.map { tok(it.second) }
    private val idf: Map<String, Double> = HashMap<String, Int>().also { df -> sets.forEach { s -> s.forEach { df.merge(it, 1, Int::plus) } } }
        .mapValues { Math.log(1.0 + chunks.size.toDouble() / it.value) }

    /** Best [k] chunk indices for [query], best first; ties by ref; chunks sharing no token are never returned. */
    fun top(query: String, k: Int): List<Pair<String, String>> {
        val q = tok(query)
        return chunks.indices.map { i -> i to q.sumOf { if (it in sets[i]) idf[it] ?: 0.0 else 0.0 } }
            .filter { it.second > 0 }.sortedWith(compareBy({ -it.second }, { chunks[it.first].first })).take(k).map { chunks[it.first] }
    }
}
