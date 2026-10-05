package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.ChunkOrigin
import com.hotatticgames.llmtrainer.studio.api.ChunkRole
import com.hotatticgames.llmtrainer.studio.api.DatasetOptions
import com.hotatticgames.llmtrainer.studio.api.DatasetStatus
import com.hotatticgames.llmtrainer.studio.api.ReviewFlag
import com.hotatticgames.llmtrainer.studio.api.RightsStatus
import java.io.File

/** One chunk as the dataset sees it (the unit of REVIEW). `split` is null when the chunk feeds no training/eval example. */
data class DChunk(
    val sourceId: String, val sourceName: String, val chunk: Chunk, val group: String?, val split: String?,
    val flags: List<ReviewFlag>, val defaultIncluded: Boolean, val reason: String?, val trainable: Boolean,
) {
    val ref: String get() = "$sourceId/${chunk.id}"
    val role: ChunkRole get() = when (split) { "train" -> ChunkRole.TRAIN; "validation" -> ChunkRole.VALIDATION; "test" -> ChunkRole.HELD_OUT_EVAL; else -> ChunkRole.REFERENCE }
}

/** A template-built question/answer pair from a chunk. Honestly SYNTHETIC: the prompt wording is not source text. */
data class Synth(
    val id: String, val chunkRef: String, val sourceId: String, val sourceName: String, val section: String, val page: Int?,
    val prompt: String, val response: String,
)

data class Example(
    val id: String, val groupId: String, val prompt: String, val response: String, val origin: String,
    val sourceId: String, val chunkId: String, val task: String, val quality: Double, val split: String,
)

data class Decision(val included: Boolean, val textSha: String)

class DatasetMeta(
    val version: Int, val options: DatasetOptions, val builtAt: Long, @Volatile var approvedAt: Long?, @Volatile var status: DatasetStatus,
    val groupBy: String, val assignment: Map<String, String>, val notes: List<String>, val splitsAvailable: Boolean,
    val sourceSnapshot: List<Map<String, Any?>>,
)

class DatasetState(val meta: DatasetMeta, val chunks: List<DChunk>, val synth: List<Synth>, val decisions: MutableMap<String, Decision>) {
    private val byRef = chunks.associateBy { it.ref }
    fun chunk(ref: String) = byRef[ref]
    fun isIncluded(c: DChunk): Boolean = decisions[c.ref]?.takeIf { it.textSha == c.chunk.sha256 }?.included ?: c.defaultIncluded
    fun isIncluded(s: Synth): Boolean = decisions[s.id]?.included ?: false
    val includedChunks: List<DChunk> get() = chunks.filter { isIncluded(it) }
}

object ExampleGen {
    const val BUILDER_NAME = "studio-template-builder"
    const val BUILDER_VERSION = "1"
    const val MIN_QUALITY = 0.4
    const val CONTAINMENT_THRESHOLD = 0.8
    private val SENT = Regex("(?<=[.!?])\\s+")
    private val PREFIX = Regex("^Complete the statement from section \".*?\": ")

    class Raw(val idx: Int, val prompt: String, val response: String, val quality: Double)

    fun sentences(text: String): List<String> = text.replace(Regex("\\s+"), " ").trim().split(SENT).filter { it.isNotEmpty() }

    /** Deterministic cloze continuation: the example text IS source text split in half. Pipeline fixture quality, not a quality generator. */
    fun cloze(section: String, text: String): List<Raw> {
        val out = ArrayList<Raw>()
        sentences(text).forEachIndexed { idx, sent ->
            val w = sent.split(Regex("\\s+")).filter { it.isNotEmpty() }
            if (w.size < 8) return@forEachIndexed
            val cut = w.size / 2
            out.add(Raw(idx, "Complete the statement from section \"$section\": " + w.subList(0, cut).joinToString(" "), w.subList(cut, w.size).joinToString(" "), minOf(1.0, w.size / 20.0)))
        }
        return out
    }

    fun leakText(prompt: String, response: String): String = PREFIX.replaceFirst(prompt, "") + " " + response

    fun exampleId(sid: String, cid: String, task: String, idx: Int) = "ex-" + Hashing.sha256("$sid|$cid|$task|$idx").take(12)

    /** Template Q/A: asks about the chunk's most frequent salient term and answers with the first sentence containing it. */
    fun qa(sourceId: String, chunkId: String, section: String, text: String): Pair<String, String>? {
        val sents = sentences(text)
        val freq = HashMap<String, Int>()
        for (w in Regex("[a-z][a-z0-9]{4,}").findAll(text.lowercase())) if (w.value !in Terms_STOP) freq.merge(w.value, 1, Int::plus)
        val term = freq.entries.sortedWith(compareBy({ -it.value }, { it.key })).firstOrNull()?.key ?: return null
        val ans = sents.firstOrNull { it.lowercase().contains(term) && it.split(" ").size >= 6 } ?: return null
        return "According to the section \"$section\", what is stated about $term?" to ans
    }
    private val Terms_STOP = setOf("which", "their", "there", "these", "those", "where", "while", "would", "should", "could", "about", "other", "after", "before")
}

object DatasetEngine {
    const val NEAR_DUP_THRESHOLD = 0.8
    const val SHINGLE = 5

    fun isTrainableRights(r: RightsStatus) = r == RightsStatus.OWNER_AUTHORED || r == RightsStatus.LICENSED_FOR_TRAINING || r == RightsStatus.PERMISSION_GRANTED

    private fun garbledRatio(t: String): Double {
        val toks = t.split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (toks.size < 8) return 0.0
        val bad = toks.count { tk -> (tk.length == 1 && tk[0].isLetter() && tk !in setOf("a", "A", "I")) || Regex("[A-Za-z]+[0-9]+[A-Za-z]+").containsMatchIn(tk) }
        return bad.toDouble() / toks.size
    }

    class BuildResult(val chunks: List<DChunk>, val synth: List<Synth>, val groupBy: String, val assignment: Map<String, String>, val notes: List<String>, val splitsAvailable: Boolean)

    fun ratios(o: DatasetOptions): LinkedHashMap<String, Double> {
        val train = 1.0 - o.heldOutFraction - o.validationFraction
        return linkedMapOf("train" to train, "validation" to o.validationFraction, "test" to o.heldOutFraction)
    }

    fun validateOptions(o: DatasetOptions): String? = when {
        o.heldOutFraction <= 0.0 || o.validationFraction <= 0.0 -> "Held-out and validation fractions must both be greater than zero"
        o.heldOutFraction + o.validationFraction >= 0.9 -> "Held-out plus validation fractions must leave at least 10% for training"
        o.chunkTargetTokens < 50 || o.chunkTargetTokens > 4000 -> "Chunk size must be between 50 and 4000 tokens"
        else -> null
    }

    fun build(sources: List<Pair<SourceDoc, List<CBlock>>>, o: DatasetOptions, onProgress: (Long, Long) -> Unit = { _, _ -> }): BuildResult {
        val maxChars = o.chunkTargetTokens * Chunker.CHARS_PER_TOKEN
        class Tmp(val src: SourceDoc, val chunk: Chunk, var flags: MutableList<ReviewFlag>, var included: Boolean, var reason: String?, val trainable: Boolean, var split: String? = null, var group: String? = null)
        val all = ArrayList<Tmp>()
        sources.sortedBy { it.first.sourceId }.forEachIndexed { i, (src, blocks) ->
            onProgress(i.toLong(), sources.size.toLong())
            val (chunks, _) = Chunker.chunk(blocks, maxChars)
            for (c in chunks) {
                val flags = ArrayList<ReviewFlag>()
                if (c.kind == "table") flags.add(ReviewFlag.TABLE)
                var included = true
                var reason: String? = null
                var trainable = src.trainable && c.role == ChunkRoles.TRAIN
                if (!src.trainable) { included = !o.excludeReferenceOnlySources; reason = "source is reference-only (never used for training)" }
                else if (c.role == ChunkRoles.EXCLUDED) {
                    included = false; reason = c.excludeReason
                    flags.add(if (c.excludeReason == "duplicate_chunk") ReviewFlag.DUPLICATE else ReviewFlag.LOW_CONFIDENCE)
                }
                if (c.role == ChunkRoles.TRAIN && garbledRatio(c.text) > 0.15) flags.add(ReviewFlag.OCR_NOISE)
                all.add(Tmp(src, c, flags, included, reason, trainable))
            }
        }
        // header/footer residue: short normalised lines repeated across the corpus
        val norm = all.associateWith { Cleaner.noiseKey(it.chunk.text) }
        val freq = HashMap<String, Int>()
        for ((t, k) in norm) if (t.chunk.text.length < 120) freq.merge(k, 1, Int::plus)
        for ((t, k) in norm) if (t.chunk.text.length < 120 && (freq[k] ?: 0) >= 3) { t.flags.add(ReviewFlag.HEADER_FOOTER_NOISE); if (t.chunk.role == ChunkRoles.TRAIN) { t.included = false; t.reason = t.reason ?: "repeated short text (header/footer residue)" } }

        // candidates: trainable prose that still forms examples
        val notes = ArrayList<String>()
        val cand = all.filter { it.trainable && it.included }
        val exCount = HashMap<Tmp, Int>()
        for (t in cand) exCount[t] = ExampleGen.cloze(t.chunk.section, t.chunk.text).count { it.quality >= ExampleGen.MIN_QUALITY }
        val withEx = cand.filter { (exCount[it] ?: 0) > 0 }
        val nDocs = withEx.map { it.src.sourceId }.toSet().size
        var groupBy = "document"
        if (nDocs < 3) { groupBy = "section"; notes.add("only $nDocs source document(s) with usable text; grouping by section so train/validation/test never share a section") }
        fun groupOf(t: Tmp) = if (groupBy == "document") t.src.sourceId else "${t.src.sourceId}#${Text.slugify(t.chunk.section)}"
        val weights = LinkedHashMap<String, Int>()
        for (t in withEx) weights.merge(groupOf(t), exCount.getValue(t), Int::plus)
        var assignment: Map<String, String> = emptyMap()
        var splitsAvailable = true
        if (weights.size < 3) {
            splitsAvailable = false
            notes.add("fewer than 3 independent groups (${weights.size}); a train/validation/held-out split is impossible, so NO training or held-out export is available. The reference package still works. Add more documents or more distinct sections.")
        } else {
            assignment = Similarity.assignGroups(weights, ratios(o), o.seed)
            for (t in withEx) { t.group = groupOf(t); t.split = assignment.getValue(t.group!!) }
        }
        for (t in cand) if (t.split == null) {
            // no examples (too short) or no split possible: keep it for retrieval only, never silently drop
            t.reason = t.reason ?: if ((exCount[t] ?: 0) == 0) "too short to form training examples (retrieval only)" else "no split available (retrieval only)"
        }
        // near-duplicate resolution among split chunks, protecting held-out > validation > train
        val inSplit = cand.filter { it.split != null }
        if (inSplit.isNotEmpty()) {
            val texts = inSplit.associate { "${it.src.sourceId}/${it.chunk.id}" to it.chunk.text }
            val splitOf = inSplit.associate { "${it.src.sourceId}/${it.chunk.id}" to it.split!! }
            val byRef = inSplit.associateBy { "${it.src.sourceId}/${it.chunk.id}" }
            val dropped = HashSet<String>()
            var cross = 0; var same = 0
            for (p in Similarity.findNearDuplicates(texts, NEAR_DUP_THRESHOLD, SHINGLE)) {
                if (p.a in dropped || p.b in dropped) continue
                val sa = splitOf.getValue(p.a); val sb = splitOf.getValue(p.b)
                val victim: String; val leak: Boolean
                if (sa != sb) { victim = if (Similarity.SPLIT_PRIORITY.getValue(sa) < Similarity.SPLIT_PRIORITY.getValue(sb)) p.a else p.b; leak = true; cross++ }
                else { victim = p.b; leak = false; same++ }
                dropped.add(victim)
                val t = byRef.getValue(victim)
                t.flags.add(if (leak) ReviewFlag.POSSIBLE_LEAKAGE else ReviewFlag.DUPLICATE)
                if (leak) t.flags.add(ReviewFlag.DUPLICATE)
                t.included = false
                t.reason = if (leak) "near-duplicate of a chunk in a more protected split (${if (sa != sb) "cross-split" else ""} jaccard>=$NEAR_DUP_THRESHOLD)" else "near-duplicate of ${if (victim == p.a) p.b else p.a}"
            }
            if (same + cross > 0) notes.add("$same near-duplicate chunk(s) and $cross cross-split near-duplicate(s) were excluded by default; they stay listed for review")
            // containment: a short chunk copied out of a protected chunk
            val remaining = inSplit.filter { it.included }
            val prot = remaining.filter { it.split != "train" }.associate { "${it.src.sourceId}/${it.chunk.id}" to it.chunk.text }
            val cands2 = remaining.associate { "${it.src.sourceId}/${it.chunk.id}" to it.chunk.text }
            var contained = 0
            for (h in Similarity.containmentLeaks(cands2, prot, ExampleGen.CONTAINMENT_THRESHOLD, SHINGLE)) {
                if (h.a == h.b) continue
                val c = byRef.getValue(h.a)
                val p = byRef.getValue(h.b)
                if (Similarity.SPLIT_PRIORITY.getValue(p.split!!) > Similarity.SPLIT_PRIORITY.getValue(c.split!!) && c.included) {
                    c.included = false; c.flags.add(ReviewFlag.POSSIBLE_LEAKAGE); c.reason = "largely contained in a held-out/validation chunk (${h.score})"; contained++
                }
            }
            if (contained > 0) notes.add("$contained chunk(s) largely contained in held-out/validation text were excluded by default")
        }
        val dchunks = all.map { t ->
            DChunk(t.src.sourceId, t.src.name, t.chunk, t.group, t.split, t.flags.distinct(), t.included, t.reason, t.trainable)
        }
        val synth = ArrayList<Synth>()
        if (o.includeSynthetic && splitsAvailable) {
            for (d in dchunks) if (d.split == "train" && d.defaultIncluded) {
                val qa = ExampleGen.qa(d.sourceId, d.chunk.id, d.chunk.section, d.chunk.text) ?: continue
                synth.add(Synth("${d.ref}#qa", d.ref, d.sourceId, d.sourceName, d.chunk.section, d.chunk.page, qa.first, qa.second))
            }
        }
        return BuildResult(dchunks, synth, groupBy, assignment, notes, splitsAvailable)
    }

    // ---- assembling final examples (the actual training/validation/test rows) ----------------------------------------

    class Assembled(
        val bySplit: Map<String, List<Example>>, val dropped: List<String>, val notes: List<String>,
        val nearDupFound: Int, val residual: Int, val groupOverlap: Int,
    )

    fun assemble(ds: DatasetState): Assembled {
        val notes = ArrayList<String>()
        val cands = ArrayList<Example>()
        for (c in ds.chunks) {
            val split = c.split ?: continue
            if (!ds.isIncluded(c)) continue
            for (r in ExampleGen.cloze(c.chunk.section, c.chunk.text)) {
                if (r.quality < ExampleGen.MIN_QUALITY) continue
                cands.add(Example(ExampleGen.exampleId(c.sourceId, c.chunk.id, "cloze_continuation", r.idx), c.group!!, r.prompt, r.response, "source_derived",
                    c.sourceId, c.chunk.id, "cloze_continuation", r.quality, split))
            }
        }
        for (s in ds.synth) {
            if (!ds.isIncluded(s)) continue
            val c = ds.chunk(s.chunkRef) ?: continue
            if (c.split != "train" || !ds.isIncluded(c)) continue
            cands.add(Example("ex-" + Hashing.sha256("synthetic|${s.id}|${s.prompt}|${s.response}").take(12), c.group!!, s.prompt, s.response, "synthetic", c.sourceId, c.chunk.id, "qa_template", 0.5, "train"))
        }
        val splitOf = LinkedHashMap<String, String>()
        val texts = LinkedHashMap<String, String>()
        for (e in cands) { splitOf[e.id] = e.split; texts[e.id] = ExampleGen.leakText(e.prompt, e.response) }
        // exact in-split duplicates carry no information: keep the first by id
        val seen = HashMap<Pair<String, String>, String>()
        val dup = ArrayList<String>()
        for (id in texts.keys.sorted()) {
            val key = splitOf.getValue(id) to Similarity.words(texts.getValue(id)).joinToString(" ")
            if (seen.containsKey(key)) dup.add(id) else seen[key] = id
        }
        dup.forEach { splitOf.remove(it); texts.remove(it) }
        if (dup.isNotEmpty()) notes.add("dropped ${dup.size} exact in-split duplicate example(s)")
        // containment against the CHUNK text of held-out groups
        val held = ds.chunks.filter { it.split == "validation" || it.split == "test" }.associate { it.ref to it.chunk.text }
        val chunkSplit = ds.chunks.filter { it.split != null }.associate { it.ref to it.split!! }
        val contained = ArrayList<String>()
        for (h in Similarity.containmentLeaks(texts.filterKeys { splitOf[it] != "test" }, held, ExampleGen.CONTAINMENT_THRESHOLD, SHINGLE)) {
            if (Similarity.SPLIT_PRIORITY.getValue(chunkSplit.getValue(h.b)) > Similarity.SPLIT_PRIORITY.getValue(splitOf.getValue(h.a))) contained.add(h.a)
        }
        contained.forEach { splitOf.remove(it); texts.remove(it) }
        if (contained.isNotEmpty()) notes.add("dropped ${contained.size} example(s) largely contained in held-out chunk text")
        val res = Similarity.resolveLeakage(texts, splitOf, NEAR_DUP_THRESHOLD, SHINGLE)
        val dropped = (res.dropped + contained + dup).toSortedSet().toList()
        if (res.dropped.isNotEmpty()) notes.add("dropped ${res.dropped.size} example(s) that were near-duplicates across splits")
        val kept = cands.filter { res.remaining.containsKey(it.id) }
        val residual = Similarity.crossSplitPairs(Similarity.findNearDuplicates(texts.filterKeys { it in res.remaining }, NEAR_DUP_THRESHOLD, SHINGLE), res.remaining).size
        val groupsBySplit = kept.groupBy { it.split }.mapValues { e -> e.value.map { it.groupId }.toSet() }
        val overlap = listOf("train" to "validation", "train" to "test", "validation" to "test").sumOf { (a, b) -> ((groupsBySplit[a] ?: emptySet()) intersect (groupsBySplit[b] ?: emptySet())).size }
        val by = listOf("train", "validation", "test").associateWith { s -> kept.filter { it.split == s }.sortedBy { it.id } }
        return Assembled(by, dropped, notes, res.found, residual, overlap)
    }

    /** Re-check from the final rows (Python `verify_no_leakage`). Empty list == clean. */
    fun verify(a: Assembled): List<String> {
        val problems = ArrayList<String>()
        val groups = a.bySplit.mapValues { e -> e.value.map { it.groupId }.toSet() }
        for ((x, y) in listOf("train" to "validation", "train" to "test", "validation" to "test")) {
            val both = groups.getValue(x) intersect groups.getValue(y)
            if (both.isNotEmpty()) problems.add("group(s) ${both.sorted()} appear in both $x and $y")
        }
        for (e in a.bySplit.getValue("test")) if (e.origin == "synthetic") problems.add("synthetic example ${e.id} in test split")
        val texts = LinkedHashMap<String, String>(); val splitOf = LinkedHashMap<String, String>()
        for ((s, rows) in a.bySplit) for (e in rows) { texts[e.id] = ExampleGen.leakText(e.prompt, e.response); splitOf[e.id] = s }
        for (p in Similarity.crossSplitPairs(Similarity.findNearDuplicates(texts, NEAR_DUP_THRESHOLD, SHINGLE), splitOf))
            problems.add("near-duplicate across splits: ${p.a} (${splitOf[p.a]}) ~ ${p.b} (${splitOf[p.b]}) jaccard=${p.score}")
        return problems
    }

    // ---- identity ------------------------------------------------------------------------------------------------------------

    fun optionsMap(o: DatasetOptions): Map<String, Any?> = linkedMapOf("seed" to o.seed, "chunk_target_tokens" to o.chunkTargetTokens, "held_out_fraction" to o.heldOutFraction,
        "validation_fraction" to o.validationFraction, "include_synthetic" to o.includeSynthetic, "exclude_reference_only_sources" to o.excludeReferenceOnlySources)

    /** Changes whenever any input (source text, rights, options, grouping, review decision) changes. */
    fun datasetSha(ds: DatasetState): String = Hashing.sha256(J.canonical(linkedMapOf(
        "builder" to listOf(ExampleGen.BUILDER_NAME, ExampleGen.BUILDER_VERSION), "options" to optionsMap(ds.meta.options), "group_by" to ds.meta.groupBy,
        "assignment" to ds.meta.assignment,
        "chunks" to ds.chunks.map { listOf(it.ref, it.chunk.sha256, it.split, ds.isIncluded(it)) },
        "synth" to ds.synth.map { listOf(it.id, ds.isIncluded(it), Hashing.sha256(it.prompt + "\u0000" + it.response)) },
    )))

    // ---- held-out evaluation items ---------------------------------------------------------------------------------------------

    class EvalItem(val itemId: String, val kind: String, val question: String, val goldRefs: List<String>, val contentText: String,
                   val expectedValue: Double?, val expectedUnit: String?, val requiredTerms: List<String>)

    private val EVAL_STOP = ("the a an of to in and or is are was were be been it its for on with as by at from that this these those which not can may should must will when where how " +
        "what why do does did if then than also each all any more most other use used using before after during between into over under about one two three first second would could " +
        "has have had their there they them you your we our i he she his her but so such only same both either neither per via being").split(" ").toSet()
    private fun contentWords(t: String) = Regex("[a-z][a-z0-9]+").findAll(t.lowercase()).map { it.value }.filter { it !in EVAL_STOP && it.length >= 3 }.toList()

    fun buildEvalItems(chunks: List<Pair<String, Pair<String, String>>>, domain: String, maxFactsPerChunk: Int = 2): List<EvalItem> {
        // chunks: (ref, (section, text))
        val sorted = chunks.sortedBy { it.first }
        val df = HashMap<String, Int>()
        for (c in sorted) for (w in contentWords(c.second.second).toSet()) df.merge(w, 1, Int::plus)
        val items = ArrayList<EvalItem>()
        for ((ref, st) in sorted) {
            val (sec, text) = st
            val tf = HashMap<String, Int>()
            for (w in contentWords("$text $sec")) if (!w.all { it.isDigit() } && w.length >= 5 && w !in EVAL_STOP) tf.merge(w, 1, Int::plus)
            val terms = tf.keys.sortedWith(compareBy({ -(tf.getValue(it) * (1.0 / (df[it] ?: 1))) }, { it })).take(5)
            if (terms.size >= 2) items.add(EvalItem("it-" + Hashing.sha256("$ref|concept").take(12), "concept",
                "Describe, as a $domain specialist would, what the section \"$sec\" covers.", listOf(ref), text, null, null, terms))
            var n = 0
            for (sent in ExampleGen.sentences(text)) {
                val qs = Quantities.extract(sent)
                if (qs.size != 1 || n >= maxFactsPerChunk) continue
                val masked = Quantities.MASK.replaceFirst(sent, "____")
                if (!masked.contains("____")) continue
                items.add(EvalItem("it-" + Hashing.sha256("$ref|fact|$sent").take(12), "fact",
                    "In the $domain documentation, section \"$sec\": \"$masked\" What is the missing value (number and unit)?", listOf(ref), sent, qs[0].first, qs[0].second, emptyList()))
                n++
            }
        }
        return items
    }
}

/** Quantity extraction (value + unit) for held-out fact items; unknown units are skipped, never guessed. Ported from evalsuite.metrics. */
object Quantities {
    private val UNITS = listOf("mm", "millimeter", "millimeters", "millimetre", "millimetres", "cm", "centimeter", "centimeters", "m", "meter", "meters", "metre", "metres",
        "in", "inch", "inches", "n·m", "n-m", "nm", "n.m", "n m", "newton-meter", "newton-meters", "lb-ft", "lb·ft", "ft-lb", "ft·lb", "ft-lbs", "lbf-ft", "ft-lbf",
        "in-lb", "in·lb", "lb-in", "in-lbs", "lbf-in", "kpa", "bar", "psi", "mpa", "v", "volt", "volts", "a", "amp", "amps", "ampere", "amperes", "ma", "milliamp", "milliamps",
        "ohm", "ohms", "ω", "kohm", "kω", "hz", "rpm", "g", "gram", "grams", "kg", "kilogram", "kilograms", "lb", "lbs", "pound", "pounds", "ml", "milliliter", "milliliters",
        "l", "liter", "liters", "litre", "litres", "qt", "quart", "quarts", "s", "sec", "second", "seconds", "min", "mins", "minute", "minutes", "h", "hr", "hrs", "hour", "hours",
        "day", "days", "%", "percent", "°c", "degc", "celsius", "°f", "degf", "fahrenheit")
    private val UNIT_RE = UNITS.sortedByDescending { it.length }.joinToString("|") { Regex.escape(it) }
    private val QTY = Regex("(?<![\\w.])([-−]?\\d+(?:[.,]\\d+)?|\\.\\d+)\\s*($UNIT_RE)(?![a-zA-Z])", RegexOption.IGNORE_CASE)
    val MASK = Regex("([-−]?\\d+(?:[.,]\\d+)?|\\.\\d+)\\s*[^\\s,;.]+")

    fun extract(text: String): List<Pair<Double, String>> = QTY.findAll(text).mapNotNull { m ->
        val v = m.groupValues[1].replace('−', '-').replace(',', '.').toDoubleOrNull() ?: return@mapNotNull null
        v to m.groupValues[2].lowercase()
    }.toList()
}

/** On-disk dataset for one project: `dataset/{build.json, chunks.jsonl, synthetic.jsonl, review.json}`. */
class DatasetStore(private val dir: File, private val problems: MutableList<String>) {
    companion object { const val SCHEMA = 1 }
    private val buildFile get() = File(dir, "build.json")
    private val chunksFile get() = File(dir, "chunks.jsonl")
    private val synthFile get() = File(dir, "synthetic.jsonl")
    private val reviewFile get() = File(dir, "review.json")

    fun exists() = buildFile.isFile || File(dir, "build.json.bak").isFile

    fun save(ds: DatasetState) {
        val m = ds.meta
        Fs.writeAtomic(chunksFile) { out ->
            for (d in ds.chunks) out.write((J.dump(linkedMapOf(
                "source_id" to d.sourceId, "source_name" to d.sourceName, "id" to d.chunk.id, "section_path" to d.chunk.sectionPath, "page" to d.chunk.page,
                "char_start" to d.chunk.charStart, "char_end" to d.chunk.charEnd, "role" to d.chunk.role, "kind" to d.chunk.kind, "exclude_reason" to d.chunk.excludeReason,
                "duplicate_of" to d.chunk.duplicateOf, "text" to d.chunk.text, "sha256" to d.chunk.sha256, "origin" to d.chunk.origin, "transforms" to d.chunk.transforms,
                "group" to d.group, "split" to d.split, "flags" to d.flags.map { it.name }, "default_included" to d.defaultIncluded, "reason" to d.reason, "trainable" to d.trainable)) + "\n").toByteArray(Charsets.UTF_8))
        }
        Fs.writeAtomic(synthFile) { out ->
            for (s in ds.synth) out.write((J.dump(linkedMapOf("id" to s.id, "chunk_ref" to s.chunkRef, "source_id" to s.sourceId, "source_name" to s.sourceName,
                "section" to s.section, "page" to s.page, "prompt" to s.prompt, "response" to s.response)) + "\n").toByteArray(Charsets.UTF_8))
        }
        saveReview(ds)
        saveMeta(ds)
    }

    fun saveReview(ds: DatasetState) {
        Fs.writeJson(reviewFile, linkedMapOf("schema" to SCHEMA, "decisions" to ds.decisions.toSortedMap().mapValues { mapOf("included" to it.value.included, "text_sha" to it.value.textSha) }))
    }

    fun saveMeta(ds: DatasetState) {
        val m = ds.meta
        Fs.writeJson(buildFile, linkedMapOf(
            "schema" to SCHEMA, "version" to m.version, "options" to DatasetEngine.optionsMap(m.options), "built_at" to m.builtAt, "approved_at" to m.approvedAt,
            "status" to m.status.name, "group_by" to m.groupBy, "assignment" to m.assignment, "notes" to m.notes, "splits_available" to m.splitsAvailable,
            "source_snapshot" to m.sourceSnapshot, "builder" to mapOf("name" to ExampleGen.BUILDER_NAME, "version" to ExampleGen.BUILDER_VERSION, "type" to "template")))
    }

    fun load(): DatasetState? {
        val b = Fs.readJson(buildFile) { problems.add("dataset: $it") } ?: return null
        try {
            val oo = b.obj("options")!!
            val opts = DatasetOptions(oo.lng("seed")!!, oo.int("chunk_target_tokens")!!, oo.dbl("held_out_fraction")!!, oo.dbl("validation_fraction")!!,
                oo.bool("include_synthetic")!!, oo.bool("exclude_reference_only_sources")!!)
            val asg = b.obj("assignment")?.let { a -> a.keyList().associateWith { a.getString(it) } } ?: emptyMap()
            val meta = DatasetMeta(b.int("version")!!, opts, b.lng("built_at")!!, b.lng("approved_at"), DatasetStatus.valueOf(b.str("status")!!), b.str("group_by") ?: "document", asg,
                b.strList("notes"), b.bool("splits_available") ?: true, b.objList("source_snapshot").map { o -> o.keyList().associateWith { o.opt(it) } })
            val chunks = Fs.readLines(chunksFile).map { line ->
                val o = org.json.JSONObject(line)
                val c = Chunk(o.str("id")!!, o.strList("section_path"), o.int("page"), o.int("char_start")!!, o.int("char_end")!!, o.str("role")!!, o.str("kind")!!, o.str("exclude_reason"),
                    o.str("duplicate_of"), o.str("text")!!, o.str("sha256")!!, o.str("origin") ?: "extracted", o.strList("transforms"))
                DChunk(o.str("source_id")!!, o.str("source_name") ?: "", c, o.str("group"), o.str("split"), o.strList("flags").mapNotNull { f -> runCatching { ReviewFlag.valueOf(f) }.getOrNull() },
                    o.bool("default_included") ?: true, o.str("reason"), o.bool("trainable") ?: false)
            }
            val synth = Fs.readLines(synthFile).map { line -> val o = org.json.JSONObject(line)
                Synth(o.str("id")!!, o.str("chunk_ref")!!, o.str("source_id")!!, o.str("source_name") ?: "", o.str("section") ?: "", o.int("page"), o.str("prompt")!!, o.str("response")!!) }
            val dec = java.util.concurrent.ConcurrentHashMap<String, Decision>()
            Fs.readJson(reviewFile) { problems.add("dataset review: $it") }?.obj("decisions")?.let { d ->
                for (k in d.keyList()) d.obj(k)?.let { dec[k] = Decision(it.bool("included") ?: true, it.str("text_sha") ?: "") }
            }
            return DatasetState(meta, chunks, synth, dec)
        } catch (e: Exception) {
            problems.add("dataset files are unreadable (${e.javaClass.simpleName}); rebuild the dataset")
            return null
        }
    }

    fun clear() { dir.deleteRecursively() }
}
