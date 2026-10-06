package com.hotatticgames.llmtrainer.studio.core

import org.json.JSONObject

/**
 * Independent re-implementation of the desktop import rules in docs/studio/PACKAGE_FORMATS.md section 1.6 (zip hygiene,
 * checksums, schema, role/rights checks, leakage re-checks). Returns problems; empty list == the Python importer would accept.
 */
object JobValidator {
    private val ID = Regex(Packages.ID_PATTERN)
    private val HASH = Regex("^sha256:[0-9a-f]{64}$")
    private val ALLOWED = setOf("manifest.json", "chunks.jsonl", "dataset_manifest.json", "dataset/train.jsonl", "dataset/validation.jsonl", "dataset/test.jsonl",
        "eval/heldout.jsonl", "license/license_text.txt", "checksums.json")

    fun lines(b: ByteArray?): List<JSONObject> = b?.toString(Charsets.UTF_8)?.lines()?.filter { it.isNotBlank() }?.map { JSONObject(it) } ?: emptyList()

    fun validate(zip: ByteArray): List<String> {
        val p = ArrayList<String>()
        val files = try { Zips.readAll(zip.inputStream()) } catch (e: PackageException) { return listOf("zip: ${e.message}") }
        p.addAll(Zips.verifyChecksums(files))
        for (n in files.keys) if (n !in ALLOWED) p.add("member outside layout: $n")
        for (n in listOf("manifest.json", "chunks.jsonl", "dataset_manifest.json", "dataset/train.jsonl", "dataset/validation.jsonl", "dataset/test.jsonl", "eval/heldout.jsonl"))
            if (n !in files) p.add("missing $n")
        if (p.isNotEmpty()) return p
        val m = JSONObject(files.getValue("manifest.json").toString(Charsets.UTF_8))
        if (m.getString("format") != "llmtrainer-training-job" || m.getInt("version") != 1) p.add("format/version")
        if (!ID.matches(m.getString("job_id"))) p.add("job_id pattern")
        val proj = m.getJSONObject("project")
        if (!ID.matches(proj.getString("id"))) p.add("project.id pattern")
        val bm = m.getJSONObject("base_model")
        if (bm.getString("registry_id") != bm.getString("family") + "@" + bm.getString("exact_version")) p.add("registry_id != family@exact_version")
        val allowedTop = setOf("format", "version", "job_id", "created_at", "created_by", "project", "device", "base_model", "license_evidence", "method", "sources", "extensions")
        for (k in m.keySet()) if (k !in allowedTop) p.add("unknown manifest field $k")
        if (m.getJSONObject("method").getString("requested") !in setOf("rag_only", "prompt_only", "lora", "qlora")) p.add("method.requested")
        // license evidence: all-or-nothing
        m.optJSONObject("license_evidence")?.let { le ->
            if (le.getString("model_id") != bm.getString("registry_id")) p.add("license_evidence.model_id")
            if (!le.getString("license_url").startsWith("https://")) p.add("license_url not https")
            if (!HASH.matches(le.getString("text_sha256"))) p.add("text_sha256")
            if (le.getString("evidence_level") != "primary_license_text_read" || le.getString("attested_by") != "owner") p.add("evidence_level/attested_by")
            if (le.getString("scope_note").isBlank()) p.add("scope_note")
            val perms = le.getJSONObject("permissions")
            for (k in listOf("commercial_use", "fine_tuning_permitted", "derivative_adapter_permitted", "redistribution_permitted", "attribution_required"))
                if (perms.getString(k) !in setOf("yes", "no", "conditional")) p.add("permission $k")
            files["license/license_text.txt"]?.let { if (Hashing.prefixed(Hashing.sha256(it)) != le.getString("text_sha256")) p.add("license text hash != text_sha256") }
        }
        val sources = m.getJSONArray("sources")
        val srcIds = (0 until sources.length()).map { sources.getJSONObject(it).getString("id") }
        if (srcIds.toSet().size != srcIds.size) p.add("duplicate source ids")
        val rights = (0 until sources.length()).associate { sources.getJSONObject(it).getString("id") to sources.getJSONObject(it).getJSONObject("rights") }
        for (id in srcIds) if (!ID.matches(id)) p.add("source id pattern $id")
        // chunks
        val chunks = lines(files["chunks.jsonl"])
        val chunkByRef = HashMap<String, JSONObject>()
        for (c in chunks) {
            val ref = c.getString("source_id") + "/" + c.getString("id")
            if (chunkByRef.put(ref, c) != null) p.add("duplicate chunk $ref")
            if (c.getString("source_id") !in srcIds) p.add("chunk source missing $ref")
            if (c.getString("sha256") != Hashing.prefixed(Hashing.sha256(c.getString("text")))) p.add("chunk sha $ref")
            val role = c.getString("role")
            if (role !in setOf("train", "reference", "excluded")) p.add("chunk role $ref")
            if (role == "excluded" && c.isNull("exclude_reason")) p.add("excluded without reason $ref")
            if (c.getString("origin") !in setOf("extracted", "ocr", "table", "edited", "synthetic")) p.add("chunk origin $ref")
            if (!ID.matches(c.getString("id"))) p.add("chunk id pattern")
            if (c.getInt("char_start") > c.getInt("char_end")) p.add("chunk offsets $ref")
        }
        // dataset
        val dm = JSONObject(files.getValue("dataset_manifest.json").toString(Charsets.UTF_8))
        val ratios = dm.getJSONObject("split_config").getJSONObject("ratios")
        if (Math.abs(ratios.getDouble("train") + ratios.getDouble("validation") + ratios.getDouble("test") - 1.0) > 1e-9) p.add("ratios do not sum to 1")
        val groupsOf = dm.getJSONObject("split_assignment").getJSONObject("groups").let { g -> g.keySet().associateWith { g.getString(it) } }
        val rows = mapOf("train" to lines(files["dataset/train.jsonl"]), "validation" to lines(files["dataset/validation.jsonl"]), "test" to lines(files["dataset/test.jsonl"]))
        val counts = dm.optJSONObject("counts")
        val splitOfEx = HashMap<String, String>()
        val textOf = HashMap<String, String>()
        val refSplits = HashMap<String, MutableSet<String>>()
        for ((s, rs) in rows) {
            if (rs.isEmpty()) p.add("empty split $s")
            if (counts != null && counts.getInt(s) != rs.size) p.add("count mismatch $s")
            for (r in rs) {
                val gid = r.getString("group_id")
                if (groupsOf[gid] != s) p.add("group $gid assigned ${groupsOf[gid]} but example in $s")
                if (!ID.matches(r.getString("example_id"))) p.add("example id")
                if (r.getString("origin") !in setOf("source_derived", "synthetic")) p.add("example origin")
                if (s == "test" && r.getString("origin") == "synthetic") p.add("synthetic in test")
                val df = r.getJSONArray("derived_from")
                if (df.length() == 0) p.add("empty derived_from")
                for (i in 0 until df.length()) {
                    val ref = df.getJSONObject(i).getString("source_id") + "/" + df.getJSONObject(i).getString("chunk_id")
                    val c = chunkByRef[ref]
                    if (c == null) p.add("example refs missing chunk $ref") else {
                        if (c.getString("role") != "train") p.add("example refs non-train chunk $ref")
                        val groupBy = dm.getJSONObject("split_assignment").getString("group_by")
                        val chunkGroup = if (groupBy == "document") c.getString("source_id") else c.optString("group_id", "").ifEmpty { c.getString("source_id") + "#" + Text.slugify(c.getJSONArray("section_path").let { a -> if (a.length() == 0) "Document" else (0 until a.length()).joinToString(" > ") { i -> a.getString(i) } }) }
                        if (chunkGroup != gid) p.add("chunk group != example group for $ref")
                        // rights gate: only permitted_training == yes may feed examples
                        if (rights.getValue(c.getString("source_id")).getString("permitted_training") != "yes") p.add("example from source without training rights $ref")
                    }
                    refSplits.getOrPut(ref) { HashSet() }.add(s)
                }
                val id = r.getString("example_id")
                splitOfEx[id] = s
                textOf[id] = ExampleGen.leakText(r.getString("prompt"), r.getString("response"))
            }
        }
        for ((ref, ss) in refSplits) if (ss.size > 1) p.add("chunk $ref backs examples in several splits $ss")
        for (a in listOf("train", "validation", "test")) for (b in listOf("train", "validation", "test")) if (a < b) {
            val ga = rows.getValue(a).map { it.getString("group_id") }.toSet(); val gb = rows.getValue(b).map { it.getString("group_id") }.toSet()
            if ((ga intersect gb).isNotEmpty()) p.add("group overlap $a/$b")
        }
        for (x in Similarity.crossSplitPairs(Similarity.findNearDuplicates(textOf, 0.8, 5), splitOfEx)) p.add("near-duplicate across splits ${x.a}~${x.b}")
        val testChunkTexts = refSplits.filter { "test" in it.value }.keys.associateWith { chunkByRef.getValue(it).getString("text") }
        val nonTest = splitOfEx.filter { it.value != "test" }.keys.associateWith { textOf.getValue(it) }
        for (x in Similarity.containmentLeaks(nonTest, testChunkTexts, 0.8, 5)) p.add("train/validation example contained in test chunk: ${x.a}")
        // held-out items
        val items = lines(files["eval/heldout.jsonl"])
        val trainValRefs = refSplits.filter { it.value.any { s -> s != "test" } }.keys
        for (it in items) {
            if (!ID.matches(it.getString("item_id"))) p.add("item id")
            if (it.getString("origin") != "source_derived") p.add("item origin")
            when (it.getString("kind")) {
                "fact" -> if (it.isNull("expected")) p.add("fact without expected")
                "concept" -> if (it.getJSONArray("required_terms").length() == 0) p.add("concept without terms")
                else -> p.add("item kind")
            }
            val gr = it.getJSONArray("gold_refs")
            if (gr.length() == 0) p.add("item without gold_refs")
            for (i in 0 until gr.length()) {
                val ref = gr.getString(i)
                if ("test" !in (refSplits[ref] ?: emptySet())) p.add("gold ref not a test chunk $ref")
                if (ref in trainValRefs) p.add("gold ref used by train/validation $ref")
            }
        }
        return p
    }
}
