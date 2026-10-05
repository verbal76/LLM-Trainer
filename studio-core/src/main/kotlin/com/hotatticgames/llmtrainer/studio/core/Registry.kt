package com.hotatticgames.llmtrainer.studio.core

import org.json.JSONObject

/** Read-only view of one registry base-model JSON entry (schema_version 2). Unknowns stay null. */
data class RegVariant(
    val variantId: String, val format: String, val quantization: String?, val sourceUrl: String?,
    val sizeBytes: Long?, val sha256: String?, val sizeEvidence: String?,
)

data class RegVerification(
    val state: String, val evidenceLevel: String, val sourceUrls: List<String>, val verifiedOn: String?,
    val textSha256: String?, val textUrl: String?, val scope: String?, val disallowedReason: String?, val uncertainties: List<String>,
)

data class RegEntry(
    val entryId: String, val family: String, val version: String, val hfRepo: String?, val parameterCount: String?,
    val licenseId: String?, val licenseUrlRaw: String?,
    val commercial: String, val fineTune: String, val adapter: String, val redistribution: String, val attribution: String,
    val restrictions: List<String>, val formats: List<String>, val contextLength: Int?,
    val layers: Int?, val kvHeads: Int?, val headDim: Int?, val archNotes: String?,
    val variants: List<RegVariant>, val androidState: String, val androidNotes: String?,
    val verification: RegVerification,
) {
    /** The URL the license TEXT is fetched from: explicit text URL, else the first https URL that looks like a license file. */
    val authoritativeUrl: String get() = Registry.authoritativeUrl(this)
}

object Registry {
    fun parse(json: String): RegEntry {
        val o = JSONObject(json)
        val fam = o.str("model_family") ?: error("model_family missing")
        val ver = o.str("exact_version") ?: error("exact_version missing")
        val arch = o.obj("architecture")
        val v = o.obj("verification") ?: error("verification missing")
        val android = o.obj("android_inference")
        return RegEntry(
            entryId = "$fam@$ver", family = fam, version = ver, hfRepo = o.str("hf_repo_or_source"),
            parameterCount = o.str("parameter_count"), licenseId = o.str("license_id"), licenseUrlRaw = o.str("license_url"),
            commercial = o.str("commercial_use") ?: "unverified", fineTune = o.str("fine_tuning_permitted") ?: "unverified",
            adapter = o.str("derivative_adapter_permitted") ?: "unverified", redistribution = o.str("redistribution_permitted") ?: "unverified",
            attribution = o.str("attribution_required") ?: "unverified",
            restrictions = o.strList("restrictions"), formats = o.strList("supported_formats"), contextLength = o.int("context_length"),
            layers = arch?.int("layers"), kvHeads = arch?.int("kv_heads"), headDim = arch?.int("head_dim"), archNotes = arch?.str("notes"),
            variants = o.objList("variants").map {
                RegVariant(it.str("variant_id") ?: "default", it.str("format") ?: "unknown", it.str("quantization"), it.str("source_url"),
                    it.lng("size_bytes"), it.str("sha256")?.let(Hashing::bare), it.str("size_evidence"))
            },
            androidState = android?.str("state") ?: "unverified", androidNotes = android?.str("notes"),
            verification = RegVerification(
                v.str("state") ?: "UNVERIFIED", v.str("evidence_level") ?: "none", v.strList("source_urls"), v.str("verified_on"),
                v.str("license_text_sha256")?.let(Hashing::bare), v.str("license_text_url"), v.str("verified_scope"),
                v.str("disallowed_reason"), v.strList("uncertainties"),
            ),
        )
    }

    /** All entries embedded at build time, sorted by id. A malformed entry is reported, never silently kept half-parsed. */
    fun loadEmbedded(onProblem: (String) -> Unit = {}): List<RegEntry> =
        parseAll(EmbeddedRegistry.files, onProblem)

    fun parseAll(files: List<Pair<String, String>>, onProblem: (String) -> Unit = {}): List<RegEntry> {
        val out = ArrayList<RegEntry>()
        for ((name, json) in files) {
            try { out.add(parse(json)) } catch (e: Exception) { onProblem("registry entry $name is invalid: ${e.message}") }
        }
        return out.sortedBy { it.entryId }
    }

    private val URL_RE = Regex("""https?://[^\s()\[\];,"']+""")

    /** Raw HF "blob" pages are HTML; the same file is served as text under /raw/. */
    fun rawTextUrl(u: String): String {
        val m = Regex("""^(https://huggingface\.co/[^/]+/[^/]+)/blob/(.*)$""").find(u)
        return if (m != null) m.groupValues[1] + "/raw/" + m.groupValues[2] else u
    }

    fun authoritativeUrl(e: RegEntry): String {
        e.verification.textUrl?.let { if (it.startsWith("https://")) return rawTextUrl(it) }
        val urls = URL_RE.findAll(e.licenseUrlRaw ?: "").map { it.value.trimEnd('.', ')') }.toList()
        val licenseLike = urls.firstOrNull { it.substringAfter("://").substringAfter('/').lowercase().contains("licen") }
        val pick = licenseLike ?: urls.firstOrNull() ?: return ""
        return if (pick.startsWith("https://")) rawTextUrl(pick) else ""
    }

    /** Parameters in billions from the recorded string; null for "effective" counts or unparseable text (never guessed). */
    fun paramsB(e: RegEntry): Double? {
        val s = e.parameterCount ?: return null
        if (s.lowercase().contains("effective")) return null
        val m = Regex("""^\s*([0-9]+(?:\.[0-9]+)?)\s*B\b""").find(s) ?: return null
        return m.groupValues[1].toDouble()
    }
}
