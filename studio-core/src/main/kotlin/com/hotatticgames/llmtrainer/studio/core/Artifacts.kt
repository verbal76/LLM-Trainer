package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.qualify.ArtifactSpec
import com.hotatticgames.llmtrainer.studio.api.LicenseInfo
import com.hotatticgames.llmtrainer.studio.api.LicenseState
import com.hotatticgames.llmtrainer.studio.api.Permission
import com.hotatticgames.llmtrainer.studio.api.Tri
import org.json.JSONObject

/**
 * Downloadable artifacts: exact GGUF files from `registry/artifacts/<id>.json`, written by the catalog-refresh CI job
 * (scripts/catalog/refresh_catalog.py). The committed files are `unrefreshed` placeholders until the first CI run; an
 * unrefreshed artifact has NO hash, size or revision and is NEVER downloadable (never ship guessed hashes).
 */
data class ArtifactRec(
    val id: String, val refreshState: String, val optional: Boolean, val published: Boolean?, val family: String, val baseRepo: String,
    val repo: String, val file: String?, val fileGlob: String?, val revision: String?, val downloadUrl: String?,
    val quantization: String, val precision: String, val sizeBytes: Long?, val sha256: String?,
    val nominalParamsB: Double, val parameterCount: Double?,
    val layers: Int?, val kvHeads: Int?, val headDim: Int?, val embd: Int?, val ffn: Int?, val vocab: Int?, val ctxTrain: Int?, val archName: String?,
    val chatTemplatePresent: Boolean?, val runtimeLlamaCpp: String, val runtimeHagEngine: String,
    val trainingClass: String, val trainingReasons: List<String>, val trainingSourceArtifactId: String?,
    val licenseState: String, val licenseExpectedSpdx: String?, val licenseSpdx: String?, val licenseReasons: List<String>,
    val licenseTextUrl: String?, val licenseTextSha256: String?, val canonicalSha256: String?, val canonicalSourceUrl: String?,
    val licenseMatch: String?, val modelCardAgrees: Boolean, val licenseFetchedAt: String?, val refreshedAt: String?,
) {
    /** Why this artifact cannot be downloaded (null = it can). Fail closed: every identity field must be present and well-formed. */
    val notDownloadableReason: String?
        get() = when {
            refreshState != "refreshed" -> "unrefreshed"
            published == false -> "not_published"
            file.isNullOrBlank() -> "no_file_name"
            revision == null || !Regex("^[0-9a-f]{40}$").matches(revision) -> "no_revision"
            sha256 == null || !Regex("^(sha256:)?[0-9a-fA-F]{64}$").matches(sha256) -> "no_sha256"
            sizeBytes == null || sizeBytes <= 0 -> "no_size"
            downloadUrl == null || !downloadUrl.startsWith("https://") || !downloadUrl.contains("/resolve/$revision/") || !downloadUrl.endsWith("/$file") -> "no_immutable_url"
            else -> null
        }
    val downloadable: Boolean get() = notDownloadableReason == null
    val bareSha256: String? get() = sha256?.let(Hashing::bare)

    /** A license is VERIFIED only if CI recorded a canonical-text match AND the model card agrees (re-checked here, fail closed). */
    val licenseVerified: Boolean
        get() = refreshState == "refreshed" && licenseState == "VERIFIED" && licenseSpdx != null && licenseSpdx == licenseExpectedSpdx &&
            modelCardAgrees && (licenseMatch == "exact" || licenseMatch == "mit_copyright_line") && !licenseTextSha256.isNullOrBlank() &&
            canonicalSha256 != null && !licenseTextUrl.isNullOrBlank() &&
            // exact matches must be hash-identical to the canonical text
            (licenseMatch != "exact" || Hashing.bare(licenseTextSha256) == Hashing.bare(canonicalSha256))

    fun toSpec(licenseState: LicenseState, modelId: String): ArtifactSpec = ArtifactSpec(
        id, modelId, quantization, precision, sha256?.let(Hashing::prefixed), sizeBytes, parameterCount, nominalParamsB, layers, kvHeads, headDim, embd, ffn, vocab,
        ctxTrain, trainingClass, trainingSourceArtifactId, downloadable, licenseState.name,
        when {
            runtimeHagEngine == "no" || runtimeLlamaCpp == "no" -> "no"
            runtimeHagEngine == "yes" -> "yes"
            else -> "unverified"
        },
    )
}

data class CanonicalLicense(
    val spdxId: String, val sha256: String, val sourceUrl: String, val cardIds: List<String>, val permissions: Map<Permission, Tri>,
    val attributionRequired: Boolean, val conditions: List<String>, val matchMode: String,
)

object ArtifactRegistry {
    fun parseCanonical(json: String): Map<String, CanonicalLicense> {
        val o = JSONObject(json)
        val out = LinkedHashMap<String, CanonicalLicense>()
        for (l in o.objList("licenses")) {
            val id = l.str("spdx_id") ?: continue
            val p = l.obj("permissions")
            fun t(k: String) = triOf(p?.str(k))
            out[id] = CanonicalLicense(id, Hashing.bare(l.str("sha256") ?: continue), l.str("source_url") ?: "", l.strList("card_ids"),
                mapOf(Permission.COMMERCIAL to t("commercial_use"), Permission.FINE_TUNE to t("fine_tuning_permitted"),
                    Permission.ADAPTER to t("derivative_adapter_permitted"), Permission.REDISTRIBUTION to t("redistribution_permitted")),
                l.str("attribution_required") == "yes" || p?.str("attribution_required") == "yes", l.strList("conditions"), l.str("match_mode") ?: "exact")
        }
        return out
    }

    fun parse(json: String): ArtifactRec {
        val o = JSONObject(json)
        val src = o.obj("source") ?: error("source missing")
        val arch = o.obj("architecture")
        val rt = o.obj("runtime_compat")
        val tr = o.obj("training")
        val lic = o.obj("license")
        val ev = lic?.obj("evidence")
        return ArtifactRec(
            id = o.str("artifact_id") ?: error("artifact_id missing"), refreshState = o.str("refresh_state") ?: "unrefreshed",
            optional = o.bool("optional") ?: false, published = o.bool("published"), family = o.str("family") ?: "", baseRepo = o.str("base_repo") ?: "",
            repo = src.str("repo") ?: "", file = src.str("file"), fileGlob = src.str("file_glob"), revision = src.str("revision"),
            downloadUrl = src.str("download_url"), quantization = o.str("quantization") ?: "unknown", precision = o.str("precision") ?: "quantized",
            sizeBytes = o.lng("size_bytes"), sha256 = o.str("sha256"), nominalParamsB = o.dbl("parameter_count_nominal_b") ?: 0.0,
            parameterCount = o.dbl("parameter_count"), layers = arch?.int("layers"), kvHeads = arch?.int("kv_heads"), headDim = arch?.int("head_dim"),
            embd = arch?.int("embd"), ffn = arch?.int("ffn"), vocab = arch?.int("vocab"), ctxTrain = arch?.int("ctx_train"), archName = arch?.str("name"),
            chatTemplatePresent = o.bool("chat_template_present"), runtimeLlamaCpp = rt?.obj("llama.cpp")?.str("supported") ?: "unverified",
            runtimeHagEngine = rt?.obj("hag-engine")?.str("supported") ?: "unverified", trainingClass = tr?.str("class") ?: "inference_only",
            trainingReasons = tr?.strList("reasons").orEmpty(), trainingSourceArtifactId = tr?.str("training_source_artifact_id"),
            licenseState = lic?.str("state") ?: "UNVERIFIED", licenseExpectedSpdx = lic?.str("expected_spdx"), licenseSpdx = lic?.str("spdx_id"),
            licenseReasons = lic?.strList("reasons").orEmpty(), licenseTextUrl = ev?.str("license_text_url"), licenseTextSha256 = ev?.str("license_text_sha256"),
            canonicalSha256 = ev?.str("canonical_sha256"), canonicalSourceUrl = ev?.str("canonical_source_url"), licenseMatch = ev?.str("match"),
            modelCardAgrees = ev?.bool("model_card_agrees") ?: false, licenseFetchedAt = ev?.str("fetched_at"), refreshedAt = o.str("refreshed_at"),
        )
    }

    fun parseAll(files: List<Pair<String, String>>, onProblem: (String) -> Unit = {}): List<ArtifactRec> {
        val out = ArrayList<ArtifactRec>()
        for ((name, json) in files) {
            try { out.add(parse(json)) } catch (e: Exception) { onProblem("artifact entry $name is invalid: ${e.message}") }
        }
        return out.sortedBy { it.id }
    }
}

/** Artifacts grouped by base-model repo, plus the license evidence derived from them. */
class ArtifactCatalog(val artifacts: List<ArtifactRec>, val canonical: Map<String, CanonicalLicense>) {
    private val byId = artifacts.associateBy { it.id }
    fun get(id: String): ArtifactRec? = byId[id]
    fun forRepo(baseRepo: String?): List<ArtifactRec> = if (baseRepo == null) emptyList() else artifacts.filter { it.baseRepo.equals(baseRepo, true) }

    /**
     * CI license evidence for a registry entry: VERIFIED only when at least one artifact of the entry's repo is
     * refreshed, EVERY refreshed artifact of it is license-verified for the SAME canonical license, and that canonical
     * license is known to this app build. Otherwise null (the owner-attestation flow stays in charge).
     */
    fun licenseEvidence(e: RegEntry): LicenseInfo? {
        val refreshed = forRepo(e.hfRepo).filter { it.refreshState == "refreshed" && it.published != false }
        if (refreshed.isEmpty() || !refreshed.all { it.licenseVerified }) return null
        val spdx = refreshed.map { it.licenseSpdx }.distinct().singleOrNull() ?: return null
        val canon = canonical[spdx] ?: return null
        val first = refreshed.first()
        // the canonical hash recorded by CI must be the hash this app build ships for that license
        if (refreshed.any { it.canonicalSha256 == null || Hashing.bare(it.canonicalSha256) != canon.sha256 }) return null
        val scope = "CI-verified: license text fetched from ${first.licenseTextUrl} is ${first.licenseMatch} to the canonical $spdx text " +
            "(sha256 ${Hashing.prefixed(canon.sha256)}, from ${canon.sourceUrl}) and the model-card license id agrees; permissions are those of the canonical license"
        return LicenseInfo(
            LicenseState.VERIFIED, spdx, first.licenseTextUrl ?: "", Evidence.CI_CANONICAL, scope, Hashing.bare(first.licenseTextSha256 ?: canon.sha256),
            Iso.parseMs(first.licenseFetchedAt), canon.permissions, canon.attributionRequired, null,
        )
    }
}
