package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.Blocker
import com.hotatticgames.llmtrainer.studio.api.IntendedUse
import com.hotatticgames.llmtrainer.studio.api.LicenseAttestation
import com.hotatticgames.llmtrainer.studio.api.LicenseInfo
import com.hotatticgames.llmtrainer.studio.api.LicenseState
import com.hotatticgames.llmtrainer.studio.api.LicenseTextFetch
import com.hotatticgames.llmtrainer.studio.api.Permission
import com.hotatticgames.llmtrainer.studio.api.StudioError
import com.hotatticgames.llmtrainer.studio.api.StudioResult
import com.hotatticgames.llmtrainer.studio.api.Tri
import java.io.File
import java.io.InputStream

/** Evidence level strings carried on [LicenseInfo.evidenceLevel]. */
object Evidence {
    const val NONE = "none"
    const val SECONDARY = "secondary"
    /** The registry maintainers read the primary license text (Python `primary_*` levels). */
    const val PRIMARY_REGISTRY = "primary_text_read"
    const val OWNER_TEXT = "owner_reviewed_text"
    const val OWNER_ATTESTED_DISALLOWED = "owner_attested"
    val VERIFIED_LEVELS = setOf(PRIMARY_REGISTRY, OWNER_TEXT)
}

object LicenseCodes {
    const val UNVERIFIED = "LICENSE_UNVERIFIED"
    const val DISALLOWED = "LICENSE_DISALLOWED"
    const val EVIDENCE_INVALID = "LICENSE_EVIDENCE_INVALID"
    const val NOT_GRANTED = "PERMISSION_NOT_GRANTED"
    const val CONDITIONAL = "PERMISSION_CONDITIONAL"
    const val PERM_UNVERIFIED = "PERMISSION_UNVERIFIED"
}

fun Tri.toWire(): String = when (this) { Tri.YES -> "yes"; Tri.NO -> "no"; Tri.CONDITIONAL -> "conditional"; Tri.UNVERIFIED -> "unverified" }
fun triOf(s: String?): Tri = when (s?.lowercase()) { "yes" -> Tri.YES; "no" -> Tri.NO; "conditional" -> Tri.CONDITIONAL; else -> Tri.UNVERIFIED }

/**
 * The license gate, ported from `factory/llmtrainer/licenses.py` (fail closed):
 * a state other than VERIFIED blocks every use and the recorded permission values are ignored (they are claims);
 * a VERIFIED entry must carry complete evidence; then every requested use must be exactly YES
 * (CONDITIONAL / NO / UNVERIFIED all block).
 */
object LicenseGate {
    fun evidenceProblems(l: LicenseInfo): List<String> {
        val p = ArrayList<String>()
        if (l.evidenceLevel !in Evidence.VERIFIED_LEVELS) p.add("VERIFIED requires primary or owner-reviewed evidence, got ${l.evidenceLevel}")
        if (l.textSha256.isNullOrBlank()) p.add("VERIFIED requires the license text sha256")
        if (l.fetchedAt == null) p.add("VERIFIED requires a fetched/verified date")
        if (l.authoritativeUrl.isBlank()) p.add("VERIFIED requires a source URL")
        if (l.scopeText.isNullOrBlank()) p.add("VERIFIED requires a scope note")
        return p
    }

    fun check(l: LicenseInfo, use: IntendedUse): List<Blocker> {
        when (l.state) {
            LicenseState.DISALLOWED -> return listOf(Blocker(LicenseCodes.DISALLOWED,
                "DISALLOWED: known incompatible with the requested use (${l.disallowedReason ?: "no reason recorded"}); recorded permission claims are ignored"))
            LicenseState.UNVERIFIED -> return listOf(Blocker(LicenseCodes.UNVERIFIED,
                "License not verified (evidence: ${l.evidenceLevel}). Fetch or import the license text, read it, and record your attestation; recorded permission claims are ignored until then"))
            LicenseState.VERIFIED -> Unit
        }
        val problems = evidenceProblems(l)
        if (problems.isNotEmpty()) return listOf(Blocker(LicenseCodes.EVIDENCE_INVALID, "Marked VERIFIED but the evidence is inconsistent: ${problems.joinToString("; ")}"))
        val out = ArrayList<Blocker>()
        fun need(on: Boolean, p: Permission, what: String) {
            if (!on) return
            when (l.permissions[p] ?: Tri.UNVERIFIED) {
                Tri.YES -> Unit
                Tri.CONDITIONAL -> out.add(Blocker(LicenseCodes.CONDITIONAL, "$what: permission is conditional; resolve the conditions and re-attest as YES"))
                Tri.NO -> out.add(Blocker(LicenseCodes.NOT_GRANTED, "$what: not permitted by the license"))
                Tri.UNVERIFIED -> out.add(Blocker(LicenseCodes.PERM_UNVERIFIED, "$what: permission unverified (treated as blocked)"))
            }
        }
        need(use.fineTune, Permission.FINE_TUNE, "Fine-tuning")
        need(use.adapter, Permission.ADAPTER, "Creating a derivative adapter")
        need(use.commercial, Permission.COMMERCIAL, "Commercial use")
        need(use.redistribute, Permission.REDISTRIBUTION, "Redistribution")
        return out
    }
}

/** Owner attestation persisted per model in the workspace (also exported inside job/specialist packages). */
data class OwnerAttestation(
    val modelId: String, val state: LicenseState, val url: String, val fetchedAt: Long, val textSha256: String,
    val permissions: Map<Permission, Tri>, val attributionRequired: Boolean, val scopeNote: String,
    val attestedAt: Long, val note: String?, val textBytes: Long, val importedFromFile: Boolean,
    val disallowedReason: String?,
)

private data class PendingFetch(val modelId: String, val url: String, val fetchedAt: Long, val sha: String, val bytes: Long, val file: String?, val imported: Boolean)

class LicenseService(
    private val dir: File, private val registry: () -> Map<String, RegEntry>, private val http: Http, private val clock: Clock,
) {
    private val storeFile = File(dir, "licenses.json")
    private val textDir = File(dir, "license_texts")
    private val attestations = LinkedHashMap<String, OwnerAttestation>()
    private val pending = LinkedHashMap<String, PendingFetch>()
    val problems = ArrayList<String>()

    companion object {
        const val MAX_LICENSE_BYTES = 8L * 1024 * 1024
        const val DISPLAY_CHARS = 200_000
    }

    init { load() }

    @Synchronized private fun load() {
        val o = Fs.readJson(storeFile) { problems.add(it) } ?: return
        o.obj("attestations")?.let { a ->
            for (k in a.keyList()) {
                val r = a.obj(k) ?: continue
                try {
                    attestations[k] = OwnerAttestation(
                        k, LicenseState.valueOf(r.str("state")!!), r.str("url")!!, r.lng("fetched_at")!!, r.str("sha256")!!,
                        permsFrom(r.obj("permissions")), r.bool("attribution_required") ?: false, r.str("scope_note") ?: "",
                        r.lng("attested_at") ?: 0L, r.str("note"), r.lng("text_bytes") ?: 0L, r.bool("imported") ?: false, r.str("disallowed_reason"))
                } catch (e: Exception) { problems.add("license attestation for $k is unreadable and was ignored (treated as UNVERIFIED)") }
            }
        }
        o.obj("pending")?.let { a ->
            for (k in a.keyList()) {
                val r = a.obj(k) ?: continue
                try { pending[k] = PendingFetch(k, r.str("url")!!, r.lng("fetched_at")!!, r.str("sha256")!!, r.lng("bytes") ?: 0L, r.str("file"), r.bool("imported") ?: false) }
                catch (e: Exception) { /* a lost pending fetch only means the owner re-fetches */ }
            }
        }
    }

    private fun permsFrom(o: org.json.JSONObject?): Map<Permission, Tri> =
        Permission.values().associateWith { triOf(o?.str(it.name)) }

    @Synchronized private fun save() {
        val m = linkedMapOf<String, Any?>(
            "schema" to 1,
            "attestations" to attestations.mapValues { (_, a) -> linkedMapOf(
                "state" to a.state.name, "url" to a.url, "fetched_at" to a.fetchedAt, "sha256" to a.textSha256,
                "permissions" to a.permissions.mapKeys { it.key.name }.mapValues { it.value.name },
                "attribution_required" to a.attributionRequired, "scope_note" to a.scopeNote, "attested_at" to a.attestedAt,
                "note" to a.note, "text_bytes" to a.textBytes, "imported" to a.importedFromFile, "disallowed_reason" to a.disallowedReason) },
            "pending" to pending.mapValues { (_, p) -> linkedMapOf("url" to p.url, "fetched_at" to p.fetchedAt, "sha256" to p.sha, "bytes" to p.bytes, "file" to p.file, "imported" to p.imported) },
        )
        // "permissions" values were written by enum name; Tri.valueOf round-trip below
        Fs.writeJson(storeFile, m)
    }

    // ---- effective license ------------------------------------------------------------------------------------

    fun attestationOf(modelId: String): OwnerAttestation? = synchronized(this) { attestations[modelId] }

    fun effective(e: RegEntry): LicenseInfo {
        val v = e.verification
        val url = e.authoritativeUrl
        val claims = mapOf(Permission.COMMERCIAL to triOf(e.commercial), Permission.FINE_TUNE to triOf(e.fineTune),
            Permission.ADAPTER to triOf(e.adapter), Permission.REDISTRIBUTION to triOf(e.redistribution))
        val attr = when (e.attribution.lowercase()) { "yes" -> true; "no" -> false; else -> null }
        val name = e.licenseId ?: "unknown"
        if (v.state == "DISALLOWED") {
            return LicenseInfo(LicenseState.DISALLOWED, name, url, Evidence.PRIMARY_REGISTRY, null, v.textSha256, Iso.parseMs(v.verifiedOn),
                claims, attr, v.disallowedReason ?: "recorded as incompatible in the registry")
        }
        val att = attestationOf(e.entryId)
        if (att != null) {
            return when (att.state) {
                LicenseState.VERIFIED -> LicenseInfo(LicenseState.VERIFIED, name, if (att.importedFromFile) att.url else att.url, Evidence.OWNER_TEXT,
                    att.scopeNote, att.textSha256, att.fetchedAt, att.permissions, att.attributionRequired, null)
                else -> LicenseInfo(LicenseState.DISALLOWED, name, att.url, Evidence.OWNER_ATTESTED_DISALLOWED, null, att.textSha256, att.fetchedAt,
                    att.permissions, att.attributionRequired, att.disallowedReason ?: "Owner attested that fine-tuning/adapters are not permitted")
            }
        }
        val primary = v.evidenceLevel.startsWith("primary_")
        val level = when { primary -> Evidence.PRIMARY_REGISTRY; v.evidenceLevel == "secondary_source_only" -> Evidence.SECONDARY; else -> Evidence.NONE }
        val regInfo = LicenseInfo(
            if (v.state == "VERIFIED") LicenseState.VERIFIED else LicenseState.UNVERIFIED, name, url, level,
            v.scope, v.textSha256, Iso.parseMs(v.verifiedOn), claims, attr, null)
        // Fail closed: a registry VERIFIED entry with incomplete evidence is shown as UNVERIFIED.
        return if (regInfo.state == LicenseState.VERIFIED && LicenseGate.evidenceProblems(regInfo).isNotEmpty()) regInfo.copy(state = LicenseState.UNVERIFIED) else regInfo
    }

    fun gate(e: RegEntry, use: IntendedUse): List<Blocker> = LicenseGate.check(effective(e), use)

    // ---- evidence flow -------------------------------------------------------------------------------------------

    private fun remember(e: RegEntry, url: String, bytes: ByteArray, fileName: String?, imported: Boolean): LicenseTextFetch {
        val sha = Hashing.sha256(bytes)
        textDir.mkdirs()
        val tf = File(textDir, "$sha.txt")
        if (!tf.isFile) Fs.writeBytes(tf, bytes)
        val now = clock.nowMs()
        synchronized(this) {
            pending[e.entryId] = PendingFetch(e.entryId, url, now, sha, bytes.size.toLong(), tf.name, imported)
            save()
        }
        val full = String(bytes, Charsets.UTF_8)
        val truncated = full.length > DISPLAY_CHARS
        return LicenseTextFetch(url, now, sha, if (truncated) full.substring(0, DISPLAY_CHARS) else full, truncated)
    }

    fun fetchText(e: RegEntry): StudioResult<LicenseTextFetch> {
        val url = e.authoritativeUrl
        if (url.isBlank()) return StudioResult.Err(StudioError.Invalid("NO_LICENSE_URL",
            "No authoritative HTTPS license URL is recorded for ${e.entryId}. Import the license file instead."))
        val r = try { http.getBytes(url, MAX_LICENSE_BYTES) } catch (ex: HttpException) {
            return StudioResult.Err(when (ex.kind) {
                HttpException.Kind.NOT_HTTPS -> StudioError.Invalid("NOT_HTTPS", ex.message ?: "HTTPS required")
                else -> StudioError.Network(ex.message ?: "Could not fetch the license text")
            })
        }
        if (r.bytes.isEmpty()) return StudioResult.Err(StudioError.Network("The server returned an empty license text"))
        return try { StudioResult.Ok(remember(e, r.finalUrl, r.bytes, null, false)) }
        catch (ex: java.io.IOException) { StudioResult.Err(StudioError.Io("Could not store the license text: ${ex.message}")) }
    }

    fun importText(e: RegEntry, fileName: String, input: InputStream): StudioResult<LicenseTextFetch> {
        val bytes = try { Fs.readLimited(input, MAX_LICENSE_BYTES) } catch (ex: java.io.IOException) { return StudioResult.Err(StudioError.Io("Could not read the file: ${ex.message}")) }
            ?: return StudioResult.Err(StudioError.Invalid("TOO_LARGE", "License file is larger than ${MAX_LICENSE_BYTES / 1024 / 1024} MB"))
        if (bytes.isEmpty()) return StudioResult.Err(StudioError.Invalid("EMPTY_FILE", "The license file is empty"))
        return try { StudioResult.Ok(remember(e, "file:" + Fs.safeName(fileName), bytes, fileName, true)) }
        catch (ex: java.io.IOException) { StudioResult.Err(StudioError.Io("Could not store the license text: ${ex.message}")) }
    }

    fun attest(e: RegEntry, a: LicenseAttestation): StudioResult<LicenseInfo> {
        val cur = effective(e)
        if (e.verification.state == "DISALLOWED")
            return StudioResult.Err(StudioError.Blocked(LicenseCodes.DISALLOWED, "The registry marks this license DISALLOWED; it cannot be verified here",
                listOf(Blocker(LicenseCodes.DISALLOWED, e.verification.disallowedReason ?: "Disallowed"))))
        val p = synchronized(this) { pending[e.entryId] }
        if (p == null || Hashing.bare(p.sha) != Hashing.bare(a.textSha256))
            return StudioResult.Err(StudioError.Invalid("EVIDENCE_MISMATCH", "Fetch or import the license text first, and attest that exact text (sha256 must match)"))
        val perms = Permission.values().associateWith { a.permissions[it] ?: Tri.UNVERIFIED }
        val tuneNo = perms[Permission.FINE_TUNE] == Tri.NO || perms[Permission.ADAPTER] == Tri.NO
        val missing = perms.filterValues { it == Tri.UNVERIFIED }.keys
        if (!tuneNo && missing.isNotEmpty())
            return StudioResult.Err(StudioError.Invalid("INCOMPLETE_ATTESTATION", "State YES, NO or CONDITIONAL for every permission: ${missing.joinToString()}"))
        val conditional = perms[Permission.FINE_TUNE] == Tri.CONDITIONAL || perms[Permission.ADAPTER] == Tri.CONDITIONAL
        if (!tuneNo && conditional)
            return StudioResult.Err(StudioError.Blocked("CONDITIONAL_PERMISSIONS",
                "Fine-tuning/adapter permission is conditional, so the license cannot be recorded as VERIFIED",
                listOf(Blocker(LicenseCodes.CONDITIONAL, "Resolve the license conditions, then attest YES (or NO if they cannot be met). Nothing was recorded."))))
        val scope = buildString {
            append("owner-reviewed license text from ${p.url}, hash ${Hashing.prefixed(p.sha)}; permissions as attested by owner")
            if (p.imported) append(" (text imported from a file by the owner; not fetched from the registry URL)")
            if (!a.note.isNullOrBlank()) append(". Owner note: ${a.note}")
        }
        val state = if (tuneNo) LicenseState.DISALLOWED else LicenseState.VERIFIED
        val rec = OwnerAttestation(e.entryId, state, p.url, p.fetchedAt, Hashing.bare(p.sha), perms, a.attributionRequired, scope, clock.nowMs(), a.note, p.bytes, p.imported,
            if (tuneNo) "Owner attested that fine-tuning/adapters are not permitted" else null)
        synchronized(this) {
            attestations[e.entryId] = rec
            try { save() } catch (ex: java.io.IOException) { attestations.remove(e.entryId); return StudioResult.Err(StudioError.Io("Could not save the attestation: ${ex.message}")) }
        }
        return StudioResult.Ok(effective(e))
    }

    /** Text the owner read for [modelId] (exact bytes), if still stored. */
    fun storedText(sha: String): ByteArray? { val f = File(textDir, Hashing.bare(sha) + ".txt"); return if (f.isFile) f.readBytes() else null }

    /**
     * Job-package `license_evidence` (PACKAGE_FORMATS 1.1). Only a COMPLETE owner attestation produces one; anything
     * else yields null and the desktop keeps the model UNVERIFIED.
     */
    fun jobEvidence(e: RegEntry): Pair<Map<String, Any?>, ByteArray?>? {
        val att = attestationOf(e.entryId) ?: return null
        if (att.state != LicenseState.VERIFIED) return null
        val url = if (att.url.startsWith("https://")) att.url else e.authoritativeUrl
        if (!url.startsWith("https://")) return null
        if (att.scopeNote.isBlank() || att.permissions.values.any { it == Tri.UNVERIFIED }) return null
        val text = storedText(att.textSha256)
        val ev = linkedMapOf<String, Any?>(
            "model_id" to e.entryId, "license_url" to url, "fetched_at" to Iso.ts(att.fetchedAt),
            "text_sha256" to Hashing.prefixed(att.textSha256), "evidence_level" to "primary_license_text_read",
            "attested_by" to "owner", "attested_on" to Iso.date(att.attestedAt),
            "permissions" to linkedMapOf(
                "commercial_use" to att.permissions.getValue(Permission.COMMERCIAL).toWire(),
                "fine_tuning_permitted" to att.permissions.getValue(Permission.FINE_TUNE).toWire(),
                "derivative_adapter_permitted" to att.permissions.getValue(Permission.ADAPTER).toWire(),
                "redistribution_permitted" to att.permissions.getValue(Permission.REDISTRIBUTION).toWire(),
                "attribution_required" to (if (att.attributionRequired) "yes" else "no")),
            "scope_note" to att.scopeNote,
        )
        return Pair(ev, text)
    }
}
