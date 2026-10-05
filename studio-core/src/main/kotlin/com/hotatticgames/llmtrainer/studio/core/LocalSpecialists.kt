package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.studio.api.*
import java.io.File
import java.io.IOException
import java.io.OutputStream

/** Cached file hashes (big GGUF files are hashed once; the key includes size and mtime so a replaced file is re-hashed). */
class FileHashes(private val cacheFile: File) {
    private val map = HashMap<String, String>()
    private var loaded = false
    @Synchronized private fun load() {
        if (loaded) return
        loaded = true
        Fs.readJson(cacheFile)?.let { o -> o.keyList().forEach { k -> o.str(k)?.let { map[k] = it } } }
    }
    /** [known] = a hash recorded when the file was downloaded/imported (verified then); used instead of re-hashing gigabytes. */
    @Synchronized fun sha256(f: File, known: String? = null): String {
        if (known != null && known.length == 64) return known.lowercase()
        load()
        val key = f.absolutePath + "|" + f.length() + "|" + f.lastModified()
        map[key]?.let { return it }
        val h = Hashing.sha256File(f)
        map[key] = h
        try { Fs.writeJson(cacheFile, map) } catch (_: IOException) { }
        return h
    }
}

class SpecialistRec(
    val id: String, val projectId: ProjectId, val name: String, val version: String, val createdAt: Long,
    val baseModelId: String, val baseVariantId: String?, val baseSha256: String, val patchSha256: String, val patchSize: Long,
    val datasetSha256: String, val runId: String, val method: TrainingMethodKind, val config: TrainingSettings,
    val finalTrainLoss: Double?, val finalValLoss: Double?, val steps: Int, val examples: Long,
    val sources: Map<String, String>, val dir: File,
) {
    @Volatile var verified = false
    @Volatile var verifyMessage = "Not verified yet"
    @Volatile var locallyEvaluated = false
    val patchFile: File get() = File(dir, PATCH_NAME)
    companion object { const val PATCH_NAME = "patch.hagpatch"; const val META = "specialist.json" }
}

/**
 * Per-project specialist artifacts: `projects/<id>/specialists/<spId>/{specialist.json, patch.hagpatch}` + `specialists/registry.json`
 * (selection and the monotonic version counter, so two different artifacts never share an identity even after deletions).
 */
class SpecialistRegistry(
    private val host: LocalHost, private val eng: EngineCore, private val hashes: FileHashes, private val clock: Clock, private val problems: MutableList<String>,
) {
    private val recs = LinkedHashMap<String, SpecialistRec>()
    private val selected = HashMap<String, String>()
    private val seq = HashMap<String, Int>()

    private fun dirOf(pid: ProjectId): File? = host.projectDir(pid)?.let { File(it, "specialists") }

    @Synchronized fun load(pid: ProjectId) {
        val root = dirOf(pid) ?: return
        recs.values.removeAll { it.projectId == pid }
        root.listFiles { f -> f.isDirectory && !f.name.startsWith(".") }?.sortedBy { it.name }?.forEach { d ->
            val o = Fs.readJson(File(d, SpecialistRec.META)) { problems.add("specialist ${d.name}: $it") } ?: return@forEach
            try {
                val r = SpecialistRec(o.str("id") ?: d.name, pid, o.str("name") ?: "Specialist", o.str("version") ?: "0", o.lng("created_at") ?: 0L, o.str("base_model_id") ?: "", o.str("base_variant_id"),
                    o.str("base_sha256") ?: "", o.str("patch_sha256") ?: "", o.lng("patch_size") ?: 0L, o.str("dataset_sha256") ?: "", o.str("run_id") ?: "",
                    LJ_kind(o.str("method")), LJ.settings(o.obj("config")), LJ.d(o, "final_train_loss"), LJ.d(o, "final_val_loss"), o.int("steps") ?: 0, o.lng("examples") ?: 0L,
                    o.obj("sources")?.let { s -> s.keyList().associateWith { s.str(it) ?: "" } } ?: emptyMap(), d)
                r.locallyEvaluated = o.bool("locally_evaluated") ?: false
                r.verified = false; r.verifyMessage = "Not verified since the app started"
                recs[r.id] = r
            } catch (e: Exception) { problems.add("specialist ${d.name} unreadable: ${e.message}") }
        }
        Fs.readJson(File(root, "registry.json"))?.let { o -> o.str("selected")?.let { selected[pid.value] = it }; seq[pid.value] = o.int("seq") ?: 0 }
        if (selected[pid.value]?.let { recs.containsKey(it) } == false) selected.remove(pid.value)
    }

    @Synchronized fun forget(pid: ProjectId) { recs.values.removeAll { it.projectId == pid }; selected.remove(pid.value); seq.remove(pid.value) }

    private fun LJ_kind(s: String?) = TrainingMethodKind.values().firstOrNull { it.name == s } ?: TrainingMethodKind.LOCAL_PARTIAL

    @Synchronized private fun saveRegistry(pid: ProjectId) {
        val root = dirOf(pid) ?: return
        Fs.writeJson(File(root, "registry.json"), linkedMapOf("schema" to 1, "selected" to selected[pid.value], "seq" to (seq[pid.value] ?: 0)))
    }

    @Synchronized private fun saveRec(r: SpecialistRec) {
        Fs.writeJson(File(r.dir, SpecialistRec.META), linkedMapOf("schema" to 1, "id" to r.id, "name" to r.name, "version" to r.version, "created_at" to r.createdAt, "base_model_id" to r.baseModelId,
            "base_variant_id" to r.baseVariantId, "base_sha256" to r.baseSha256, "patch_sha256" to r.patchSha256, "patch_size" to r.patchSize, "dataset_sha256" to r.datasetSha256, "run_id" to r.runId,
            "method" to r.method.name, "config" to LJ.settings(r.config), "final_train_loss" to r.finalTrainLoss, "final_val_loss" to r.finalValLoss, "steps" to r.steps, "examples" to r.examples,
            "sources" to r.sources, "locally_evaluated" to r.locallyEvaluated))
    }

    /** Next unique version string for the project ("1.0.N", N never reused). Reserved before the artifact exists. */
    @Synchronized fun nextVersion(pid: ProjectId): String { val n = (seq[pid.value] ?: 0) + 1; seq[pid.value] = n; saveRegistry(pid); return "1.0.$n" }

    @Synchronized fun newDir(pid: ProjectId, id: String): File = File(dirOf(pid) ?: throw StudioException(StudioError.NotFound("project ${pid.value}")), id).also { it.mkdirs() }

    @Synchronized fun add(r: SpecialistRec, select: Boolean) {
        saveRec(r); recs[r.id] = r
        if (select || selected[r.projectId.value] == null) selected[r.projectId.value] = r.id
        saveRegistry(r.projectId)
    }

    @Synchronized fun get(id: String): SpecialistRec? = recs[id]
    @Synchronized fun selectedId(pid: ProjectId): String? = selected[pid.value]?.takeIf { recs.containsKey(it) }

    @Synchronized fun markEvaluated(id: String) { recs[id]?.let { it.locallyEvaluated = true; saveRec(it) } }

    private fun infoOf(r: SpecialistRec, facts: Map<String, SourceFact>?): SpecialistInfo {
        var stale = false
        var reason: String? = null
        if (facts != null) {
            val gone = r.sources.keys.filter { it !in facts }
            val changed = r.sources.filter { (k, sha) -> facts[k]?.let { it.sha256 != sha || !it.trainable } == true }.keys
            if (gone.isNotEmpty()) { stale = true; reason = "${gone.size} source(s) this specialist was trained on have been removed. Retrain without them; this artifact still contains what it learned from them." }
            else if (changed.isNotEmpty()) { stale = true; reason = "${changed.size} source(s) used for training changed or are no longer cleared for training. Retrain." }
        }
        return SpecialistInfo(r.id, r.projectId, r.name, r.version, r.createdAt, r.baseModelId, r.baseVariantId, r.baseSha256, r.patchSha256, r.patchSize, r.datasetSha256, r.runId, r.method, r.config,
            r.finalTrainLoss, r.finalValLoss, r.steps, r.examples, r.verified, r.verifyMessage, selected[r.projectId.value] == r.id, r.locallyEvaluated,
            "Parameters changed on this device by fine-tuning on your approved dataset (${when (r.method) { TrainingMethodKind.LOCAL_FULL -> "all layers"; TrainingMethodKind.LOCAL_LORA -> "LoRA adapter, rank ${if (r.config.loraRank > 0) r.config.loraRank else DEFAULT_LORA_RANK}"; else -> "last ${r.config.trainableLastLayers} layers" }}). " +
                "The base model file is untouched; this is a patch on top of base ${r.baseSha256.take(12)}.", stale, reason, r.sources.keys.sorted())
    }

    fun list(pid: ProjectId): List<SpecialistInfo> {
        val facts = try { host.sourceFacts(pid) } catch (_: RuntimeException) { null }
        return synchronized(this) { recs.values.filter { it.projectId == pid } }.map { infoOf(it, facts) }
    }

    fun info(id: String): SpecialistInfo? = get(id)?.let { r -> infoOf(r, try { host.sourceFacts(r.projectId) } catch (_: RuntimeException) { null }) }

    /**
     * Re-hash the patch, compare the patch's recorded base hash with the base file, and (engine permitting) reload base+patch.
     * Never throws for a bad artifact: the result says what is wrong.
     */
    fun verify(id: String): SpecialistInfo {
        val r = get(id) ?: throw StudioException(StudioError.NotFound("specialist $id"))
        var ok = false
        val msg: String = try {
            val base = host.baseRef(r.projectId)
            val baseFile = base?.file
            when {
                !r.patchFile.isFile -> "Patch file is missing"
                Hashing.sha256File(r.patchFile) != r.patchSha256 -> "Patch file hash does not match the registry (file changed or damaged)"
                baseFile == null || !baseFile.isFile -> "Base model is not installed, so the patch cannot be checked against it"
                hashes.sha256(baseFile, base.sha256) != r.baseSha256 -> "The installed base model is not the one this specialist was trained on (hash differs)"
                else -> {
                    val pi = eng.trainer?.takeIf { it.status().available }?.patchInfo(r.patchFile.absolutePath)
                    if (pi != null && pi.baseSha256 != r.baseSha256) "Patch metadata names a different base model than the registry"
                    else if (!eng.inferenceStatus().available) "Hashes match, but the engine is not available to reload the patch"
                    else {
                        eng.withModel(baseFile, r.patchFile, 512, keepLoaded = false) { it.info.patched }.let { patched ->
                            if (patched) { ok = true; "Patch hash, base hash and structure verified; reloaded by the engine" } else "The engine loaded the model but did not report the patch as applied"
                        }
                    }
                }
            }
        } catch (e: BackendException) { "Reload failed: ${e.message}" }
        r.verified = ok; r.verifyMessage = msg
        return info(id)!!
    }

    fun select(pid: ProjectId, id: String?): List<SpecialistInfo> {
        if (host.projectDir(pid) == null) throw StudioException(StudioError.NotFound("project ${pid.value}"))
        synchronized(this) {
            if (id == null) selected.remove(pid.value)
            else {
                val r = recs[id]
                if (r == null || r.projectId != pid) throw StudioException(StudioError.NotFound("specialist $id"))
                selected[pid.value] = id
            }
            saveRegistry(pid)
        }
        return list(pid)
    }

    fun delete(id: String) {
        val r = get(id) ?: throw StudioException(StudioError.NotFound("specialist $id"))
        synchronized(this) {
            recs.remove(id)
            if (selected[r.projectId.value] == id) { selected.remove(r.projectId.value); saveRegistry(r.projectId) }
        }
        r.dir.deleteRecursively()
    }

    /** Portable zip: patch + manifest + checksums.json. Never contains the base model. */
    fun export(id: String, out: OutputStream, extra: Map<String, Any?>): ExportedPackage {
        val r = get(id) ?: throw StudioException(StudioError.NotFound("specialist $id"))
        if (!r.patchFile.isFile || Hashing.sha256File(r.patchFile) != r.patchSha256) throw StudioException(StudioError.Invalid("PATCH_DAMAGED", "The patch file is missing or does not match its recorded hash; not exporting a damaged artifact."))
        val i = info(id)!!
        val manifest = linkedMapOf<String, Any?>("format" to "llmtrainer-specialist-patch", "version" to 1, "specialist_id" to r.id, "name" to r.name, "specialist_version" to r.version, "created_at" to Iso.ts(r.createdAt),
            "kind" to "parameter-patch", "statement" to i.statement,
            "base_model" to linkedMapOf("model_id" to r.baseModelId, "variant_id" to r.baseVariantId, "sha256" to r.baseSha256, "note" to "The base model is NOT included; obtain it separately under its own license."),
            "patch" to linkedMapOf("file" to SpecialistRec.PATCH_NAME, "sha256" to r.patchSha256, "size" to r.patchSize), "dataset_sha256" to r.datasetSha256,
            "training" to linkedMapOf("run_id" to r.runId, "method" to r.method.name, "config" to LJ.settings(r.config), "final_train_loss" to r.finalTrainLoss, "final_val_loss" to r.finalValLoss, "steps" to r.steps, "examples" to r.examples),
            "source_ids" to r.sources.keys.sorted(), "source_sha256" to r.sources, "stale" to i.stale, "stale_reason" to i.staleReason, "locally_evaluated" to r.locallyEvaluated) + extra
        // Streamed (the patch can be hundreds of MB; never held in memory).
        val manifestBytes = J.bytes(manifest) + "\n".toByteArray()
        val checks = J.bytes(linkedMapOf("algorithm" to "sha256", "files" to linkedMapOf("manifest.json" to Hashing.prefixed(Hashing.sha256(manifestBytes)),
            SpecialistRec.PATCH_NAME to Hashing.prefixed(r.patchSha256))), true) + "\n".toByteArray()
        val md = java.security.MessageDigest.getInstance("SHA-256")
        var size = 0L
        val counting = object : OutputStream() {
            override fun write(b: Int) { out.write(b); md.update(b.toByte()); size++ }
            override fun write(b: ByteArray, off: Int, len: Int) { out.write(b, off, len); md.update(b, off, len); size += len }
            override fun flush() = out.flush()
        }
        val zos = java.util.zip.ZipOutputStream(counting)
        fun entry(name: String) = java.util.zip.ZipEntry(name).also { it.time = 315532800000L; zos.putNextEntry(it) }
        entry("checksums.json"); zos.write(checks); zos.closeEntry()
        entry("manifest.json"); zos.write(manifestBytes); zos.closeEntry()
        entry(SpecialistRec.PATCH_NAME); r.patchFile.inputStream().use { it.copyTo(zos) }; zos.closeEntry()
        zos.finish(); zos.flush(); out.flush()
        val wr = object { val sizeBytes = size; val sha256 = Hashing.hex(md.digest()); val files = listOf("checksums.json", "manifest.json", SpecialistRec.PATCH_NAME) }
        val warnings = ArrayList<String>()
        if (i.stale) warnings.add(i.staleReason ?: "stale")
        if (!i.verified) warnings.add("Not verified on this run: ${i.verifyMessage}")
        return ExportedPackage("specialist-patch", "${Text.slugify(r.name)}-${r.version}.patch.zip", wr.sizeBytes, wr.sha256, wr.files, warnings)
    }
}
