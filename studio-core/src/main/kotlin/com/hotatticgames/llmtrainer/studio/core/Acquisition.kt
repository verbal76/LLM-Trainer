package com.hotatticgames.llmtrainer.studio.core

import com.hotatticgames.llmtrainer.qualify.DeviceSnapshot
import com.hotatticgames.llmtrainer.qualify.Qualifier
import com.hotatticgames.llmtrainer.qualify.SafetyPolicy
import com.hotatticgames.llmtrainer.studio.api.AcquisitionPlan
import com.hotatticgames.llmtrainer.studio.api.Blocker
import com.hotatticgames.llmtrainer.studio.api.IntendedUse
import com.hotatticgames.llmtrainer.studio.api.LicenseState
import com.hotatticgames.llmtrainer.studio.api.Operation
import com.hotatticgames.llmtrainer.studio.api.OperationKind
import com.hotatticgames.llmtrainer.studio.api.OperationState
import com.hotatticgames.llmtrainer.studio.api.ProjectId
import com.hotatticgames.llmtrainer.studio.api.Progress
import com.hotatticgames.llmtrainer.studio.api.SourceInput
import com.hotatticgames.llmtrainer.studio.api.StudioError
import com.hotatticgames.llmtrainer.studio.api.StudioResult
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Model acquisition: plan, resumable HTTPS download (Range + `.part`), import from a stream. Never automatic: a download
 * needs a VERIFIED license that grants the intended use AND explicit confirmation. Operations are persisted one file each
 * under workspace/operations/ and reloaded after process death (downloads as PAUSED+resumable; imports as FAILED because
 * the picked stream cannot be reopened).
 */
class AcquisitionService(
    private val wsDir: File, private val catalog: CatalogService, private val licenses: LicenseService, private val files: ModelFiles,
    private val http: Http, private val clock: Clock, private val storage: StorageProbe, private val runner: TaskRunner,
    private val ids: IdSource, private val policy: SafetyPolicy, private val snapshot: () -> String,
) {
    private val opsDir = File(wsDir, "operations")
    private val lock = Any()
    private val ops = LinkedHashMap<String, Op>()
    private val cancelFlags = ConcurrentHashMap<String, AtomicBoolean>()
    private val importInputs = ConcurrentHashMap<String, SourceInput>()
    val problems = ArrayList<String>()

    private class Op(
        var op: Operation, val variantId: String, val url: String?, val use: IntendedUse, val expectedSize: Long?, val expectedSha: String?,
        val fileName: String,
    )

    companion object {
        const val MIB = 1024L * 1024L
        const val GIB = 1024L * MIB
        const val MAX_DOWNLOAD = 64L * GIB
    }

    init { load() }

    // ---- persistence ----------------------------------------------------------------------------------------------

    private fun load() {
        val list = opsDir.listFiles { f -> f.isFile && f.name.endsWith(".json") } ?: return
        pruneOld(list)
        for (f in list.sortedBy { it.name }) {
            val o = Fs.readJson(f) { problems.add(it) } ?: continue
            try {
                val kind = OperationKind.valueOf(o.str("kind")!!)
                var state = OperationState.valueOf(o.str("state")!!)
                var msg = o.str("message") ?: ""
                var err: StudioError? = o.obj("error")?.let { errFrom(it) }
                var resumable = o.bool("resumable") ?: false
                val id = o.str("id")!!
                var dirty = false
                if (state == OperationState.RUNNING || state == OperationState.QUEUED) {
                    dirty = true
                    if (kind == OperationKind.DOWNLOAD) { state = OperationState.PAUSED; resumable = true; msg = "Interrupted when the app closed; resume to continue" }
                    else { state = OperationState.FAILED; resumable = false; msg = "Interrupted when the app closed; choose the file again"; err = StudioError.Io(msg) }
                }
                val done = o.lng("done") ?: 0L
                val part = partFile(o.str("variant_id")!!, o.str("file_name")!!)
                val progressDone = if (kind == OperationKind.DOWNLOAD && part.isFile && state == OperationState.PAUSED) part.length() else done
                val use = o.obj("use")?.let { IntendedUse(it.bool("fine_tune") ?: true, it.bool("adapter") ?: true, it.bool("commercial") ?: false, it.bool("redistribute") ?: false) } ?: IntendedUse()
                val op = Operation(id, kind, o.str("project_id")?.let(::ProjectId), o.str("subject") ?: "", state,
                    Progress(progressDone, o.lng("total") ?: -1L, "bytes"), msg, err, resumable, o.lng("created_at") ?: 0L, o.lng("updated_at") ?: 0L)
                val rec = Op(op, o.str("variant_id")!!, o.str("url"), use, o.lng("expected_size"), o.str("expected_sha"), o.str("file_name")!!)
                ops[id] = rec
                if (dirty) persist(rec)
            } catch (e: Exception) { problems.add("operation file ${f.name} is unreadable and was ignored") }
        }
    }

    /** Finished operations older than 30 days (or beyond the newest 100) are forgotten so the list stays useful. */
    private fun pruneOld(files: Array<File>) {
        val finished = files.mapNotNull { f ->
            val o = try { org.json.JSONObject(f.readText()) } catch (e: Exception) { return@mapNotNull null }
            val st = o.str("state") ?: return@mapNotNull null
            if (st == "SUCCEEDED" || st == "CANCELLED" || (st == "FAILED" && o.bool("resumable") != true)) Pair(f, o.lng("updated_at") ?: 0L) else null
        }.sortedByDescending { it.second }
        val cutoff = clock.nowMs() - 30L * 24 * 3600 * 1000
        finished.forEachIndexed { i, (f, at) -> if (i >= 100 || at < cutoff) { f.delete(); File(f.path + ".bak").delete() } }
    }

    private fun errFrom(o: org.json.JSONObject): StudioError = when (o.str("code")) {
        "NETWORK_ERROR" -> StudioError.Network(o.str("message") ?: "")
        "CANCELLED" -> StudioError.Cancelled()
        "CONFLICT" -> StudioError.Conflict(o.str("message") ?: "")
        "NOT_FOUND" -> StudioError.NotFound(o.str("message")?.removePrefix("Not found: ") ?: "")
        "IO_ERROR" -> StudioError.Io(o.str("message") ?: "")
        else -> StudioError.Invalid(o.str("code") ?: "ERROR", o.str("message") ?: "")
    }

    private fun persist(r: Op) {
        val o = r.op
        try {
            Fs.writeJson(File(opsDir, "${o.id}.json"), linkedMapOf(
                "schema" to 1, "id" to o.id, "kind" to o.kind.name, "state" to o.state.name, "project_id" to o.projectId?.value, "subject" to o.subject,
                "done" to o.progress.done, "total" to o.progress.total, "message" to o.message,
                "error" to o.error?.let { mapOf("code" to it.code, "message" to it.message) }, "resumable" to o.resumable,
                "created_at" to o.createdAt, "updated_at" to o.updatedAt, "variant_id" to r.variantId, "url" to r.url,
                "use" to mapOf("fine_tune" to r.use.fineTune, "adapter" to r.use.adapter, "commercial" to r.use.commercial, "redistribute" to r.use.redistribute),
                "expected_size" to r.expectedSize, "expected_sha" to r.expectedSha, "file_name" to r.fileName))
        } catch (e: IOException) { problems.add("could not persist operation ${o.id}: ${e.message}") }
    }

    private fun update(id: String, force: Boolean = true, f: (Operation) -> Operation): Operation = synchronized(lock) {
        val r = ops.getValue(id)
        r.op = f(r.op).copy(updatedAt = clock.nowMs())
        if (force) persist(r)
        r.op
    }

    private fun partFile(variantId: String, fileName: String) = File(files.dirOf(variantId), "$fileName.part")

    // ---- planning ---------------------------------------------------------------------------------------------------

    private fun reserveBytes(): Long {
        val p = DeviceSnapshot.profileFromSnapshot(snapshot()).profile
        return if (p != null) (Qualifier.storageReserveMb(p, policy) * MIB).toLong() else 2 * GIB
    }

    private fun freeBytes(): Long {
        val probe = storage.freeBytes(files.dir)
        val snap = DeviceSnapshot.profileFromSnapshot(snapshot()).profile?.let { it.storageFreeMb * MIB }
        return if (snap != null && snap > 0) minOf(probe, snap).let { if (probe > 0) it else snap } else probe
    }

    fun plan(variantId: String, use: IntendedUse): StudioResult<AcquisitionPlan> {
        val (e, v) = catalog.findVariant(variantId) ?: return StudioResult.Err(StudioError.NotFound("variant $variantId"))
        val view = catalog.variantView(e, v)
        val lic = licenses.effective(e)
        val blocking = ArrayList<Blocker>()
        blocking.addAll(licenses.gate(e, use))
        val url = v.directUrl
        when {
            v.sourceUrl == null -> blocking.add(Blocker("NO_DOWNLOAD_URL", "No source URL is recorded for this variant; import the model file instead"))
            !v.sourceUrl.startsWith("https://") -> blocking.add(Blocker("NOT_HTTPS", "Only HTTPS downloads are allowed"))
            url == null -> blocking.add(Blocker("NO_DOWNLOAD_URL", "The recorded source is a repository page, not a downloadable file; import the model file instead"))
        }
        val size = view.sizeBytes
        val free = freeBytes()
        val reserve = reserveBytes()
        val partial = (view.sizeBytes).let { if (url != null) partFile(variantId, fileNameFor(variantId, url)).let { p -> if (p.isFile) p.length() else 0L } else 0L }
        val after = free - (size - partial)
        if (after < reserve) blocking.add(Blocker("INSUFFICIENT_STORAGE",
            "Not enough free storage: after this download ${fmtBytes(maxOf(after, 0))} would remain, below the ${fmtBytes(reserve)} safety reserve"))
        if (files.isAcquired(variantId)) blocking.add(Blocker("ALREADY_ACQUIRED", "This variant is already on the device"))
        val compat = view.android.let { f -> if (view.provenance.evidenceLevel == "estimate") f.copy(reasons = f.reasons + "Download size shown is an ESTIMATE; the registry does not record the real size.") else f }
        return StudioResult.Ok(AcquisitionPlan(variantId, e.entryId.substringAfter('@'), url, size, free, after, reserve, compat, lic.state, use, blocking.isEmpty(), blocking, true))
    }

    private fun fmtBytes(b: Long) = if (b >= GIB) "%.2f GiB".format(b.toDouble() / GIB) else "%.1f MiB".format(b.toDouble() / MIB)

    private fun fileNameFor(variantId: String, url: String): String =
        Fs.safeName(url.substringBefore('?').substringAfterLast('/').ifBlank { variantId })

    // ---- download -----------------------------------------------------------------------------------------------------

    fun startDownload(variantId: String, use: IntendedUse, confirmed: Boolean): StudioResult<Operation> {
        val plan = when (val p = plan(variantId, use)) { is StudioResult.Ok -> p.value; is StudioResult.Err -> return p }
        val (_, v) = catalog.findVariant(variantId)!!
        val benign = plan.blocking.filter { it.code != "ALREADY_ACQUIRED" }
        if (plan.blocking.any { it.code == "ALREADY_ACQUIRED" } && benign.isEmpty()) return StudioResult.Err(StudioError.Conflict("This model file is already on the device"))
        if (!plan.allowed) return StudioResult.Err(StudioError.Blocked("DOWNLOAD_BLOCKED", "Download is blocked", plan.blocking))
        if (!confirmed) return StudioResult.Err(StudioError.Blocked("CONFIRMATION_REQUIRED", "Explicit confirmation is required",
            listOf(Blocker("CONFIRMATION_REQUIRED", "Confirm size, storage impact and license before downloading"))))
        val url = plan.url!!
        val rec = synchronized(lock) {
            if (ops.values.any { it.op.kind == OperationKind.DOWNLOAD && it.variantId == variantId && !it.op.isTerminal })
                return StudioResult.Err(StudioError.Conflict("A download for this variant is already in progress"))
            val id = "op-" + ids.hex(10)
            val now = clock.nowMs()
            val op = Operation(id, OperationKind.DOWNLOAD, null, variantId, OperationState.QUEUED, Progress(0, v.sizeBytes ?: -1L, "bytes"), "Queued", null, true, now, now)
            Op(op, variantId, url, use, v.sizeBytes, v.sha256, fileNameFor(variantId, url)).also { ops[id] = it; persist(it) }
        }
        cancelFlags[rec.op.id] = AtomicBoolean(false)
        runner.submit { runDownload(rec.op.id) }
        return StudioResult.Ok(synchronized(lock) { ops.getValue(rec.op.id).op })
    }

    private fun runDownload(opId: String) {
        val r = synchronized(lock) { ops[opId] } ?: return
        val flag = cancelFlags.getOrPut(opId) { AtomicBoolean(false) }
        if (flag.get()) return
        val dir = files.dirOf(r.variantId)
        val part = File(dir, r.fileName + ".part")
        try {
            dir.mkdirs()
            update(opId) { it.copy(state = OperationState.RUNNING, message = if (part.length() > 0) "Resuming from ${fmtBytes(part.length())}" else "Starting download", error = null, resumable = true) }
            val reserve = reserveBytes()
            val remaining = (r.expectedSize ?: 0L) - part.length()
            if (r.expectedSize != null && storage.freeBytes(dir) - remaining < reserve) {
                update(opId) { it.copy(state = OperationState.FAILED, error = StudioError.Io("Not enough free storage to finish this download while keeping the safety reserve"), message = "Not enough free storage", resumable = part.isFile) }
                return
            }
            var lastPersist = 0L
            val out = http.downloadToFile(r.url!!, part, r.expectedSize?.let { it + MIB } ?: MAX_DOWNLOAD, { flag.get() }, { done, total ->
                val now = clock.nowMs()
                val force = now - lastPersist > 750
                if (force) lastPersist = now
                update(opId, force) { it.copy(progress = Progress(done, if (total >= 0) total else it.progress.total, "bytes"), message = "Downloading") }
            })
            if (r.expectedSize != null && out.bytesOnDisk != r.expectedSize) {
                part.delete()
                update(opId) { it.copy(state = OperationState.FAILED, error = StudioError.Invalid("SIZE_MISMATCH", "Downloaded ${out.bytesOnDisk} bytes but ${r.expectedSize} were expected; the partial file was removed"), message = "Size mismatch", resumable = false) }
                return
            }
            update(opId) { it.copy(message = "Verifying checksum") }
            val sha = Hashing.sha256File(part)
            if (r.expectedSha != null && r.expectedSha != sha) {
                part.delete()
                update(opId) { it.copy(state = OperationState.FAILED, error = StudioError.Invalid("HASH_MISMATCH", "The downloaded file does not match the recorded checksum; it was removed"), message = "Checksum mismatch", resumable = false) }
                return
            }
            val final = File(dir, r.fileName)
            java.nio.file.Files.move(part.toPath(), final.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING)
            Fs.writeJson(files.acquiredRecord(r.variantId), linkedMapOf("schema" to 1, "file" to final.name, "sha256" to sha, "size" to final.length(),
                "source" to "download", "url" to r.url, "at" to clock.nowMs(), "checksum_verified_against_registry" to (r.expectedSha != null)))
            update(opId) { it.copy(state = OperationState.SUCCEEDED, progress = Progress(final.length(), final.length(), "bytes"), message = "Downloaded" + if (r.expectedSha == null) " (no registry checksum to verify against; sha256 recorded)" else " and verified", resumable = false, error = null) }
        } catch (e: HttpException) {
            if (e.kind == HttpException.Kind.CANCELLED || flag.get()) {
                part.delete()
                update(opId) { it.copy(state = OperationState.CANCELLED, resumable = false, message = "Cancelled; partial file removed", error = null) }
            } else {
                val keep = part.isFile && e.kind != HttpException.Kind.NOT_HTTPS && e.kind != HttpException.Kind.TOO_LARGE
                if (!keep) part.delete()
                update(opId) { it.copy(state = OperationState.FAILED, error = StudioError.Network(e.message ?: "Network error"), message = e.message ?: "Network error", resumable = keep) }
            }
        } catch (e: IOException) {
            update(opId) { it.copy(state = OperationState.FAILED, error = StudioError.Io(e.message ?: "I/O error"), message = e.message ?: "I/O error", resumable = part.isFile) }
        } catch (e: RuntimeException) {
            update(opId) { it.copy(state = OperationState.FAILED, error = StudioError.Io("Unexpected error: ${e.javaClass.simpleName}: ${e.message}"), message = "Unexpected error", resumable = part.isFile) }
        } finally { importInputs.remove(opId) }
    }

    // ---- import -----------------------------------------------------------------------------------------------------------

    fun importModel(variantId: String, input: SourceInput): StudioResult<Operation> {
        val (e, v) = catalog.findVariant(variantId) ?: return StudioResult.Err(StudioError.NotFound("variant $variantId"))
        if (licenses.effective(e).state == LicenseState.DISALLOWED)
            return StudioResult.Err(StudioError.Blocked("LICENSE_DISALLOWED", "This model's license is DISALLOWED for the intended use",
                listOf(Blocker(LicenseCodes.DISALLOWED, licenses.effective(e).disallowedReason ?: "Disallowed"))))
        if (input.sizeBytes == 0L) return StudioResult.Err(StudioError.Invalid("EMPTY_FILE", "The chosen file is empty"))
        if (input.sizeBytes > 0 && storage.freeBytes(files.dir) - input.sizeBytes < reserveBytes())
            return StudioResult.Err(StudioError.Blocked("INSUFFICIENT_STORAGE", "Not enough free storage to import this file and keep the safety reserve",
                listOf(Blocker("INSUFFICIENT_STORAGE", "File is ${fmtBytes(input.sizeBytes)}; free ${fmtBytes(storage.freeBytes(files.dir))}; reserve ${fmtBytes(reserveBytes())}"))))
        val fileName = Fs.safeName(input.name)
        val rec = synchronized(lock) {
            if (ops.values.any { it.op.kind == OperationKind.IMPORT_MODEL && it.variantId == variantId && !it.op.isTerminal })
                return StudioResult.Err(StudioError.Conflict("An import for this variant is already in progress"))
            val id = "op-" + ids.hex(10)
            val now = clock.nowMs()
            val op = Operation(id, OperationKind.IMPORT_MODEL, null, variantId, OperationState.QUEUED, Progress(0, input.sizeBytes, "bytes"), "Queued", null, false, now, now)
            Op(op, variantId, null, IntendedUse(), v.sizeBytes, v.sha256, fileName).also { ops[id] = it; persist(it) }
        }
        cancelFlags[rec.op.id] = AtomicBoolean(false)
        importInputs[rec.op.id] = input
        runner.submit { runImport(rec.op.id) }
        return StudioResult.Ok(synchronized(lock) { ops.getValue(rec.op.id).op })
    }

    private fun runImport(opId: String) {
        val r = synchronized(lock) { ops[opId] } ?: return
        val input = importInputs[opId] ?: return
        val flag = cancelFlags.getOrPut(opId) { AtomicBoolean(false) }
        val dir = files.dirOf(r.variantId)
        val part = File(dir, r.fileName + ".part")
        try {
            dir.mkdirs()
            update(opId) { it.copy(state = OperationState.RUNNING, message = "Copying ${input.name}") }
            var done = 0L
            var last = 0L
            val md = java.security.MessageDigest.getInstance("SHA-256")
            input.open().use { ins ->
                FileOutputStream(part).use { fos ->
                    val buf = ByteArray(1 shl 16)
                    while (true) {
                        if (flag.get()) throw HttpException(HttpException.Kind.CANCELLED, "Cancelled")
                        val n = ins.read(buf)
                        if (n < 0) break
                        fos.write(buf, 0, n); md.update(buf, 0, n); done += n
                        val now = clock.nowMs()
                        if (now - last > 750) { last = now; update(opId) { it.copy(progress = Progress(done, it.progress.total, "bytes")) } }
                    }
                    try { fos.fd.sync() } catch (_: Exception) { }
                }
            }
            if (done == 0L) { part.delete(); update(opId) { it.copy(state = OperationState.FAILED, error = StudioError.Invalid("EMPTY_FILE", "The chosen file is empty"), message = "Empty file", resumable = false) }; return }
            val sha = Hashing.hex(md.digest())
            if (r.expectedSha != null && r.expectedSha != sha) {
                part.delete()
                update(opId) { it.copy(state = OperationState.FAILED, error = StudioError.Invalid("HASH_MISMATCH", "The file does not match the checksum recorded for this variant; it was not imported"), message = "Checksum mismatch", resumable = false) }
                return
            }
            val final = File(dir, r.fileName)
            java.nio.file.Files.move(part.toPath(), final.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING)
            Fs.writeJson(files.acquiredRecord(r.variantId), linkedMapOf("schema" to 1, "file" to final.name, "sha256" to sha, "size" to done, "source" to "import",
                "original_name" to input.name, "at" to clock.nowMs(), "checksum_verified_against_registry" to (r.expectedSha != null)))
            update(opId) { it.copy(state = OperationState.SUCCEEDED, progress = Progress(done, done, "bytes"), message = "Imported ${input.name}" + if (r.expectedSha == null) " (no registry checksum to verify against; sha256 recorded)" else " and verified", resumable = false, error = null) }
        } catch (e: HttpException) {
            part.delete()
            update(opId) { it.copy(state = OperationState.CANCELLED, resumable = false, message = "Cancelled; partial file removed") }
        } catch (e: IOException) {
            part.delete()
            update(opId) { it.copy(state = OperationState.FAILED, error = StudioError.Io(e.message ?: "I/O error"), message = e.message ?: "I/O error", resumable = false) }
        } finally { importInputs.remove(opId) }
    }

    // ---- common operations API ----------------------------------------------------------------------------------------------

    fun operations(): List<Operation> = synchronized(lock) { ops.values.map { it.op }.sortedBy { it.createdAt } }
    fun operation(id: String): StudioResult<Operation> = synchronized(lock) { ops[id]?.op }?.let { StudioResult.Ok(it) } ?: StudioResult.Err(StudioError.NotFound("operation $id"))

    fun cancel(id: String): StudioResult<Operation> {
        val r = synchronized(lock) { ops[id] } ?: return StudioResult.Err(StudioError.NotFound("operation $id"))
        // A FAILED download that kept a resumable partial file may still be cancelled (discarding the partial).
        if (r.op.isTerminal && !(r.op.state == OperationState.FAILED && r.op.resumable)) return StudioResult.Err(StudioError.Conflict("Operation already finished"))
        cancelFlags.getOrPut(id) { AtomicBoolean(false) }.set(true)
        // A task that is not currently running (queued, paused, failed-resumable) is cancelled here; a running one notices the flag.
        val running = synchronized(lock) { r.op.state == OperationState.RUNNING }
        if (!running) {
            partFile(r.variantId, r.fileName).delete()
            return StudioResult.Ok(update(id) { it.copy(state = OperationState.CANCELLED, resumable = false, message = "Cancelled; partial file removed", error = null) })
        }
        return StudioResult.Ok(synchronized(lock) { ops.getValue(id).op })
    }

    fun resume(id: String): StudioResult<Operation> {
        val r = synchronized(lock) { ops[id] } ?: return StudioResult.Err(StudioError.NotFound("operation $id"))
        val st = r.op.state
        if (r.op.kind != OperationKind.DOWNLOAD) return StudioResult.Err(StudioError.Conflict("Only downloads can resume; choose the file again to retry an import"))
        if (!(st == OperationState.PAUSED || (st == OperationState.FAILED && r.op.resumable))) return StudioResult.Err(StudioError.Conflict("Only paused or failed (resumable) operations can resume"))
        val plan = when (val p = plan(r.variantId, r.use)) { is StudioResult.Ok -> p.value; is StudioResult.Err -> return p }
        val blocking = plan.blocking.filter { it.code != "ALREADY_ACQUIRED" }
        if (blocking.isNotEmpty()) return StudioResult.Err(StudioError.Blocked("DOWNLOAD_BLOCKED", "Download is blocked", blocking))
        cancelFlags[id] = AtomicBoolean(false)
        update(id) { it.copy(state = OperationState.QUEUED, message = "Queued to resume", error = null) }
        runner.submit { runDownload(id) }
        return StudioResult.Ok(synchronized(lock) { ops.getValue(id).op })
    }
}
