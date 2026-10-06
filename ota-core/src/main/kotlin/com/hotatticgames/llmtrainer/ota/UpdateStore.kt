package com.hotatticgames.llmtrainer.ota

import kotlinx.serialization.Serializable
import java.io.File

@Serializable
data class SlotInfo(
    val id: String,
    val bundleVersion: Int,
    val versionName: String,
    val sha256: String,
    val entryClass: String,
    val installedAtMs: Long,
    /** Boots started with this slot that never reached markHealthy(). */
    val bootsSinceHealthy: Int = 0,
    val everHealthy: Boolean = false,
    val files: List<FileEntry> = emptyList(),
    /** Native ABI the bundle was built for. Slots written by v1 hosts predate this field and are ABI 1 by definition. */
    val nativeAbi: Int = 1,
)

@Serializable
data class HistoryEvent(val atMs: Long, val event: String, val detail: String = "")

@Serializable
data class StoreState(
    val schema: Int = STATE_SCHEMA,
    val active: String? = null,
    val lastKnownGood: String? = null,
    val pending: String? = null,
    val slots: Map<String, SlotInfo> = emptyMap(),
    val quarantined: Map<String, String> = emptyMap(),
    val history: List<HistoryEvent> = emptyList(),
)

sealed interface BootPlan {
    /** Run the bundle embedded in the APK. Always available, always the last resort. */
    data class Builtin(val reason: String) : BootPlan
    data class Slot(val slot: SlotInfo, val dir: File, val trial: Boolean) : BootPlan
}

/**
 * Durable bundle store + boot/rollback state machine. Pure java.io so it is unit-tested on the JVM.
 *
 * Boot order: pending (trial) -> active -> lastKnownGood -> builtin.
 * A slot is only trusted as "good" after the running bundle calls [markHealthy]. The boot counter is
 * persisted BEFORE the bundle is loaded, so a hard crash / process kill counts against the slot.
 */
class UpdateStore(
    private val root: File,
    private val clock: () -> Long = { System.currentTimeMillis() },
) {
    companion object {
        const val MAX_BOOTS_TRIAL = 2
        const val MAX_BOOTS_ACTIVE = 3
        const val MAX_HISTORY = 60
    }

    private val stateFile = File(root, "state.json")
    private val bundlesDir = File(root, "bundles")

    @Synchronized
    fun load(): StoreState {
        if (!stateFile.exists()) return StoreState()
        return try {
            OtaJson.decodeFromString(StoreState.serializer(), stateFile.readText())
        } catch (_: Exception) {
            // Corrupt state must never brick the app: reset to builtin-only and say so.
            StoreState(history = listOf(HistoryEvent(clock(), "STATE_CORRUPT_RESET")))
        }
    }

    @Synchronized
    private fun save(s: StoreState) {
        root.mkdirs()
        val tmp = File(root, "state.json.tmp")
        val trimmed = s.copy(history = s.history.takeLast(MAX_HISTORY))
        tmp.writeText(OtaJson.encodeToString(StoreState.serializer(), trimmed))
        if (!tmp.renameTo(stateFile)) {
            stateFile.delete()
            check(tmp.renameTo(stateFile)) { "cannot persist update state" }
        }
    }

    private fun StoreState.log(event: String, detail: String = "") =
        copy(history = history + HistoryEvent(clock(), event, detail))

    fun slotDir(id: String) = File(bundlesDir, id)

    /** Highest bundle version we already hold or run; a new bundle must beat it. */
    @Synchronized
    fun highestKnownVersion(builtin: Int): Int {
        val s = load()
        return maxOf(builtin, s.slots.values.maxOfOrNull { it.bundleVersion } ?: 0)
    }

    /**
     * Drop anything older than the bundle embedded in a (newly installed) APK, so replacing the APK
     * can never leave an older OTA bundle shadowing the newer built-in one.
     */
    @Synchronized
    fun reconcileBuiltin(builtinVersion: Int, hostNativeAbi: Int? = null) {
        var s = load()
        // Also drop slots built for another native runtime (e.g. an abi-1 slot left by a v1 APK that was upgraded
        // in place to an abi-2 APK): exact-ABI semantics must hold for what is already on disk, not only downloads.
        val stale = s.slots.values.filter {
            it.bundleVersion <= builtinVersion || (hostNativeAbi != null && it.nativeAbi != hostNativeAbi)
        }.map { it.id }
        if (stale.isEmpty()) return
        for (id in stale) {
            slotDir(id).deleteRecursively()
            s = s.copy(
                slots = s.slots - id,
                active = s.active.takeUnless { it == id },
                lastKnownGood = s.lastKnownGood.takeUnless { it == id },
                pending = s.pending.takeUnless { it == id },
            ).log("SUPERSEDED_BY_BUILTIN", id)
        }
        save(s)
    }

    /** Install a verified bundle as the pending (trial) slot. Takes effect on next launch. */
    @Synchronized
    fun stage(v: BundleFormat.Verified, bundleSha256: String, builtinVersion: Int, hostNativeAbi: Int? = null): Result<SlotInfo> {
        val m = v.manifest
        if (hostNativeAbi != null && m.requires.nativeAbi != hostNativeAbi) {
            return Result.failure(RejectedException(Reject(RejectCode.NATIVE_ABI_MISMATCH, "bundle ${m.requires.nativeAbi} != host $hostNativeAbi")))
        }
        val s0 = load()
        val id = "v${m.bundleVersion}-${bundleSha256.take(12)}"
        if (isVersionQuarantined(s0, m.bundleVersion)) return Result.failure(RejectedException(Reject(RejectCode.QUARANTINED, id)))
        if (m.bundleVersion <= highestKnownVersion(builtinVersion)) {
            return Result.failure(RejectedException(Reject(RejectCode.NOT_NEWER, "v${m.bundleVersion}")))
        }
        val dir = slotDir(id)
        dir.deleteRecursively()
        try {
            for ((path, data) in v.entries) {
                val f = File(dir, path)
                f.parentFile.mkdirs()
                f.writeBytes(data)
                // Android 14+ refuses to load writable dex files.
                f.setReadOnly()
            }
        } catch (e: Exception) {
            dir.deleteRecursively()
            return Result.failure(e)
        }
        val slot = SlotInfo(
            id = id, bundleVersion = m.bundleVersion, versionName = m.bundleVersionName, sha256 = bundleSha256,
            entryClass = m.entryClass, installedAtMs = clock(), files = m.files,
            nativeAbi = m.requires.nativeAbi,
        )
        var s = s0
        // Replace any older un-promoted pending slot.
        s.pending?.let { old ->
            if (old != s.active && old != s.lastKnownGood) {
                slotDir(old).deleteRecursively()
                s = s.copy(slots = s.slots - old)
            }
        }
        s = s.copy(slots = s.slots + (id to slot), pending = id).log("STAGED", "${m.bundleVersionName} ($id)")
        save(s)
        gc()
        return Result.success(slot)
    }

    /** A version that failed on a device is never retried, whatever its hash: fixes ship as a higher version. */
    fun isVersionQuarantined(s: StoreState, version: Int) = s.quarantined.keys.any { it.startsWith("v$version-") }

    class RejectedException(val reject: Reject) : Exception(reject.toString())

    /**
     * Decide what to run for this launch and persist the attempt BEFORE anything is loaded.
     * Slots that exhausted their boot budget or fail an on-disk integrity check are quarantined.
     */
    @Synchronized
    fun planBoot(): BootPlan {
        var s = load()
        val order = listOfNotNull(s.pending, s.active, s.lastKnownGood).distinct()
        for (id in order) {
            val slot = s.slots[id] ?: continue
            if (id in s.quarantined) continue
            val trial = id == s.pending && !slot.everHealthy
            val budget = if (trial) MAX_BOOTS_TRIAL else MAX_BOOTS_ACTIVE
            if (slot.bootsSinceHealthy >= budget) {
                s = quarantineInternal(s, id, "boot budget exhausted (${slot.bootsSinceHealthy} unhealthy boots)")
                continue
            }
            if (!intact(slot)) {
                s = quarantineInternal(s, id, "on-disk integrity check failed")
                continue
            }
            val bumped = slot.copy(bootsSinceHealthy = slot.bootsSinceHealthy + 1)
            s = s.copy(slots = s.slots + (id to bumped)).log(if (trial) "BOOT_TRIAL" else "BOOT", id)
            save(s)
            return BootPlan.Slot(bumped, slotDir(id), trial)
        }
        save(s)
        return BootPlan.Builtin(if (order.isEmpty()) "no OTA bundle installed" else "no usable OTA bundle")
    }

    private fun intact(slot: SlotInfo): Boolean {
        val dir = slotDir(slot.id)
        return slot.files.isNotEmpty() && slot.files.all { f ->
            val file = File(dir, f.path)
            file.isFile && file.length() == f.size && Hashing.sha256Hex(file) == f.sha256
        }
    }

    private fun quarantineInternal(s0: StoreState, id: String, reason: String): StoreState {
        var s = s0.copy(
            quarantined = s0.quarantined + (id to reason),
            active = s0.active.takeUnless { it == id },
            lastKnownGood = s0.lastKnownGood.takeUnless { it == id },
            pending = s0.pending.takeUnless { it == id },
        ).log("QUARANTINE", "$id: $reason")
        // Promote lastKnownGood back to active if active was lost.
        if (s.active == null && s.lastKnownGood != null) s = s.copy(active = s.lastKnownGood)
        slotDir(id).deleteRecursively()
        return s.copy(slots = s.slots - id)
    }

    /** The running bundle proves it works (UI drawn, self-test passed): promote it. */
    @Synchronized
    fun markHealthy(id: String) {
        var s = load()
        val slot = s.slots[id] ?: return
        s = s.copy(
            slots = s.slots + (id to slot.copy(bootsSinceHealthy = 0, everHealthy = true)),
            active = id,
            lastKnownGood = id,
            pending = s.pending.takeUnless { it == id },
        ).log("HEALTHY", id)
        save(s)
        gc()
    }

    /** The running bundle failed in-process (load error, self-test failure, uncaught fatal). */
    @Synchronized
    fun reportFailure(id: String, reason: String) {
        val s = load()
        if (id !in s.slots) return
        save(quarantineInternal(s, id, reason))
        gc()
    }

    @Synchronized
    fun record(event: String, detail: String = "") = save(load().log(event, detail))

    private fun gc() {
        val s = load()
        val keep = setOfNotNull(s.active, s.lastKnownGood, s.pending)
        bundlesDir.listFiles()?.forEach { if (it.name !in keep) it.deleteRecursively() }
        val dead = s.slots.keys - keep
        if (dead.isNotEmpty()) save(s.copy(slots = s.slots - dead))
    }
}
