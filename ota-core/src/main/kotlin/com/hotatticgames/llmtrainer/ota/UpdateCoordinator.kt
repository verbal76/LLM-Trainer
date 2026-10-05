package com.hotatticgames.llmtrainer.ota

import java.io.File
import java.io.IOException

/** Thrown by a [Fetcher] for non-2xx responses so callers can distinguish "not found" from real failures. */
class HttpStatusException(val status: Int, message: String) : IOException(message)

/** Network seam. The Android host implements it with HttpURLConnection; tests use a fake. */
interface Fetcher {
    @Throws(IOException::class)
    fun getBytes(url: String, maxBytes: Long): ByteArray

    @Throws(IOException::class)
    fun download(url: String, dest: File, maxBytes: Long)
}

sealed interface CheckResult {
    data class UpToDate(val version: Int) : CheckResult
    /** A compatible, verified bundle was staged; it runs on next launch. */
    data class Staged(val slot: SlotInfo) : CheckResult
    /** Newer bundles exist but all need a newer APK (native/runtime change). Never forced via OTA. */
    data class NeedsNewApk(val newestBundle: String, val reasons: List<Reject>, val apkUrl: String?) : CheckResult
    data class Failed(val reasons: List<Reject>) : CheckResult
}

class UpdateCoordinator(
    private val host: HostInfo,
    private val trusted: List<TrustedKey>,
    private val store: UpdateStore,
    private val fetcher: Fetcher,
    private val workDir: File,
    private val indexUrl: String,
) {
    companion object {
        const val MAX_INDEX_BYTES = 1L * 1024 * 1024
        const val MAX_BUNDLE_BYTES = BundleFormat.MAX_TOTAL_BYTES
    }

    /**
     * Fetch the channel index, pick the highest bundle compatible with THIS host, download,
     * fully verify, stage. Incompatible bundles are never downloaded for install; if the only newer
     * bundles need a new APK we report NeedsNewApk instead of touching anything.
     */
    fun check(): CheckResult {
        val index = try {
            val raw = fetcher.getBytes(indexUrl, MAX_INDEX_BYTES).toString(Charsets.UTF_8)
            OtaJson.decodeFromString(ChannelIndex.serializer(), raw)
        } catch (e: HttpStatusException) {
            if (e.status == 404) {
                // Channel not published yet: nothing newer exists. A clean "up to date", not a failure.
                val floor = store.highestKnownVersion(host.builtinBundleVersion)
                store.record("CHECK_NO_CHANNEL", "index 404")
                return CheckResult.UpToDate(floor)
            }
            store.record("CHECK_FAILED", "index: ${e.message}")
            return CheckResult.Failed(listOf(Reject(RejectCode.INDEX_INVALID, e.message ?: "fetch failed")))
        } catch (e: Exception) {
            store.record("CHECK_FAILED", "index: ${e.message}")
            return CheckResult.Failed(listOf(Reject(RejectCode.INDEX_INVALID, e.message ?: "fetch/parse failed")))
        }
        if (index.schema != CHANNEL_SCHEMA || index.bundleId != host.bundleId || index.channel != host.channel) {
            store.record("CHECK_FAILED", "index identity mismatch")
            return CheckResult.Failed(listOf(Reject(RejectCode.INDEX_INVALID, "schema/bundleId/channel mismatch")))
        }

        val floor = store.highestKnownVersion(host.builtinBundleVersion)
        val newer = index.bundles.filter { it.bundleVersion > floor }.sortedByDescending { it.bundleVersion }
        if (newer.isEmpty()) {
            store.record("CHECK_UP_TO_DATE", "v$floor")
            return CheckResult.UpToDate(floor)
        }

        val state = store.load()
        val candidates = newer.filter { Compatibility.evaluate(it.requires, host).isEmpty() }
            .filter { !store.isVersionQuarantined(state, it.bundleVersion) }
        if (candidates.isEmpty()) {
            val top = newer.first()
            val reasons = Compatibility.evaluate(top.requires, host)
            if (reasons.isNotEmpty() && Compatibility.needsNewHost(reasons)) {
                store.record("NEEDS_NEW_APK", "${top.bundleVersionName}: ${reasons.joinToString()}")
                return CheckResult.NeedsNewApk(top.bundleVersionName, reasons, index.hostRelease?.url)
            }
            store.record("CHECK_NO_CANDIDATE", "all newer bundles quarantined")
            return CheckResult.UpToDate(floor)
        }

        // Try the best candidate first; fall back to the next-best compatible one on failure.
        val failures = mutableListOf<Reject>()
        for (entry in candidates) {
            workDir.mkdirs()
            val tmp = File(workDir, "download-${entry.bundleVersion}.hagb")
            try {
                fetcher.download(entry.url, tmp, minOf(entry.size + 1024, MAX_BUNDLE_BYTES))
            } catch (e: Exception) {
                tmp.delete()
                failures += Reject(RejectCode.DOWNLOAD_FAILED, "v${entry.bundleVersion}: ${e.message}")
                continue
            }
            try {
                val sha = Hashing.sha256Hex(tmp)
                if (!sha.equals(entry.sha256, ignoreCase = true)) {
                    failures += Reject(RejectCode.HASH_MISMATCH, "v${entry.bundleVersion} download")
                    continue
                }
                when (val r = BundleFormat.verify(tmp, trusted, host)) {
                    is BundleFormat.Result.Rejected -> {
                        store.record("REJECTED", "v${entry.bundleVersion}: ${r.reasons.joinToString()}")
                        failures += r.reasons
                    }
                    is BundleFormat.Result.Ok -> {
                        if (r.bundle.manifest.bundleVersion != entry.bundleVersion) {
                            failures += Reject(RejectCode.INDEX_INVALID, "index/manifest version disagree")
                            continue
                        }
                        val staged = store.stage(r.bundle, sha, host.builtinBundleVersion)
                        staged.onSuccess { return CheckResult.Staged(it) }
                        staged.onFailure {
                            failures += ((it as? UpdateStore.RejectedException)?.reject
                                ?: Reject(RejectCode.BAD_ARCHIVE, it.message ?: "stage failed"))
                        }
                    }
                }
            } finally {
                tmp.delete()
            }
        }
        store.record("CHECK_FAILED", failures.joinToString())
        return CheckResult.Failed(failures)
    }
}
