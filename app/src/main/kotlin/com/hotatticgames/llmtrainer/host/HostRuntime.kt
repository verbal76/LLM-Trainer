package com.hotatticgames.llmtrainer.host

import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.View
import com.hotatticgames.llmtrainer.BuildConfig
import com.hotatticgames.llmtrainer.hostapi.BundleApp
import com.hotatticgames.llmtrainer.hostapi.BundleEntry
import com.hotatticgames.llmtrainer.hostapi.Capabilities
import com.hotatticgames.llmtrainer.hostapi.HOST_API_LEVEL
import com.hotatticgames.llmtrainer.hostapi.HostServices
import com.hotatticgames.llmtrainer.hostapi.UpdateStatus
import com.hotatticgames.llmtrainer.ota.BootPlan
import com.hotatticgames.llmtrainer.ota.BundleFormat
import com.hotatticgames.llmtrainer.ota.CheckResult
import com.hotatticgames.llmtrainer.ota.Diagnostics
import com.hotatticgames.llmtrainer.ota.Fetcher
import com.hotatticgames.llmtrainer.ota.Hashing
import com.hotatticgames.llmtrainer.ota.HostInfo
import com.hotatticgames.llmtrainer.ota.TrustedKey
import com.hotatticgames.llmtrainer.ota.UpdateCoordinator
import com.hotatticgames.llmtrainer.ota.UpdateStore
import dalvik.system.DexClassLoader
import java.io.File
import java.util.concurrent.Executors

private const val TAG = "HagOta"

/** A bundle that is loaded, self-tested, and has a content view ready to attach. */
class Running(
    val source: String, // "builtin" | "ota"
    val version: Int,
    val versionName: String,
    val slotId: String?,
    val entryClass: String,
    val app: BundleApp,
    var view: View?,
)

/**
 * Owns the OTA lifecycle on the device: verifies/loads the right bundle, falls back safely, serves
 * the host API to bundles, and runs update checks. Constructed with injectable root/fetcher/keys so
 * the instrumented OTA qualification test can drive the real code paths without the network.
 */
class HostRuntime(
    private val ctx: Context,
    val rootDir: File = File(ctx.filesDir, "ota"),
    private val fetcher: Fetcher = HttpFetcher(),
    private val channelUrl: String = BuildConfig.OTA_CHANNEL_URL,
    private val trusted: List<TrustedKey> = listOf(TrustedKey(BuildConfig.OTA_KEY_ID, BuildConfig.OTA_PUBLIC_KEY)),
    private val builtinAssetPath: String = "builtin/llmtrainer-main.hagb",
    private val builtinBundleVersion: Int = BuildConfig.BUILTIN_BUNDLE_VERSION,
) {
    val hostInfo = HostInfo(
        hostVersionCode = BuildConfig.VERSION_CODE,
        hostVersionName = BuildConfig.VERSION_NAME,
        hostApiLevel = HOST_API_LEVEL,
        nativeAbi = BuildConfig.NATIVE_ABI,
        nativeRuntimeId = BuildConfig.NATIVE_RUNTIME_ID,
        capabilities = setOf(Capabilities.CORE_V1, Capabilities.DEVICE_SNAPSHOT_V1, Capabilities.UPDATE_CHECK_V1),
        sdkInt = Build.VERSION.SDK_INT,
        bundleId = "llmtrainer-main",
        channel = "stable",
        builtinBundleVersion = builtinBundleVersion,
        sourceSha = BuildConfig.GIT_SHA,
    )
    val store = UpdateStore(rootDir)

    @Volatile var running: Running? = null
        private set

    /**
     * Identity of the bundle currently being brought up (loaded / self-tested / building its view). Published BEFORE any
     * bundle code runs so diagnostics can never claim "none / v0" while a bundle is alive (entry.create, selfTest and
     * createContentView may all read diagnosticsJson()).
     */
    private class Starting(val source: String, val version: Int, val versionName: String, val slotId: String?)
    @Volatile private var starting: Starting? = null
    private var startedAtMs = 0L
    private val io = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())

    val services: HostServices = object : HostServices {
        override val hostVersionName = hostInfo.hostVersionName
        override val hostApiLevel = hostInfo.hostApiLevel
        override val nativeAbi = hostInfo.nativeAbi
        override val nativeRuntimeId = hostInfo.nativeRuntimeId
        override fun diagnosticsJson() = this@HostRuntime.diagnosticsJson()
        override fun deviceSnapshotJson() = DeviceProbe.snapshotJson(ctx)
        override fun checkForUpdates(callback: (UpdateStatus) -> Unit) = this@HostRuntime.checkForUpdates(callback)
        override fun restartApp() = this@HostRuntime.restartProcess()
    }

    fun diagnosticsJson(): String = Diagnostics.toJson(diagnosticsReport())

    fun diagnosticsReport() = run {
        val r = running
        val s = starting
        val state = store.load()
        when {
            r != null -> Diagnostics.build(hostInfo, state, r.source, r.version, r.versionName, r.slotId)
            // Bundle coming up: report what is starting, with an explicit source so it is never mistaken for a settled state.
            s != null -> Diagnostics.build(hostInfo, state, "starting:${s.source}", s.version, s.versionName, s.slotId)
            else -> Diagnostics.build(hostInfo, state, "none", 0, "-", null)
        }
    }

    /** The five-identity block of docs/VERSIONING.md (native / application / OTA sequence / runtime / source). */
    fun identityBlock(): String = diagnosticsReport().identityBlock

    /**
     * Pick, load and self-test a bundle, falling back until something works. Heavy; call off the main thread.
     * Returns null only if even the built-in bundle fails (host then shows safe mode).
     * [makeView] runs the bundle's createContentView on the main thread.
     */
    fun boot(makeView: (BundleApp) -> View): Running? {
        startedAtMs = SystemClock.elapsedRealtime()
        runCatching { store.reconcileBuiltin(builtinBundleVersion, hostInfo.nativeAbi) }
        repeat(6) {
            when (val plan = store.planBoot()) {
                is BootPlan.Slot -> {
                    Log.i(TAG, "boot plan: ${if (plan.trial) "TRIAL" else "ACTIVE"} ${plan.slot.id}")
                    val dexes = plan.slot.files.map { it.path }.filter { it.endsWith(".dex") }.sorted().map { File(plan.dir, it) }
                    val r = tryLoad("ota", plan.slot.bundleVersion, plan.slot.versionName, plan.slot.id, plan.slot.entryClass, dexes, makeView)
                    if (r != null) return r.also { running = it }
                    // tryLoad already quarantined the slot; loop picks the next candidate.
                }
                is BootPlan.Builtin -> {
                    Log.i(TAG, "boot plan: BUILTIN (${plan.reason})")
                    return loadBuiltin(makeView)?.also { running = it }
                }
            }
        }
        return loadBuiltin(makeView)?.also { running = it }
    }

    private fun tryLoad(
        source: String, version: Int, name: String, slotId: String?, entryClass: String,
        dexes: List<File>, makeView: (BundleApp) -> View,
    ): Running? {
        return try {
            starting = Starting(source, version, name, slotId)
            val loader = DexClassLoader(
                dexes.joinToString(File.pathSeparator) { it.absolutePath }, null, null, HostRuntime::class.java.classLoader,
            )
            val entry = loader.loadClass(entryClass).getDeclaredConstructor().newInstance() as BundleEntry
            val app = entry.create(services)
            app.selfTest()?.let { throw IllegalStateException("selfTest failed: $it") }
            // Publish identity BEFORE the UI is built so diagnosticsJson() is truthful inside createContentView.
            val r = Running(source, version, name, slotId, entryClass, app, null)
            running = r
            r.view = makeView(app)
            starting = null
            r
        } catch (t: Throwable) {
            running = null
            starting = null
            Log.e(TAG, "bundle $source v$version failed to start", t)
            if (slotId != null) store.reportFailure(slotId, "${t.javaClass.simpleName}: ${t.message}")
            else store.record("BUILTIN_FAILED", "${t.javaClass.simpleName}: ${t.message}")
            null
        }
    }

    private fun loadBuiltin(makeView: (BundleApp) -> View): Running? = try {
        val tmp = File(ctx.cacheDir, "builtin-verify.hagb")
        ctx.assets.open(builtinAssetPath).use { i -> tmp.outputStream().use { o -> i.copyTo(o) } }
        val sha = Hashing.sha256Hex(tmp)
        val v = (BundleFormat.verify(tmp, trusted, hostInfo) as? BundleFormat.Result.Ok)?.bundle
            ?: error("built-in bundle failed verification")
        tmp.delete()
        val dir = File(rootDir, "builtin/v${v.manifest.bundleVersion}-${sha.take(12)}")
        if (!dir.isDirectory || v.manifest.files.any { !File(dir, it.path).isFile }) {
            File(rootDir, "builtin").deleteRecursively()
            for ((p, data) in v.entries) File(dir, p).apply { parentFile!!.mkdirs(); writeBytes(data); setReadOnly() }
        }
        val dexes = v.manifest.files.map { it.path }.filter { it.endsWith(".dex") }.sorted().map { File(dir, it) }
        tryLoad("builtin", v.manifest.bundleVersion, v.manifest.bundleVersionName, null, v.manifest.entryClass, dexes, makeView)
    } catch (t: Throwable) {
        Log.e(TAG, "built-in bundle unusable", t)
        store.record("BUILTIN_FAILED", "${t.javaClass.simpleName}: ${t.message}")
        null
    }

    /** Called once the content view drew its first frame: the bundle has proven itself. */
    fun onFirstFrame() {
        val r = running ?: return
        r.slotId?.let { store.markHealthy(it) }
        Log.i(TAG, "healthy: ${r.source} v${r.version}")
    }

    /** If a freshly started OTA bundle takes the process down, roll it back immediately. */
    fun onUncaught(e: Throwable) {
        try {
            val r = running ?: return
            val slot = r.slotId ?: return
            val pkg = r.entryClass.substringBeforeLast('.')
            var t: Throwable? = e
            var fromBundle = false
            while (t != null && !fromBundle) {
                fromBundle = t.stackTrace.any { it.className.startsWith(pkg) }
                t = t.cause
            }
            val young = SystemClock.elapsedRealtime() - startedAtMs < 120_000
            if (fromBundle && young) store.reportFailure(slot, "uncaught: ${e.javaClass.simpleName}: ${e.message}")
        } catch (_: Throwable) {
        }
    }

    fun checkForUpdates(callback: (UpdateStatus) -> Unit) {
        io.execute {
            val result = try {
                UpdateCoordinator(hostInfo, trusted, store, fetcher, File(ctx.cacheDir, "ota-dl"), channelUrl).check()
            } catch (t: Throwable) {
                CheckResult.Failed(listOf(com.hotatticgames.llmtrainer.ota.Reject(com.hotatticgames.llmtrainer.ota.RejectCode.DOWNLOAD_FAILED, t.message ?: "error")))
            }
            val status = when (result) {
                is CheckResult.UpToDate -> UpdateStatus(
                    "UP_TO_DATE",
                    if (result.channelPublished) "bundle #${result.version}" else "No update channel is published yet, so there is nothing newer to install.",
                )
                is CheckResult.Staged -> UpdateStatus("STAGED", "${result.slot.versionName} (v${result.slot.bundleVersion})")
                is CheckResult.NeedsNewApk -> UpdateStatus("NEEDS_NEW_APK", "${result.newestBundle}: ${result.reasons.joinToString()}")
                is CheckResult.Failed -> UpdateStatus("FAILED", failureMessage(result.reasons))
            }
            main.post { callback(status) }
        }
    }

    private fun failureMessage(reasons: List<com.hotatticgames.llmtrainer.ota.Reject>): String {
        val r = reasons.firstOrNull() ?: return "unknown error"
        return when (r.code) {
            com.hotatticgames.llmtrainer.ota.RejectCode.NETWORK_UNAVAILABLE -> "Could not reach the update server (offline or network error). Try again when connected."
            com.hotatticgames.llmtrainer.ota.RejectCode.SERVER_ERROR -> "The update server had a problem (${r.detail}). Try again later."
            else -> reasons.joinToString()
        }
    }

    fun restartProcess() {
        val i = ctx.packageManager.getLaunchIntentForPackage(ctx.packageName)!!
            .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK or android.content.Intent.FLAG_ACTIVITY_CLEAR_TASK)
        ctx.startActivity(i)
        main.postDelayed({ android.os.Process.killProcess(android.os.Process.myPid()) }, 200)
    }
}
