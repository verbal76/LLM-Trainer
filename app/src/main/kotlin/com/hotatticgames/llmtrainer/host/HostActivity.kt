package com.hotatticgames.llmtrainer.host

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.view.ViewTreeObserver
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.hotatticgames.llmtrainer.hostapi.BundleApp
import java.util.concurrent.CountDownLatch
import kotlin.concurrent.thread

/**
 * Launch order: native/system startup window (plain brand-neutral background, no icon) -> Hot Attic Games studio
 * splash (exact canonical asset) -> LLM Trainer product experience (the OTA-updatable bundle).
 *
 * Splash contract (docs/RELEASE.md, instrumented in HostSplashTest):
 *  - COLD PROCESS START ONLY. The flag lives in the process, so an Activity re-created inside a live process
 *    (relaunch after back, recreate(), config change that is not self-handled, singleTask re-delivery) never replays it.
 *    A restore after process death is a new process and therefore a cold start.
 *  - It is an opaque overlay (neutral backdrop, logo fit-inside with aspect preserved, alpha untouched) that consumes
 *    touches while it is up and is removed once BOTH the minimum brand time has passed AND the product view drew its
 *    first frame, with a hard cap so it can never linger. It is never shown over a running product UI.
 *  - The bitmap is decoded unscaled/unmodified from the canonical asset (no crop, stretch, recolor, resample).
 */
class HostActivity : Activity() {
    companion object {
        const val MIN_SPLASH_MS = 1200L
        /** Hard ceiling from the moment the product view is attached (or safe mode shown): splash is gone by then. */
        const val MAX_SPLASH_AFTER_READY_MS = 3000L
        const val FADE_MS = 250L
        const val SPLASH_TAG = "hag:splash"
        const val BACKDROP = "#0E0E12"
        const val LOGO_ASSET = "branding/studio-logo.png"
        private const val TAG = "HagHost"
        private const val SAVED_FRAGMENTS_KEY = "android:fragments" // Activity.FRAGMENTS_TAG (platform fragments)

        private var handlerInstalled = false
        @Volatile private var currentRuntime: HostRuntime? = null

        /** Process-wide: true until the first HostActivity of this process consumed the studio splash. */
        @Volatile private var coldLaunchPending = true

        /** Test seam: minimum brand time. Production value is [MIN_SPLASH_MS]. */
        @Volatile internal var minSplashMs: Long = MIN_SPLASH_MS

        /** Test seam: pretend the process was just started (instrumentation shares one process across tests). */
        internal fun resetColdLaunchForTest() { coldLaunchPending = true; minSplashMs = MIN_SPLASH_MS }

        /** Test seam: milestones of the most recent splash, in SystemClock.elapsedRealtime(). -1 = did not happen. */
        @Volatile internal var lastSplashShownAtMs = -1L
        @Volatile internal var lastProductAttachedAtMs = -1L
        @Volatile internal var lastSplashRemovedAtMs = -1L
    }

    private lateinit var container: FrameLayout
    private var splash: ImageView? = null
    private lateinit var runtime: HostRuntime
    private var createdAtMs = 0L
    private var app: BundleApp? = null
    private var splashActive = false
    private var splashRemoving = false
    private var productDrawn = false
    private var readyAtMs = -1L
    private var resumed = false
    private val ui = android.os.Handler(android.os.Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        // Android restores Fragments (and view state) from the saved bundle through THIS class loader. Fragments that
        // the OTA bundle added (e.g. its file picker) can only be loaded by the bundle's DexClassLoader, so restoring
        // them after process death crashes the host. The host therefore never accepts saved state: the product UI is
        // rebuilt from persisted app data, not from framework instance state.
        super.onCreate(null)
        createdAtMs = SystemClock.elapsedRealtime()
        runtime = HostRuntime(applicationContext)
        currentRuntime = runtime
        installCrashHandler()

        if (Build.VERSION.SDK_INT >= 31) {
            // The system splash (blank icon on the brand-neutral background) hands over to ours without its own
            // exit animation, so nothing else is ever drawn between it and the studio logo.
            splashScreen.setOnExitAnimationListener { it.remove() }
        }

        container = FrameLayout(this).apply { setBackgroundColor(Color.parseColor(BACKDROP)) }
        splashActive = coldLaunchPending
        coldLaunchPending = false
        if (splashActive) {
            splash = buildSplash()?.also { container.addView(it, FrameLayout.LayoutParams(-1, -1)) }
            if (splash == null) splashActive = false
        }
        lastSplashShownAtMs = if (splashActive) SystemClock.elapsedRealtime() else -1L
        lastProductAttachedAtMs = -1L
        lastSplashRemovedAtMs = -1L
        setContentView(container)
        startBoot()
    }

    /** Exact canonical asset on an opaque neutral backdrop. Returns null (no splash) rather than ever failing the launch. */
    private fun buildSplash(): ImageView? {
        val bmp: Bitmap = try {
            val o = BitmapFactory.Options().apply {
                inScaled = false // never resample for density
                inPreferredConfig = Bitmap.Config.ARGB_8888 // keep alpha
                inPremultiplied = true
            }
            assets.open(LOGO_ASSET).use { BitmapFactory.decodeStream(it, null, o) } ?: return null
        } catch (t: Throwable) {
            Log.e(TAG, "studio logo unavailable; continuing without splash", t)
            return null
        }
        return ImageView(this).apply {
            tag = SPLASH_TAG
            contentDescription = "Hot Attic Games"
            // Fit inside, centred, aspect ratio preserved: never crop, never stretch.
            scaleType = ImageView.ScaleType.FIT_CENTER
            adjustViewBounds = false
            setPadding(dp(24), dp(24), dp(24), dp(24))
            setImageBitmap(bmp)
            // Opaque backdrop: the product UI underneath must never show through the (alpha) logo.
            setBackgroundColor(Color.parseColor(BACKDROP))
            // Swallow touches while it is up so nothing underneath is tapped blind.
            isClickable = true
            isFocusable = false
        }
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun installCrashHandler() {
        if (handlerInstalled) return
        handlerInstalled = true
        val prev = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            currentRuntime?.onUncaught(e) // the newest runtime, not the first Activity's
            prev?.uncaughtException(t, e)
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        // Never persist bundle-owned Fragments (see onCreate): the saved state would otherwise outlive this process.
        outState.remove(SAVED_FRAGMENTS_KEY)
        outState.remove("android:support:fragments")
    }

    /** View state is not restored either: the bundle's views are rebuilt from scratch. */
    override fun onRestoreInstanceState(savedInstanceState: Bundle) {}

    private fun startBoot() {
        thread(name = "hag-boot") {
            val r = runtime.boot { bundleApp -> onMainBlocking { bundleApp.createContentView(this) } }
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                if (r == null) showSafeMode() else showBundle(r.app, r.view!!)
            }
        }
    }

    private fun <T> onMainBlocking(block: () -> T): T {
        var result: Result<T>? = null
        val latch = CountDownLatch(1)
        runOnUiThread { result = runCatching(block); latch.countDown() }
        latch.await()
        return result!!.getOrThrow()
    }

    private fun showBundle(bundleApp: BundleApp, view: View) {
        app = bundleApp
        container.addView(view, 0, FrameLayout.LayoutParams(-1, -1))
        lastProductAttachedAtMs = SystemClock.elapsedRealtime()
        view.viewTreeObserver.addOnPreDrawListener(object : ViewTreeObserver.OnPreDrawListener {
            override fun onPreDraw(): Boolean {
                view.viewTreeObserver.removeOnPreDrawListener(this)
                // First frame of the bundle is about to draw: it has proven it can render.
                view.post {
                    runtime.onFirstFrame()
                    productDrawn = true
                    tryRemoveSplash()
                }
                return true
            }
        })
        scheduleSplashRemoval()
        if (resumed) bundleApp.onResume()
    }

    /** Removal needs min brand time AND a drawn product frame; the cap guarantees it never depends on a draw callback. */
    private fun scheduleSplashRemoval() {
        if (!splashActive) return
        readyAtMs = SystemClock.elapsedRealtime()
        val sinceCreate = readyAtMs - createdAtMs
        ui.postDelayed({ tryRemoveSplash() }, (minSplashMs - sinceCreate).coerceAtLeast(0))
        ui.postDelayed({ productDrawn = true; tryRemoveSplash() }, (minSplashMs - sinceCreate).coerceAtLeast(0) + MAX_SPLASH_AFTER_READY_MS)
    }

    private fun tryRemoveSplash() {
        if (!splashActive || splashRemoving) return
        val minElapsed = SystemClock.elapsedRealtime() - createdAtMs >= minSplashMs
        if (!minElapsed || !productDrawn) return
        splashRemoving = true
        val s = splash ?: return
        s.animate().alpha(0f).setDuration(FADE_MS).withEndAction { removeSplashNow() }.start()
        ui.postDelayed({ removeSplashNow() }, FADE_MS + 400) // animator scale 0 / cancelled animators
    }

    private fun removeSplashNow() {
        val s = splash ?: return
        splash = null
        splashActive = false
        s.animate().cancel()
        s.visibility = View.GONE
        (s.parent as? FrameLayout)?.removeView(s)
        lastSplashRemovedAtMs = SystemClock.elapsedRealtime()
    }

    /** Last resort: even the built-in bundle failed. Never a blank screen; always diagnosable. */
    private fun showSafeMode() {
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(16), dp(32), dp(16), dp(16)) }
        col.addView(TextView(this).apply { text = "LLM Trainer – safe mode"; textSize = 22f; setTextColor(Color.WHITE); setTypeface(typeface, Typeface.BOLD) })
        col.addView(TextView(this).apply { text = "The app could not start its main experience. Details below; you can retry or check for a fixed update."; setTextColor(Color.LTGRAY) })
        val status = TextView(this).apply { setTextColor(Color.parseColor("#FF8A1F")) }
        col.addView(Button(this).apply {
            text = "Check for updates"; isAllCaps = false
            setOnClickListener { runtime.checkForUpdates { status.text = "${it.kind}: ${it.message}" } }
        })
        col.addView(Button(this).apply { text = "Restart"; isAllCaps = false; setOnClickListener { runtime.restartProcess() } })
        col.addView(status)
        col.addView(TextView(this).apply { text = runtime.identityBlock() + "\n\n" + runtime.diagnosticsJson(); typeface = Typeface.MONOSPACE; textSize = 11f; setTextColor(Color.GRAY); setTextIsSelectable(true) })
        container.addView(ScrollView(this).apply { setBackgroundColor(Color.parseColor(BACKDROP)); addView(col) }, 0)
        productDrawn = true
        scheduleSplashRemoval()
    }

    override fun onResume() { super.onResume(); resumed = true; app?.onResume() }
    override fun onPause() { resumed = false; app?.onPause(); super.onPause() }

    override fun onDestroy() {
        ui.removeCallbacksAndMessages(null)
        splash?.animate()?.cancel()
        super.onDestroy()
    }

    // With targetSdk 36 predictive back is on (also declared in the manifest). The platform Activity's default
    // back callback still routes to onBackPressed(), which is where the bundle gets first refusal.
    @Deprecated("framework back handling is sufficient for the single-activity host")
    override fun onBackPressed() {
        if (app?.onBackPressed() != true) super.onBackPressed()
    }
}
