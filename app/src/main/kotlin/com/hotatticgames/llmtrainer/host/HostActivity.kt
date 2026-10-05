package com.hotatticgames.llmtrainer.host

import android.app.Activity
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.os.SystemClock
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
 * Launch order: native/system startup window (plain brand background) -> Hot Attic Games studio splash
 * (exact canonical asset) -> LLM Trainer product experience (the OTA-updatable bundle).
 * One activity: the splash is an overlay that fades once the bundle's first frame is drawn, so there is
 * no second splash and no artificial dead time beyond MIN_SPLASH_MS (brand legibility floor).
 */
class HostActivity : Activity() {
    companion object {
        const val MIN_SPLASH_MS = 1200L
        private const val FADE_MS = 250L
        private const val LOGO_ASSET = "branding/studio-logo.png"
        private var handlerInstalled = false
        /** Process-wide: the studio splash is a COLD-launch affair only (not config recreates / task resumes). */
        private var splashConsumed = false
    }

    private lateinit var container: FrameLayout
    private lateinit var splash: ImageView
    private lateinit var runtime: HostRuntime
    private var shownAtMs = 0L
    private var app: BundleApp? = null
    private var splashActive = false
    private val ui = android.os.Handler(android.os.Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        runtime = HostRuntime(applicationContext)
        installCrashHandler()

        container = FrameLayout(this).apply { setBackgroundColor(Color.parseColor("#0E0E12")) }
        splash = ImageView(this).apply {
            // Exact asset, never cropped/stretched/recolored: fit inside, aspect ratio preserved, alpha intact.
            scaleType = ImageView.ScaleType.FIT_CENTER
            adjustViewBounds = false
            setPadding(dp(24), dp(24), dp(24), dp(24))
            assets.open(LOGO_ASSET).use { setImageBitmap(BitmapFactory.decodeStream(it)) }
        }
        // Opaque backdrop: the product UI underneath must never show through the (alpha) logo.
        splash.setBackgroundColor(Color.parseColor("#0E0E12"))
        splashActive = !splashConsumed && savedInstanceState == null
        splashConsumed = true
        if (splashActive) container.addView(splash, FrameLayout.LayoutParams(-1, -1))
        setContentView(container)
        shownAtMs = SystemClock.elapsedRealtime()
        startBoot()
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun installCrashHandler() {
        if (handlerInstalled) return
        handlerInstalled = true
        val prev = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            runtime.onUncaught(e)
            prev?.uncaughtException(t, e)
        }
    }

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
        view.viewTreeObserver.addOnPreDrawListener(object : ViewTreeObserver.OnPreDrawListener {
            override fun onPreDraw(): Boolean {
                view.viewTreeObserver.removeOnPreDrawListener(this)
                // First frame of the bundle is about to draw: it has proven it can render.
                view.post { runtime.onFirstFrame() }
                return true
            }
        })
        // Splash removal is timer-driven (never dependent on a draw callback) so it cannot get stuck.
        val wait = (MIN_SPLASH_MS - (SystemClock.elapsedRealtime() - shownAtMs)).coerceAtLeast(0)
        ui.postDelayed({ fadeSplash() }, wait)
        bundleApp.onResume()
    }

    private fun fadeSplash() {
        if (!splashActive) return
        splashActive = false
        splash.animate().alpha(0f).setDuration(FADE_MS).withEndAction { removeSplashNow() }.start()
        ui.postDelayed({ removeSplashNow() }, FADE_MS + 400) // animation scale 0 / interrupted animators
    }

    private fun removeSplashNow() {
        splash.visibility = View.GONE
        (splash.parent as? FrameLayout)?.removeView(splash)
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
        col.addView(TextView(this).apply { text = runtime.diagnosticsJson(); typeface = Typeface.MONOSPACE; textSize = 11f; setTextColor(Color.GRAY); setTextIsSelectable(true) })
        container.addView(ScrollView(this).apply { setBackgroundColor(Color.parseColor("#0E0E12")); addView(col) }, 0)
        fadeSplash()
    }

    override fun onResume() { super.onResume(); app?.onResume() }
    override fun onPause() { app?.onPause(); super.onPause() }

    @Deprecated("framework back handling is sufficient for the single-activity host")
    override fun onBackPressed() {
        if (app?.onBackPressed() != true) super.onBackPressed()
    }
}
