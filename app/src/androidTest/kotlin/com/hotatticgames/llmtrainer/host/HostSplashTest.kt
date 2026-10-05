package com.hotatticgames.llmtrainer.host

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.ColorDrawable
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import androidx.lifecycle.Lifecycle
import com.hotatticgames.llmtrainer.ota.Hashing
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer

/**
 * The studio splash on the REAL HostActivity: exact canonical bitmap, opaque neutral backdrop, aspect preserved,
 * gone in bounded time with the product UI visible, and COLD PROCESS START ONLY (not a second Activity in the same
 * process, not recreate(), not a background/foreground round trip).
 *
 * The instrumentation process is shared by all tests, so [HostActivity.resetColdLaunchForTest] simulates "the process
 * was just started". The minimum brand time is stretched so the presence assertions are not racing the timer.
 */
class HostSplashTest {
    private val instr = InstrumentationRegistry.getInstrumentation()
    private val target = instr.targetContext
    private var scenario: ActivityScenario<HostActivity>? = null
    private val minSplash = 3_000L

    @Before fun setUp() {
        File(target.filesDir, "ota").deleteRecursively()
        HostActivity.resetColdLaunchForTest()
        HostActivity.minSplashMs = minSplash
    }

    @After fun tearDown() {
        scenario?.close()
        HostActivity.resetColdLaunchForTest()
        File(target.filesDir, "ota").deleteRecursively()
    }

    // ---- helpers (view access on the main thread) -----------------------------------------------------------
    private fun walk(v: View, f: (View) -> Boolean): View? {
        if (f(v)) return v
        if (v is ViewGroup) for (i in 0 until v.childCount) walk(v.getChildAt(i), f)?.let { return it }
        return null
    }

    private fun <T> onActivity(block: (HostActivity) -> T): T {
        var r: T? = null
        scenario!!.onActivity { r = block(it) }
        @Suppress("UNCHECKED_CAST")
        return r as T
    }

    private fun splashOrNull(): ImageView? = onActivity { a ->
        walk(a.window.decorView) { it.tag == HostActivity.SPLASH_TAG } as ImageView?
    }

    private fun hostLoader() = HostActivity::class.java.classLoader

    /** The bundle's root view: any view whose class was defined by a loader other than the host's. */
    private fun bundleViewOrNull(): View? = onActivity { a ->
        walk(a.window.decorView) { it.javaClass.classLoader !== hostLoader() && it.javaClass.classLoader !== Any::class.java.classLoader && it.javaClass.name.startsWith("com.hotatticgames.llmtrainer.app.") }
    }

    private fun <T> waitUntil(what: String, timeoutMs: Long, probe: () -> T?): T {
        val end = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < end) {
            probe()?.let { return it }
            Thread.sleep(100)
        }
        throw AssertionError("Timed out waiting for $what")
    }

    private fun pixelHash(b: Bitmap): String {
        val buf = ByteBuffer.allocate(b.byteCount)
        b.copyPixelsToBuffer(buf)
        return Hashing.sha256Hex(buf.array())
    }

    private fun referenceBitmap(): Bitmap {
        val o = BitmapFactory.Options().apply { inScaled = false; inPreferredConfig = Bitmap.Config.ARGB_8888; inPremultiplied = true }
        return target.assets.open(HostActivity.LOGO_ASSET).use { BitmapFactory.decodeStream(it, null, o) }!!
    }

    private fun assertProductVisibleAndNoSplash() {
        waitUntil("product UI visible", 30_000) {
            bundleViewOrNull()?.takeIf { v -> onActivity { v.isShown && v.width > 0 && v.height > 0 } }
        }
        assertNull("splash must be gone", splashOrNull())
        // Nothing but the product view remains in the host container: no overlay lingers over the live UI.
        val children = onActivity { a -> (a.findViewById<ViewGroup>(android.R.id.content).getChildAt(0) as ViewGroup).childCount }
        assertEquals(1, children)
    }

    // ---- tests ----------------------------------------------------------------------------------------------
    @Test fun coldLaunchShowsTheExactCanonicalLogoThenYieldsToTheProductUi() {
        scenario = ActivityScenario.launch(HostActivity::class.java)
        val splash = splashOrNull()
        assertNotNull("splash must be present at cold start", splash)

        // 1. The asset itself is the canonical file.
        val assetBytes = target.assets.open(HostActivity.LOGO_ASSET).use { it.readBytes() }
        assertEquals("e3d9bb5653eafb783eede827606e7ac73a4e45564a1c25b1ed13ad1429f48c4e", Hashing.sha256Hex(assetBytes))

        // 2. The displayed bitmap is that asset, unmodified: same dimensions, same pixels.
        val ref = referenceBitmap()
        assertEquals(1536, ref.width); assertEquals(1024, ref.height)
        onActivity {
            val bmp = ((splash!!.drawable) as BitmapDrawable).bitmap
            assertEquals(ref.width, bmp.width); assertEquals(ref.height, bmp.height)
            assertTrue("alpha channel must be preserved", bmp.hasAlpha())
            assertEquals(pixelHash(ref), pixelHash(bmp))

            // 3. Opaque neutral backdrop (product UI can never show through the transparent logo areas).
            val bg = splash.background as ColorDrawable
            assertEquals(255, android.graphics.Color.alpha(bg.color))
            assertEquals(android.graphics.Color.parseColor(HostActivity.BACKDROP), bg.color)
        }

        // 4. Aspect ratio preserved (uniform scale, fits inside, no crop): measured on the laid-out view.
        waitUntil("splash laid out", 5_000) { onActivity { if (splash!!.width > 0 && splash.height > 0) true else null } }
        onActivity {
            assertEquals(ImageView.ScaleType.FIT_CENTER, splash!!.scaleType)
            val m = FloatArray(9); splash.imageMatrix.getValues(m)
            assertEquals("uniform scale (no stretch)", m[android.graphics.Matrix.MSCALE_X].toDouble(), m[android.graphics.Matrix.MSCALE_Y].toDouble(), 1e-3)
            val sw = ref.width * m[android.graphics.Matrix.MSCALE_X]; val sh = ref.height * m[android.graphics.Matrix.MSCALE_Y]
            assertTrue("logo fits inside the view (no crop): ${sw}x$sh in ${splash.width}x${splash.height}",
                sw <= splash.width - splash.paddingLeft - splash.paddingRight + 1 && sh <= splash.height - splash.paddingTop - splash.paddingBottom + 1)
            assertTrue("splash consumes touches while up", splash.isClickable)
        }

        // 5. Still present within the brand minimum, then removed within a bounded time after the product is attached.
        assertTrue(splashOrNull() != null || HostActivity.lastSplashRemovedAtMs >= 0)
        waitUntil("splash removed", minSplash + HostActivity.MAX_SPLASH_AFTER_READY_MS + 20_000) { if (splashOrNull() == null) true else null }
        val attached = HostActivity.lastProductAttachedAtMs
        val removed = HostActivity.lastSplashRemovedAtMs
        assertTrue("product attached before the splash was removed ($attached, $removed)", attached in 0..removed)
        assertTrue("splash removal bounded after product ready: ${removed - attached}ms",
            removed - attached <= minSplash + HostActivity.MAX_SPLASH_AFTER_READY_MS + HostActivity.FADE_MS + 1_500)
        assertTrue("brand minimum honoured", removed - HostActivity.lastSplashShownAtMs >= minSplash - 50)

        // 6. Product content is visible and nothing lingers over it.
        assertProductVisibleAndNoSplash()
    }

    @Test fun secondActivityStartInTheSameProcessDoesNotShowTheSplash() {
        HostActivity.minSplashMs = 0 // keep this one quick; the cold launch is covered above
        scenario = ActivityScenario.launch(HostActivity::class.java)
        assertTrue("the first (cold) start did show the splash", HostActivity.lastSplashShownAtMs >= 0)
        assertProductVisibleAndNoSplash()
        scenario!!.close()

        // Same process, brand-new Activity instance (e.g. relaunched from the launcher after back).
        HostActivity.minSplashMs = minSplash // a splash, had it been shown, would be clearly observable
        scenario = ActivityScenario.launch(HostActivity::class.java)
        assertNull("no splash on a non-cold start", splashOrNull())
        assertEquals("splash was never shown", -1L, HostActivity.lastSplashShownAtMs)
        assertProductVisibleAndNoSplash()
    }

    @Test fun recreateAndBackgroundRoundTripDoNotReplayTheSplash() {
        HostActivity.minSplashMs = 0
        scenario = ActivityScenario.launch(HostActivity::class.java)
        assertProductVisibleAndNoSplash()

        HostActivity.minSplashMs = minSplash
        scenario!!.recreate()
        assertNull(splashOrNull())
        assertEquals(-1L, HostActivity.lastSplashShownAtMs)
        assertProductVisibleAndNoSplash()

        scenario!!.moveToState(Lifecycle.State.CREATED) // stopped (backgrounded)
        scenario!!.moveToState(Lifecycle.State.RESUMED) // foregrounded
        assertNull(splashOrNull())
        assertEquals(-1L, HostActivity.lastSplashShownAtMs)
        assertProductVisibleAndNoSplash()
    }

    @Test fun coldStartAgainAfterProcessRestartShowsItAgain() {
        // resetColdLaunchForTest() stands in for a new process; nothing else (saved state etc.) may suppress the splash.
        HostActivity.minSplashMs = 0
        scenario = ActivityScenario.launch(HostActivity::class.java)
        assertProductVisibleAndNoSplash()
        scenario!!.close()
        HostActivity.resetColdLaunchForTest()
        HostActivity.minSplashMs = minSplash
        scenario = ActivityScenario.launch(HostActivity::class.java)
        assertNotNull(splashOrNull())
        assertTrue(HostActivity.lastSplashShownAtMs >= 0)
    }
}
