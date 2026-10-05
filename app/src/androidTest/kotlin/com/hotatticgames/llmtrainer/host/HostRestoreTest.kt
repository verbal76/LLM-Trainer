package com.hotatticgames.llmtrainer.host

import android.app.Fragment
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Process-death / recreation guard. Android restores Fragments from saved instance state through the HOST class
 * loader; a Fragment added by the OTA bundle (its file picker) can only be loaded by the bundle's DexClassLoader, so
 * an unguarded host crashes on restore. The restore path is identical for process death and for recreate(), so the
 * test adds a genuinely bundle-only Fragment, forces a save+recreate, and requires a clean, working Activity.
 */
class HostRestoreTest {
    private val instr = InstrumentationRegistry.getInstrumentation()
    private val target = instr.targetContext
    private var scenario: ActivityScenario<HostActivity>? = null

    @Before fun setUp() {
        File(target.filesDir, "ota").deleteRecursively()
        HostActivity.resetColdLaunchForTest()
        HostActivity.minSplashMs = 0
    }

    @After fun tearDown() {
        scenario?.close()
        HostActivity.resetColdLaunchForTest()
        File(target.filesDir, "ota").deleteRecursively()
    }

    private fun walk(v: View, f: (View) -> Boolean): View? {
        if (f(v)) return v
        if (v is ViewGroup) for (i in 0 until v.childCount) walk(v.getChildAt(i), f)?.let { return it }
        return null
    }

    private fun bundleLoader(a: HostActivity): ClassLoader? =
        walk(a.window.decorView) { it.javaClass.name.startsWith("com.hotatticgames.llmtrainer.app.") }?.javaClass?.classLoader

    private fun waitForProductUi() {
        val end = System.currentTimeMillis() + 30_000
        while (System.currentTimeMillis() < end) {
            var ready = false
            scenario!!.onActivity { a -> ready = bundleLoader(a) != null && walk(a.window.decorView) { it.tag == HostActivity.SPLASH_TAG } == null }
            if (ready) return
            Thread.sleep(100)
        }
        fail("product UI did not come up")
    }

    @Test fun recreateWithABundleOwnedFragmentDoesNotCrashAndDropsIt() {
        scenario = ActivityScenario.launch(HostActivity::class.java)
        waitForProductUi()

        scenario!!.onActivity { a ->
            val loader = bundleLoader(a)!!
            val name = "com.hotatticgames.llmtrainer.app.PickerFragment" // the bundle's real headless picker fragment
            try {
                Class.forName(name, false, HostActivity::class.java.classLoader)
                fail("precondition: the host class loader must NOT be able to load a bundle-only class")
            } catch (_: ClassNotFoundException) {
            }
            val f = Class.forName(name, true, loader).getDeclaredConstructor().newInstance() as Fragment
            a.fragmentManager.beginTransaction().add(f, "bundle-fragment").commitNow()
            assertNotNull(a.fragmentManager.findFragmentByTag("bundle-fragment"))
        }

        // Real save + destroy + re-create through the framework. Unguarded, FragmentManager.restoreAllState throws
        // Fragment.InstantiationException (ClassNotFoundException) here and the instrumentation process crashes.
        scenario!!.recreate()
        waitForProductUi()
        scenario!!.onActivity { a ->
            assertNull("bundle-owned fragment must not be restored", a.fragmentManager.findFragmentByTag("bundle-fragment"))
            assertEquals(0, a.fragmentManager.fragments.size)
        }
    }

    @Test fun savedInstanceStateNeverCarriesFragmentsOrViewState() {
        scenario = ActivityScenario.launch(HostActivity::class.java)
        waitForProductUi()
        val out = Bundle()
        scenario!!.onActivity { a ->
            val loader = bundleLoader(a)!!
            val f = Class.forName("com.hotatticgames.llmtrainer.app.PickerFragment", true, loader).getDeclaredConstructor().newInstance() as Fragment
            a.fragmentManager.beginTransaction().add(f, "bundle-fragment").commitNow()
            instr.callActivityOnSaveInstanceState(a, out)
        }
        assertFalse(out.containsKey("android:fragments"))
        assertFalse(out.containsKey("android:support:fragments"))
    }

    @Test fun createWithAHostileSavedBundleIsIgnored() {
        // Process-death restore delivers a saved Bundle to onCreate. Whatever it contains must be ignored, not parsed.
        scenario = ActivityScenario.launch(HostActivity::class.java)
        waitForProductUi()
        var created = false
        scenario!!.onActivity { a ->
            val hostile = Bundle().apply {
                putString("android:fragments", "not even a parcelable")
                putBundle("android:viewHierarchyState", Bundle().apply { putString("x", "y") })
            }
            // onRestoreInstanceState is the second entry point of restored state; it must be a harmless no-op.
            instr.callActivityOnRestoreInstanceState(a, hostile)
            created = true
        }
        assertTrue(created)
        waitForProductUi()
    }
}
