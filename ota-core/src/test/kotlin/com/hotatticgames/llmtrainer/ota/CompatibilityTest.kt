package com.hotatticgames.llmtrainer.ota

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CompatibilityTest {
    private val req = Requires(IntRangeSpec(2, 4), 7, listOf("a", "b"), 28)
    private fun host(api: Int = 3, abi: Int = 7, caps: Set<String> = setOf("a", "b", "c"), sdk: Int = 30) =
        TestKit.host(api, abi, caps, sdk)

    @Test fun compatible() = assertTrue(Compatibility.evaluate(req, host()).isEmpty())
    @Test fun apiBoundsInclusive() {
        assertTrue(Compatibility.evaluate(req, host(api = 2)).isEmpty())
        assertTrue(Compatibility.evaluate(req, host(api = 4)).isEmpty())
    }
    @Test fun apiTooOld() = assertEquals(RejectCode.HOST_API_TOO_OLD, Compatibility.evaluate(req, host(api = 1)).single().code)
    @Test fun apiTooNew() = assertEquals(RejectCode.HOST_API_TOO_NEW, Compatibility.evaluate(req, host(api = 5)).single().code)
    @Test fun nativeAbiMismatchBothDirections() {
        assertEquals(RejectCode.NATIVE_ABI_MISMATCH, Compatibility.evaluate(req, host(abi = 6)).single().code)
        assertEquals(RejectCode.NATIVE_ABI_MISMATCH, Compatibility.evaluate(req, host(abi = 8)).single().code)
    }
    @Test fun missingCapability() {
        val r = Compatibility.evaluate(req, host(caps = setOf("a")))
        assertEquals(listOf(Reject(RejectCode.MISSING_CAPABILITY, "b")), r)
    }
    @Test fun sdkTooLow() = assertEquals(RejectCode.SDK_TOO_LOW, Compatibility.evaluate(req, host(sdk = 27)).single().code)
    @Test fun needsNewHostOnlyForHostSideReasons() {
        assertTrue(Compatibility.needsNewHost(Compatibility.evaluate(req, host(abi = 8))))
        assertTrue(!Compatibility.needsNewHost(emptyList()))
        assertTrue(!Compatibility.needsNewHost(listOf(Reject(RejectCode.BAD_SIGNATURE))))
    }
}
