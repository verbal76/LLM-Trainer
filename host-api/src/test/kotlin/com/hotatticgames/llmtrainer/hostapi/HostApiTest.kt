package com.hotatticgames.llmtrainer.hostapi

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class HostApiTest {
    @Test fun levelIsTwoAndMajorIsUnchanged() {
        assertEquals(2, HOST_API_LEVEL)
        assertEquals(0, HOST_API_LEVEL / 100) // still API major 0: level-1 bundles stay installable
    }

    @Test fun capabilityNamesAreTheDocumentedStrings() {
        assertEquals("inference.gguf.v1", Capabilities.INFERENCE_GGUF_V1)
        assertEquals("training.patch.v1", Capabilities.TRAINING_PATCH_V1)
        assertEquals("core.v1", Capabilities.CORE_V1)
    }

    @Test fun engineCapabilitiesAreAdvertisedOnlyWhenTheEngineInitialised() {
        val without = HostCapabilities.advertised(engineAvailable = false)
        assertFalse(Capabilities.INFERENCE_GGUF_V1 in without)
        assertFalse(Capabilities.TRAINING_PATCH_V1 in without)
        assertTrue(Capabilities.CORE_V1 in without)

        val with = HostCapabilities.advertised(engineAvailable = true)
        assertTrue(Capabilities.INFERENCE_GGUF_V1 in with && Capabilities.TRAINING_PATCH_V1 in with)
        assertTrue(without.all { it in with })
    }
}
