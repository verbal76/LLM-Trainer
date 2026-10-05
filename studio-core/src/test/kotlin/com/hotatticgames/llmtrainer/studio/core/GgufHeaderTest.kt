package com.hotatticgames.llmtrainer.studio.core

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class GgufHeaderTest {
    private fun tmp(bytes: ByteArray) = File.createTempFile("mdl", ".gguf").also { it.deleteOnExit(); it.writeBytes(bytes) }

    @Test fun readsArchitectureGeometryParametersAndChatTemplate() {
        val i = assertNotNull(GgufHeader.read(tmp(ArtifactKit.gguf(arch = "qwen3", layers = 28, heads = 16, kvHeads = 8, embd = 128, vocab = 300))))
        assertEquals("qwen3", i.architecture); assertEquals(28, i.layers); assertEquals(16, i.heads); assertEquals(8, i.kvHeads)
        assertEquals(128, i.embd); assertEquals(8, i.headDim); assertEquals(4096, i.ctxTrain); assertEquals(128, i.ffn)
        assertEquals(300L, i.vocab); assertEquals(128L * 300 + 128L * 128, i.parameterCount); assertTrue(i.chatTemplatePresent); assertEquals("tiny", i.sizeLabel)
        assertEquals(3, i.version); assertEquals(2L, i.tensorCount)
    }

    @Test fun noTemplateIsReported() {
        assertFalse(assertNotNull(GgufHeader.read(tmp(ArtifactKit.gguf(template = false)))).chatTemplatePresent)
    }

    @Test fun nonGgufTruncatedAndGarbageReturnNullNeverThrow() {
        assertNull(GgufHeader.read(tmp("hello world, not a model".toByteArray())))
        assertNull(GgufHeader.read(tmp(ArtifactKit.gguf().copyOf(100))))
        assertNull(GgufHeader.read(tmp(ByteArray(0))))
        assertNull(GgufHeader.read(File("/definitely/not/here.gguf")))
        val badVersion = ArtifactKit.gguf().also { it[4] = 9 }
        assertNull(GgufHeader.read(tmp(badVersion)))
        assertTrue(GgufHeader.hasMagic(tmp(ArtifactKit.gguf())) && !GgufHeader.hasMagic(tmp(ByteArray(10))))
    }
}
