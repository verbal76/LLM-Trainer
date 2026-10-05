package com.hotatticgames.hag.runtime

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class HandlesTest {
    private val released = mutableListOf<Long>()
    private fun h(ptr: Long, parent: Handle? = null) = Handle(ptr, "x", { released += it }, parent)

    @Test fun useOfFreedHandleThrowsAndFreeIsIdempotent() {
        val t = HandleTable()
        val id = t.put(h(11))
        t.use(id) { assertEquals(11, it.ptr) }
        t.free(id); t.free(id)
        assertEquals(listOf(11L), released)
        val e = assertFailsWith<HagException> { t.use(id) { } }
        assertEquals(HagException.USAGE, e.code)
        assertFailsWith<HagException> { t.use(9999) { } }
    }

    @Test fun idsAreNeverReusedEvenForTheSamePointer() {
        val t = HandleTable()
        val a = t.put(h(5)); t.free(a)
        val b = t.put(h(5))
        assertTrue(a != b)
        assertFailsWith<HagException> { t.use(a) { } }
    }

    @Test fun freeDuringUseIsDeferredUntilTheCallEnds() {
        val t = HandleTable()
        val id = t.put(h(7))
        t.use(id) {
            t.free(id) // e.g. from inside a streaming callback
            assertTrue(released.isEmpty(), "must not free while a native call is in flight")
        }
        assertEquals(listOf(7L), released)
    }

    @Test fun childKeepsParentAliveAndCountsAsLiveChild() {
        val t = HandleTable()
        val mid = t.put(h(1))
        val model = t.use(mid) { it }
        val sid = t.put(h(2, parent = model))
        assertEquals(1, model.liveChildren.get())
        t.free(mid)
        assertTrue(released.isEmpty(), "model must outlive its session")
        t.free(sid)
        assertEquals(listOf(2L, 1L), released) // session first, then the model
        assertEquals(0, model.liveChildren.get())
    }

    @Test fun concurrentUseAndFreeNeverDoubleReleases() {
        repeat(200) {
            released.clear()
            val t = HandleTable()
            val id = t.put(h(3))
            val th = (1..4).map { Thread { repeat(50) { runCatching { t.use(id) { Thread.yield() } } } } }
            th.forEach { it.start() }
            t.free(id)
            th.forEach { it.join() }
            assertEquals(1, released.size)
        }
    }
}

class Utf8StreamTest {
    @Test fun passesWholeCharactersThrough() {
        val s = Utf8Stream()
        assertEquals("héllo 😀", s.push("héllo 😀".toByteArray()))
        assertEquals("", s.finish())
    }

    @Test fun neverSplitsAMultiByteCharacterEvenWhenChunksDo() {
        val bytes = "a😀é".toByteArray() // 1 + 4 + 2 bytes
        val s = Utf8Stream()
        val sb = StringBuilder()
        for (b in bytes) sb.append(s.push(byteArrayOf(b))) // one byte at a time
        assertEquals("a😀é", sb.toString())
        assertEquals("", s.finish())
    }

    @Test fun danglingTailSurfacesAsReplacementCharacter() {
        val s = Utf8Stream()
        assertEquals("", s.push(byteArrayOf(0xF0.toByte(), 0x9F.toByte())))
        assertEquals("�", s.finish())
    }
}

class HagRuntimeUnavailableTest {
    @Test fun missingRuntimeLibraryIsUnavailableNotACrash() {
        val r = HagRuntime.load(loader = { throw UnsatisfiedLinkError("no libhagrt") })
        val u = r as EngineStatus.Unavailable
        assertEquals("loadRuntime", u.stage)
        assertTrue(u.reason.contains("no libhagrt"))
    }

    @Test fun cpuGateFailureNeverLoadsTheEngineLibrary() {
        val loaded = mutableListOf<String>()
        val r = HagRuntime.load(loader = { loaded += it }, cpuGateOverride = { "CPU lacks dotprod" })
        val u = r as EngineStatus.Unavailable
        assertEquals("cpuGate", u.stage)
        assertEquals(listOf(HagRuntime.RUNTIME_LIB), loaded, "libhagengine.so must not be loaded when the gate fails")
    }

    @Test fun missingEngineLibraryIsReported() {
        val r = HagRuntime.load(
            loader = { if (it == HagRuntime.ENGINE_LIB) throw UnsatisfiedLinkError("nope") },
            cpuGateOverride = { null },
        )
        assertEquals("loadEngine", (r as EngineStatus.Unavailable).stage)
    }
}
