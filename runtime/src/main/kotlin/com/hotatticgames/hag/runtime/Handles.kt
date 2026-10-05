package com.hotatticgames.hag.runtime

import java.nio.ByteBuffer
import java.nio.CharBuffer
import java.nio.charset.CodingErrorAction
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * A native resource with deferred release: [close] marks it closed immediately (no new use is possible) and the
 * native free runs when the last in-flight use ends. A free issued from inside a callback therefore cannot pull the
 * object out from under the running native call, and a child (session) keeps its parent (model) alive.
 */
internal class Handle(
    val ptr: Long,
    val kind: String,
    private val release: (Long) -> Unit,
    private val parent: Handle? = null,
) {
    private var refs = 1 // the table's own reference
    private var closed = false
    val liveChildren = AtomicInteger(0)

    init {
        // A child pins its parent for its whole life.
        parent?.let { it.acquire(); it.liveChildren.incrementAndGet() }
    }

    @Synchronized fun acquire() {
        if (closed) throw HagException(HagException.USAGE, "$kind handle was already freed")
        refs++
    }

    fun unuse() {
        val last = synchronized(this) { --refs == 0 }
        if (last) {
            try {
                release(ptr)
            } finally {
                parent?.let { it.liveChildren.decrementAndGet(); it.unuse() }
            }
        }
    }

    fun close() {
        val first = synchronized(this) { if (closed) false else { closed = true; true } }
        if (first) unuse()
    }
}

/** Opaque, never-reused Long ids -> handles. Ids are NOT pointers, so a freed id can never alias a new object. */
internal class HandleTable {
    private val next = AtomicLong(1)
    private val map = ConcurrentHashMap<Long, Handle>()

    fun put(h: Handle): Long = next.getAndIncrement().also { map[it] = h }

    fun <T> use(id: Long, f: (Handle) -> T): T {
        val h = map[id] ?: throw HagException(HagException.USAGE, "invalid or freed handle $id")
        h.acquire()
        try {
            return f(h)
        } finally {
            h.unuse()
        }
    }

    /** Freeing an unknown/already freed id is a no-op (idempotent close). */
    fun free(id: Long) { map.remove(id)?.close() }

    fun size() = map.size
}

/** Decodes a stream of UTF-8 byte chunks into text, never splitting a character even if chunks do. */
internal class Utf8Stream {
    private val dec = Charsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPLACE)
        .onUnmappableCharacter(CodingErrorAction.REPLACE)
    private var carry = ByteArray(0)

    fun push(chunk: ByteArray): String {
        val all = if (carry.isEmpty()) chunk else carry + chunk
        val inBuf = ByteBuffer.wrap(all)
        val out = CharBuffer.allocate(all.size + 4)
        dec.decode(inBuf, out, false)
        carry = if (inBuf.hasRemaining()) all.copyOfRange(inBuf.position(), all.size) else ByteArray(0)
        out.flip()
        return out.toString()
    }

    /** A dangling incomplete tail (engine bug) is surfaced as the replacement character instead of being lost. */
    fun finish(): String = if (carry.isEmpty()) "" else { carry = ByteArray(0); "�" }
}
