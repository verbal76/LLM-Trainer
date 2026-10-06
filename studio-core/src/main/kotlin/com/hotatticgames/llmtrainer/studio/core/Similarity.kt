package com.hotatticgames.llmtrainer.studio.core

/** BLAKE2b (RFC 7693), unkeyed, variable digest length. Needed because the Python splits use `blake2b(digest_size=8)`. */
object Blake2b {
    private val IV = longArrayOf(
        0x6a09e667f3bcc908UL.toLong(), 0xbb67ae8584caa73bUL.toLong(), 0x3c6ef372fe94f82bUL.toLong(), 0xa54ff53a5f1d36f1UL.toLong(),
        0x510e527fade682d1UL.toLong(), 0x9b05688c2b3e6c1fUL.toLong(), 0x1f83d9abfb41bd6bUL.toLong(), 0x5be0cd19137e2179UL.toLong(),
    )
    private val SIGMA = arrayOf(
        intArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        intArrayOf(14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3),
        intArrayOf(11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4),
        intArrayOf(7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8),
        intArrayOf(9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13),
        intArrayOf(2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9),
        intArrayOf(12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11),
        intArrayOf(13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10),
        intArrayOf(6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5),
        intArrayOf(10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0),
    )

    fun digest(data: ByteArray, outLen: Int): ByteArray {
        require(outLen in 1..64)
        val h = IV.copyOf()
        h[0] = h[0] xor 0x01010000L xor outLen.toLong()
        var t = 0L
        val n = data.size
        var off = 0
        val block = ByteArray(128)
        if (n == 0) {
            compress(h, block, 0, true)
        } else {
            while (n - off > 128) {
                System.arraycopy(data, off, block, 0, 128)
                t += 128
                compress(h, block, t, false)
                off += 128
            }
            val rem = n - off
            java.util.Arrays.fill(block, 0)
            System.arraycopy(data, off, block, 0, rem)
            t += rem
            compress(h, block, t, true)
        }
        val out = ByteArray(outLen)
        for (i in 0 until outLen) out[i] = (h[i / 8] ushr (8 * (i % 8))).toByte()
        return out
    }

    private fun compress(h: LongArray, block: ByteArray, t: Long, last: Boolean) {
        val m = LongArray(16)
        for (i in 0 until 16) {
            var v = 0L
            for (j in 7 downTo 0) v = (v shl 8) or (block[i * 8 + j].toLong() and 0xFF)
            m[i] = v
        }
        val v = LongArray(16)
        for (i in 0 until 8) { v[i] = h[i]; v[i + 8] = IV[i] }
        v[12] = v[12] xor t
        if (last) v[14] = v[14].inv()
        for (r in 0 until 12) {
            val s = SIGMA[r % 10]
            g(v, 0, 4, 8, 12, m[s[0]], m[s[1]]); g(v, 1, 5, 9, 13, m[s[2]], m[s[3]])
            g(v, 2, 6, 10, 14, m[s[4]], m[s[5]]); g(v, 3, 7, 11, 15, m[s[6]], m[s[7]])
            g(v, 0, 5, 10, 15, m[s[8]], m[s[9]]); g(v, 1, 6, 11, 12, m[s[10]], m[s[11]])
            g(v, 2, 7, 8, 13, m[s[12]], m[s[13]]); g(v, 3, 4, 9, 14, m[s[14]], m[s[15]])
        }
        for (i in 0 until 8) h[i] = h[i] xor v[i] xor v[i + 8]
    }

    private fun g(v: LongArray, a: Int, b: Int, c: Int, d: Int, x: Long, y: Long) {
        v[a] = v[a] + v[b] + x
        v[d] = java.lang.Long.rotateRight(v[d] xor v[a], 32)
        v[c] = v[c] + v[d]
        v[b] = java.lang.Long.rotateRight(v[b] xor v[c], 24)
        v[a] = v[a] + v[b] + y
        v[d] = java.lang.Long.rotateRight(v[d] xor v[a], 16)
        v[c] = v[c] + v[d]
        v[b] = java.lang.Long.rotateRight(v[b] xor v[c], 63)
    }

    /** Big-endian unsigned 64-bit read of an 8-byte digest (as Python `int.from_bytes(.., "big")`), kept in a Long bit pattern. */
    fun h64(s: String): Long {
        val d = digest(s.toByteArray(Charsets.UTF_8), 8)
        var v = 0L
        for (b in d) v = (v shl 8) or (b.toLong() and 0xFF)
        return v
    }
}

/**
 * Port of `factory/llmtrainer/splits.py`: shingling, MinHash-LSH near-duplicate detection, containment, deterministic
 * group assignment and leakage resolution. Output must equal the Python reference (golden-tested).
 */
object Similarity {
    private const val P = (1L shl 61) - 1
    const val NUM_PERM = 64
    const val BANDS = 16
    const val ROWS = NUM_PERM / BANDS
    val SPLIT_PRIORITY = mapOf("train" to 0, "validation" to 1, "test" to 2)

    private val PERM_A = LongArray(NUM_PERM) { i -> java.lang.Long.remainderUnsigned(Blake2b.h64("a$i"), P) or 1L }
    private val PERM_B = LongArray(NUM_PERM) { i -> java.lang.Long.remainderUnsigned(Blake2b.h64("b$i"), P) }

    /** (a * b) mod (2^61 - 1) for 0 <= a, b < 2^61 without 128-bit arithmetic. */
    fun mulmod(a: Long, b: Long): Long {
        val aHi = a ushr 31; val aLo = a and 0x7FFFFFFFL
        val bHi = b ushr 31; val bLo = b and 0x7FFFFFFFL
        val hh = aHi * bHi                       // < 2^60, weight 2^62 == 2 (mod P)
        val mid = aHi * bLo + aLo * bHi          // < 2^62, weight 2^31
        val ll = aLo * bLo                       // < 2^62
        val midHi = mid ushr 30; val midLo = mid and 0x3FFFFFFFL   // mid*2^31 = midHi*2^61 + midLo*2^31
        var s = (hh shl 1) + midHi + (midLo shl 31) + ll
        s = (s and P) + (s ushr 61)
        s = (s and P) + (s ushr 61)
        if (s >= P) s -= P
        return s
    }

    private fun modAdd(a: Long, b: Long): Long { var s = a + b; if (s >= P) s -= P; return s }

    private val WORD = Regex("[a-z0-9]+")
    fun words(text: String): List<String> = WORD.findAll(text.lowercase()).map { it.value }.toList()

    fun shingles(text: String, k: Int = 5): Set<Long> {
        val w = words(text)
        if (w.size <= k) return setOf(Blake2b.h64(w.joinToString(" ")))
        val out = LinkedHashSet<Long>()
        for (i in 0..w.size - k) out.add(Blake2b.h64(w.subList(i, i + k).joinToString(" ")))
        return out
    }

    fun jaccard(a: Set<Long>, b: Set<Long>): Double {
        if (a.isEmpty() && b.isEmpty()) return 1.0
        var inter = 0
        val (small, big) = if (a.size <= b.size) a to b else b to a
        for (x in small) if (x in big) inter++
        return inter.toDouble() / (a.size + b.size - inter)
    }

    fun minhash(sh: Set<Long>): LongArray {
        val out = LongArray(NUM_PERM) { Long.MAX_VALUE }
        val xs = LongArray(sh.size)
        var n = 0
        for (x in sh) xs[n++] = java.lang.Long.remainderUnsigned(x, P)
        for (p in 0 until NUM_PERM) {
            var best = Long.MAX_VALUE
            for (x in xs) { val v = modAdd(mulmod(PERM_A[p], x), PERM_B[p]); if (v < best) best = v }
            out[p] = best
        }
        return out
    }

    data class Pair3(val a: String, val b: String, val score: Double)

    fun round6(x: Double): Double = Math.round(x * 1e6) / 1e6

    /** All id pairs with exact shingle-Jaccard >= threshold (LSH candidates, exactly verified). Sorted by (a, b). */
    fun findNearDuplicates(texts: Map<String, String>, threshold: Double = 0.8, k: Int = 5): List<Pair3> {
        val sh = texts.mapValues { shingles(it.value, k) }
        val sig = sh.mapValues { minhash(it.value) }
        val buckets = HashMap<String, MutableList<String>>()
        for ((id, s) in sig) for (b in 0 until BANDS) {
            val key = StringBuilder().append(b)
            for (r in 0 until ROWS) key.append(':').append(s[b * ROWS + r])
            buckets.getOrPut(key.toString()) { ArrayList() }.add(id)
        }
        val cands = java.util.TreeSet<Pair<String, String>>(compareBy({ it.first }, { it.second }))
        for (ids in buckets.values) if (ids.size > 1) {
            val sorted = ids.sorted()
            for (i in sorted.indices) for (j in i + 1 until sorted.size) if (sorted[i] != sorted[j]) cands.add(sorted[i] to sorted[j])
        }
        val out = ArrayList<Pair3>()
        for ((a, b) in cands) {
            val j = jaccard(sh.getValue(a), sh.getValue(b))
            if (j >= threshold) out.add(Pair3(a, b, round6(j)))
        }
        return out
    }

    /** Deterministic whole-group assignment (Python `assign_groups`). Needs >= 3 groups; ratios keep their insertion order. */
    fun assignGroups(groupWeights: Map<String, Int>, ratios: Map<String, Double>, seed: Long): Map<String, String> {
        require(groupWeights.size >= 3) { "need at least 3 groups to form train/validation/test, got ${groupWeights.size}" }
        val order = groupWeights.keys.sortedWith { x, y ->
            val c = java.lang.Long.compareUnsigned(Blake2b.h64("$seed|$x"), Blake2b.h64("$seed|$y"))
            if (c != 0) c else x.compareTo(y)
        }
        val total = groupWeights.values.sum()
        val target = ratios.mapValues { it.value * total }
        val got = ratios.keys.associateWith { 0 }.toMutableMap()
        val assign = LinkedHashMap<String, String>()
        val seeds = listOf("test", "validation", "train")
        for ((g, s) in order.take(3).zip(seeds)) { assign[g] = s; got[s] = got.getValue(s) + groupWeights.getValue(g) }
        for (g in order.drop(3)) {
            var best: String? = null
            var bestKey = Double.NEGATIVE_INFINITY
            var bestTie = false
            for (s in ratios.keys) {
                val key = target.getValue(s) - got.getValue(s)
                val tie = SPLIT_PRIORITY[s] == 0
                if (best == null || key > bestKey || (key == bestKey && tie && !bestTie)) { best = s; bestKey = key; bestTie = tie }
            }
            assign[g] = best!!
            got[best] = got.getValue(best) + groupWeights.getValue(g)
        }
        return assign
    }

    fun crossSplitPairs(pairs: List<Pair3>, splitOf: Map<String, String>): List<Pair3> =
        pairs.filter { splitOf[it.a] != splitOf[it.b] }

    class Resolved(val remaining: Map<String, String>, val dropped: List<String>, val found: Int)

    /** Drops the less-protected side of every cross-split near-duplicate pair (test > validation > train). */
    fun resolveLeakage(texts: Map<String, String>, splitOf: Map<String, String>, threshold: Double, k: Int): Resolved {
        val pairs = findNearDuplicates(texts, threshold, k)
        val dropped = LinkedHashSet<String>()
        var found = 0
        for (p in crossSplitPairs(pairs, splitOf)) {
            found++
            if (p.a in dropped || p.b in dropped) continue
            val victim = if (SPLIT_PRIORITY.getValue(splitOf.getValue(p.a)) < SPLIT_PRIORITY.getValue(splitOf.getValue(p.b))) p.a else p.b
            dropped.add(victim)
        }
        return Resolved(splitOf.filterKeys { it !in dropped }, dropped.sorted(), found)
    }

    /** Candidates whose shingles are (almost) contained in a protected text: (candidate id, protected id, containment). */
    fun containmentLeaks(candidates: Map<String, String>, protectedTexts: Map<String, String>, threshold: Double, k: Int): List<Pair3> {
        val index = HashMap<Long, MutableSet<String>>()
        for ((pid, t) in protectedTexts) for (s in shingles(t, k)) index.getOrPut(s) { HashSet() }.add(pid)
        val out = ArrayList<Pair3>()
        for (cid in candidates.keys.sorted()) {
            val sh = shingles(candidates.getValue(cid), k)
            val hits = HashMap<String, Int>()
            for (s in sh) index[s]?.let { for (pid in it) hits[pid] = (hits[pid] ?: 0) + 1 }
            if (hits.isNotEmpty()) {
                var bestPid = ""
                var bestN = -1
                for (pid in hits.keys.sorted()) if (hits.getValue(pid) > bestN) { bestN = hits.getValue(pid); bestPid = pid }
                val c = bestN.toDouble() / sh.size
                if (c >= threshold) out.add(Pair3(cid, bestPid, round6(c)))
            }
        }
        return out
    }
}
