package com.hotatticgames.llmtrainer.studio.core

import org.json.JSONObject
import java.math.BigInteger
import java.util.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** The Kotlin port must reproduce factory/llmtrainer/splits.py exactly (golden: factory/tools/gen_golden_splits.py). */
class SimilarityGoldenTest {
    private val g: JSONObject = JSONObject(javaClass.getResourceAsStream("/golden/splits.v1.json")!!.readBytes().toString(Charsets.UTF_8))

    private fun ulong(s: String) = BigInteger(s).toLong()   // Python unsigned value -> Long bit pattern

    @Test fun blake2bMatchesPythonDigestSize8() {
        val rows = g.getJSONArray("blake2b8")
        assertTrue(rows.length() >= 10)
        for (i in 0 until rows.length()) {
            val r = rows.getJSONObject(i)
            assertEquals(ulong(r.get("h").toString()), Blake2b.h64(r.getString("s")), "blake2b of '${r.getString("s").take(20)}'")
        }
    }

    @Test fun blake2bFullLengthKnownVector() {
        // RFC 7693 appendix A: BLAKE2b-512("abc")
        val d = Hashing.hex(Blake2b.digest("abc".toByteArray(), 64))
        assertEquals("ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923", d)
    }

    @Test fun mulmodMatchesBigInteger() {
        val p = BigInteger.ONE.shiftLeft(61).subtract(BigInteger.ONE)
        val r = Random(5)
        repeat(20_000) {
            val a = (r.nextLong() ushr 3); val b = (r.nextLong() ushr 3)
            val aa = a % p.toLong(); val bb = b % p.toLong()
            assertEquals(BigInteger.valueOf(aa).multiply(BigInteger.valueOf(bb)).mod(p).toLong(), Similarity.mulmod(aa, bb))
        }
        val max = p.toLong() - 1
        assertEquals(BigInteger.valueOf(max).multiply(BigInteger.valueOf(max)).mod(p).toLong(), Similarity.mulmod(max, max))
    }

    @Test fun shinglesAndMinhashMatch() {
        val texts = g.getJSONObject("texts")
        for (k in texts.keySet()) {
            val sh = g.getJSONObject("shingles").getJSONArray(k)
            val expected = (0 until sh.length()).map { ulong(sh.get(it).toString()) }.toSet()
            val got = Similarity.shingles(texts.getString(k), 5)
            assertEquals(expected, got, "shingles $k")
            val mh = g.getJSONObject("minhash").getJSONArray(k)
            val sig = Similarity.minhash(got)
            for (i in 0 until 64) assertEquals(ulong(mh.get(i).toString()), sig[i], "minhash $k[$i]")
        }
    }

    private fun pairs(a: org.json.JSONArray) = (0 until a.length()).map { val r = a.getJSONArray(it); Triple(r.getString(0), r.getString(1), r.getDouble(2)) }

    @Test fun nearDuplicatesMatch() {
        val texts = g.getJSONObject("texts").let { o -> o.keySet().associateWith { o.getString(it) } }
        for ((key, thr) in listOf("near_dups_08" to 0.8, "near_dups_05" to 0.5)) {
            val want = pairs(g.getJSONArray(key))
            val got = Similarity.findNearDuplicates(texts, thr, 5).map { Triple(it.a, it.b, it.score) }
            assertEquals(want, got, key)
        }
    }

    @Test fun containmentMatches() {
        val c = g.getJSONObject("containment")
        val cand = c.getJSONObject("candidates").let { o -> o.keySet().associateWith { o.getString(it) } }
        val prot = c.getJSONObject("protected").let { o -> o.keySet().associateWith { o.getString(it) } }
        val got = Similarity.containmentLeaks(cand, prot, 0.8, 5).map { Triple(it.a, it.b, it.score) }
        assertEquals(pairs(c.getJSONArray("result")), got)
    }

    @Test fun groupAssignmentMatchesForManySeedsAndRatios() {
        val arr = g.getJSONArray("assign")
        for (i in 0 until arr.length()) {
            val c = arr.getJSONObject(i)
            val w = c.getJSONObject("weights").let { o -> o.keys().asSequence().associateWith { o.getInt(it) } }
            val rObj = c.getJSONObject("ratios")
            val ratios = linkedMapOf("train" to rObj.getDouble("train"), "validation" to rObj.getDouble("validation"), "test" to rObj.getDouble("test"))
            val want = c.getJSONObject("result").let { o -> o.keys().asSequence().associateWith { o.getString(it) } }
            assertEquals(want, Similarity.assignGroups(w, ratios, c.getLong("seed")), "seed ${c.getLong("seed")}")
        }
    }

    @Test fun assignNeedsThreeGroups() {
        assertFailsWith<IllegalArgumentException> { Similarity.assignGroups(mapOf("a" to 1, "b" to 2), linkedMapOf("train" to 0.7, "validation" to 0.15, "test" to 0.15), 1) }
    }

    @Test fun everySplitNonEmptyWithThreeGroupsAndOrderIndependent() {
        val w = linkedMapOf("a" to 5, "b" to 5, "c" to 5)
        val r = linkedMapOf("train" to 0.8, "validation" to 0.1, "test" to 0.1)
        val x = Similarity.assignGroups(w, r, 42)
        assertEquals(setOf("train", "validation", "test"), x.values.toSet())
        val y = Similarity.assignGroups(linkedMapOf("c" to 5, "a" to 5, "b" to 5), r, 42)
        assertEquals(x, y)
    }

    @Test fun resolveLeakageDropsLessProtectedSide() {
        val t = "Tighten the rear axle nut to 85 N-m and then fit a new split pin before refitting the wheel on the swing arm of the machine."
        val res = Similarity.resolveLeakage(mapOf("tr" to t, "te" to t + " today"), mapOf("tr" to "train", "te" to "test"), 0.8, 5)
        assertEquals(listOf("tr"), res.dropped)
        assertEquals(1, res.found)
        val res2 = Similarity.resolveLeakage(mapOf("v" to t, "te" to t + " today"), mapOf("v" to "validation", "te" to "test"), 0.8, 5)
        assertEquals(listOf("v"), res2.dropped)
    }
}
