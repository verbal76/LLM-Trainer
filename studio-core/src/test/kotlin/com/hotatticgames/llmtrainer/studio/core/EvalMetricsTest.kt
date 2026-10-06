package com.hotatticgames.llmtrainer.studio.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Ports of the Python evalsuite semantics (factory/llmtrainer/evalsuite/metrics.py). */
class EvalMetricsTest {
    private fun q(v: Double, u: String) = EvalMetrics.qty(v, u)!!

    @Test fun unitsNormalizeAcrossSystems() {
        assertTrue(EvalMetrics.factCorrect("Tighten to 85 N·m.", q(85.0, "n·m")))
        assertTrue(EvalMetrics.factCorrect("The torque is 62.7 lb-ft", q(85.0, "n·m")), "lb-ft converted within the 2% rounding tolerance")
        assertFalse(EvalMetrics.factCorrect("Tighten to 60 N-m.", q(85.0, "n·m")))
        assertTrue(EvalMetrics.factCorrect("about 3 mm", q(0.003, "m")))
        assertTrue(EvalMetrics.factCorrect("the engine is at 212 °F", q(100.0, "°c")), "temperature F->C")
    }

    @Test fun answerAssertingAConflictingValueOfTheSameDimensionIsWrong() {
        assertFalse(EvalMetrics.factCorrect("It is 85 N-m, or maybe 60 N-m", q(85.0, "n·m")))
        assertTrue(EvalMetrics.factCorrect("It is 85 N-m at 3 mm gap", q(85.0, "n·m")), "other dimensions do not conflict")
        assertFalse(EvalMetrics.factCorrect("no number here", q(85.0, "n·m")))
    }

    @Test fun unknownUnitsAreNotGuessed() {
        assertEquals(null, EvalMetrics.qty(5.0, "furlongs"))
        assertTrue(EvalMetrics.extractQuantities("5 furlongs and 3 mm").map { it.unit } == listOf("mm"))
    }

    @Test fun unsupportedClaimsAreFlaggedAgainstTheProvidedSources() {
        val support = "The rear axle nut must be tightened to 85 N-m using a calibrated torque wrench."
        val good = EvalMetrics.unsupportedClaims("The rear axle nut must be tightened to 85 N-m with a calibrated torque wrench.", support)
        assertEquals(0 to 1, good)
        val badNumber = EvalMetrics.unsupportedClaims("The rear axle nut must be tightened to 120 N-m with a calibrated torque wrench.", support)
        assertEquals(1 to 1, badNumber)
        val invented = EvalMetrics.unsupportedClaims("Hydraulic suspension bladders require nitrogen pressure checks monthly.", support)
        assertEquals(1 to 1, invented)
        assertEquals(0 to 0, EvalMetrics.unsupportedClaims("I do not know.", support), "hedges are not claims")
        assertEquals(0 to 0, EvalMetrics.unsupportedClaims("Is it 85?", support), "questions are not claims")
    }

    @Test fun terminologyCoverageCountsDistinctWholeWords() {
        assertEquals(2 to 3, EvalMetrics.terminologyUse("Check the caliper and the piston.", listOf("caliper", "piston", "gasket")))
        assertEquals(0 to 1, EvalMetrics.terminologyUse("calipers", listOf("caliper")), "substring is not a use")
    }

    @Test fun probesMatchLikeThePythonSuite() {
        val m = EvalMetrics.PROBES.associate { it.first to it.second }
        assertTrue(EvalMetrics.probePass("27", m.getValue("What is 12 plus 15? Answer with just the number.")))
        assertFalse(EvalMetrics.probePass("272", m.getValue("What is 12 plus 15? Answer with just the number.")))
        assertTrue(EvalMetrics.probePass("PARIS", m.getValue("What is the capital of France? One word.")))
        assertFalse(EvalMetrics.probePass("banana", m.getValue("Write the word 'banana' in all capital letters.")), "case-sensitive probe")
        assertEquals(12, EvalMetrics.PROBES.size)
    }

    @Test fun bootstrapIsDeterministicAndPaired() {
        val a = EvalMetrics.Pairs(); val b = EvalMetrics.Pairs()
        for (i in 0 until 60) { a.add(if (i % 4 == 0) 1.0 else 0.0, 1.0); b.add(if (i % 4 != 3) 1.0 else 0.0, 1.0) }
        val c1 = EvalMetrics.pairedDeltaCi(a, b, 7, 1000)
        val c2 = EvalMetrics.pairedDeltaCi(a, b, 7, 1000)
        assertEquals(c1, c2)
        assertTrue(c1.first > 0, "specialist clearly better: interval excludes zero ${c1}")
        val same = EvalMetrics.pairedDeltaCi(a, a, 7, 1000)
        assertEquals(0.0, same.first); assertEquals(0.0, same.second)
        val tiny = EvalMetrics.Pairs().also { it.add(0.0, 1.0); it.add(1.0, 1.0) }
        val tiny2 = EvalMetrics.Pairs().also { it.add(1.0, 1.0); it.add(1.0, 1.0) }
        val ci = EvalMetrics.pairedDeltaCi(tiny, tiny2, 1, 1000)
        assertTrue(ci.first <= 0.0, "n=2 cannot be significant: $ci")
        assertEquals(0.0 to 0.0, EvalMetrics.pairedDeltaCi(EvalMetrics.Pairs(), EvalMetrics.Pairs(), 1, 100))
    }

    @Test fun lexicalRetrieverIsDeterministicAndIgnoresUnrelatedChunks() {
        val r = LexicalRetriever(listOf("b/1" to "brake caliper piston seal", "a/1" to "carburetor float bowl", "c/1" to "brake pad thickness"))
        assertEquals(listOf("b/1", "c/1"), r.top("replace the brake caliper", 3).map { it.first })
        assertTrue(r.top("zzzz qqqq", 3).isEmpty())
    }
}
