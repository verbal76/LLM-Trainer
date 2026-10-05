package com.hotatticgames.llmtrainer.qualify

import java.math.BigDecimal
import java.math.MathContext
import java.math.RoundingMode

/**
 * Python-compatible number rendering/rounding so reasons and rounded figures are byte-identical to the
 * Python spec on every platform (JVM and ART): exact binary value, round-half-even.
 */
internal object PyFormat {
    /** Python `round(x, nd)` for floats. */
    fun round(x: Double, nd: Int): Double {
        if (x.isNaN() || x.isInfinite()) return x
        return BigDecimal(x).setScale(nd, RoundingMode.HALF_EVEN).toDouble()
    }

    /** Python `round(x)` (banker's rounding to an integer). */
    fun roundInt(x: Double): Long = BigDecimal(x).setScale(0, RoundingMode.HALF_EVEN).toLong()

    /** Python `f"{x:.{nd}f}"`. */
    fun fixed(x: Double, nd: Int): String {
        val bd = BigDecimal(x).setScale(nd, RoundingMode.HALF_EVEN)
        val s = bd.toPlainString()
        return if (bd.signum() == 0 && (x < 0.0 || 1.0 / x < 0.0)) "-$s" else s
    }

    /** Python `f"{x:.0%}"`. */
    fun percent0(x: Double): String = fixed(x * 100.0, 0) + "%"

    /** Python `repr(float)` for ordinary magnitudes: the shortest decimal that round-trips. */
    fun repr(x: Double): String {
        if (x == 0.0) return if (1.0 / x < 0.0) "-0.0" else "0.0"
        if (x.isNaN() || x.isInfinite()) return x.toString()
        val exact = BigDecimal(x)
        for (digits in 1..17) {
            val r = exact.round(MathContext(digits, RoundingMode.HALF_EVEN))
            if (r.toDouble() == x) {
                val s = r.stripTrailingZeros().toPlainString()
                return if ('.' in s) s else "$s.0"
            }
        }
        return exact.toPlainString()
    }
}
