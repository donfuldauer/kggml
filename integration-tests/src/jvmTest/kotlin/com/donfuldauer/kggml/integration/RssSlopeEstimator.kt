package com.donfuldauer.kggml.integration

/**
 * Estimates the steady-state per-cycle RSS growth rate (in KB per cycle) from equally spaced `VmRSS` readings.
 *
 * Uses the Theil-Sen estimator — the median of all pairwise slopes `(y[j] - y[i]) / (j - i)` for `0 <= i < j < n` —
 * evaluated independently on the first and second halves of the measured window.
 *
 * - Taking pairwise slopes across multi-cycle spans removes the upward bias that asymmetric JVM GC / allocator
 *   sawtooth oscillations (gradual ramp over several cycles followed by a sharp drop) cause in consecutive single-cycle
 *   deltas (`y[i + 1] - y[i]`).
 * - Taking the minimum of the two halves requires growth to be sustained across the whole measured window: a single
 *   one-off allocator or arena step (even tens of MB, e.g. glibc promoting a compute buffer into a heap arena) can
 *   affect at most one half while the other stays flat, whereas a genuine per-cycle leak raises both halves equally.
 */
internal object RssSlopeEstimator {
    data class SustainedSlope(
        val firstHalfKbPerCycle: Double,
        val secondHalfKbPerCycle: Double,
    ) {
        val sustainedKbPerCycle: Double
            get() = minOf(firstHalfKbPerCycle, secondHalfKbPerCycle)
    }

    /**
     * Returns the Theil-Sen slope in KB per cycle for [samplesKb], which must contain at least 2 readings.
     */
    fun theilSenSlopeKbPerCycle(samplesKb: List<Long>): Double {
        val n = samplesKb.size
        require(n >= 2) { "theilSenSlopeKbPerCycle requires at least 2 samples, got $n" }
        val slopes = DoubleArray(n * (n - 1) / 2)
        var k = 0
        for (i in 0 until n - 1) {
            val yi = samplesKb[i]
            for (j in i + 1 until n) {
                slopes[k++] = (samplesKb[j] - yi).toDouble() / (j - i)
            }
        }
        slopes.sort()
        val mid = slopes.size / 2
        return if (slopes.size % 2 == 1) {
            slopes[mid]
        } else {
            (slopes[mid - 1] + slopes[mid]) / 2.0
        }
    }

    /**
     * Splits [samplesKb] (at least 4 readings) into two contiguous halves and computes the Theil-Sen slope on each.
     */
    fun sustainedSlopeKbPerCycle(samplesKb: List<Long>): SustainedSlope {
        val n = samplesKb.size
        require(n >= 4) { "sustainedSlopeKbPerCycle requires at least 4 samples, got $n" }
        val mid = n / 2
        return SustainedSlope(
            firstHalfKbPerCycle = theilSenSlopeKbPerCycle(samplesKb.subList(0, mid)),
            secondHalfKbPerCycle = theilSenSlopeKbPerCycle(samplesKb.subList(mid, n)),
        )
    }
}
