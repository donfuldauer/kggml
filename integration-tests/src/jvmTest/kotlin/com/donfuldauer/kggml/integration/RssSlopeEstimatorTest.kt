package com.donfuldauer.kggml.integration

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class RssSlopeEstimatorTest {
    @Test
    fun rejectsTooFewSamples() {
        assertFailsWith<IllegalArgumentException> { RssSlopeEstimator.theilSenSlopeKbPerCycle(emptyList()) }
        assertFailsWith<IllegalArgumentException> { RssSlopeEstimator.theilSenSlopeKbPerCycle(listOf(500_000L)) }
        assertFailsWith<IllegalArgumentException> { RssSlopeEstimator.sustainedSlopeKbPerCycle(listOf(1L, 2L, 3L)) }
    }

    @Test
    fun flatAndExactLinearSeriesReturnExactSlopes() {
        val flat = List(26) { 540L * 1024 }
        assertEquals(0.0, RssSlopeEstimator.theilSenSlopeKbPerCycle(flat), 1e-6)
        assertEquals(0.0, RssSlopeEstimator.sustainedSlopeKbPerCycle(flat).sustainedKbPerCycle, 1e-6)

        val growing = List(26) { i -> 540L * 1024 + i * 2048L }
        assertEquals(2048.0, RssSlopeEstimator.theilSenSlopeKbPerCycle(growing), 1e-6)
        assertEquals(2048.0, RssSlopeEstimator.sustainedSlopeKbPerCycle(growing).sustainedKbPerCycle, 1e-6)

        val shrinking = List(26) { i -> 540L * 1024 - i * 512L }
        assertEquals(-512.0, RssSlopeEstimator.theilSenSlopeKbPerCycle(shrinking), 1e-6)
        assertEquals(-512.0, RssSlopeEstimator.sustainedSlopeKbPerCycle(shrinking).sustainedKbPerCycle, 1e-6)

        assertEquals(1500.0, RssSlopeEstimator.theilSenSlopeKbPerCycle(listOf(1000L, 2500L)), 1e-6)
    }

    @Test
    fun periodicGcSawtoothWithZeroNetGrowthStaysWellBelowLeakBound() {
        // Asymmetric sawtooth: climbs +2 MB/cycle for 4 cycles, then GC drops -8 MB in 1 cycle.
        // Consecutive-delta median would falsely report +2048 KB/cycle on this zero-leak signal.
        val patternMb = listOf(535L, 537L, 539L, 541L, 543L)
        val samplesKb = List(26) { i -> patternMb[i % patternMb.size] * 1024L }
        val fullSlope = RssSlopeEstimator.theilSenSlopeKbPerCycle(samplesKb)
        val sustained = RssSlopeEstimator.sustainedSlopeKbPerCycle(samplesKb).sustainedKbPerCycle
        assertTrue(abs(fullSlope) < 256.0, "expected near-zero full Theil-Sen slope, got $fullSlope KB/cycle")
        assertTrue(sustained < 512.0, "expected sustained slope well below 1024 KB/cycle, got $sustained KB/cycle")
    }

    @Test
    fun ubuntuCiFlakeTraceFromIssue127StaysWellBelowLeakBound() {
        // Exact VmRSS trace (in MB) from Ubuntu CI PR #126 (issue #127): flat/oscillating ~14 MB net over 20 cycles,
        // where the old consecutive-delta median falsely reported 1474 KB/cycle.
        val ciTraceMb = listOf(
            541L, 532L, 533L, 543L, 535L, 535L, 545L, 535L, 535L, 535L,
            538L, 548L, 540L, 541L, 541L, 543L, 553L, 551L, 545L, 555L,
        )
        val sustained = RssSlopeEstimator.sustainedSlopeKbPerCycle(ciTraceMb.map { it * 1024L }).sustainedKbPerCycle
        assertTrue(sustained < 1024.0, "expected Ubuntu CI flake trace well below 1024 KB/cycle, got $sustained KB/cycle")
    }

    @Test
    fun oneOffAllocatorOrArenaStepOfAnySizeDoesNotTriggerLeakBound() {
        // Real Linux trace where glibc arena promotion caused a one-off 56 MB step (300 -> 333 -> 356) then stayed flat.
        val devboxArenaStepMb = listOf(
            298L, 300L, 300L, 300L, 300L, 300L, 300L, 300L, 333L, 356L, 356L, 356L, 357L,
            357L, 357L, 357L, 358L, 358L, 358L, 358L, 358L, 358L, 358L, 358L, 358L, 358L,
        )
        val devboxSustained = RssSlopeEstimator.sustainedSlopeKbPerCycle(devboxArenaStepMb.map { it * 1024L }).sustainedKbPerCycle
        assertTrue(devboxSustained < 256.0, "expected one-off arena step to stay flat, got $devboxSustained KB/cycle")

        // Sweep a 60 MB 2-cycle ramp across every possible cycle position, including straddling the midpoint.
        for (start in 1 until 24) {
            val samplesKb = List(26) { i ->
                val step = when {
                    i < start -> 0L
                    i == start -> 30L
                    else -> 60L
                }
                (300L + step) * 1024L
            }
            val sustained = RssSlopeEstimator.sustainedSlopeKbPerCycle(samplesKb).sustainedKbPerCycle
            assertTrue(
                sustained < 1024.0,
                "one-off 60 MB step starting at cycle $start should stay under 1024 KB/cycle, got $sustained",
            )
        }
    }

    @Test
    fun catchesSteadyLinearAndPeriodicLeaksEvenWithSawtoothNoise() {
        val sawtoothMb = listOf(0L, 3L, 6L, 9L)
        val linearLeakKb = List(26) { i -> (535L + sawtoothMb[i % sawtoothMb.size]) * 1024L + i * 2048L }
        val linearSlope = RssSlopeEstimator.sustainedSlopeKbPerCycle(linearLeakKb).sustainedKbPerCycle
        assertTrue(linearSlope >= 1536.0, "expected 2 MB/cycle leak + sawtooth to exceed 1536 KB/cycle, got $linearSlope")

        // Periodic/chunked leak: +6 MB every 3 cycles (2 MB/cycle average), which consecutive-delta median misses (0 KB/cycle).
        val chunkedLeakKb = List(26) { i -> (535L + 6L * (i / 3)) * 1024L }
        val chunkedSlope = RssSlopeEstimator.sustainedSlopeKbPerCycle(chunkedLeakKb).sustainedKbPerCycle
        assertTrue(chunkedSlope >= 1536.0, "expected 6 MB / 3 cycles leak to exceed 1536 KB/cycle, got $chunkedSlope")
    }
}
