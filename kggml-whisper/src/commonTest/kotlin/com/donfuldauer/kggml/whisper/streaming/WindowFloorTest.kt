package com.donfuldauer.kggml.whisper.streaming

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotSame
import kotlin.test.assertTrue

class WindowFloorTest {
    @Test
    fun constantsMatchSixteenKilohertz() {
        assertEquals(16_000, WindowFloor.MIN_WINDOW_SAMPLES)
        assertEquals(480_000, WindowFloor.MAX_WINDOW_SAMPLES)
        assertEquals(0, WindowFloor.samplesForMs(0))
        assertEquals(3_200, WindowFloor.samplesForMs(200))
        assertEquals(15_984, WindowFloor.samplesForMs(999))
        assertEquals(16_000, WindowFloor.samplesForMs(1_000))
        assertEquals(480_000, WindowFloor.samplesForMs(30_000))
        assertFailsWith<IllegalArgumentException> { WindowFloor.samplesForMs(-1) }
    }

    @Test
    fun boundaryLengthsArePaddedToExactlyOneSecondOrKept() {
        for ((ms, expectedSize) in listOf(0 to 16_000, 200 to 16_000, 999 to 16_000, 1_000 to 16_000, 30_000 to 480_000)) {
            val n = WindowFloor.samplesForMs(ms)
            val src = FloatArray(n) { 0.5f + it % 7 }
            val out = WindowFloor.padToMinimum(src)
            assertEquals(expectedSize, out.size, "$ms ms")
            assertContentEquals(src, out.copyOf(n), "$ms ms audio preserved")
            assertTrue(out.drop(n).all { it == 0f }, "$ms ms tail is zero padding")
            assertNotSame(src, out)
        }
    }

    @Test
    fun paddingUsesOnlyTheRequestedPrefix() {
        val src = FloatArray(20_000) { 1f }
        val out = WindowFloor.padToMinimum(src, 100)
        assertEquals(16_000, out.size)
        assertEquals(100, out.count { it == 1f })
        assertEquals(20_000, WindowFloor.padToMinimum(src).size)
        assertFailsWith<IllegalArgumentException> { WindowFloor.padToMinimum(src, 20_001) }
        assertFailsWith<IllegalArgumentException> { WindowFloor.padToMinimum(src, -1) }
    }
}
