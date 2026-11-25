package com.donfuldauer.kggml.samples.showcase

import kotlin.test.Test
import kotlin.test.assertEquals

class FormatTest {
    @Test
    fun formatsBytesCountsAndDecimals() {
        assertEquals("512 B", formatBytes(512))
        assertEquals("1.50 KiB", formatBytes(1536))
        assertEquals("74.1 MiB", formatBytes(77_700_000))
        assertEquals("3.98 GiB", formatBytes(4_270_000_000))
        assertEquals("-2.00 KiB", formatBytes(-2048))
        assertEquals("494.0 M", formatCount(494_000_000))
        assertEquals("1.24 B", formatCount(1_240_000_000))
        assertEquals("12", formatCount(12))
        assertEquals("0.9659", formatDecimal(0.96594, 4))
        assertEquals("-0.5", formatDecimal(-0.5, 1))
        assertEquals("3", formatDecimal(2.6, 0))
    }

    @Test
    fun describeNamesTheExceptionType() {
        assertEquals("IllegalStateException: boom", IllegalStateException("boom").describe())
    }

    @Test
    fun toRgbaExpandsGrayAndRgb() {
        assertEquals(listOf<Byte>(1, 2, 3, -1), toRgba(1, 1, 3, byteArrayOf(1, 2, 3)).toList())
        assertEquals(listOf<Byte>(9, 9, 9, -1), toRgba(1, 1, 1, byteArrayOf(9)).toList())
        assertEquals(listOf<Byte>(1, 2, 3, 4), toRgba(1, 1, 4, byteArrayOf(1, 2, 3, 4)).toList())
    }
}
