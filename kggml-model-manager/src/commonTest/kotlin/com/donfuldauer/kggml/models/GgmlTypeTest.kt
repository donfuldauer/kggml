package com.donfuldauer.kggml.models

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class GgmlTypeTest {
    @Test
    fun blockLayoutsMatchGgml() {
        assertEquals(4L * 10, GgmlType.F32.byteSize(10))
        assertEquals(18L * 2, GgmlType.Q4_0.byteSize(64))
        assertEquals(144L, GgmlType.Q4_K.byteSize(256))
        assertEquals(210L * 4, GgmlType.Q6_K.byteSize(1024))
        assertEquals(36L, GgmlType.NVFP4.byteSize(64))
        assertEquals(0L, GgmlType.Q8_0.byteSize(0))
        // Bits per weight sanity: Q4_K = 4.5 bpw, Q8_0 = 8.5 bpw.
        assertEquals(4.5, GgmlType.Q4_K.typeSize * 8.0 / GgmlType.Q4_K.blockSize)
        assertEquals(8.5, GgmlType.Q8_0.typeSize * 8.0 / GgmlType.Q8_0.blockSize)
    }

    @Test
    fun idsRoundTripAndRemovedIdsAreUnknown() {
        GgmlType.entries.forEach { assertEquals(it, GgmlType.fromId(it.id)) }
        listOf(4, 5, 31, 32, 33, 36, 37, 38, 41, -1).forEach { assertNull(GgmlType.fromId(it)) }
        assertEquals(GgmlType.entries.size, GgmlType.entries.map { it.id }.toSet().size)
        assertTrue(GgmlType.Q2_K.isQuantized)
        assertFalse(GgmlType.BF16.isQuantized)
    }

    @Test
    fun invalidCountsAndOverflowFail() {
        assertFailsWith<ModelFormatException> { GgmlType.Q4_0.byteSize(33) }
        assertFailsWith<ModelFormatException> { GgmlType.F32.byteSize(-1) }
        assertFailsWith<ModelFormatException> { GgmlType.F64.byteSize(Long.MAX_VALUE / 2) }
        assertFailsWith<ModelFormatException> { checkedAdd(Long.MAX_VALUE, 1) }
    }
}
