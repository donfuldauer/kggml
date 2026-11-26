package com.donfuldauer.kggml.models

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class LittleEndianReaderTest {
    @Test
    fun byteArraySourceReadsWithinBoundsAndValidatesArguments() {
        val source = ByteArrayModelSource(byteArrayOf(1, 2, 3, 4, 5))
        val buffer = ByteArray(4)
        assertEquals(3, source.read(2, buffer, 1, 3))
        assertContentEquals(byteArrayOf(0, 3, 4, 5), buffer)
        assertEquals(1, source.read(4, buffer, 0, 4), "short read at the end")
        assertEquals(-1, source.read(5, buffer, 0, 4), "at the end")
        assertEquals(-1, source.read(99, buffer, 0, 4), "past the end")
        assertEquals(0, source.read(0, buffer, 4, 0), "empty range at the end of the buffer")
        assertFailsWith<IllegalArgumentException> { source.read(-1, buffer, 0, 1) }
        assertFailsWith<IllegalArgumentException> { source.read(0, buffer, -1, 1) }
        assertFailsWith<IllegalArgumentException> { source.read(0, buffer, 0, -1) }
        assertFailsWith<IllegalArgumentException> { source.read(0, buffer, 2, 3) }
        assertEquals(5L, source.size)
    }

    @Test
    fun readsLittleEndianPrimitives() {
        val bytes = LeWriter().u8(0xFE).u8(0xFE).i16(0xFFFE).i16(0xFFFE).i32(-2).i32(-2).i64(-2).f32(1.5f).f64(-3.25)
            .i64(Long.MAX_VALUE).toByteArray()
        val r = LittleEndianReader(ByteArrayModelSource(bytes), bufferSize = 3) // a tiny buffer forces refills mid-value
        assertEquals(254, r.readU8())
        assertEquals(-2, r.readI8())
        assertEquals(65534, r.readU16())
        assertEquals(-2, r.readI16())
        assertEquals(-2, r.readI32())
        assertEquals(4_294_967_294L, r.readU32())
        assertEquals(-2L, r.readI64())
        assertEquals(1.5f, r.readF32())
        assertEquals(-3.25, r.readF64())
        assertEquals(Long.MAX_VALUE, r.readU64AsLong("max"))
        assertEquals(bytes.size.toLong(), r.position)
        assertEquals(0L, r.remaining)
    }

    @Test
    fun readU64AsLongRejectsValuesAboveLongMax() {
        val r = LittleEndianReader(ByteArrayModelSource(LeWriter().i64(Long.MIN_VALUE).toByteArray()))
        val e = assertFailsWith<ModelFormatException> { r.readU64AsLong("count") }
        assertEquals("count is too large (9223372036854775808)", e.message)
    }

    @Test
    fun readBytesSpansBufferRefills() {
        val data = ByteArray(10) { it.toByte() }
        val r = LittleEndianReader(ByteArrayModelSource(data), start = 1, bufferSize = 4)
        assertContentEquals(byteArrayOf(1, 2, 3, 4, 5, 6, 7), r.readBytes(7))
        assertEquals(8L, r.position)
        assertContentEquals(ByteArray(0), r.readBytes(0))
        val e = assertFailsWith<ModelFormatException> { r.readBytes(3) }
        assertEquals("truncated file: 3-byte field needs 3 bytes at offset 8, 2 left", e.message)
        assertEquals(8L, r.position, "a failed read consumes nothing")
    }

    @Test
    fun skipOnAKnownSizeSourceChecksTheRemainingBytes() {
        val r = LittleEndianReader(ByteArrayModelSource(ByteArray(10)))
        r.skip(0, "nothing")
        r.skip(10, "all")
        assertEquals(10L, r.position)
        assertEquals("negative size for x", assertFailsWith<ModelFormatException> { r.skip(-1, "x") }.message)
        assertEquals(
            "truncated file: data needs 1 bytes at offset 10, 0 left",
            assertFailsWith<ModelFormatException> { r.skip(1, "data") }.message,
        )
    }

    @Test
    fun skipOnAnUnknownSizeSourceProbesTheLastSkippedByte() {
        val r = LittleEndianReader(UnknownSizeSource(ByteArray(10) { it.toByte() }))
        assertNull(r.remaining)
        r.skip(0, "nothing")
        assertEquals(0L, r.position, "skipping zero bytes reads nothing")
        r.skip(4, "four")
        assertEquals(4L, r.position)
        assertEquals(4, r.readU8(), "the byte after the skipped range")
        r.skip(5, "rest")
        assertEquals(10L, r.position)
        val e = assertFailsWith<ModelFormatException> { r.skip(1, "tail") }
        assertEquals("truncated file: unexpected end of data reading tail at offset 10", e.message)
        r.ensureAvailable(Long.MAX_VALUE, "unknown size cannot be checked up front")
    }

    @Test
    fun checkedArithmeticDetectsOverflowAndNegativeFactors() {
        assertEquals(0L, checkedMultiply(0, Long.MAX_VALUE))
        assertEquals(0L, checkedMultiply(Long.MAX_VALUE, 0))
        assertEquals(Long.MAX_VALUE, checkedMultiply(Long.MAX_VALUE, 1))
        assertEquals(6L * 1_000_000_007L, checkedMultiply(6, 1_000_000_007L))
        assertFailsWith<ModelFormatException> { checkedMultiply(-1, 2) }
        assertFailsWith<ModelFormatException> { checkedMultiply(2, -1) }
        val e = assertFailsWith<ModelFormatException> { checkedMultiply(1L shl 32, 1L shl 31) }
        assertEquals("size overflow (4294967296 * 2147483648)", e.message)
        assertEquals(Long.MAX_VALUE, checkedAdd(Long.MAX_VALUE - 1, 1))
        assertEquals(-1L, checkedAdd(Long.MAX_VALUE, Long.MIN_VALUE), "mixed signs never overflow")
        assertFailsWith<ModelFormatException> { checkedAdd(Long.MIN_VALUE, -1) }
    }
}
