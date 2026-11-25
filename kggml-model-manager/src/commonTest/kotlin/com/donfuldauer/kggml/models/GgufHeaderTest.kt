package com.donfuldauer.kggml.models

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class GgufHeaderTest {
    private fun parse(bytes: ByteArray) = GgufModelHeader.parse(ByteArrayModelSource(bytes))

    @Test
    fun parsesLlamaMetadataAndTensors() {
        for (version in 1..3) {
            val h = parse(tinyLlamaGguf(version))
            assertEquals(version, h.version)
            assertEquals("llama", h.architecture)
            assertEquals("tiny", h.name)
            assertEquals(15, h.fileTypeId)
            assertEquals("Q4_K_M", h.quantization)
            assertEquals(64L, h.contextLength)
            assertEquals(2L, h.blockCount)
            assertEquals(64L, h.embeddingLength)
            assertEquals(4L, h.headCount)
            assertEquals(2L, h.headCountKv)
            assertEquals(16L, h.keyLength)
            assertEquals(16L, h.valueLength)
            assertEquals(8L, h.vocabularySize)
            assertEquals(4, h.tensors.size)
            assertEquals(listOf(256L, 64L), h.tensors[1].dimensions)
            assertEquals(GgmlType.Q4_K, h.tensors[1].type)
            assertEquals(64L * 8 + 2 * 256 * 64 + 64, h.parameterCount)
            assertEquals(64L * 8 * 2 + 2 * 144 * 64 + 64 * 4, h.tensorDataBytes)
            assertEquals(GgmlType.Q4_K, h.dominantTensorType)
            assertEquals(32, h.alignment)
            assertEquals(0L, h.dataOffset % 32)
            assertEquals(0L, h.tensors.first().offset)
        }
    }

    @Test
    fun dataOffsetPointsAtFirstTensor() {
        val bytes = tinyLlamaGguf(withData = true)
        val h = parse(bytes)
        assertEquals(bytes.size.toLong(), h.dataOffset + h.tensors.last().offset + h.tensors.last().byteSize!! +
            (32 - h.tensors.last().byteSize!! % 32) % 32)
    }

    @Test
    fun keepsSmallArraysAndSkipsLargeOnes() {
        val big = List(GgufModelHeader.MAX_RETAINED_ARRAY_ELEMENTS + 1) { "tok$it" }
        val h = parse(
            GgufBuilder().string("general.architecture", "qwen2")
                .stringArray("tokenizer.ggml.tokens", big)
                .intArray("qwen2.attention.head_count_kv", listOf(2, 8, 4))
                .intArray("big.ints", List(5000) { it })
                .u32("qwen2.attention.head_count", 8)
                .u64("qwen2.vocab_size", 151_936)
                .u32("general.file_type", 99)
                .misc("x")
                .f32("x.f32", 1.5f)
                .bool("x.bool", true)
                .tensor("w", listOf(32), GgmlType.Q8_0)
                .build(),
        )
        val tokens = h.metadata["tokenizer.ggml.tokens"] as GgufValue.Array
        assertEquals(big.size.toLong(), tokens.size)
        assertNull(tokens.values)
        assertEquals(GgufValueType.STRING, tokens.elementType)
        assertNull((h.metadata["big.ints"] as GgufValue.Array).values)
        assertEquals(8L, h.headCountKv, "per-layer arrays report the maximum")
        assertEquals(151_936L, h.vocabularySize, "vocab_size wins over the token array")
        assertNull(h.keyLength, "no embedding_length → no default head size")
        assertEquals("Q8_0", h.quantization, "unknown file_type falls back to the dominant tensor type")
        assertEquals(GgufValue.Integer(GgufValueType.UINT8, 200), h.metadata["x.u8"])
        assertEquals(GgufValue.Integer(GgufValueType.INT8, -3), h.metadata["x.i8"])
        assertEquals(GgufValue.Integer(GgufValueType.UINT16, 60000), h.metadata["x.u16"])
        assertEquals(GgufValue.Integer(GgufValueType.INT16, -2), h.metadata["x.i16"])
        assertEquals(GgufValue.Integer(GgufValueType.INT64, -5), h.metadata["x.i64"])
        assertEquals(GgufValue.Real(GgufValueType.FLOAT64, 2.5), h.metadata["x.f64"])
        assertEquals(GgufValue.Real(GgufValueType.FLOAT32, 1.5), h.metadata["x.f32"])
        assertEquals(GgufValue.Bool(true), h.metadata["x.bool"])
        assertEquals("qwen2", h.string("general.architecture"))
        assertNull(h.long("general.architecture"))
    }

    @Test
    fun customAlignmentAndUnknownTensorType() {
        val b = GgufBuilder().apply { alignment = 64 }
            .u32("general.alignment", 64)
            .tensor("a", listOf(16), GgmlType.F32)
            .tensor("b", listOf(16), 77)
        val h = parse(b.build())
        assertEquals(64, h.alignment)
        assertEquals(0L, h.dataOffset % 64)
        assertNull(h.tensors[1].type)
        assertNull(h.tensors[1].byteSize)
        assertEquals(64L, h.tensorDataBytes, "unknown types count as 0 bytes")
        assertNull(h.architecture)
        assertNull(h.contextLength)
    }

    @Test
    fun rejectsCorruptHeaders() {
        val good = tinyLlamaGguf()
        val badMagic = good.copyOf().also { it[0] = 'X'.code.toByte() }
        assertTrue(assertFailsWith<ModelFormatException> { parse(badMagic) }.message!!.contains("not a GGUF"))
        val badVersion = good.copyOf().also { it[4] = 9 }
        assertTrue(assertFailsWith<ModelFormatException> { parse(badVersion) }.message!!.contains("version"))
        assertFailsWith<ModelFormatException> { parse(GgufBuilder().rawKv("k", 42).build()) }
        assertFailsWith<ModelFormatException> { parse(GgufBuilder().u32("general.alignment", 48).build()) }
        assertFailsWith<ModelFormatException> { parse(GgufBuilder().string("general.alignment", "x").build()) }
        assertFailsWith<ModelFormatException> { parse(GgufBuilder().u32("a", 1).u32("a", 2).build()) }
        assertFailsWith<ModelFormatException> { parse(GgufBuilder().tensor("q", listOf(33), GgmlType.Q4_0).build()) }
        assertFailsWith<ModelFormatException> { parse(GgufBuilder().tensor("five", listOf(1, 1, 1, 1, 1), GgmlType.F32).build()) }
        // Implausible counts are rejected before any allocation.
        val hugeCount = LeWriter().bytes("GGUF".encodeToByteArray()).i32(3).i64(Long.MAX_VALUE).i64(0).toByteArray()
        assertFailsWith<ModelFormatException> { parse(hugeCount) }
        val hugeString = LeWriter().bytes("GGUF".encodeToByteArray()).i32(3).i64(0).i64(1).i64(1L shl 40).toByteArray()
        assertFailsWith<ModelFormatException> { parse(hugeString) }
    }

    @Test
    fun everyTruncationFailsCleanly() {
        val good = tinyLlamaGguf()
        val header = parse(good)
        // Every strict prefix of the header (up to the end of the tensor infos) must fail with ModelFormatException.
        val end = good.size
        for (cut in 0 until end) {
            assertFailsWith<ModelFormatException>("prefix $cut") { parse(good.copyOf(cut)) }
            assertFailsWith<ModelFormatException>("unknown-size prefix $cut") {
                GgufModelHeader.parse(UnknownSizeSource(good.copyOf(cut)))
            }
        }
        assertEquals(header.tensors, GgufModelHeader.parse(UnknownSizeSource(good)).tensors)
    }
}
