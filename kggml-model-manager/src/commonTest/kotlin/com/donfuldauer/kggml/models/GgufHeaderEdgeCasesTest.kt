package com.donfuldauer.kggml.models

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * GGUF value types, array handling, tensor-info validation and derived properties, checked against the GGUF layout
 * documented in `ggml/include/gguf.h` of the pinned llama.cpp.
 */
class GgufHeaderEdgeCasesTest {
    private fun parse(bytes: ByteArray) = GgufModelHeader.parse(ByteArrayModelSource(bytes))

    private fun parseError(bytes: ByteArray): String =
        assertFailsWith<ModelFormatException> { parse(bytes) }.message!!

    /** Hand-written GGUF file: header, then [body] writes the metadata key/values and tensor infos. */
    private class Raw(val version: Int = 3) {
        val w = LeWriter()
        fun len(n: Long) = if (version == 1) w.i32(n.toInt()) else w.i64(n)
        fun str(s: String) = apply { val b = s.encodeToByteArray(); len(b.size.toLong()); w.bytes(b) }
        fun header(tensors: Long, kvs: Long) = apply { w.bytes("GGUF".encodeToByteArray()).i32(version); len(tensors); len(kvs) }
        fun key(name: String, typeId: Int) = apply { str(name); w.i32(typeId) }
        fun arrayKey(name: String, elementTypeId: Int, count: Long) = apply { key(name, 9); w.i32(elementTypeId); len(count) }
        fun tensor(name: String, dims: List<Long>, typeId: Int, offset: Long) = apply {
            str(name); w.i32(dims.size); dims.forEach { len(it) }; w.i32(typeId).i64(offset)
        }
        fun bytes() = w.toByteArray()
    }

    private val sentinel = 0x00C0FFEE

    @Test
    fun valueTypeIdsMatchGgufH() {
        val expected = listOf(
            "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32", "FLOAT32", "BOOL", "STRING", "ARRAY",
            "UINT64", "INT64", "FLOAT64",
        )
        expected.forEachIndexed { id, name -> assertEquals(name, GgufValueType.fromId(id)?.name, "gguf_type $id") }
        assertNull(GgufValueType.fromId(13), "GGUF_TYPE_COUNT is not a value type")
        assertNull(GgufValueType.fromId(-1))
    }

    @Test
    fun smallArraysOfEveryScalarTypeAreRetained() {
        val r = Raw().header(tensors = 0, kvs = 13)
        r.arrayKey("u8", 0, 2).w.u8(0).u8(255)
        r.arrayKey("i8", 1, 2).w.u8(0x80).u8(0x7F)
        r.arrayKey("u16", 2, 1).w.i16(0xFFFF)
        r.arrayKey("i16", 3, 1).w.i16(0x8000)
        r.arrayKey("u32", 4, 2).w.i32(-1).i32(7)
        r.arrayKey("i32", 5, 2).w.i32(-1).i32(7)
        r.arrayKey("f32", 6, 1).w.f32(0.5f)
        r.arrayKey("bool", 7, 3).w.u8(1).u8(0).u8(2)
        r.arrayKey("str", 8, 2).str("").str("hé")
        r.arrayKey("u64", 10, 1).w.i64(-1L)
        r.arrayKey("i64", 11, 1).w.i64(Long.MIN_VALUE)
        r.arrayKey("f64", 12, 1).w.f64(-0.25)
        r.arrayKey("empty", 5, 0)
        val h = parse(r.bytes())

        fun ints(t: GgufValueType, vararg v: Long) = GgufValue.Array(t, v.size.toLong(), v.map { GgufValue.Integer(t, it) })
        assertEquals(ints(GgufValueType.UINT8, 0, 255), h.metadata["u8"])
        assertEquals(ints(GgufValueType.INT8, -128, 127), h.metadata["i8"])
        assertEquals(ints(GgufValueType.UINT16, 65535), h.metadata["u16"])
        assertEquals(ints(GgufValueType.INT16, -32768), h.metadata["i16"])
        assertEquals(ints(GgufValueType.UINT32, 4_294_967_295L, 7), h.metadata["u32"])
        assertEquals(ints(GgufValueType.INT32, -1, 7), h.metadata["i32"])
        // UINT64 above Long.MAX_VALUE keeps its raw bits (documented on GgufValue.Integer).
        assertEquals(ints(GgufValueType.UINT64, -1L), h.metadata["u64"])
        assertEquals(ints(GgufValueType.INT64, Long.MIN_VALUE), h.metadata["i64"])
        assertEquals(GgufValue.Array(GgufValueType.FLOAT32, 1, listOf(GgufValue.Real(GgufValueType.FLOAT32, 0.5))), h.metadata["f32"])
        assertEquals(GgufValue.Array(GgufValueType.FLOAT64, 1, listOf(GgufValue.Real(GgufValueType.FLOAT64, -0.25))), h.metadata["f64"])
        // Any non-zero byte is true.
        assertEquals(
            GgufValue.Array(GgufValueType.BOOL, 3, listOf(GgufValue.Bool(true), GgufValue.Bool(false), GgufValue.Bool(true))),
            h.metadata["bool"],
        )
        assertEquals(GgufValue.Array(GgufValueType.STRING, 2, listOf(GgufValue.Text(""), GgufValue.Text("hé"))), h.metadata["str"])
        assertEquals(GgufValue.Array(GgufValueType.INT32, 0, emptyList()), h.metadata["empty"])
        assertEquals(GgufValueType.ARRAY, h.metadata["u8"]!!.type)
        assertEquals(GgufValueType.BOOL, (h.metadata["bool"] as GgufValue.Array).values!![0].type)
        assertEquals(GgufValueType.STRING, (h.metadata["str"] as GgufValue.Array).values!![0].type)

        // long(): the maximum of an integer array, null for non-integer or empty arrays.
        assertEquals(4_294_967_295L, h.long("u32"))
        assertEquals(127L, h.long("i8"))
        assertNull(h.long("f32"))
        assertNull(h.long("bool"))
        assertNull(h.long("empty"))
        assertNull(h.long("missing"))
        assertNull(h.string("str"), "an array is not a string")
    }

    @Test
    fun largeArraysAreSkippedByElementWidth() {
        val n = GgufModelHeader.MAX_RETAINED_ARRAY_ELEMENTS + 1L
        // (type id, bytes per element) from gguf.h; the key after the array only parses if the skip width is right.
        val widths = listOf(0 to 1, 1 to 1, 7 to 1, 2 to 2, 3 to 2, 4 to 4, 5 to 4, 6 to 4, 10 to 8, 11 to 8, 12 to 8)
        for ((typeId, width) in widths) {
            val r = Raw().header(tensors = 0, kvs = 2)
            r.arrayKey("big", typeId, n).w.zeros((n * width).toInt())
            r.key("after", 4).w.i32(sentinel)
            val h = parse(r.bytes())
            val big = h.metadata["big"] as GgufValue.Array
            assertEquals(GgufValueType.fromId(typeId), big.elementType)
            assertEquals(n, big.size)
            assertNull(big.values, "type $typeId")
            assertNull(h.long("big"), "skipped arrays have no values to take the maximum of")
            assertEquals(sentinel.toLong(), h.long("after"), "type $typeId skipped with width $width")
        }
    }

    @Test
    fun largeStringArraysAreSkippedInEveryVersion() {
        val n = GgufModelHeader.MAX_RETAINED_ARRAY_ELEMENTS + 1L
        for (version in 1..3) {
            val r = Raw(version).header(tensors = 0, kvs = 2)
            r.arrayKey("tokenizer.ggml.tokens", 8, n)
            for (i in 0 until n) r.str("tok$i")
            r.key("after", 4).w.i32(sentinel)
            val h = parse(r.bytes())
            assertEquals(sentinel.toLong(), h.long("after"), "version $version")
            assertNull((h.metadata["tokenizer.ggml.tokens"] as GgufValue.Array).values)
        }
    }

    @Test
    fun malformedArraysAreRejected() {
        val nested = Raw().header(0, 1).arrayKey("k", 9, 1).bytes()
        assertEquals("nested GGUF arrays are not supported (k)", parseError(nested))

        val unknownElement = Raw().header(0, 1).key("k", 9).also { it.w.i32(13) }.bytes()
        assertEquals("unknown GGUF value type 13 for k[]", parseError(unknownElement))

        val unknownScalar = Raw().header(0, 1).key("k", 99).bytes()
        assertEquals("unknown GGUF value type 99 for k", parseError(unknownScalar))

        // Array length above 2^31 is rejected before anything is allocated or skipped.
        val hugeLength = Raw().header(0, 1).arrayKey("k", 0, (1L shl 31) + 1).bytes()
        assertEquals("k array length 2147483649 is implausibly large", parseError(hugeLength))

        // A string inside a skipped string array claiming more than 64 MiB.
        val r = Raw().header(0, 1).arrayKey("k", 8, GgufModelHeader.MAX_RETAINED_ARRAY_ELEMENTS + 1L)
        r.len(64L * 1024 * 1024 + 1)
        assertEquals("k[0] length 67108865 is implausibly large", parseError(r.bytes()))

        // Skipped fixed-width array longer than the file.
        val short = Raw().header(0, 1).arrayKey("k", 10, 2000).also { it.w.zeros(8) }.bytes()
        assertTrue(parseError(short).startsWith("truncated file: k array needs 16000 bytes"))
    }

    @Test
    fun unsupportedVersionsAreRejected() {
        for (version in listOf(0, 4, -1)) {
            val bytes = LeWriter().bytes("GGUF".encodeToByteArray()).i32(version).i64(0).i64(0).toByteArray()
            assertEquals("unsupported GGUF version $version", parseError(bytes))
        }
        // A uint64 count with the top bit set does not fit a Long.
        val hugeU64 = LeWriter().bytes("GGUF".encodeToByteArray()).i32(3).i64(-1L).i64(0).toByteArray()
        assertEquals("tensor count is too large (18446744073709551615)", parseError(hugeU64))
    }

    @Test
    fun alignmentMustBeAPositivePowerOfTwo() {
        assertEquals("general.alignment 0 is not a power of two", parseError(GgufBuilder().u32("general.alignment", 0).build()))
        // UINT32 0xFFFFFFFF does not fit an Int alignment.
        assertEquals("general.alignment -1 is not a power of two", parseError(GgufBuilder().u32("general.alignment", -1).build()))
        assertEquals("general.alignment must be an integer", parseError(GgufBuilder().f32("general.alignment", 32f).build()))
        val ok = parse(GgufBuilder().apply { alignment = 1 }.u32("general.alignment", 1).tensor("t", listOf(3), GgmlType.I8).build())
        assertEquals(1, ok.alignment)
    }

    @Test
    fun tensorInfosAreValidated() {
        val zeroDims = Raw().header(1, 0).tensor("t", emptyList(), 0, 0).bytes()
        assertEquals("tensor t has 0 dimensions", parseError(zeroDims))

        // Offsets must be multiples of the alignment (default 32, GGUF_DEFAULT_ALIGNMENT).
        val unaligned = Raw().header(1, 0).tensor("t", listOf(4), 0, 16).bytes()
        assertEquals("tensor t offset 16 is not aligned", parseError(unaligned))

        val hugeDim = Raw().header(1, 0).tensor("t", listOf(-1L), 0, 0).bytes()
        assertEquals("tensor t dimension is too large (18446744073709551615)", parseError(hugeDim))

        val overflow = Raw().header(1, 0).tensor("t", listOf(1L shl 32, 1L shl 32), 0, 0).bytes()
        assertTrue(parseError(overflow).startsWith("size overflow"))

        val hugeOffset = Raw().header(1, 0).also { it.str("t"); it.w.i32(1); it.len(4); it.w.i32(0).i64(Long.MIN_VALUE) }.bytes()
        assertEquals("tensor t offset is too large (9223372036854775808)", parseError(hugeOffset))

        // Version 1 stores dimensions as uint32.
        val v1 = parse(Raw(1).header(1, 0).tensor("t", listOf(0xFFFF_FFFFL), GgmlType.I8.id, 0).bytes())
        assertEquals(listOf(4_294_967_295L), v1.tensors.single().dimensions)
        assertEquals(4_294_967_295L, v1.tensors.single().byteSize)
    }

    @Test
    fun headSizesPreferExplicitKeysAndNeedValidDefaults() {
        val explicit = parse(
            GgufBuilder().string("general.architecture", "gemma3")
                .u32("gemma3.embedding_length", 1024)
                .u32("gemma3.attention.head_count", 8)
                .u32("gemma3.attention.key_length", 256)
                .u32("gemma3.attention.value_length", 192)
                .build(),
        )
        assertEquals(256L, explicit.keyLength)
        assertEquals(192L, explicit.valueLength)
        assertEquals(8L, explicit.headCountKv, "head_count_kv defaults to head_count")

        val zeroHeads = parse(
            GgufBuilder().string("general.architecture", "a").u32("a.embedding_length", 64).u32("a.attention.head_count", 0).build(),
        )
        assertNull(zeroHeads.keyLength, "no division by a zero head count")
        assertNull(zeroHeads.valueLength)

        val noHeads = parse(GgufBuilder().string("general.architecture", "a").u32("a.embedding_length", 64).build())
        assertNull(noHeads.keyLength)
        assertNull(noHeads.headCountKv)
        assertNull(noHeads.vocabularySize)
    }

    @Test
    fun quantizationFallsBackToTheTensorHoldingMostBytes() {
        // Two Q8_0 tensors (2 * 34 bytes) vs one F16 tensor of 64 elements (128 bytes): F16 holds more bytes.
        val h = parse(
            GgufBuilder().tensor("a", listOf(32), GgmlType.Q8_0).tensor("b", listOf(32), GgmlType.Q8_0)
                .tensor("c", listOf(64), GgmlType.F16).build(),
        )
        assertNull(h.fileTypeId)
        assertEquals(GgmlType.F16, h.dominantTensorType)
        assertEquals("F16", h.quantization)

        val unknownOnly = parse(GgufBuilder().tensor("x", listOf(8), 77).build())
        assertNull(unknownOnly.dominantTensorType)
        assertNull(unknownOnly.quantization)
        assertEquals(8L, unknownOnly.parameterCount)
        assertEquals(0L, unknownOnly.tensorDataBytes)

        val empty = parse(GgufBuilder().build())
        assertNull(empty.quantization)
        assertEquals(0L, empty.parameterCount)
        assertEquals(32L, empty.dataOffset, "24-byte header rounded up to the 32-byte alignment")
    }

    @Test
    fun llamaFileTypeNamesMatchLlamaH() {
        // llama_ftype from include/llama.h of the pinned llama.cpp.
        val expected = mapOf(
            0 to "F32", 1 to "F16", 2 to "Q4_0", 3 to "Q4_1", 7 to "Q8_0", 8 to "Q5_0", 9 to "Q5_1", 10 to "Q2_K",
            11 to "Q3_K_S", 12 to "Q3_K_M", 13 to "Q3_K_L", 14 to "Q4_K_S", 15 to "Q4_K_M", 16 to "Q5_K_S",
            17 to "Q5_K_M", 18 to "Q6_K", 19 to "IQ2_XXS", 20 to "IQ2_XS", 21 to "Q2_K_S", 22 to "IQ3_XS",
            23 to "IQ3_XXS", 24 to "IQ1_S", 25 to "IQ4_NL", 26 to "IQ3_S", 27 to "IQ3_M", 28 to "IQ2_S", 29 to "IQ2_M",
            30 to "IQ4_XS", 31 to "IQ1_M", 32 to "BF16", 36 to "TQ1_0", 37 to "TQ2_0", 38 to "MXFP4_MOE", 39 to "NVFP4",
        )
        for (id in -1..40) assertEquals(expected[id], llamaFileTypeName(id), "llama_ftype $id")
        // Removed (Q4_1_SOME_F16, Q4_2, Q4_3, Q4_0_4_4/4_8/8_8) and LLAMA_FTYPE_GUESSED are not file types.
        listOf(4, 5, 6, 33, 34, 35, 1024).forEach { assertNull(llamaFileTypeName(it)) }

        val h = parse(GgufBuilder().u32("general.file_type", 32).tensor("w", listOf(32), GgmlType.Q8_0).build())
        assertEquals("BF16", h.quantization, "general.file_type wins over the dominant tensor type")
    }
}
