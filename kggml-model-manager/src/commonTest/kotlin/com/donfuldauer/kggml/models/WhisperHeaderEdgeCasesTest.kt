package com.donfuldauer.kggml.models

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

/**
 * Whisper `ggml` header bounds and derived properties, checked against the legacy layout read by `whisper.cpp`'s
 * `whisper_model_load` and the `ggml_ftype` / model-size tables of the pinned sources.
 */
class WhisperHeaderEdgeCasesTest {
    private fun parse(bytes: ByteArray) = WhisperModelHeader.parse(ByteArrayModelSource(bytes))

    private fun parseError(bytes: ByteArray): String =
        assertFailsWith<ModelFormatException> { parse(bytes) }.message!!

    private val tinyHparams = listOf(51865, 1500, 384, 6, 4, 448, 384, 6, 4, 80, 1)

    /** Magic, [hparams] and a mel filter shape; the caller appends the rest. */
    private fun prefix(hparams: List<Int> = tinyHparams, nMel: Int = 1, nFft: Int = 1): LeWriter {
        val w = LeWriter().i32(0x67676d6c)
        hparams.forEach { w.i32(it) }
        return w.i32(nMel).i32(nFft)
    }

    /** [prefix] with a 1x1 mel filter and an empty vocabulary: a complete header, ready for tensors. */
    private fun header(): LeWriter = prefix().f32(0f).i32(0)

    @Test
    fun modelTypeFollowsEncoderDepth() {
        // n_audio_layer per model size (whisper.cpp: tiny 4, base 6, small 12, medium 24, large 32).
        val sizes = mapOf(4 to "tiny", 6 to "base", 12 to "small", 24 to "medium", 32 to "large")
        for ((layers, name) in sizes) assertEquals(name, parse(tinyWhisperGgml(audioLayers = layers)).modelType)
        for (layers in listOf(1, 5, 31, 33)) assertNull(parse(tinyWhisperGgml(audioLayers = layers)).modelType)
    }

    @Test
    fun ggmlFileTypeNamesMatchGgmlH() {
        // enum ggml_ftype from ggml/include/ggml.h of the pinned ggml.
        val expected = mapOf(
            0 to "F32", 1 to "F16", 2 to "Q4_0", 3 to "Q4_1", 4 to "Q4_1_SOME_F16", 7 to "Q8_0", 8 to "Q5_0",
            9 to "Q5_1", 10 to "Q2_K", 11 to "Q3_K", 12 to "Q4_K", 13 to "Q5_K", 14 to "Q6_K", 15 to "IQ2_XXS",
            16 to "IQ2_XS", 17 to "IQ3_XXS", 18 to "IQ1_S", 19 to "IQ4_NL", 20 to "IQ3_S", 21 to "IQ2_S",
            22 to "IQ4_XS", 23 to "IQ1_M", 24 to "BF16", 25 to "MXFP4", 26 to "NVFP4",
        )
        for (id in -1..27) assertEquals(expected[id], ggmlFileTypeName(id), "ggml_ftype $id")
        // GGML_FTYPE_UNKNOWN (-1) and the unused ids 5 and 6 have no name.
        listOf(-1, 5, 6).forEach { assertNull(ggmlFileTypeName(it)) }
        // ftype 1014 = GGML_QNT_VERSION 1 * 1000 + Q6_K.
        val q6k = parse(tinyWhisperGgml(ftype = 1014))
        assertEquals("Q6_K", q6k.quantization)
        assertEquals(1, q6k.hyperParameters.quantizationVersion)
    }

    @Test
    fun hyperParametersAboveTheLimitAreRejected() {
        val names = listOf(
            "n_vocab", "n_audio_ctx", "n_audio_state", "n_audio_head", "n_audio_layer",
            "n_text_ctx", "n_text_state", "n_text_head", "n_text_layer", "n_mels",
        )
        val limit = 1 shl 24
        for (i in names.indices) {
            val hp = tinyHparams.toMutableList().also { it[i] = limit + 1 }
            assertEquals("invalid ${names[i]} ${limit + 1}", parseError(prefix(hp).toByteArray()))
            val negative = tinyHparams.toMutableList().also { it[i] = -1 }
            assertEquals("invalid ${names[i]} -1", parseError(prefix(negative).toByteArray()))
        }
        // The limit itself is accepted (header-only parse; the vocabulary is empty).
        val atLimit = tinyHparams.toMutableList().also { it[0] = limit }
        val h = WhisperModelHeader.parse(ByteArrayModelSource(prefix(atLimit).f32(0f).i32(0).toByteArray()), scanTensors = false)
        assertEquals(limit, h.hyperParameters.vocabularySize)
    }

    @Test
    fun melFilterShapeIsBounded() {
        val limit = 1 shl 24
        for ((mel, fft) in listOf(0 to 1, 1 to 0, -1 to 1, 1 to -1, limit + 1 to 1, 1 to limit + 1)) {
            assertEquals("invalid mel filter shape $mel x $fft", parseError(prefix(nMel = mel, nFft = fft).toByteArray()))
        }
        // n_mel * n_fft * 4 bytes of filters must be present.
        assertEquals(
            "truncated file: mel filters needs 24 bytes at offset 56, 4 left",
            parseError(prefix(nMel = 2, nFft = 3).zeros(4).toByteArray()),
        )
    }

    @Test
    fun vocabularyIsBounded() {
        assertEquals("invalid vocabulary count -1", parseError(prefix().f32(0f).i32(-1).toByteArray()))
        assertEquals("invalid vocabulary count ${(1 shl 24) + 1}", parseError(prefix().f32(0f).i32((1 shl 24) + 1).toByteArray()))
        // Token lengths are uint32 and limited to 1 MiB.
        val tooLong = prefix().f32(0f).i32(2).i32(1).bytes("a".encodeToByteArray()).i32(1024 * 1024 + 1)
        assertEquals("token #1 length 1048577 is implausibly large", parseError(tooLong.toByteArray()))
        val negativeAsU32 = prefix().f32(0f).i32(1).i32(-1)
        assertEquals("token #0 length 4294967295 is implausibly large", parseError(negativeAsU32.toByteArray()))
        // A token of exactly 1 MiB is fine.
        val big = prefix().f32(0f).i32(1).i32(1024 * 1024).zeros(1024 * 1024)
        assertEquals(1, parse(big.toByteArray()).vocabularyEntries)
    }

    @Test
    fun tensorHeadersAreBounded() {
        // n_dims must be 1..4.
        assertEquals("tensor #0 has 0 dimensions", parseError(header().i32(0).i32(1).i32(0).toByteArray()))
        assertEquals("tensor #0 has -2 dimensions", parseError(header().i32(-2).i32(1).i32(0).toByteArray()))
        // Name length must be 1..1024 bytes.
        assertEquals("tensor #0 name length 1025", parseError(header().i32(1).i32(1025).i32(0).toByteArray()))
        assertEquals("tensor #0 name length -1", parseError(header().i32(1).i32(-1).i32(0).toByteArray()))
        // A dimension of zero is invalid even after a valid first tensor; the index counts tensors read so far.
        val second = header().i32(1).i32(1).i32(GgmlType.F32.id).i32(2).bytes("a".encodeToByteArray()).zeros(8)
            .i32(2).i32(1).i32(GgmlType.F32.id).i32(3).i32(0)
        assertEquals("tensor #1 has dimension 0", parseError(second.toByteArray()))
        // Four dimensions, a 1024-byte name, and a tensor that ends exactly at end of file are accepted.
        val name = "n".repeat(1024)
        val ok = header().i32(4).i32(1024).i32(GgmlType.F16.id).i32(2).i32(3).i32(1).i32(1)
            .bytes(name.encodeToByteArray()).zeros(12)
        val t = parse(ok.toByteArray()).tensors.single()
        assertEquals(listOf(2L, 3L, 1L, 1L), t.dimensions)
        assertEquals(name, t.name)
        assertEquals(12L, t.byteSize)
        assertEquals(ok.size - 12L, t.dataOffset)
    }

    @Test
    fun tensorsOfUnknownTypeCountAsZeroBytes() {
        // The parser rejects unknown types, but a header built from known parts still follows the documented rule.
        val parsed = parse(tinyWhisperGgml())
        val unknown = WhisperTensorInfo("x", listOf(4, 4), typeId = 99, dataOffset = 0)
        assertNull(unknown.type)
        assertNull(unknown.byteSize)
        assertEquals(16L, unknown.elementCount)
        val header = WhisperModelHeader(parsed.hyperParameters, parsed.melFilterShape, 0, parsed.tensors + unknown)
        assertEquals(parsed.tensorDataBytes, header.tensorDataBytes)
        assertEquals(parsed.parameterCount + 16, header.parameterCount)
    }
}
