package com.donfuldauer.kggml.models

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class WhisperHeaderTest {
    private fun parse(bytes: ByteArray, scan: Boolean = true) = WhisperModelHeader.parse(ByteArrayModelSource(bytes), scan)

    @Test
    fun parsesHyperParametersVocabularyAndTensors() {
        val h = parse(tinyWhisperGgml())
        val hp = h.hyperParameters
        assertEquals(51865, hp.vocabularySize)
        assertEquals(1500, hp.audioContext)
        assertEquals(384, hp.audioState)
        assertEquals(4, hp.audioLayers)
        assertEquals(448, hp.textContext)
        assertEquals(80, hp.melBands)
        assertEquals(1, hp.fileTypeId)
        assertEquals(0, hp.quantizationVersion)
        assertEquals("tiny", h.modelType)
        assertTrue(h.isMultilingual)
        assertEquals("F16", h.quantization)
        assertEquals(80 to 201, h.melFilterShape)
        assertEquals(3, h.vocabularyEntries)
        assertEquals(listOf("encoder.conv1.weight", "decoder.token_embedding.weight", "decoder.ln.bias"), h.tensors.map { it.name })
        assertEquals(3L * 80 * 8 + 32 * 4 + 16, h.parameterCount)
        assertEquals(3L * 80 * 8 * 2 + 4 * 34 + 16 * 4, h.tensorDataBytes)
        assertEquals(GgmlType.Q8_0, h.tensors[1].type)
    }

    @Test
    fun quantizedFileTypeCarriesQuantizationVersion() {
        val h = parse(tinyWhisperGgml(ftype = 2008, audioLayers = 32, nVocabHparam = 51864))
        assertEquals(8, h.hyperParameters.fileTypeId)
        assertEquals(2, h.hyperParameters.quantizationVersion)
        assertEquals("Q5_0", h.quantization)
        assertEquals("large", h.modelType)
        assertFalse(h.isMultilingual)
        assertNull(parse(tinyWhisperGgml(ftype = 5, audioLayers = 7)).let { it.quantization ?: it.modelType })
    }

    @Test
    fun headerOnlyModeSkipsTensors() {
        val bytes = tinyWhisperGgml()
        val headerEnd = tinyWhisperGgml(tensors = emptyList()).size
        val h = parse(bytes.copyOf(headerEnd), scan = false)
        assertTrue(h.tensors.isEmpty())
        assertEquals(0L, h.parameterCount)
        // Exact header with no tensors is also a valid (empty) file when scanning.
        assertTrue(parse(bytes.copyOf(headerEnd)).tensors.isEmpty())
        assertEquals(3, WhisperModelHeader.parse(UnknownSizeSource(bytes)).tensors.size)
    }

    @Test
    fun rejectsCorruptFiles() {
        val good = tinyWhisperGgml()
        assertFailsWith<ModelFormatException> { parse(good.copyOf().also { it[0] = 0 }) }
        // n_audio_ctx = 0
        assertFailsWith<ModelFormatException> { parse(good.copyOf().also { for (i in 8 until 12) it[i] = 0 }) }
        // negative ftype
        assertFailsWith<ModelFormatException> { parse(good.copyOf().also { for (i in 44 until 48) it[i] = -1 }) }
        // invalid mel shape
        assertFailsWith<ModelFormatException> { parse(good.copyOf().also { for (i in 48 until 52) it[i] = 0 }) }
        assertFailsWith<ModelFormatException> {
            parse(tinyWhisperGgml(tensors = listOf(Triple("bad", listOf(33), GgmlType.Q4_0))))
        }
        val unknownType = tinyWhisperGgml(tensors = emptyList()) + LeWriter().i32(1).i32(1).i32(99).i32(4).bytes("x".encodeToByteArray()).toByteArray()
        assertFailsWith<ModelFormatException> { parse(unknownType) }
        val tooManyDims = tinyWhisperGgml(tensors = emptyList()) + LeWriter().i32(5).i32(1).i32(0).toByteArray()
        assertFailsWith<ModelFormatException> { parse(tooManyDims) }
        val badName = tinyWhisperGgml(tensors = emptyList()) + LeWriter().i32(1).i32(0).i32(0).toByteArray()
        assertFailsWith<ModelFormatException> { parse(badName) }
        val badDim = tinyWhisperGgml(tensors = emptyList()) + LeWriter().i32(1).i32(1).i32(0).i32(-4).toByteArray()
        assertFailsWith<ModelFormatException> { parse(badDim) }
    }

    @Test
    fun truncatedFilesFail() {
        val good = tinyWhisperGgml()
        val cuts = listOf(0, 3, 4, 20, 47, 52, 60, 52 + 80 * 201 * 4 + 6, good.size - 1, good.size - 100)
        for (cut in cuts) {
            assertFailsWith<ModelFormatException>("cut $cut") { parse(good.copyOf(cut)) }
        }
        assertFailsWith<ModelFormatException> { WhisperModelHeader.parse(UnknownSizeSource(good.copyOf(good.size - 1))) }
    }
}
