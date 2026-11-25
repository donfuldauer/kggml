package com.donfuldauer.kggml.models

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs

class ModelInspectorTest {
    @Test
    fun detectsFormats() {
        val gguf = ModelInspector.inspect(ByteArrayModelSource(tinyLlamaGguf()))
        assertIs<ModelHeaderInfo.Gguf>(gguf)
        assertEquals(ModelFormat.GGUF, gguf.format)
        assertEquals("Q4_K_M", gguf.quantization)
        assertEquals(64L, gguf.contextLength)
        assertEquals(gguf.header.parameterCount, gguf.parameterCount)
        assertEquals(estimateLlmPeakRam(gguf.header), gguf.estimatedPeakRam)

        val whisper = ModelInspector.inspect(ByteArrayModelSource(tinyWhisperGgml()))
        assertIs<ModelHeaderInfo.Whisper>(whisper)
        assertEquals(ModelFormat.WHISPER_GGML, whisper.format)
        assertEquals("F16", whisper.quantization)
        assertEquals(448L, whisper.contextLength)
        assertEquals(whisper.header.parameterCount, whisper.parameterCount)
        assertEquals(estimateWhisperPeakRam(whisper.header), whisper.estimatedPeakRam)
    }

    @Test
    fun rejectsUnknownOrShortFiles() {
        assertFailsWith<ModelFormatException> { ModelInspector.inspect(ByteArrayModelSource(ByteArray(3))) }
        assertFailsWith<ModelFormatException> { ModelInspector.inspect(ByteArrayModelSource("PK\u0003\u0004rest".encodeToByteArray())) }
    }

    @Test
    fun inspectsFilesThroughTheFileSystem() {
        val fs = FakeFileSystem()
        fs.files["/models/w.bin"] = tinyWhisperGgml()
        assertIs<ModelHeaderInfo.Whisper>(ModelInspector.inspectFile("/models/w.bin", fs))
        assertFailsWith<ModelFileSystemException> { ModelInspector.inspectFile("/missing", fs) }
    }
}
