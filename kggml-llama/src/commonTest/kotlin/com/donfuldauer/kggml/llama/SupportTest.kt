package com.donfuldauer.kggml.llama

import com.donfuldauer.kggml.core.InvalidInputException
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class Utf8StreamDecoderTest {
    @Test
    fun holdsBackIncompleteSequencesUntilComplete() {
        val bytes = "aé€🙂".encodeToByteArray()
        val d = Utf8StreamDecoder()
        val out = bytes.map { d.decode(byteArrayOf(it)) }
        assertEquals(listOf("a", "", "é", "", "", "€", "", "", "", "🙂"), out)
        assertEquals(0, d.pendingBytes)
    }

    @Test
    fun malformedInputIsReplacedNotHeldForever() {
        val d = Utf8StreamDecoder()
        assertEquals("\uFFFD", d.decode(byteArrayOf(0xFF.toByte())))
        assertEquals("", d.decode(byteArrayOf(0xE2.toByte(), 0x82.toByte())))
        assertEquals(2, d.pendingBytes)
        assertEquals("\uFFFD", d.flush())
        assertEquals(0, d.pendingBytes)
        assertTrue(d.decode(byteArrayOf(0x80.toByte(), 0x80.toByte(), 0x80.toByte(), 0x80.toByte())).isNotEmpty())
    }
}

class SessionCodecTest {
    private val fp = SessionCodec.Fingerprint(288, 32, 1)

    @Test
    fun roundTrips() {
        val encoded = SessionCodec.encode(fp, intArrayOf(1, 70000, -5), byteArrayOf(9, 8, 7))
        val decoded = SessionCodec.decode(encoded)
        assertEquals(fp, decoded.fingerprint)
        assertContentEquals(intArrayOf(1, 70000, -5), decoded.tokens)
        assertContentEquals(byteArrayOf(9, 8, 7), decoded.nativeState)
        assertEquals('K'.code.toByte(), encoded[0])
        assertEquals(SessionCodec.VERSION.toByte(), encoded[4])
        assertEquals(0xCBF43926.toInt(), SessionCodec.crc32("123456789".encodeToByteArray()), "standard CRC-32 check value")
    }

    @Test
    fun rejectsCorruptContainers() {
        val good = SessionCodec.encode(fp, intArrayOf(1, 2), byteArrayOf(5))
        val badMagic = good.copyOf().also { it[0] = 'X'.code.toByte() }
        val badVersion = good.copyOf().also { it[4] = 9 }
        val truncated = good.copyOf(good.size - 1)
        val damaged = good.copyOf().also { it[it.size - 1] = 6 }
        val hugeCount = good.copyOf().also { it[20] = 0x7F; it[23] = 0x7F }
        for (bad in listOf(byteArrayOf(), badMagic, badVersion, truncated, damaged, hugeCount, good + 0)) {
            assertFailsWith<InvalidInputException> { SessionCodec.decode(bad) }
        }
    }
}

class OptionsTest {
    @Test
    fun generationOptionsValidate() {
        GenerationOptions(maxTokens = GenerationOptions.UNLIMITED, seed = 0, temperature = -1f)
        assertTrue(GenerationOptions.GREEDY.isGreedy)
        for (bad in listOf<() -> Unit>(
            { GenerationOptions(maxTokens = 0) },
            { GenerationOptions(seed = -1) },
            { GenerationOptions(seed = 0xFFFFFFFFL) },
            { GenerationOptions(temperature = Float.NaN) },
            { GenerationOptions(repeatPenalty = 0f) },
            { GenerationOptions(repeatLastN = -1) },
            { GenerationOptions(stopSequences = listOf("")) },
            { GenerationOptions(topP = -0.1f) },
            { GenerationOptions(minP = 1.5f) },
        )) {
            assertFailsWith<IllegalArgumentException> { bad() }
        }
    }

    @Test
    fun contextConfigAndEngineOptionsValidate() {
        assertEquals(4096, LlamaContextConfig.DEFAULT.contextLength)
        LlamaContextConfig(contextLength = LlamaContextConfig.MODEL_CONTEXT_LENGTH)
        assertFailsWith<IllegalArgumentException> { LlamaContextConfig(contextLength = -1) }
        assertFailsWith<IllegalArgumentException> { LlamaContextConfig(batchSize = 0) }
        assertFailsWith<IllegalArgumentException> { LlamaContextConfig(microBatchSize = -1) }
        assertFailsWith<IllegalArgumentException> { LlamaContextConfig(embeddingContextLength = 0) }
        assertFailsWith<IllegalArgumentException> { LlamaEngineOptions(gpuDevice = -1) }
        assertEquals(listOf(-1, 0, 1, 2, 3), LlamaPooling.entries.map { it.nativeValue })
    }

    @Test
    fun bridgeHolderWrapsLoadFailures() {
        val holder = LlamaBridgeHolder { throw UnsupportedOperationException("no lib") }
        val e = assertFailsWith<com.donfuldauer.kggml.core.NativeFailureException> { holder.get() }
        assertTrue(e.message!!.startsWith("Failed to load the Kggml llama native library"))
    }
}
