package com.donfuldauer.kggml.models

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class RamEstimateTest {
    private val mib = 1024L * 1024

    private fun llama(ctx: Int) = GgufModelHeader.parse(
        ByteArrayModelSource(
            GgufBuilder()
                .string("general.architecture", "llama")
                .u32("llama.context_length", ctx)
                .u32("llama.embedding_length", 4096)
                .u32("llama.block_count", 32)
                .u32("llama.attention.head_count", 32)
                .u32("llama.attention.head_count_kv", 8)
                .u32("llama.vocab_size", 128_256)
                .tensor("w", listOf(4096, 4096), GgmlType.Q4_K)
                .build(),
        ),
    )

    @Test
    fun llmKvCacheMatchesLlamaCppFormula() {
        val h = llama(131_072)
        // Default: context capped at 4096; F16 KV = 32 layers * 4096 * 8 kv heads * (128 + 128) * 2 bytes = 512 MiB.
        val e = estimateLlmPeakRam(h)
        assertEquals(512 * mib, e.kvCacheBytes)
        assertEquals(4096L * 4096 / 256 * 144, e.weightsBytes)
        val logits = 128_256L * 512 * 4
        val activations = 4096L * 512 * 4 * 8
        val scores = 32L * 512 * 4096 * 4
        assertEquals(logits + activations + scores, e.computeBytes)
        assertEquals(64 * mib, e.overheadBytes)
        assertEquals(e.weightsBytes + e.kvCacheBytes + e.computeBytes + e.overheadBytes, e.totalBytes)

        val fa = estimateLlmPeakRam(h, LlmRamEstimateOptions(contextLength = 2048, flashAttention = true, kvCacheElementBytes = 1.0))
        assertEquals(128 * mib, fa.kvCacheBytes)
        assertEquals(logits + activations, fa.computeBytes)

        val full = estimateLlmPeakRam(h, LlmRamEstimateOptions(contextLength = 0))
        assertEquals(512 * mib * 32, full.kvCacheBytes)
    }

    @Test
    fun smallContextCapsBatchAndModelDefault() {
        val h = llama(256)
        val e = estimateLlmPeakRam(h)
        assertEquals(32L * 256 * 8 * 256 * 2, e.kvCacheBytes, "model context below the cap is used as-is")
        assertEquals(128_256L * 256 * 4 + 4096L * 256 * 4 * 8 + 32L * 256 * 256 * 4, e.computeBytes, "batch capped at n_ctx")
    }

    @Test
    fun nonLlmGgufDegradesToWeightsAndOverhead() {
        val h = GgufModelHeader.parse(ByteArrayModelSource(GgufBuilder().tensor("w", listOf(64), GgmlType.F16).build()))
        val e = estimateLlmPeakRam(h)
        assertEquals(0L, e.kvCacheBytes)
        assertEquals(0L, e.computeBytes)
        assertEquals(128L + 64 * mib, e.totalBytes)
    }

    @Test
    fun whisperEstimate() {
        val h = WhisperModelHeader.parse(ByteArrayModelSource(tinyWhisperGgml()))
        val e = estimateWhisperPeakRam(h)
        val selfKv = 4L * 448 * 384 * 2
        val crossKv = 4L * 1500 * 384 * 2
        assertEquals((selfKv + crossKv) * 2, e.kvCacheBytes)
        assertEquals(1500L * 384 * 4 * 24 + 51865L * 448 * 4, e.computeBytes)
        assertEquals(h.tensorDataBytes, e.weightsBytes)
        assertTrue(e.totalBytes > 16 * mib)
    }

    @Test
    fun optionsValidate() {
        assertFailsWith<IllegalArgumentException> { LlmRamEstimateOptions(contextLength = -1) }
        assertFailsWith<IllegalArgumentException> { LlmRamEstimateOptions(batchSize = 0) }
        assertFailsWith<IllegalArgumentException> { LlmRamEstimateOptions(kvCacheElementBytes = Double.NaN) }
    }

    @Test
    fun zeroOrMissingTrainedContext() {
        val h = GgufModelHeader.parse(
            ByteArrayModelSource(
                GgufBuilder()
                    .string("general.architecture", "llama")
                    .u32("llama.embedding_length", 64)
                    .u32("llama.block_count", 2)
                    .u32("llama.attention.head_count", 4)
                    .u32("llama.vocab_size", 100)
                    .build(),
            ),
        )
        // contextLength = 0 means "the trained context", which is absent (0): no KV cache, no attention scores,
        // and the batch is not capped by a zero context.
        val full = estimateLlmPeakRam(h, LlmRamEstimateOptions(contextLength = 0, batchSize = 8))
        assertEquals(0L, full.kvCacheBytes)
        assertEquals(100L * 8 * 4 + 64L * 8 * 4 * 8, full.computeBytes)
        // The default falls back to the 4096-token cap: 2 layers * 4096 * 4 kv heads * (16 + 16) * 2 bytes.
        val default = estimateLlmPeakRam(h, LlmRamEstimateOptions(batchSize = 8))
        assertEquals(2L * 4096 * 4 * (16 + 16) * 2, default.kvCacheBytes)
        assertEquals(100L * 8 * 4 + 64L * 8 * 4 * 8 + 4L * 8 * 4096 * 4, default.computeBytes)
    }

    @Test
    fun kvElementSizeMustBePositiveAndFinite() {
        for (bytes in listOf(0.0, -1.0, Double.POSITIVE_INFINITY)) {
            assertFailsWith<IllegalArgumentException>("$bytes") { LlmRamEstimateOptions(kvCacheElementBytes = bytes) }
        }
        assertEquals(0.5, LlmRamEstimateOptions(kvCacheElementBytes = 0.5).kvCacheElementBytes)
    }
}
