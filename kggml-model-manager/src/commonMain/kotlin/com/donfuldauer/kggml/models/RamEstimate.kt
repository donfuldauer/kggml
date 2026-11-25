package com.donfuldauer.kggml.models

/**
 * Pre-load estimate of the peak memory an engine needs for a model, computed from the header only.
 *
 * All values are estimates for planning (refusing a load on a device that is too small, choosing a context size);
 * the real numbers depend on the backend, its buffer alignment, and runtime options.
 *
 * @property weightsBytes tensor data. With `useMmap` these pages are file-backed and can be reclaimed by the OS, but
 *   they still count toward the resident footprint while the model is used.
 * @property kvCacheBytes key/value cache (self-attention, plus cross-attention for Whisper).
 * @property computeBytes scratch / compute graph buffers (heuristic).
 * @property overheadBytes fixed context, tokenizer and allocator overhead (heuristic).
 */
public data class PeakRamEstimate(
    val weightsBytes: Long,
    val kvCacheBytes: Long,
    val computeBytes: Long,
    val overheadBytes: Long,
) {
    /** Sum of all components. */
    val totalBytes: Long get() = checkedAdd(checkedAdd(weightsBytes, kvCacheBytes), checkedAdd(computeBytes, overheadBytes))
}

/**
 * Options for [estimateLlmPeakRam].
 *
 * @property contextLength KV cache length in tokens; `null` uses `min(model context, 4096)` like `LlamaContextConfig`'s
 *   default, `0` uses the full training context.
 * @property batchSize logical batch size (`n_batch`), which sizes the logits and activation buffers.
 * @property kvCacheElementBytes bytes per KV cache element (2 for the default F16 cache, 1 for Q8_0-ish caches).
 * @property flashAttention with flash attention the `n_head x n_batch x n_ctx` attention score matrix is not
 *   materialized.
 */
public data class LlmRamEstimateOptions(
    val contextLength: Long? = null,
    val batchSize: Long = 512,
    val kvCacheElementBytes: Double = 2.0,
    val flashAttention: Boolean = false,
) {
    init {
        require(contextLength == null || contextLength >= 0) { "contextLength must be >= 0" }
        require(batchSize > 0) { "batchSize must be > 0" }
        require(kvCacheElementBytes > 0 && kvCacheElementBytes.isFinite()) { "kvCacheElementBytes must be > 0" }
    }

    public companion object {
        /** Context cap applied when [contextLength] is `null`. */
        public const val DEFAULT_CONTEXT_CAP: Long = 4096
    }
}

private const val MIB = 1024L * 1024L
private const val F32_BYTES = 4L
private const val F16_BYTES = 2L

/**
 * Estimates the peak RAM needed to run the LLM described by [header] with llama.cpp.
 *
 * - weights: [GgufModelHeader.tensorDataBytes];
 * - KV cache: `n_layer * n_ctx * n_head_kv * (key_length + value_length) * kvCacheElementBytes`;
 * - compute: logits (`n_vocab * n_batch * 4`), activations (`n_embd * n_batch * 4 * 8`), and without flash attention
 *   the attention scores (`n_head * n_batch * n_ctx * 4`, with `n_batch` capped at `n_ctx`);
 * - overhead: 64 MiB.
 *
 * Missing hyper-parameters (non-LLM GGUF files such as CLIP projectors or diffusion models) count as zero, so the
 * estimate degrades to the weights plus overhead.
 */
public fun estimateLlmPeakRam(header: GgufModelHeader, options: LlmRamEstimateOptions = LlmRamEstimateOptions()): PeakRamEstimate {
    val trained = header.contextLength ?: 0L
    val nCtx = when (val requested = options.contextLength) {
        null -> if (trained > 0) minOf(trained, LlmRamEstimateOptions.DEFAULT_CONTEXT_CAP) else LlmRamEstimateOptions.DEFAULT_CONTEXT_CAP
        0L -> trained
        else -> requested
    }
    val layers = header.blockCount ?: 0L
    val headsKv = header.headCountKv ?: 0L
    val kvWidth = headsKv * ((header.keyLength ?: 0L) + (header.valueLength ?: 0L))
    val kvElements = checkedMultiply(checkedMultiply(layers, nCtx), kvWidth)
    val kvBytes = (kvElements * options.kvCacheElementBytes).toLong()

    val batch = if (nCtx > 0) minOf(options.batchSize, nCtx) else options.batchSize
    val logits = checkedMultiply(header.vocabularySize ?: 0L, batch * F32_BYTES)
    val activations = checkedMultiply(header.embeddingLength ?: 0L, batch * F32_BYTES * 8)
    val scores = if (options.flashAttention) 0L else checkedMultiply(checkedMultiply(header.headCount ?: 0L, batch), nCtx * F32_BYTES)
    return PeakRamEstimate(
        weightsBytes = header.tensorDataBytes,
        kvCacheBytes = kvBytes,
        computeBytes = checkedAdd(checkedAdd(logits, activations), scores),
        overheadBytes = 64 * MIB,
    )
}

/**
 * Estimates the peak RAM needed to run the Whisper model described by [header] with whisper.cpp (one state).
 *
 * - weights: [WhisperModelHeader.tensorDataBytes];
 * - KV cache (F16): self-attention `n_text_layer * n_text_ctx * n_text_state * 2` plus cross-attention
 *   `n_text_layer * n_audio_ctx * n_text_state * 2` elements;
 * - compute: encoder activations `n_audio_ctx * n_audio_state * 4 * 24` plus decoder logits
 *   `n_vocab * n_text_ctx * 4` (heuristic);
 * - overhead: 16 MiB.
 */
public fun estimateWhisperPeakRam(header: WhisperModelHeader): PeakRamEstimate {
    val hp = header.hyperParameters
    val selfKv = hp.textLayers.toLong() * hp.textContext * hp.textState * 2
    val crossKv = hp.textLayers.toLong() * hp.audioContext * hp.textState * 2
    val encoder = hp.audioContext.toLong() * hp.audioState * F32_BYTES * 24
    val logits = hp.vocabularySize.toLong() * hp.textContext * F32_BYTES
    return PeakRamEstimate(
        weightsBytes = header.tensorDataBytes,
        kvCacheBytes = checkedMultiply(selfKv + crossKv, F16_BYTES),
        computeBytes = encoder + logits,
        overheadBytes = 16 * MIB,
    )
}
