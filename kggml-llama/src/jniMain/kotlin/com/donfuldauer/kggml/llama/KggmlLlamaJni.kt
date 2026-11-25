package com.donfuldauer.kggml.llama

/**
 * `external` declarations implemented by `kggml-llama/native/jni/kggml_llama_jni.cpp`
 * (`Java_com_donfuldauer_kggml_llama_KggmlLlamaJni_*`).
 *
 * Members are deliberately public inside this internal object: Kotlin mangles the JVM names of `internal`
 * members, which would break the JNI symbol names. Text crosses as UTF-8 `ByteArray`s.
 */
internal object KggmlLlamaJni {
    const val VERSION_MODULE: Int = 0
    const val VERSION_UPSTREAM: Int = 1
    const val VERSION_UPSTREAM_COMMIT: Int = 2
    const val VERSION_SYSTEM_INFO: Int = 3

    /** backend, gpu_layers, gpu_device, use_mmap, use_mlock. */
    const val MODEL_PARAM_COUNT: Int = 5

    /** n_vocab, n_ctx_train, n_embd, n_layer, has_encoder, has_decoder, backend, uses_gpu, bos, eos. */
    const val MODEL_INFO_INT_COUNT: Int = 10

    /** size_bytes, n_params. */
    const val MODEL_INFO_LONG_COUNT: Int = 2

    /** n_ctx, n_batch, n_ubatch, n_seq_max, n_threads, n_threads_batch, flash_attn, embeddings, pooling. */
    const val CONTEXT_PARAM_COUNT: Int = 9

    /** top_k, repeat_last_n, seed. */
    const val SAMPLER_INT_COUNT: Int = 3

    /** temperature, top_p, min_p, repeat_penalty. */
    const val SAMPLER_FLOAT_COUNT: Int = 4

    @JvmStatic external fun nativeVersionString(which: Int): String?

    @JvmStatic external fun nativeStatusName(status: Int): String?

    @JvmStatic external fun nativeLastError(): String?

    @JvmStatic external fun nativeModelLoad(pathUtf8: ByteArray, ints: IntArray, outHandle: LongArray): Int

    @JvmStatic external fun nativeModelFree(handle: Long)

    @JvmStatic external fun nativeModelInfo(handle: Long, outInts: IntArray, outLongs: LongArray): Int

    @JvmStatic external fun nativeModelDesc(handle: Long, out: Array<ByteArray?>): Int

    @JvmStatic external fun nativeTokenIsEog(model: Long, token: Int): Int

    @JvmStatic external fun nativeTokenize(model: Long, text: ByteArray, addSpecial: Int, parseSpecial: Int, out: Array<IntArray?>): Int

    @JvmStatic external fun nativeTokenToPiece(model: Long, token: Int, special: Int, out: Array<ByteArray?>): Int

    @JvmStatic external fun nativeContextCreate(model: Long, ints: IntArray, outHandle: LongArray): Int

    @JvmStatic external fun nativeContextFree(handle: Long)

    @JvmStatic external fun nativeContextNCtx(handle: Long, out: IntArray): Int

    @JvmStatic external fun nativeRequestAbort(handle: Long)

    @JvmStatic external fun nativeClearAbort(handle: Long)

    @JvmStatic external fun nativeAbortRequested(handle: Long): Int

    @JvmStatic external fun nativeDecode(
        handle: Long,
        tokens: IntArray,
        offset: Int,
        count: Int,
        pos0: Int,
        seqId: Int,
        logitsLast: Int,
    ): Int

    @JvmStatic external fun nativeMemorySeqRm(handle: Long, seqId: Int, p0: Int, p1: Int): Int

    @JvmStatic external fun nativeMemoryClear(handle: Long): Int

    @JvmStatic external fun nativeMemorySeqPosMax(handle: Long, seqId: Int, out: IntArray): Int

    @JvmStatic external fun nativeGetEmbeddings(handle: Long, seqId: Int, normalize: Int, out: Array<FloatArray?>): Int

    @JvmStatic external fun nativeStateSave(handle: Long, seqId: Int, out: Array<ByteArray?>): Int

    @JvmStatic external fun nativeStateLoad(handle: Long, seqId: Int, data: ByteArray): Int

    @JvmStatic external fun nativeSamplerCreate(
        model: Long,
        ints: IntArray,
        floats: FloatArray,
        grammarUtf8: ByteArray?,
        rootUtf8: ByteArray?,
        outHandle: LongArray,
    ): Int

    @JvmStatic external fun nativeSamplerFree(handle: Long)

    @JvmStatic external fun nativeSamplerSample(sampler: Long, context: Long, idx: Int, out: IntArray): Int

    @JvmStatic external fun nativeSamplerAccept(sampler: Long, token: Int, applyGrammar: Int): Int

    @JvmStatic external fun nativeSamplerReset(sampler: Long): Int

    /** `true` only in builds with `KGGML_LLAMA_TEST_HOOKS` (never in release libraries). */
    @JvmStatic external fun nativeTestHooksAvailable(): Boolean

    /** Test hook: live native models / contexts / samplers in this process, or -1 when unavailable. */
    @JvmStatic external fun nativeTestLiveObjects(): Long

    /** Test hook: abort-callback polls on the context since it was created, or -1 when unavailable. */
    @JvmStatic external fun nativeTestAbortPollCount(handle: Long): Long

    /** Test hook: the context requests abort on itself after [n] further polls (-1 disarms). */
    @JvmStatic external fun nativeTestAbortAfterPolls(handle: Long, n: Int): Int
}
