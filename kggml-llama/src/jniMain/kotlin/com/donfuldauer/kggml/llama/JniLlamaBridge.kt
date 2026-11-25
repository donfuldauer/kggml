package com.donfuldauer.kggml.llama

/**
 * [NativeLlamaBridge] over the `kggml_llama_jni` JNI library (Android + JVM Desktop).
 *
 * Token arrays are passed by reference; the native side pins them with `GetPrimitiveArrayCritical` (zero-copy) and
 * releases them on every path.
 *
 * @param loadLibraries loads `kggml_llama_jni` and its dependencies; invoked once, before any native call.
 */
internal class JniLlamaBridge(loadLibraries: () -> Unit) : NativeLlamaBridge {
    init {
        loadLibraries()
    }

    override fun moduleVersion(): String = version(KggmlLlamaJni.VERSION_MODULE)

    override fun upstreamVersion(): String = version(KggmlLlamaJni.VERSION_UPSTREAM)

    override fun upstreamCommit(): String = version(KggmlLlamaJni.VERSION_UPSTREAM_COMMIT)

    override fun systemInfo(): String = version(KggmlLlamaJni.VERSION_SYSTEM_INFO)

    override fun statusName(status: Int): String = KggmlLlamaJni.nativeStatusName(status) ?: "KGGML_STATUS_$status"

    override fun lastError(): String = KggmlLlamaJni.nativeLastError().orEmpty()

    override fun modelLoad(path: String, params: RawModelParams): RawHandleResult {
        val ints = IntArray(KggmlLlamaJni.MODEL_PARAM_COUNT)
        ints[0] = params.backend
        ints[1] = params.gpuLayers
        ints[2] = params.gpuDevice
        ints[3] = params.useMmap.toInt()
        ints[4] = params.useMlock.toInt()
        return handleResult { KggmlLlamaJni.nativeModelLoad(path.encodeToByteArray(), ints, it) }
    }

    override fun modelFree(model: Long) = KggmlLlamaJni.nativeModelFree(model)

    override fun modelInfo(model: Long): RawModelInfo {
        val ints = IntArray(KggmlLlamaJni.MODEL_INFO_INT_COUNT)
        val longs = LongArray(KggmlLlamaJni.MODEL_INFO_LONG_COUNT)
        val status = KggmlLlamaJni.nativeModelInfo(model, ints, longs)
        if (status != LlamaStatus.OK) return RawModelInfo(status)
        return RawModelInfo(
            status = status,
            vocabularySize = ints[0],
            trainContextLength = ints[1],
            embeddingSize = ints[2],
            layerCount = ints[3],
            hasEncoder = ints[4] != 0,
            hasDecoder = ints[5] != 0,
            backend = ints[6],
            usesGpu = ints[7] != 0,
            bos = ints[8],
            eos = ints[9],
            sizeBytes = longs[0],
            parameterCount = longs[1],
        )
    }

    override fun modelDescription(model: Long): String {
        val out = arrayOfNulls<ByteArray>(1)
        val status = KggmlLlamaJni.nativeModelDesc(model, out)
        return if (status == LlamaStatus.OK) out[0]?.decodeToString().orEmpty() else ""
    }

    override fun tokenIsEog(model: Long, token: Int): Boolean = KggmlLlamaJni.nativeTokenIsEog(model, token) != 0

    override fun tokenize(model: Long, utf8: ByteArray, addSpecial: Boolean, parseSpecial: Boolean): RawTokensResult {
        val out = arrayOfNulls<IntArray>(1)
        val status = KggmlLlamaJni.nativeTokenize(model, utf8, addSpecial.toInt(), parseSpecial.toInt(), out)
        return RawTokensResult(status, if (status == LlamaStatus.OK) out[0] ?: IntArray(0) else IntArray(0))
    }

    override fun tokenToPiece(model: Long, token: Int, special: Boolean): RawBytesResult {
        val out = arrayOfNulls<ByteArray>(1)
        val status = KggmlLlamaJni.nativeTokenToPiece(model, token, special.toInt(), out)
        return RawBytesResult(status, if (status == LlamaStatus.OK) out[0] ?: ByteArray(0) else ByteArray(0))
    }

    override fun contextCreate(model: Long, params: RawContextParams): RawHandleResult {
        val ints = IntArray(KggmlLlamaJni.CONTEXT_PARAM_COUNT)
        ints[0] = params.contextLength
        ints[1] = params.batchSize
        ints[2] = params.microBatchSize
        ints[3] = params.maxSequences
        ints[4] = params.threads
        ints[5] = params.batchThreads
        ints[6] = params.flashAttention
        ints[7] = params.embeddings.toInt()
        ints[8] = params.pooling
        return handleResult { KggmlLlamaJni.nativeContextCreate(model, ints, it) }
    }

    override fun contextFree(context: Long) = KggmlLlamaJni.nativeContextFree(context)

    override fun contextSize(context: Long): RawIntResult = intResult { KggmlLlamaJni.nativeContextNCtx(context, it) }

    override fun requestAbort(context: Long) = KggmlLlamaJni.nativeRequestAbort(context)

    override fun clearAbort(context: Long) = KggmlLlamaJni.nativeClearAbort(context)

    override fun abortRequested(context: Long): Boolean = KggmlLlamaJni.nativeAbortRequested(context) != 0

    override fun decode(context: Long, tokens: IntArray, offset: Int, count: Int, pos0: Int, seqId: Int, logitsLast: Boolean): Int =
        KggmlLlamaJni.nativeDecode(context, tokens, offset, count, pos0, seqId, logitsLast.toInt())

    override fun memorySeqRemove(context: Long, seqId: Int, p0: Int, p1: Int): Int = KggmlLlamaJni.nativeMemorySeqRm(context, seqId, p0, p1)

    override fun memoryClear(context: Long): Int = KggmlLlamaJni.nativeMemoryClear(context)

    override fun memorySeqPosMax(context: Long, seqId: Int): RawIntResult = intResult { KggmlLlamaJni.nativeMemorySeqPosMax(context, seqId, it) }

    override fun embeddings(context: Long, seqId: Int, normalize: Boolean): RawFloatsResult {
        val out = arrayOfNulls<FloatArray>(1)
        val status = KggmlLlamaJni.nativeGetEmbeddings(context, seqId, normalize.toInt(), out)
        return RawFloatsResult(status, if (status == LlamaStatus.OK) out[0] ?: FloatArray(0) else FloatArray(0))
    }

    override fun stateSave(context: Long, seqId: Int): RawBytesResult {
        val out = arrayOfNulls<ByteArray>(1)
        val status = KggmlLlamaJni.nativeStateSave(context, seqId, out)
        return RawBytesResult(status, if (status == LlamaStatus.OK) out[0] ?: ByteArray(0) else ByteArray(0))
    }

    override fun stateLoad(context: Long, seqId: Int, data: ByteArray): Int = KggmlLlamaJni.nativeStateLoad(context, seqId, data)

    override fun samplerCreate(model: Long, params: RawSamplerParams): RawHandleResult {
        val ints = IntArray(KggmlLlamaJni.SAMPLER_INT_COUNT)
        ints[0] = params.topK
        ints[1] = params.repeatLastN
        ints[2] = params.seed
        val floats = FloatArray(KggmlLlamaJni.SAMPLER_FLOAT_COUNT)
        floats[0] = params.temperature
        floats[1] = params.topP
        floats[2] = params.minP
        floats[3] = params.repeatPenalty
        return handleResult {
            KggmlLlamaJni.nativeSamplerCreate(model, ints, floats, params.grammar?.encodeToByteArray(), params.grammarRoot?.encodeToByteArray(), it)
        }
    }

    override fun samplerFree(sampler: Long) = KggmlLlamaJni.nativeSamplerFree(sampler)

    override fun samplerSample(sampler: Long, context: Long): RawIntResult =
        intResult { KggmlLlamaJni.nativeSamplerSample(sampler, context, -1, it) }

    override fun samplerAccept(sampler: Long, token: Int, applyGrammar: Boolean): Int =
        KggmlLlamaJni.nativeSamplerAccept(sampler, token, applyGrammar.toInt())

    override fun samplerReset(sampler: Long): Int = KggmlLlamaJni.nativeSamplerReset(sampler)

    private fun version(which: Int): String = KggmlLlamaJni.nativeVersionString(which).orEmpty()

    private inline fun intResult(call: (IntArray) -> Int): RawIntResult {
        val out = IntArray(1)
        val status = call(out)
        return RawIntResult(status, out[0])
    }

    private inline fun handleResult(call: (LongArray) -> Int): RawHandleResult {
        val out = LongArray(1)
        val status = call(out)
        return RawHandleResult(status, if (status == LlamaStatus.OK) out[0] else 0L)
    }

    private fun Boolean.toInt(): Int = if (this) 1 else 0
}
