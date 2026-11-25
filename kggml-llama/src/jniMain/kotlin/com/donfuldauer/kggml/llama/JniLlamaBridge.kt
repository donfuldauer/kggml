package com.donfuldauer.kggml.llama

/**
 * [NativeLlamaBridge] over the `kggml_llama_jni` JNI library (Android + JVM Desktop).
 *
 * Token arrays and image bytes are passed by reference; the native side pins them with `GetPrimitiveArrayCritical`
 * (zero-copy) and releases them on every path (images only for the duration of [mtmdTokenize]).
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

    override fun mtmdInit(model: Long, projectorPath: String, params: RawMtmdParams): RawHandleResult {
        val ints = IntArray(KggmlLlamaJni.MTMD_PARAM_COUNT)
        ints[0] = params.backend
        ints[1] = params.threads
        ints[2] = params.flashAttention
        ints[3] = params.imageMinTokens
        ints[4] = params.imageMaxTokens
        ints[5] = params.warmup.toInt()
        return handleResult { KggmlLlamaJni.nativeMtmdInit(model, projectorPath.encodeToByteArray(), ints, it) }
    }

    override fun mtmdFree(mtmd: Long) = KggmlLlamaJni.nativeMtmdFree(mtmd)

    override fun mtmdCaps(mtmd: Long): RawMtmdCaps {
        val out = IntArray(KggmlLlamaJni.MTMD_CAPS_COUNT)
        val status = KggmlLlamaJni.nativeMtmdCaps(mtmd, out)
        return if (status != LlamaStatus.OK) RawMtmdCaps(status) else RawMtmdCaps(status, out[0] != 0, out[1] != 0, out[2])
    }

    override fun mtmdDefaultMarker(): String = KggmlLlamaJni.nativeMtmdDefaultMarker() ?: "<__media__>"

    override fun mtmdTokenize(mtmd: Long, prompt: String, images: List<RawImage>, addSpecial: Boolean, parseSpecial: Boolean): RawHandleResult {
        val stride = KggmlLlamaJni.IMAGE_META_STRIDE
        val meta = IntArray(images.size * stride)
        images.forEachIndexed { i, image ->
            meta[i * stride] = image.format
            meta[i * stride + 1] = image.width
            meta[i * stride + 2] = image.height
        }
        val arrays = Array(images.size) { images[it].bytes }
        return handleResult {
            KggmlLlamaJni.nativeMtmdTokenize(mtmd, prompt.encodeToByteArray(), arrays, meta, addSpecial.toInt(), parseSpecial.toInt(), it)
        }
    }

    override fun mtmdChunksFree(chunks: Long) = KggmlLlamaJni.nativeMtmdChunksFree(chunks)

    override fun mtmdChunksInfo(chunks: Long): RawChunksInfo {
        val out = IntArray(KggmlLlamaJni.CHUNKS_INFO_COUNT)
        val status = KggmlLlamaJni.nativeMtmdChunksInfo(chunks, out)
        return if (status != LlamaStatus.OK) RawChunksInfo(status) else RawChunksInfo(status, out[0], out[1], out[2])
    }

    override fun mtmdEvalChunks(mtmd: Long, context: Long, chunks: Long, nPast: Int, seqId: Int, logitsLast: Boolean): RawIntResult =
        intResult { KggmlLlamaJni.nativeMtmdEvalChunks(mtmd, context, chunks, nPast, seqId, logitsLast.toInt(), it) }

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
