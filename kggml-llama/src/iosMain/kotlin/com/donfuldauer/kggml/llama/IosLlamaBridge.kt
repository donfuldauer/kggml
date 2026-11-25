package com.donfuldauer.kggml.llama

import cnames.structs.kggml_llama_context
import cnames.structs.kggml_llama_model
import cnames.structs.kggml_llama_mtmd
import cnames.structs.kggml_llama_mtmd_chunks
import cnames.structs.kggml_llama_sampler
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_abort_requested
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_clear_abort
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_context_create
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_context_default_params
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_context_free
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_context_n_ctx
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_context_params
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_decode
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_get_embeddings
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_image
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_last_error
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_memory_clear
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_memory_seq_pos_max
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_memory_seq_rm
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_model_default_params
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_model_desc
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_model_free
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_model_get_info
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_model_info
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_model_load
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_model_params
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_mtmd_chunks_free
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_mtmd_chunks_info
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_mtmd_default_marker
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_mtmd_default_params
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_mtmd_eval_chunks
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_mtmd_free
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_mtmd_get_caps
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_mtmd_init
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_mtmd_params
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_mtmd_tokenize
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_request_abort
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_sampler_accept
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_sampler_create
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_sampler_default_params
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_sampler_free
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_sampler_params
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_sampler_reset
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_sampler_sample
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_state_load
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_state_save
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_state_size
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_system_info
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_token_is_eog
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_token_to_piece
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_tokenize
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_upstream_commit
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_upstream_version
import com.donfuldauer.kggml.llama.cinterop.kggml_llama_version
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.Pinned
import kotlinx.cinterop.UByteVar
import kotlinx.cinterop.UIntVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.convert
import kotlinx.cinterop.cstr
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.pin
import kotlinx.cinterop.ptr
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.toCPointer
import kotlinx.cinterop.toKString
import kotlinx.cinterop.toLong
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value
import platform.posix.size_tVar

/**
 * [NativeLlamaBridge] calling the `kggml_llama.h` C ABI directly through Kotlin/Native cinterop.
 *
 * Token arrays are pinned with `usePinned { it.addressOf(offset) }` and read in place (zero-copy); image bytes are
 * pinned the same way, only for [mtmdTokenize]. Every C entry
 * point catches C++ exceptions and returns a status code, so nothing can unwind into Kotlin/Native frames. The
 * library is linked dynamically (`KggmlLlama.framework`), so there is nothing to load at runtime.
 */
@OptIn(ExperimentalForeignApi::class)
internal object IosLlamaBridge : NativeLlamaBridge {
    override fun moduleVersion(): String = kggml_llama_version()?.toKString().orEmpty()

    override fun upstreamVersion(): String = kggml_llama_upstream_version()?.toKString().orEmpty()

    override fun upstreamCommit(): String = kggml_llama_upstream_commit()?.toKString().orEmpty()

    override fun systemInfo(): String = kggml_llama_system_info()?.toKString().orEmpty()

    override fun statusName(status: Int): String = STATUS_NAMES[status] ?: "KGGML_STATUS_$status"

    override fun lastError(): String = kggml_llama_last_error()?.toKString().orEmpty()

    override fun modelLoad(path: String, params: RawModelParams): RawHandleResult = memScoped {
        val p = alloc<kggml_llama_model_params>()
        val defaults = kggml_llama_model_default_params(p.ptr).toInt()
        if (defaults != LlamaStatus.OK) return@memScoped RawHandleResult(defaults, 0L)
        p.backend = params.backend
        p.gpu_layers = params.gpuLayers
        p.gpu_device = params.gpuDevice
        p.use_mmap = params.useMmap.toInt()
        p.use_mlock = params.useMlock.toInt()
        val out = alloc<CPointerVar<kggml_llama_model>>()
        val status = kggml_llama_model_load(path, p.ptr, out.ptr).toInt()
        RawHandleResult(status, if (status == LlamaStatus.OK) out.value.toLong() else 0L)
    }

    override fun modelFree(model: Long) = kggml_llama_model_free(modelPtr(model))

    override fun modelInfo(model: Long): RawModelInfo = memScoped {
        val info = alloc<kggml_llama_model_info>()
        val status = kggml_llama_model_get_info(modelPtr(model), info.ptr).toInt()
        if (status != LlamaStatus.OK) return@memScoped RawModelInfo(status)
        RawModelInfo(
            status = status,
            vocabularySize = info.n_vocab,
            trainContextLength = info.n_ctx_train,
            embeddingSize = info.n_embd,
            layerCount = info.n_layer,
            hasEncoder = info.has_encoder != 0,
            hasDecoder = info.has_decoder != 0,
            backend = info.backend,
            usesGpu = info.uses_gpu != 0,
            bos = info.bos,
            eos = info.eos,
            sizeBytes = info.size_bytes.toLong(),
            parameterCount = info.n_params.toLong(),
        )
    }

    override fun modelDescription(model: Long): String = memScoped {
        val length = alloc<size_tVar>()
        kggml_llama_model_desc(modelPtr(model), null, 0u.convert(), length.ptr)
        val capacity = length.value.toInt() + 1
        val buffer = ByteArray(capacity)
        val status = buffer.usePinned { kggml_llama_model_desc(modelPtr(model), it.addressOf(0), capacity.convert(), length.ptr).toInt() }
        if (status != LlamaStatus.OK) "" else buffer.decodeToString(0, buffer.indexOf(0).let { if (it < 0) capacity else it })
    }

    override fun tokenIsEog(model: Long, token: Int): Boolean = kggml_llama_token_is_eog(modelPtr(model), token) != 0

    override fun tokenize(model: Long, utf8: ByteArray, addSpecial: Boolean, parseSpecial: Boolean): RawTokensResult = memScoped {
        val text = if (utf8.isEmpty()) ByteArray(1) else utf8 // addressOf(0) needs an element; length stays utf8.size
        val count = alloc<IntVar>()
        var tokens = IntArray(utf8.size + 8)
        fun call(): Int = text.usePinned { t ->
            tokens.usePinned { out ->
                kggml_llama_tokenize(
                    modelPtr(model), t.addressOf(0), utf8.size.convert(), addSpecial.toInt(), parseSpecial.toInt(),
                    out.addressOf(0), tokens.size, count.ptr,
                ).toInt()
            }
        }
        var status = call()
        if (status == LlamaStatus.BUFFER_TOO_SMALL && count.value > 0) {
            tokens = IntArray(count.value)
            status = call()
        }
        if (status != LlamaStatus.OK) RawTokensResult(status) else RawTokensResult(status, tokens.copyOf(count.value))
    }

    override fun tokenToPiece(model: Long, token: Int, special: Boolean): RawBytesResult = memScoped {
        val length = alloc<IntVar>()
        var buffer = ByteArray(64)
        fun call(): Int = buffer.usePinned {
            kggml_llama_token_to_piece(modelPtr(model), token, special.toInt(), it.addressOf(0), buffer.size, length.ptr).toInt()
        }
        var status = call()
        if (status == LlamaStatus.BUFFER_TOO_SMALL && length.value > 0) {
            buffer = ByteArray(length.value)
            status = call()
        }
        if (status != LlamaStatus.OK) RawBytesResult(status) else RawBytesResult(status, buffer.copyOf(length.value))
    }

    override fun contextCreate(model: Long, params: RawContextParams): RawHandleResult = memScoped {
        val p = alloc<kggml_llama_context_params>()
        val defaults = kggml_llama_context_default_params(p.ptr).toInt()
        if (defaults != LlamaStatus.OK) return@memScoped RawHandleResult(defaults, 0L)
        p.n_ctx = params.contextLength.toUInt()
        p.n_batch = params.batchSize.toUInt()
        p.n_ubatch = params.microBatchSize.toUInt()
        p.n_seq_max = params.maxSequences.toUInt()
        p.n_threads = params.threads
        p.n_threads_batch = params.batchThreads
        p.flash_attn = params.flashAttention
        p.embeddings = params.embeddings.toInt()
        p.pooling = params.pooling
        val out = alloc<CPointerVar<kggml_llama_context>>()
        val status = kggml_llama_context_create(modelPtr(model), p.ptr, out.ptr).toInt()
        RawHandleResult(status, if (status == LlamaStatus.OK) out.value.toLong() else 0L)
    }

    override fun contextFree(context: Long) = kggml_llama_context_free(contextPtr(context))

    override fun contextSize(context: Long): RawIntResult = memScoped {
        val out = alloc<UIntVar>()
        val status = kggml_llama_context_n_ctx(contextPtr(context), out.ptr).toInt()
        RawIntResult(status, out.value.coerceAtMost(Int.MAX_VALUE.toUInt()).toInt())
    }

    override fun requestAbort(context: Long) = kggml_llama_request_abort(contextPtr(context))

    override fun clearAbort(context: Long) = kggml_llama_clear_abort(contextPtr(context))

    override fun abortRequested(context: Long): Boolean = kggml_llama_abort_requested(contextPtr(context)) != 0

    override fun decode(context: Long, tokens: IntArray, offset: Int, count: Int, pos0: Int, seqId: Int, logitsLast: Boolean): Int {
        if (offset < 0 || count <= 0 || offset.toLong() + count > tokens.size) return LlamaStatus.INVALID_ARGUMENT
        return tokens.usePinned {
            kggml_llama_decode(contextPtr(context), it.addressOf(offset), count, pos0, seqId, logitsLast.toInt()).toInt()
        }
    }

    override fun memorySeqRemove(context: Long, seqId: Int, p0: Int, p1: Int): Int =
        kggml_llama_memory_seq_rm(contextPtr(context), seqId, p0, p1).toInt()

    override fun memoryClear(context: Long): Int = kggml_llama_memory_clear(contextPtr(context)).toInt()

    override fun memorySeqPosMax(context: Long, seqId: Int): RawIntResult = memScoped {
        val out = alloc<IntVar>()
        val status = kggml_llama_memory_seq_pos_max(contextPtr(context), seqId, out.ptr).toInt()
        RawIntResult(status, out.value)
    }

    override fun embeddings(context: Long, seqId: Int, normalize: Boolean): RawFloatsResult = memScoped {
        val count = alloc<IntVar>()
        val first = kggml_llama_get_embeddings(contextPtr(context), seqId, normalize.toInt(), null, 0, count.ptr).toInt()
        if (first != LlamaStatus.OK && first != LlamaStatus.BUFFER_TOO_SMALL) return@memScoped RawFloatsResult(first)
        val values = FloatArray(count.value.coerceAtLeast(1))
        val status = values.usePinned {
            kggml_llama_get_embeddings(contextPtr(context), seqId, normalize.toInt(), it.addressOf(0), values.size, count.ptr).toInt()
        }
        if (status != LlamaStatus.OK) RawFloatsResult(status) else RawFloatsResult(status, values.copyOf(count.value))
    }

    override fun stateSave(context: Long, seqId: Int): RawBytesResult = memScoped {
        val size = alloc<size_tVar>()
        val sized = kggml_llama_state_size(contextPtr(context), seqId, size.ptr).toInt()
        if (sized != LlamaStatus.OK) return@memScoped RawBytesResult(sized)
        if (size.value > Int.MAX_VALUE.toULong()) return@memScoped RawBytesResult(LlamaStatus.OUT_OF_MEMORY)
        val buffer = ByteArray(size.value.toInt().coerceAtLeast(1))
        val written = alloc<size_tVar>()
        val status = buffer.usePinned {
            kggml_llama_state_save(contextPtr(context), seqId, it.addressOf(0).reinterpret<UByteVar>(), size.value, written.ptr).toInt()
        }
        if (status != LlamaStatus.OK) RawBytesResult(status) else RawBytesResult(status, buffer.copyOf(written.value.toInt()))
    }

    override fun stateLoad(context: Long, seqId: Int, data: ByteArray): Int {
        if (data.isEmpty()) return LlamaStatus.INVALID_ARGUMENT
        return data.usePinned {
            kggml_llama_state_load(contextPtr(context), seqId, it.addressOf(0).reinterpret<UByteVar>(), data.size.convert()).toInt()
        }
    }

    override fun samplerCreate(model: Long, params: RawSamplerParams): RawHandleResult = memScoped {
        val p = alloc<kggml_llama_sampler_params>()
        val defaults = kggml_llama_sampler_default_params(p.ptr).toInt()
        if (defaults != LlamaStatus.OK) return@memScoped RawHandleResult(defaults, 0L)
        p.temperature = params.temperature
        p.top_k = params.topK
        p.top_p = params.topP
        p.min_p = params.minP
        p.repeat_penalty = params.repeatPenalty
        p.repeat_last_n = params.repeatLastN
        p.seed = params.seed.toUInt()
        p.grammar = params.grammar?.cstr?.ptr
        p.grammar_root = params.grammarRoot?.cstr?.ptr
        val out = alloc<CPointerVar<kggml_llama_sampler>>()
        val status = kggml_llama_sampler_create(modelPtr(model), p.ptr, out.ptr).toInt()
        RawHandleResult(status, if (status == LlamaStatus.OK) out.value.toLong() else 0L)
    }

    override fun samplerFree(sampler: Long) = kggml_llama_sampler_free(samplerPtr(sampler))

    override fun samplerSample(sampler: Long, context: Long): RawIntResult = memScoped {
        val out = alloc<IntVar>()
        val status = kggml_llama_sampler_sample(samplerPtr(sampler), contextPtr(context), -1, out.ptr).toInt()
        RawIntResult(status, out.value)
    }

    override fun samplerAccept(sampler: Long, token: Int, applyGrammar: Boolean): Int =
        kggml_llama_sampler_accept(samplerPtr(sampler), token, applyGrammar.toInt()).toInt()

    override fun samplerReset(sampler: Long): Int = kggml_llama_sampler_reset(samplerPtr(sampler)).toInt()

    override fun mtmdInit(model: Long, projectorPath: String, params: RawMtmdParams): RawHandleResult = memScoped {
        val p = alloc<kggml_llama_mtmd_params>()
        val defaults = kggml_llama_mtmd_default_params(p.ptr).toInt()
        if (defaults != LlamaStatus.OK) return@memScoped RawHandleResult(defaults, 0L)
        p.backend = params.backend
        p.n_threads = params.threads
        p.flash_attn = params.flashAttention
        p.image_min_tokens = params.imageMinTokens
        p.image_max_tokens = params.imageMaxTokens
        p.warmup = params.warmup.toInt()
        val out = alloc<CPointerVar<kggml_llama_mtmd>>()
        val status = kggml_llama_mtmd_init(modelPtr(model), projectorPath, p.ptr, out.ptr).toInt()
        RawHandleResult(status, if (status == LlamaStatus.OK) out.value.toLong() else 0L)
    }

    override fun mtmdFree(mtmd: Long) = kggml_llama_mtmd_free(mtmdPtr(mtmd))

    override fun mtmdCaps(mtmd: Long): RawMtmdCaps = memScoped {
        val vision = alloc<IntVar>()
        val audio = alloc<IntVar>()
        val rate = alloc<IntVar>()
        val status = kggml_llama_mtmd_get_caps(mtmdPtr(mtmd), vision.ptr, audio.ptr, rate.ptr).toInt()
        if (status != LlamaStatus.OK) RawMtmdCaps(status) else RawMtmdCaps(status, vision.value != 0, audio.value != 0, rate.value)
    }

    override fun mtmdDefaultMarker(): String = kggml_llama_mtmd_default_marker()?.toKString() ?: "<__media__>"

    override fun mtmdTokenize(mtmd: Long, prompt: String, images: List<RawImage>, addSpecial: Boolean, parseSpecial: Boolean): RawHandleResult {
        // Every image is pinned (zero-copy) for this call only; libmtmd copies / decodes it into the chunks.
        val pins = ArrayList<Pinned<ByteArray>?>(images.size)
        try {
            for (image in images) pins += if (image.bytes.isEmpty()) null else image.bytes.pin()
            return memScoped {
                val descs = if (images.isEmpty()) null else allocArray<kggml_llama_image>(images.size)
                images.forEachIndexed { i, image ->
                    val d = descs!![i]
                    d.data = pins[i]?.addressOf(0)?.reinterpret<UByteVar>()
                    d.size = image.bytes.size.convert()
                    d.format = image.format
                    d.width = image.width.coerceAtLeast(0).toUInt()
                    d.height = image.height.coerceAtLeast(0).toUInt()
                }
                val out = alloc<CPointerVar<kggml_llama_mtmd_chunks>>()
                val status = kggml_llama_mtmd_tokenize(
                    mtmdPtr(mtmd), prompt, descs, images.size, addSpecial.toInt(), parseSpecial.toInt(), out.ptr,
                ).toInt()
                RawHandleResult(status, if (status == LlamaStatus.OK) out.value.toLong() else 0L)
            }
        } finally {
            pins.forEach { it?.unpin() }
        }
    }

    override fun mtmdChunksFree(chunks: Long) = kggml_llama_mtmd_chunks_free(chunksPtr(chunks))

    override fun mtmdChunksInfo(chunks: Long): RawChunksInfo = memScoped {
        val tokens = alloc<IntVar>()
        val positions = alloc<IntVar>()
        val count = alloc<IntVar>()
        val status = kggml_llama_mtmd_chunks_info(chunksPtr(chunks), tokens.ptr, positions.ptr, count.ptr).toInt()
        if (status != LlamaStatus.OK) RawChunksInfo(status) else RawChunksInfo(status, tokens.value, positions.value, count.value)
    }

    override fun mtmdEvalChunks(mtmd: Long, context: Long, chunks: Long, nPast: Int, seqId: Int, logitsLast: Boolean): RawIntResult =
        memScoped {
            val next = alloc<IntVar>()
            next.value = nPast
            val status = kggml_llama_mtmd_eval_chunks(
                mtmdPtr(mtmd), contextPtr(context), chunksPtr(chunks), nPast, seqId, logitsLast.toInt(), next.ptr,
            ).toInt()
            RawIntResult(status, next.value)
        }

    private fun mtmdPtr(handle: Long): CPointer<kggml_llama_mtmd>? = handle.toCPointer()

    private fun chunksPtr(handle: Long): CPointer<kggml_llama_mtmd_chunks>? = handle.toCPointer()

    private fun modelPtr(handle: Long): CPointer<kggml_llama_model>? = handle.toCPointer()

    private fun contextPtr(handle: Long): CPointer<kggml_llama_context>? = handle.toCPointer()

    private fun samplerPtr(handle: Long): CPointer<kggml_llama_sampler>? = handle.toCPointer()

    private fun Boolean.toInt(): Int = if (this) 1 else 0

    // kggml_status_name() is bound by :kggml-runtime's cinterop, not this one.
    private val STATUS_NAMES: Map<Int, String> = mapOf(
        0 to "KGGML_STATUS_OK",
        1 to "KGGML_STATUS_MODEL_LOAD_FAILED",
        2 to "KGGML_STATUS_INVALID_ARGUMENT",
        3 to "KGGML_STATUS_BACKEND_UNAVAILABLE",
        4 to "KGGML_STATUS_OUT_OF_MEMORY",
        5 to "KGGML_STATUS_CANCELLED",
        6 to "KGGML_STATUS_NATIVE_FAILURE",
        100 to "KGGML_STATUS_NOT_INITIALIZED",
        101 to "KGGML_STATUS_BUFFER_TOO_SMALL",
        102 to "KGGML_STATUS_UNKNOWN",
    )
}

internal actual fun createPlatformLlamaBridge(): NativeLlamaBridge = IosLlamaBridge
