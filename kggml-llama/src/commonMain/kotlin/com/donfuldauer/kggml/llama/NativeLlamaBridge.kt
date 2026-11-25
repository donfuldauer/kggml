package com.donfuldauer.kggml.llama

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceException
import com.donfuldauer.kggml.core.NativeFailureException
import kotlin.concurrent.atomics.AtomicReference

/**
 * Raw, 1:1 view of the `kggml_llama.h` C ABI (text generation plus the libmtmd multimodal part).
 *
 * Implemented by the JNI bridge (Android, JVM Desktop), the Kotlin/Native cinterop bridge (iOS), and by fakes in
 * tests. Methods return raw native values (status codes, opaque `Long` handles); translation into Kotlin types
 * and typed exceptions happens once, in [LlamaEngine].
 *
 * Token arrays are handed to [decode] **zero-copy**: the JNI bridge pins them with `GetPrimitiveArrayCritical`,
 * the iOS bridge with `usePinned`, and llama.cpp reads them in place. Text crosses as UTF-8 bytes. Image bytes of
 * [mtmdTokenize] are pinned the same way, only for that call (libmtmd wraps / decodes them in memory into the
 * preprocessed chunks); nothing is ever written to a temporary file.
 *
 * Contract: [lastError] must be called on the same thread, immediately after the failing call (the native
 * diagnostic message is thread-local). Every method taking a handle except [requestAbort] and [abortRequested]
 * must be called from the owning engine's serial dispatcher.
 */
internal interface NativeLlamaBridge {
    fun moduleVersion(): String

    fun upstreamVersion(): String

    fun upstreamCommit(): String

    fun systemInfo(): String

    fun statusName(status: Int): String

    fun lastError(): String

    // ---- model ----

    fun modelLoad(path: String, params: RawModelParams): RawHandleResult

    fun modelFree(model: Long)

    fun modelInfo(model: Long): RawModelInfo

    fun modelDescription(model: Long): String

    fun tokenIsEog(model: Long, token: Int): Boolean

    fun tokenize(model: Long, utf8: ByteArray, addSpecial: Boolean, parseSpecial: Boolean): RawTokensResult

    /** Raw UTF-8 bytes of [token]; may end in the middle of a multi-byte character. */
    fun tokenToPiece(model: Long, token: Int, special: Boolean): RawBytesResult

    // ---- context ----

    fun contextCreate(model: Long, params: RawContextParams): RawHandleResult

    fun contextFree(context: Long)

    fun contextSize(context: Long): RawIntResult

    /** Lock-free, any thread. */
    fun requestAbort(context: Long)

    fun clearAbort(context: Long)

    /** Lock-free, any thread. */
    fun abortRequested(context: Long): Boolean

    /** `kggml_llama_decode()` on `tokens[offset, offset + count)`, read in place. */
    fun decode(context: Long, tokens: IntArray, offset: Int, count: Int, pos0: Int, seqId: Int, logitsLast: Boolean): Int

    fun memorySeqRemove(context: Long, seqId: Int, p0: Int, p1: Int): Int

    fun memoryClear(context: Long): Int

    fun memorySeqPosMax(context: Long, seqId: Int): RawIntResult

    fun embeddings(context: Long, seqId: Int, normalize: Boolean): RawFloatsResult

    /** Serialized KV-cache state of [seqId] (`< 0` = whole context). */
    fun stateSave(context: Long, seqId: Int): RawBytesResult

    fun stateLoad(context: Long, seqId: Int, data: ByteArray): Int

    // ---- sampler ----

    fun samplerCreate(model: Long, params: RawSamplerParams): RawHandleResult

    fun samplerFree(sampler: Long)

    fun samplerSample(sampler: Long, context: Long): RawIntResult

    fun samplerAccept(sampler: Long, token: Int, applyGrammar: Boolean): Int

    fun samplerReset(sampler: Long): Int

    // ---- multimodal (libmtmd) ----

    fun mtmdInit(model: Long, projectorPath: String, params: RawMtmdParams): RawHandleResult

    fun mtmdFree(mtmd: Long)

    fun mtmdCaps(mtmd: Long): RawMtmdCaps

    /** The media marker that stands for one image in a multimodal prompt (`"<__media__>"`). */
    fun mtmdDefaultMarker(): String

    /** Splits [prompt] at its media markers and preprocesses [images] in memory into a chunks handle. */
    fun mtmdTokenize(mtmd: Long, prompt: String, images: List<RawImage>, addSpecial: Boolean, parseSpecial: Boolean): RawHandleResult

    fun mtmdChunksFree(chunks: Long)

    fun mtmdChunksInfo(chunks: Long): RawChunksInfo

    /** Encodes and evaluates [chunks] from position [nPast]; [RawIntResult.value] is the next position. */
    fun mtmdEvalChunks(mtmd: Long, context: Long, chunks: Long, nPast: Int, seqId: Int, logitsLast: Boolean): RawIntResult
}

/** `kggml_llama_mtmd_params`. */
internal data class RawMtmdParams(
    val backend: Int,
    val threads: Int,
    val flashAttention: Int,
    val imageMinTokens: Int,
    val imageMaxTokens: Int,
    val warmup: Boolean,
)

/** `kggml_llama_mtmd_get_caps()`. */
internal data class RawMtmdCaps(val status: Int, val vision: Boolean = false, val audio: Boolean = false, val audioSampleRate: Int = -1)

/** `kggml_llama_mtmd_chunks_info()`. */
internal data class RawChunksInfo(val status: Int, val tokens: Int = 0, val positions: Int = 0, val images: Int = 0)

/** `kggml_llama_image`: [bytes] is read in place (never copied by the bridges). */
internal class RawImage(val format: Int, val width: Int, val height: Int, val bytes: ByteArray) {
    companion object {
        const val FORMAT_RGB: Int = 0
        const val FORMAT_ENCODED: Int = 1
    }
}

/** `kggml_llama_model_params` (without the progress callback). */
internal data class RawModelParams(
    val backend: Int,
    val gpuLayers: Int,
    val gpuDevice: Int,
    val useMmap: Boolean,
    val useMlock: Boolean,
)

/** `kggml_llama_context_params`. */
internal data class RawContextParams(
    val contextLength: Int,
    val batchSize: Int,
    val microBatchSize: Int,
    val maxSequences: Int,
    val threads: Int,
    val batchThreads: Int,
    val flashAttention: Int,
    val embeddings: Boolean,
    val pooling: Int,
)

/** `kggml_llama_sampler_params`. [seed] carries the `uint32_t` bits. */
internal data class RawSamplerParams(
    val temperature: Float,
    val topK: Int,
    val topP: Float,
    val minP: Float,
    val repeatPenalty: Float,
    val repeatLastN: Int,
    val seed: Int,
    val grammar: String?,
    val grammarRoot: String?,
)

/** `kggml_llama_model_info`. */
internal data class RawModelInfo(
    val status: Int,
    val vocabularySize: Int = 0,
    val trainContextLength: Int = 0,
    val embeddingSize: Int = 0,
    val layerCount: Int = 0,
    val hasEncoder: Boolean = false,
    val hasDecoder: Boolean = false,
    val backend: Int = 0,
    val usesGpu: Boolean = false,
    val bos: Int = -1,
    val eos: Int = -1,
    val sizeBytes: Long = 0,
    val parameterCount: Long = 0,
)

/** A status code plus one `int32_t` out-parameter. */
internal data class RawIntResult(val status: Int, val value: Int)

/** A status code plus an opaque handle (`0` unless [status] is OK). */
internal data class RawHandleResult(val status: Int, val handle: Long)

/** A status code plus token ids (empty unless [status] is OK). */
internal class RawTokensResult(val status: Int, val tokens: IntArray = IntArray(0))

/** A status code plus bytes (empty unless [status] is OK). */
internal class RawBytesResult(val status: Int, val bytes: ByteArray = ByteArray(0))

/** A status code plus floats (empty unless [status] is OK). */
internal class RawFloatsResult(val status: Int, val values: FloatArray = FloatArray(0))

/** `kggml_status` values used by the llama ABI. Codes 0..6 equal `NativeErrorCode`. */
internal object LlamaStatus {
    const val OK: Int = 0
    const val MODEL_LOAD_FAILED: Int = 1
    const val INVALID_ARGUMENT: Int = 2
    const val BACKEND_UNAVAILABLE: Int = 3
    const val OUT_OF_MEMORY: Int = 4
    const val CANCELLED: Int = 5
    const val NATIVE_FAILURE: Int = 6
    const val BUFFER_TOO_SMALL: Int = 101
}

/** `kggml_backend_kind` values, as consumed by `kggml_llama_model_params.backend`. */
internal object LlamaBackendKind {
    const val CPU: Int = 0
    const val METAL: Int = 1
    const val VULKAN: Int = 2
    const val COREML: Int = 3

    fun of(backend: ComputeBackend): Int = when (backend) {
        ComputeBackend.CPU_ONLY -> CPU
        ComputeBackend.GPU_METAL -> METAL
        ComputeBackend.GPU_VULKAN -> VULKAN
        ComputeBackend.NPU_COREML -> COREML
        ComputeBackend.AUTO -> error("AUTO must be resolved before reaching native code")
    }

    fun toComputeBackend(kind: Int): ComputeBackend? = when (kind) {
        CPU -> ComputeBackend.CPU_ONLY
        METAL -> ComputeBackend.GPU_METAL
        VULKAN -> ComputeBackend.GPU_VULKAN
        COREML -> ComputeBackend.NPU_COREML
        else -> null
    }
}

/**
 * Creates the platform bridge, loading the native libraries if needed.
 *
 * @throws Throwable (typically `UnsatisfiedLinkError`) if the native library cannot be loaded.
 */
internal expect fun createPlatformLlamaBridge(): NativeLlamaBridge

/** Lazily creates the platform bridge once and translates load failures into [NativeFailureException]. */
internal class LlamaBridgeHolder(private val factory: () -> NativeLlamaBridge) {
    private val ref = AtomicReference<NativeLlamaBridge?>(null)

    fun get(): NativeLlamaBridge {
        ref.load()?.let { return it }
        val created = try {
            factory()
        } catch (e: InferenceException) {
            throw e
        } catch (t: Throwable) {
            throw NativeFailureException(
                message = "Failed to load the Kggml llama native library: ${t.message ?: t::class.simpleName}",
                cause = t,
            )
        }
        // Racing first calls may both create a bridge; they are stateless views of the same native library.
        ref.compareAndSet(null, created)
        return ref.load() ?: created
    }

    companion object {
        val platform: LlamaBridgeHolder = LlamaBridgeHolder(::createPlatformLlamaBridge)
    }
}

/** Builds the typed exception for a non-OK native [status], with status name and native diagnostic attached. */
internal fun NativeLlamaBridge.nativeError(
    status: Int,
    operation: String,
    backend: ComputeBackend? = null,
): InferenceException {
    val detail = runCatching { lastError() }.getOrDefault("")
    val name = runCatching { statusName(status) }.getOrDefault("KGGML_STATUS_$status")
    val message = buildString {
        append(operation).append(" failed: ").append(name).append(" (").append(status).append(')')
        if (detail.isNotEmpty()) append(": ").append(detail)
    }
    return InferenceException.fromNativeError(status, message, backend)
}

/** Throws [nativeError] unless [status] is OK. Must be called right after the native call. */
internal fun NativeLlamaBridge.check(status: Int, operation: String, backend: ComputeBackend? = null) {
    if (status != LlamaStatus.OK) throw nativeError(status, operation, backend)
}
