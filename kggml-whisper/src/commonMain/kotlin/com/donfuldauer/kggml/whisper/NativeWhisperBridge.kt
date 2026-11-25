package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceException
import com.donfuldauer.kggml.core.NativeFailureException
import kotlin.concurrent.atomics.AtomicReference

/**
 * Raw, 1:1 view of the `kggml_whisper.h` C ABI.
 *
 * Implemented by the JNI bridge (Android, JVM Desktop), the Kotlin/Native cinterop bridge (iOS), and by fakes in
 * tests. Methods return raw native values (status codes, 0/1 flags, opaque `Long` context handles); translation
 * into Kotlin types and typed exceptions happens once, in [WhisperEngine].
 *
 * PCM arrays are handed to native code **zero-copy**: the JNI bridge pins them with `GetPrimitiveArrayCritical`,
 * the iOS bridge with `usePinned`, and whisper.cpp reads them in place.
 *
 * Contract: [lastError] must be called on the same thread, immediately after the failing call (the native
 * diagnostic message is thread-local). Every method taking a handle except [requestAbort] and [abortRequested]
 * must be called from the owning engine's serial dispatcher.
 */
internal interface NativeWhisperBridge {
    fun hasCoreMl(): Boolean

    fun moduleVersion(): String

    fun upstreamVersion(): String

    fun upstreamCommit(): String

    fun systemInfo(): String

    /** `kggml_whisper_init_from_file()` with `backend` (a [WhisperBackendKind]), returning the context handle. */
    fun initFromFile(modelPath: String, backendKind: Int, gpuDevice: Int, flashAttention: Boolean): RawHandleResult

    fun free(handle: Long)

    /** `kggml_whisper_get_backend()`: the effective [WhisperBackendKind]. */
    fun effectiveBackend(handle: Long): RawIntResult

    /** `kggml_whisper_uses_gpu()`: `whisper_context_params.use_gpu` the context was created with (0/1). */
    fun usesGpu(handle: Long): RawIntResult

    /** Lock-free, any thread. */
    fun requestAbort(handle: Long)

    fun clearAbort(handle: Long)

    /** Lock-free, any thread. */
    fun abortRequested(handle: Long): Boolean

    /** `kggml_whisper_transcribe()` on the first [count] 16 kHz mono samples of [samples], read in place. */
    fun transcribe(handle: Long, samples: FloatArray, count: Int, params: RawTranscribeParams): Int

    fun segmentCount(handle: Long): RawIntResult

    fun segment(handle: Long, index: Int): RawSegmentResult

    /** `kggml_whisper_resampled_length(frames, sampleRate, 16000)`, or `-1` for invalid arguments. */
    fun resampledLength(frames: Int, sampleRate: Int): Long

    /**
     * `kggml_whisper_pcm_to_whisper()` for interleaved float PCM: converts [frames] frames of [input] into
     * [output] (16 kHz mono). [RawIntResult.value] is the number of frames written (or required).
     */
    fun convertF32(input: FloatArray, frames: Int, channels: Int, sampleRate: Int, output: FloatArray): RawIntResult

    /** Like [convertF32] for interleaved signed 16-bit PCM. */
    fun convertS16(input: ShortArray, frames: Int, channels: Int, sampleRate: Int, output: FloatArray): RawIntResult

    fun statusName(status: Int): String

    fun lastError(): String
}

/** A status code plus one `int32_t` out-parameter. */
internal data class RawIntResult(val status: Int, val value: Int)

/** A status code plus an opaque context handle (`0` unless [status] is OK). */
internal data class RawHandleResult(val status: Int, val handle: Long)

/** `kggml_whisper_transcribe_params`. */
internal data class RawTranscribeParams(
    val language: String?,
    val initialPrompt: String?,
    val threads: Int,
    val translate: Boolean,
    val noContext: Boolean,
    val singleSegment: Boolean,
    val noTimestamps: Boolean,
    val maxTokens: Int,
    val temperature: Float,
    val temperatureIncrement: Float,
)

/** One raw `kggml_whisper_segment`. [text] holds the raw UTF-8 bytes. */
internal class RawSegmentResult(
    val status: Int,
    val startMs: Long = 0,
    val endMs: Long = 0,
    val noSpeechProbability: Float = 0f,
    val text: ByteArray = ByteArray(0),
)

/** `kggml_status` values used by the whisper ABI. Codes 0..6 equal `NativeErrorCode`. */
internal object WhisperStatus {
    const val OK: Int = 0
    const val MODEL_LOAD_FAILED: Int = 1
    const val INVALID_ARGUMENT: Int = 2
    const val BACKEND_UNAVAILABLE: Int = 3
    const val OUT_OF_MEMORY: Int = 4
    const val CANCELLED: Int = 5
    const val NATIVE_FAILURE: Int = 6
    const val BUFFER_TOO_SMALL: Int = 101
}

/** `kggml_backend_kind` values, as consumed by `kggml_whisper_context_params.backend`. */
internal object WhisperBackendKind {
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
internal expect fun createPlatformWhisperBridge(): NativeWhisperBridge

/** Lazily creates the platform bridge once and translates load failures into [NativeFailureException]. */
internal class WhisperBridgeHolder(private val factory: () -> NativeWhisperBridge) {
    private val ref = AtomicReference<NativeWhisperBridge?>(null)

    fun get(): NativeWhisperBridge {
        ref.load()?.let { return it }
        val created = try {
            factory()
        } catch (e: InferenceException) {
            throw e
        } catch (t: Throwable) {
            throw NativeFailureException(
                message = "Failed to load the Kggml whisper native library: ${t.message ?: t::class.simpleName}",
                cause = t,
            )
        }
        // Racing first calls may both create a bridge; they are stateless views of the same native library.
        ref.compareAndSet(null, created)
        return ref.load() ?: created
    }

    companion object {
        val platform: WhisperBridgeHolder = WhisperBridgeHolder(::createPlatformWhisperBridge)
    }
}

/** Builds the typed exception for a non-OK native [status], with status name and native diagnostic attached. */
internal fun NativeWhisperBridge.nativeError(
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
internal fun NativeWhisperBridge.check(status: Int, operation: String, backend: ComputeBackend? = null) {
    if (status != WhisperStatus.OK) throw nativeError(status, operation, backend)
}
