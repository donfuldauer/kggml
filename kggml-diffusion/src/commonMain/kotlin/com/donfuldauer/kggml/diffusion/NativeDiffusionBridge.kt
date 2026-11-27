package com.donfuldauer.kggml.diffusion

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceException
import com.donfuldauer.kggml.core.NativeFailureException
import kotlin.concurrent.atomics.AtomicReference

/**
 * Raw, 1:1 view of the `kggml_diffusion.h` C ABI.
 *
 * Implemented by the JNI bridge (Android, JVM Desktop), the Kotlin/Native cinterop bridge (iOS), and by fakes in
 * tests. Methods return raw native values (status codes, opaque `Long` handles); translation into Kotlin types and
 * typed exceptions happens once, in [DiffusionEngine].
 *
 * [generate] copies the RGB8 result into a `ByteArray` and **always** releases the native buffer with
 * `kggml_diffusion_image_free()` before returning, on every path. Text crosses as UTF-8.
 *
 * Contract: [lastError] must be called on the same thread, immediately after the failing call (the native
 * diagnostic message is thread-local). Every method taking a handle except [requestAbort] and [abortRequested]
 * must be called from the owning engine's serial dispatcher.
 */
internal interface NativeDiffusionBridge {
    fun moduleVersion(): String

    fun upstreamVersion(): String

    fun upstreamCommit(): String

    fun systemInfo(): String

    fun statusName(status: Int): String

    fun lastError(): String

    fun contextCreate(params: RawContextParams): RawHandleResult

    fun contextFree(context: Long)

    fun contextInfo(context: Long): RawContextInfo

    /** Lock-free, any thread. */
    fun requestAbort(context: Long)

    fun clearAbort(context: Long)

    /** Lock-free, any thread. */
    fun abortRequested(context: Long): Boolean

    /**
     * `kggml_diffusion_generate()` (txt2img). [listener] receives every progress point on the calling thread while
     * the call runs. Abort is driven by the context's latched abort flag ([requestAbort]); no abort callback is used.
     */
    fun generate(context: Long, params: RawGenerateParams, listener: DiffusionProgressListener?): RawImageResult
}

/**
 * Receives `kggml_diffusion_progress_callback` notifications on the thread running [NativeDiffusionBridge.generate].
 * Implementations must not throw; a throwable is treated as an abort request by the bridges.
 *
 * The JNI bridge invokes [onProgress] by name (signature `(IIF)V`); keep it stable.
 */
internal fun interface DiffusionProgressListener {
    fun onProgress(step: Int, totalSteps: Int, secondsPerStep: Float)
}

/** `kggml_diffusion_context_params`. [paths] holds the 8 model paths in C struct order (`null` = unused). */
internal data class RawContextParams(
    val paths: List<String?>,
    val backend: Int,
    val gpuDevice: Int,
    val threads: Int,
    val useMmap: Boolean,
    val flashAttention: Boolean,
    val offloadParamsToCpu: Boolean,
    val keepClipOnCpu: Boolean,
    val keepVaeOnCpu: Boolean,
    val vaeDecodeOnly: Boolean,
    val weightType: Int,
) {
    init {
        require(paths.size == PATH_COUNT) { "expected $PATH_COUNT paths, got ${paths.size}" }
    }

    companion object {
        const val PATH_COUNT: Int = 8
    }
}

/** `kggml_diffusion_context_info`. */
internal data class RawContextInfo(
    val status: Int,
    val backend: Int = 0,
    val usesGpu: Boolean = false,
    val threads: Int = 0,
    val supportsImageGeneration: Boolean = false,
    val defaultSampleMethod: Int = -1,
)

/** `kggml_diffusion_generate_params`. */
internal data class RawGenerateParams(
    val prompt: String,
    val negativePrompt: String,
    val width: Int,
    val height: Int,
    val sampleSteps: Int,
    val cfgScale: Float,
    val distilledGuidance: Float,
    val sampleMethod: Int,
    val scheduler: Int,
    val seed: Long,
    val clipSkip: Int,
    val vaeTiling: Boolean,
)

/** A status code plus an opaque handle (`0` unless [status] is OK). */
internal data class RawHandleResult(val status: Int, val handle: Long)

/** A status code plus the copied RGB8 image (empty unless [status] is OK). */
internal class RawImageResult(
    val status: Int,
    val width: Int = 0,
    val height: Int = 0,
    val channels: Int = 0,
    val pixels: ByteArray = ByteArray(0),
)

/** `kggml_status` values used by the diffusion ABI. Codes 0..6 equal `NativeErrorCode`. */
internal object DiffusionStatus {
    const val OK: Int = 0
    const val MODEL_LOAD_FAILED: Int = 1
    const val INVALID_ARGUMENT: Int = 2
    const val BACKEND_UNAVAILABLE: Int = 3
    const val OUT_OF_MEMORY: Int = 4
    const val CANCELLED: Int = 5
    const val NATIVE_FAILURE: Int = 6
}

/** `kggml_backend_kind` values, as consumed by `kggml_diffusion_context_params.backend`. */
internal object DiffusionBackendKind {
    const val CPU: Int = 0
    const val METAL: Int = 1
    const val VULKAN: Int = 2
    const val COREML: Int = 3
    const val OPENCL: Int = 4

    fun of(backend: ComputeBackend): Int = when (backend) {
        ComputeBackend.CPU_ONLY -> CPU
        ComputeBackend.GPU_METAL -> METAL
        ComputeBackend.GPU_VULKAN -> VULKAN
        ComputeBackend.NPU_COREML -> COREML
        ComputeBackend.GPU_OPENCL -> OPENCL
        ComputeBackend.AUTO -> error("AUTO must be resolved before reaching native code")
    }

    fun toComputeBackend(kind: Int): ComputeBackend? = when (kind) {
        CPU -> ComputeBackend.CPU_ONLY
        METAL -> ComputeBackend.GPU_METAL
        VULKAN -> ComputeBackend.GPU_VULKAN
        COREML -> ComputeBackend.NPU_COREML
        OPENCL -> ComputeBackend.GPU_OPENCL
        else -> null
    }
}

/**
 * Creates the platform bridge, loading the native libraries if needed.
 *
 * @throws Throwable (typically `UnsatisfiedLinkError`) if the native library cannot be loaded.
 */
internal expect fun createPlatformDiffusionBridge(): NativeDiffusionBridge

/** Lazily creates the platform bridge once and translates load failures into [NativeFailureException]. */
internal class DiffusionBridgeHolder(private val factory: () -> NativeDiffusionBridge) {
    private val ref = AtomicReference<NativeDiffusionBridge?>(null)

    fun get(): NativeDiffusionBridge {
        ref.load()?.let { return it }
        val created = try {
            factory()
        } catch (e: InferenceException) {
            throw e
        } catch (t: Throwable) {
            throw NativeFailureException(
                message = "Failed to load the Kggml diffusion native library: ${t.message ?: t::class.simpleName}",
                cause = t,
            )
        }
        // Racing first calls may both create a bridge; they are stateless views of the same native library.
        ref.compareAndSet(null, created)
        return ref.load() ?: created
    }

    companion object {
        val platform: DiffusionBridgeHolder = DiffusionBridgeHolder(::createPlatformDiffusionBridge)
    }
}

/** Builds the typed exception for a non-OK native [status], with status name and native diagnostic attached. */
internal fun NativeDiffusionBridge.nativeError(
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
internal fun NativeDiffusionBridge.check(status: Int, operation: String, backend: ComputeBackend? = null) {
    if (status != DiffusionStatus.OK) throw nativeError(status, operation, backend)
}
