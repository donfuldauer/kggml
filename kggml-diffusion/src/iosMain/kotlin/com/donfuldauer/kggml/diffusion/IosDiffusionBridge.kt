package com.donfuldauer.kggml.diffusion

import cnames.structs.kggml_diffusion_context
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_abort_requested
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_clear_abort
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_context_create
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_context_default_params
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_context_free
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_context_get_info
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_context_info
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_context_params
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_generate
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_generate_default_params
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_generate_params
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_image
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_image_free
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_last_error
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_request_abort
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_system_info
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_upstream_commit
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_upstream_version
import com.donfuldauer.kggml.diffusion.cinterop.kggml_diffusion_version
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.StableRef
import kotlinx.cinterop.alloc
import kotlinx.cinterop.asStableRef
import kotlinx.cinterop.cstr
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.readBytes
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.staticCFunction
import kotlinx.cinterop.toCPointer
import kotlinx.cinterop.toKString
import kotlinx.cinterop.toLong
import kotlinx.cinterop.value

/**
 * [NativeDiffusionBridge] calling the `kggml_diffusion.h` C ABI directly through Kotlin/Native cinterop.
 *
 * The generated RGB8 image is copied into a `ByteArray` and its native buffer is released with
 * `kggml_diffusion_image_free()` in a `finally`, so it never outlives the call. Progress reaches the
 * [DiffusionProgressListener] through a [StableRef] passed as `user_data` (disposed on every path). Every C entry
 * point catches C++ exceptions and returns a status code, so nothing can unwind into Kotlin/Native frames. The
 * library is linked dynamically (`KggmlDiffusion.framework`), so there is nothing to load at runtime.
 */
@OptIn(ExperimentalForeignApi::class)
internal object IosDiffusionBridge : NativeDiffusionBridge {
    override fun moduleVersion(): String = kggml_diffusion_version()?.toKString().orEmpty()

    override fun upstreamVersion(): String = kggml_diffusion_upstream_version()?.toKString().orEmpty()

    override fun upstreamCommit(): String = kggml_diffusion_upstream_commit()?.toKString().orEmpty()

    override fun systemInfo(): String = kggml_diffusion_system_info()?.toKString().orEmpty()

    override fun statusName(status: Int): String = STATUS_NAMES[status] ?: "KGGML_STATUS_$status"

    override fun lastError(): String = kggml_diffusion_last_error()?.toKString().orEmpty()

    override fun contextCreate(params: RawContextParams): RawHandleResult = memScoped {
        val p = alloc<kggml_diffusion_context_params>()
        val defaults = kggml_diffusion_context_default_params(p.ptr).toInt()
        if (defaults != DiffusionStatus.OK) return@memScoped RawHandleResult(defaults, 0L)
        val paths = params.paths.map { it?.cstr?.ptr }
        p.model_path = paths[0]
        p.diffusion_model_path = paths[1]
        p.vae_path = paths[2]
        p.taesd_path = paths[3]
        p.clip_l_path = paths[4]
        p.clip_g_path = paths[5]
        p.t5xxl_path = paths[6]
        p.llm_path = paths[7]
        p.backend = params.backend
        p.gpu_device = params.gpuDevice
        p.n_threads = params.threads
        p.use_mmap = params.useMmap.toInt()
        p.flash_attn = params.flashAttention.toInt()
        p.offload_params_to_cpu = params.offloadParamsToCpu.toInt()
        p.keep_clip_on_cpu = params.keepClipOnCpu.toInt()
        p.keep_vae_on_cpu = params.keepVaeOnCpu.toInt()
        p.vae_decode_only = params.vaeDecodeOnly.toInt()
        p.weight_type = params.weightType
        val out = alloc<CPointerVar<kggml_diffusion_context>>()
        val status = kggml_diffusion_context_create(p.ptr, out.ptr).toInt()
        RawHandleResult(status, if (status == DiffusionStatus.OK) out.value.toLong() else 0L)
    }

    override fun contextFree(context: Long) = kggml_diffusion_context_free(contextPtr(context))

    override fun contextInfo(context: Long): RawContextInfo = memScoped {
        val info = alloc<kggml_diffusion_context_info>()
        val status = kggml_diffusion_context_get_info(contextPtr(context), info.ptr).toInt()
        if (status != DiffusionStatus.OK) return@memScoped RawContextInfo(status)
        RawContextInfo(
            status = status,
            backend = info.backend,
            usesGpu = info.uses_gpu != 0,
            threads = info.n_threads,
            supportsImageGeneration = info.supports_image_generation != 0,
            defaultSampleMethod = info.default_sample_method,
        )
    }

    override fun requestAbort(context: Long) = kggml_diffusion_request_abort(contextPtr(context))

    override fun clearAbort(context: Long) = kggml_diffusion_clear_abort(contextPtr(context))

    override fun abortRequested(context: Long): Boolean = kggml_diffusion_abort_requested(contextPtr(context)) != 0

    override fun generate(context: Long, params: RawGenerateParams, listener: DiffusionProgressListener?): RawImageResult =
        memScoped {
            val p = alloc<kggml_diffusion_generate_params>()
            val defaults = kggml_diffusion_generate_default_params(p.ptr).toInt()
            if (defaults != DiffusionStatus.OK) return@memScoped RawImageResult(defaults)
            p.prompt = params.prompt.cstr.ptr
            p.negative_prompt = params.negativePrompt.cstr.ptr
            p.width = params.width
            p.height = params.height
            p.sample_steps = params.sampleSteps
            p.cfg_scale = params.cfgScale
            p.distilled_guidance = params.distilledGuidance
            p.sample_method = params.sampleMethod
            p.scheduler = params.scheduler
            p.seed = params.seed
            p.clip_skip = params.clipSkip
            p.vae_tiling = params.vaeTiling.toInt()
            val image = alloc<kggml_diffusion_image>()
            // The listener reaches the C callback as a StableRef; disposed on every path once the call returns.
            val ref = listener?.let { StableRef.create(ProgressForward(context, it)) }
            try {
                val status = kggml_diffusion_generate(
                    contextPtr(context), p.ptr, if (ref != null) progressCallback else null, null, ref?.asCPointer(),
                    image.ptr,
                ).toInt()
                if (status != DiffusionStatus.OK) return@memScoped RawImageResult(status)
                copyImage(image)
            } finally {
                ref?.dispose()
                kggml_diffusion_image_free(image.ptr) // no-op for a zeroed (failed) image
            }
        }

    /** Copies the RGB8 pixels of [image] into a `ByteArray` (the caller frees the native buffer). */
    private fun copyImage(image: kggml_diffusion_image): RawImageResult {
        val width = image.width.toInt()
        val height = image.height.toInt()
        val channels = image.channels.toInt()
        val size = width.toLong() * height * channels
        val data = image.data
        if (data == null || size <= 0L || size > Int.MAX_VALUE) return RawImageResult(DiffusionStatus.NATIVE_FAILURE)
        val pixels = data.reinterpret<ByteVar>().readBytes(size.toInt())
        return RawImageResult(DiffusionStatus.OK, width, height, channels, pixels)
    }

    private fun contextPtr(handle: Long): CPointer<kggml_diffusion_context>? = handle.toCPointer()

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

/** `user_data` of [progressCallback]: the listener plus the context to abort if it throws. */
private class ProgressForward(val context: Long, val listener: DiffusionProgressListener)

/**
 * `kggml_diffusion_progress_callback` forwarding to the [DiffusionProgressListener] passed as a [StableRef] in
 * `user_data`. Runs on the thread inside `kggml_diffusion_generate()`; a Kotlin exception must never unwind into
 * C++, so a throwing listener latches an abort instead (the generation stops at its next step boundary).
 */
@OptIn(ExperimentalForeignApi::class)
private val progressCallback = staticCFunction { step: Int, total: Int, secondsPerStep: Float, userData: COpaquePointer? ->
    val forward = userData?.asStableRef<ProgressForward>()?.get()
    if (forward != null) {
        try {
            forward.listener.onProgress(step, total, secondsPerStep)
        } catch (_: Throwable) {
            kggml_diffusion_request_abort(forward.context.toCPointer<kggml_diffusion_context>())
        }
    }
}

internal actual fun createPlatformDiffusionBridge(): NativeDiffusionBridge = IosDiffusionBridge
