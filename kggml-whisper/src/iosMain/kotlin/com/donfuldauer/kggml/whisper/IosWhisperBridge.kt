package com.donfuldauer.kggml.whisper

import cnames.structs.kggml_whisper_context
import com.donfuldauer.kggml.whisper.cinterop.KGGML_PCM_F32
import com.donfuldauer.kggml.whisper.cinterop.KGGML_PCM_S16
import com.donfuldauer.kggml.whisper.cinterop.KGGML_RESAMPLE_SINC
import com.donfuldauer.kggml.whisper.cinterop.KGGML_WHISPER_SAMPLE_RATE
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_abort_requested
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_clear_abort
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_context_default_params
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_context_params
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_free
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_get_backend
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_get_segment
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_has_coreml
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_init_from_file
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_last_error
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_pcm_to_whisper
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_request_abort
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_resampled_length
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_segment
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_segment_count
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_system_info
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_transcribe
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_transcribe_default_params
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_transcribe_params
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_upstream_commit
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_upstream_version
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_uses_gpu
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_version
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.FloatVar
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.convert
import kotlinx.cinterop.cstr
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.readBytes
import kotlinx.cinterop.toCPointer
import kotlinx.cinterop.toKString
import kotlinx.cinterop.toLong
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value
import platform.posix.size_tVar

/**
 * [NativeWhisperBridge] calling the `kggml_whisper.h` C ABI directly through Kotlin/Native cinterop.
 *
 * PCM arrays are pinned with `usePinned { it.addressOf(0) }` and read in place (zero-copy). Every C entry point
 * catches C++ exceptions and returns a status code, so nothing can unwind into Kotlin/Native frames. The library
 * is linked dynamically (`KggmlWhisper.framework`), so there is nothing to load at runtime.
 */
@OptIn(ExperimentalForeignApi::class)
internal object IosWhisperBridge : NativeWhisperBridge {
    override fun hasCoreMl(): Boolean = kggml_whisper_has_coreml() != 0

    override fun moduleVersion(): String = kggml_whisper_version()?.toKString().orEmpty()

    override fun upstreamVersion(): String = kggml_whisper_upstream_version()?.toKString().orEmpty()

    override fun upstreamCommit(): String = kggml_whisper_upstream_commit()?.toKString().orEmpty()

    override fun systemInfo(): String = kggml_whisper_system_info()?.toKString().orEmpty()

    override fun initFromFile(
        modelPath: String,
        backendKind: Int,
        gpuDevice: Int,
        flashAttention: Boolean,
    ): RawHandleResult = memScoped {
        val params = alloc<kggml_whisper_context_params>()
        val defaults = kggml_whisper_context_default_params(params.ptr).toInt()
        if (defaults != WhisperStatus.OK) return@memScoped RawHandleResult(defaults, 0L)
        params.backend = backendKind
        params.gpu_device = gpuDevice
        params.flash_attn = if (flashAttention) 1 else 0
        val out = alloc<CPointerVar<kggml_whisper_context>>()
        val status = kggml_whisper_init_from_file(modelPath, params.ptr, out.ptr).toInt()
        RawHandleResult(status, if (status == WhisperStatus.OK) out.value.toLong() else 0L)
    }

    override fun free(handle: Long) = kggml_whisper_free(context(handle))

    override fun effectiveBackend(handle: Long): RawIntResult = memScoped {
        val out = alloc<IntVar>()
        val status = kggml_whisper_get_backend(context(handle), out.ptr).toInt()
        RawIntResult(status, out.value)
    }

    override fun usesGpu(handle: Long): RawIntResult = memScoped {
        val out = alloc<IntVar>()
        val status = kggml_whisper_uses_gpu(context(handle), out.ptr).toInt()
        RawIntResult(status, out.value)
    }

    override fun requestAbort(handle: Long) = kggml_whisper_request_abort(context(handle))

    override fun clearAbort(handle: Long) = kggml_whisper_clear_abort(context(handle))

    override fun abortRequested(handle: Long): Boolean = kggml_whisper_abort_requested(context(handle)) != 0

    override fun transcribe(handle: Long, samples: FloatArray, count: Int, params: RawTranscribeParams): Int =
        memScoped {
            val p = alloc<kggml_whisper_transcribe_params>()
            val defaults = kggml_whisper_transcribe_default_params(p.ptr).toInt()
            if (defaults != WhisperStatus.OK) return@memScoped defaults
            p.language = params.language?.cstr?.getPointer(this)
            p.initial_prompt = params.initialPrompt?.cstr?.getPointer(this)
            p.n_threads = params.threads
            p.translate = params.translate.toInt()
            p.no_context = params.noContext.toInt()
            p.single_segment = params.singleSegment.toInt()
            p.no_timestamps = params.noTimestamps.toInt()
            p.max_tokens = params.maxTokens
            p.temperature = params.temperature
            p.temperature_inc = params.temperatureIncrement
            withPinned(samples) { pcm ->
                kggml_whisper_transcribe(context(handle), pcm, count.convert(), p.ptr).toInt()
            }
        }

    override fun segmentCount(handle: Long): RawIntResult = memScoped {
        val out = alloc<IntVar>()
        val status = kggml_whisper_segment_count(context(handle), out.ptr).toInt()
        RawIntResult(status, out.value)
    }

    override fun segment(handle: Long, index: Int): RawSegmentResult = memScoped {
        val seg = alloc<kggml_whisper_segment>()
        val status = kggml_whisper_get_segment(context(handle), index, seg.ptr).toInt()
        if (status != WhisperStatus.OK) return@memScoped RawSegmentResult(status)
        // Raw UTF-8 bytes (not toKString(): whisper.cpp may split a multi-byte character across segments).
        val text = seg.text?.let { p ->
            var length = 0
            while (p[length] != 0.toByte()) length++
            p.readBytes(length)
        } ?: ByteArray(0)
        RawSegmentResult(status, seg.t0_ms, seg.t1_ms, seg.no_speech_prob, text)
    }

    override fun resampledLength(frames: Int, sampleRate: Int): Long = memScoped {
        if (frames < 0) return@memScoped -1L
        val out = alloc<size_tVar>()
        val status = kggml_whisper_resampled_length(frames.convert(), sampleRate, KGGML_WHISPER_SAMPLE_RATE, out.ptr)
        if (status.toInt() != WhisperStatus.OK) -1L else out.value.toLong()
    }

    override fun convertF32(
        input: FloatArray,
        frames: Int,
        channels: Int,
        sampleRate: Int,
        output: FloatArray,
    ): RawIntResult = withPinned(input) { src -> convert(src, KGGML_PCM_F32.toInt(), frames, channels, sampleRate, output) }

    override fun convertS16(
        input: ShortArray,
        frames: Int,
        channels: Int,
        sampleRate: Int,
        output: FloatArray,
    ): RawIntResult {
        if (input.isEmpty()) return convert(null, KGGML_PCM_S16.toInt(), frames, channels, sampleRate, output)
        return input.usePinned { src ->
            convert(src.addressOf(0), KGGML_PCM_S16.toInt(), frames, channels, sampleRate, output)
        }
    }

    // kggml_status_name() is declared in kggml_runtime.h, whose bindings live in :kggml-runtime's cinterop klib
    // (cinterop does not regenerate a dependency's declarations), so the names are mapped here.
    override fun statusName(status: Int): String = STATUS_NAMES[status] ?: "KGGML_STATUS_$status"

    override fun lastError(): String = kggml_whisper_last_error()?.toKString().orEmpty()

    private fun convert(
        input: COpaquePointer?,
        format: Int,
        frames: Int,
        channels: Int,
        sampleRate: Int,
        output: FloatArray,
    ): RawIntResult = memScoped {
        val produced = alloc<size_tVar>()
        val status = withPinned(output) { dst ->
            kggml_whisper_pcm_to_whisper(
                input, format, frames.convert(), channels, sampleRate, KGGML_RESAMPLE_SINC.toInt(),
                dst, output.size.convert(), produced.ptr,
            ).toInt()
        }
        RawIntResult(status, produced.value.toLong().coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
    }

    /** Pins [array] for the duration of [block]; an empty array is passed as `NULL`. */
    private inline fun <R> withPinned(array: FloatArray, block: (CPointer<FloatVar>?) -> R): R =
        if (array.isEmpty()) block(null) else array.usePinned { block(it.addressOf(0)) }

    private fun context(handle: Long): CPointer<kggml_whisper_context>? = handle.toCPointer()

    private fun Boolean.toInt(): Int = if (this) 1 else 0

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

internal actual fun createPlatformWhisperBridge(): NativeWhisperBridge = IosWhisperBridge
