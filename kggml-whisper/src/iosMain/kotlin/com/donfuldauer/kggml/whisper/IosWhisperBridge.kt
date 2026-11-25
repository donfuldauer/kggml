package com.donfuldauer.kggml.whisper

import cnames.structs.kggml_whisper_context
import cnames.structs.kggml_whisper_resampler
import cnames.structs.kggml_whisper_state
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
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_max_prompt_tokens
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_pcm_to_whisper
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_request_abort
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_resampled_length
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_resampler_create
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_resampler_flush
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_resampler_free
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_resampler_max_output
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_resampler_process
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_segment
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_segment_count
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_state_create
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_state_free
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_state_get_segment
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_state_get_segment_tokens
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_state_segment_count
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_system_info
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_transcribe
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_transcribe_default_params
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_transcribe_params
import com.donfuldauer.kggml.whisper.cinterop.kggml_whisper_transcribe_with_state
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
import kotlinx.cinterop.MemScope
import kotlinx.cinterop.StableRef
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.asStableRef
import kotlinx.cinterop.convert
import kotlinx.cinterop.cstr
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.readBytes
import kotlinx.cinterop.staticCFunction
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
            val defaults = fillParams(p, params)
            if (defaults != WhisperStatus.OK) return@memScoped defaults
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
        rawSegment(status, seg)
    }

    override fun stateCreate(handle: Long): RawHandleResult = memScoped {
        val out = alloc<CPointerVar<kggml_whisper_state>>()
        val status = kggml_whisper_state_create(context(handle), out.ptr).toInt()
        RawHandleResult(status, if (status == WhisperStatus.OK) out.value.toLong() else 0L)
    }

    override fun stateFree(state: Long) = kggml_whisper_state_free(stateOf(state))

    override fun maxPromptTokens(handle: Long): RawIntResult = memScoped {
        val out = alloc<IntVar>()
        val status = kggml_whisper_max_prompt_tokens(context(handle), out.ptr).toInt()
        RawIntResult(status, out.value)
    }

    override fun transcribeWithState(
        handle: Long,
        state: Long,
        samples: FloatArray,
        count: Int,
        params: RawTranscribeParams,
        promptTokens: IntArray,
        promptCount: Int,
        listener: WhisperSegmentListener?,
    ): Int = memScoped {
        val p = alloc<kggml_whisper_transcribe_params>()
        val defaults = fillParams(p, params)
        if (defaults != WhisperStatus.OK) return@memScoped defaults
        // The listener reaches the C callback as a StableRef; disposed on every path once the call returns.
        val ref = listener?.let { StableRef.create(it) }
        try {
            withPinned(samples) { pcm ->
                withPinnedInts(promptTokens) { prompt ->
                    kggml_whisper_transcribe_with_state(
                        context(handle), stateOf(state), pcm, count.convert(), p.ptr, prompt, promptCount,
                        if (ref != null) segmentCallback else null, ref?.asCPointer(),
                    ).toInt()
                }
            }
        } finally {
            ref?.dispose()
        }
    }

    override fun stateSegmentCount(state: Long): RawIntResult = memScoped {
        val out = alloc<IntVar>()
        val status = kggml_whisper_state_segment_count(stateOf(state), out.ptr).toInt()
        RawIntResult(status, out.value)
    }

    override fun stateSegment(state: Long, index: Int): RawSegmentResult = memScoped {
        val seg = alloc<kggml_whisper_segment>()
        val status = kggml_whisper_state_get_segment(stateOf(state), index, seg.ptr).toInt()
        if (status != WhisperStatus.OK) return@memScoped RawSegmentResult(status)
        rawSegment(status, seg)
    }

    override fun stateSegmentTokens(state: Long, index: Int): RawTokensResult = memScoped {
        val needed = alloc<IntVar>()
        val query = kggml_whisper_state_get_segment_tokens(stateOf(state), index, null, 0, needed.ptr).toInt()
        if (query != WhisperStatus.OK && query != WhisperStatus.BUFFER_TOO_SMALL) return@memScoped RawTokensResult(query)
        val tokens = IntArray(needed.value.coerceAtLeast(0))
        val got = alloc<IntVar>()
        val status = withPinnedInts(tokens) { out ->
            kggml_whisper_state_get_segment_tokens(stateOf(state), index, out, tokens.size, got.ptr).toInt()
        }
        if (status != WhisperStatus.OK) RawTokensResult(status) else RawTokensResult(status, tokens.copyOf(got.value))
    }

    override fun resamplerCreate(inputRate: Int, channels: Int, format: Int): RawHandleResult = memScoped {
        val out = alloc<CPointerVar<kggml_whisper_resampler>>()
        val status = kggml_whisper_resampler_create(
            inputRate, KGGML_WHISPER_SAMPLE_RATE, channels, format, KGGML_RESAMPLE_SINC.toInt(), out.ptr,
        ).toInt()
        RawHandleResult(status, if (status == WhisperStatus.OK) out.value.toLong() else 0L)
    }

    override fun resamplerFree(resampler: Long) = kggml_whisper_resampler_free(resamplerOf(resampler))

    override fun resamplerMaxOutput(resampler: Long, frames: Int): Long = memScoped {
        if (frames < 0) return@memScoped -1L
        val out = alloc<size_tVar>()
        val status = kggml_whisper_resampler_max_output(resamplerOf(resampler), frames.convert(), out.ptr).toInt()
        if (status != WhisperStatus.OK) -1L else out.value.toLong()
    }

    override fun resamplerProcessF32(
        resampler: Long,
        input: FloatArray,
        frames: Int,
        channels: Int,
        output: FloatArray,
    ): RawIntResult {
        if (frames < 0 || channels <= 0 || input.size.toLong() < frames.toLong() * channels) {
            return RawIntResult(WhisperStatus.INVALID_ARGUMENT, 0)
        }
        return withPinned(input) { src -> resamplerProcess(resampler, src, frames, output) }
    }

    override fun resamplerProcessS16(
        resampler: Long,
        input: ShortArray,
        frames: Int,
        channels: Int,
        output: FloatArray,
    ): RawIntResult {
        if (frames < 0 || channels <= 0 || input.size.toLong() < frames.toLong() * channels) {
            return RawIntResult(WhisperStatus.INVALID_ARGUMENT, 0)
        }
        if (input.isEmpty()) return resamplerProcess(resampler, null, frames, output)
        return input.usePinned { src -> resamplerProcess(resampler, src.addressOf(0), frames, output) }
    }

    override fun resamplerFlush(resampler: Long, output: FloatArray): RawIntResult = memScoped {
        val produced = alloc<size_tVar>()
        val status = withPinned(output) { dst ->
            kggml_whisper_resampler_flush(resamplerOf(resampler), dst, output.size.convert(), produced.ptr).toInt()
        }
        RawIntResult(status, produced.value.toLong().coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
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

    private fun resamplerProcess(resampler: Long, input: COpaquePointer?, frames: Int, output: FloatArray): RawIntResult =
        memScoped {
            val produced = alloc<size_tVar>()
            val status = withPinned(output) { dst ->
                kggml_whisper_resampler_process(
                    resamplerOf(resampler), input, frames.convert(), dst, output.size.convert(), produced.ptr,
                ).toInt()
            }
            RawIntResult(status, produced.value.toLong().coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
        }

    /** Fills `kggml_whisper_transcribe_params` (strings are allocated in the enclosing [MemScope]). */
    private fun MemScope.fillParams(p: kggml_whisper_transcribe_params, params: RawTranscribeParams): Int {
        val defaults = kggml_whisper_transcribe_default_params(p.ptr).toInt()
        if (defaults != WhisperStatus.OK) return defaults
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
        return WhisperStatus.OK
    }

    private fun rawSegment(status: Int, seg: kggml_whisper_segment): RawSegmentResult {
        // Raw UTF-8 bytes (not toKString(): whisper.cpp may split a multi-byte character across segments).
        val text = seg.text?.let { p ->
            var length = 0
            while (p[length] != 0.toByte()) length++
            p.readBytes(length)
        } ?: ByteArray(0)
        return RawSegmentResult(status, seg.t0_ms, seg.t1_ms, seg.no_speech_prob, text)
    }

    /** Pins [array] for the duration of [block]; an empty array is passed as `NULL`. */
    private inline fun <R> withPinnedInts(array: IntArray, block: (CPointer<IntVar>?) -> R): R =
        if (array.isEmpty()) block(null) else array.usePinned { block(it.addressOf(0)) }

    /** Pins [array] for the duration of [block]; an empty array is passed as `NULL`. */
    private inline fun <R> withPinned(array: FloatArray, block: (CPointer<FloatVar>?) -> R): R =
        if (array.isEmpty()) block(null) else array.usePinned { block(it.addressOf(0)) }

    private fun context(handle: Long): CPointer<kggml_whisper_context>? = handle.toCPointer()

    private fun stateOf(handle: Long): CPointer<kggml_whisper_state>? = handle.toCPointer()

    private fun resamplerOf(handle: Long): CPointer<kggml_whisper_resampler>? = handle.toCPointer()

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

/**
 * `kggml_whisper_new_segment_callback` forwarding to the [WhisperSegmentListener] passed as a [StableRef] in
 * `user_data`. Runs on the thread inside `kggml_whisper_transcribe_with_state()`; a Kotlin exception must never
 * unwind into C++, so any throwable becomes `KGGML_STATUS_NATIVE_FAILURE` (which stops the decode).
 */
@OptIn(ExperimentalForeignApi::class)
private val segmentCallback = staticCFunction { _: CPointer<kggml_whisper_state>?, first: Int, count: Int, userData: COpaquePointer? ->
    val listener = userData?.asStableRef<WhisperSegmentListener>()?.get()
    if (listener == null) {
        WhisperStatus.NATIVE_FAILURE
    } else {
        try {
            listener.onNewSegments(first, count)
        } catch (_: Throwable) {
            WhisperStatus.NATIVE_FAILURE
        }
    }
}

internal actual fun createPlatformWhisperBridge(): NativeWhisperBridge = IosWhisperBridge
