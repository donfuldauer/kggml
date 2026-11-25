package com.donfuldauer.kggml.whisper

/**
 * [NativeWhisperBridge] over the `kggml_whisper_jni` JNI library (Android + JVM Desktop).
 *
 * PCM arrays are passed by reference; the native side pins them with `GetPrimitiveArrayCritical` (zero-copy) and
 * releases them on every path.
 *
 * @param loadLibraries loads `kggml_whisper_jni` and its dependencies; invoked once, before any native call.
 */
internal class JniWhisperBridge(loadLibraries: () -> Unit) : NativeWhisperBridge {
    init {
        loadLibraries()
    }

    override fun hasCoreMl(): Boolean = KggmlWhisperJni.nativeHasCoreml() != 0

    override fun moduleVersion(): String = version(KggmlWhisperJni.VERSION_MODULE)

    override fun upstreamVersion(): String = version(KggmlWhisperJni.VERSION_UPSTREAM)

    override fun upstreamCommit(): String = version(KggmlWhisperJni.VERSION_UPSTREAM_COMMIT)

    override fun systemInfo(): String = version(KggmlWhisperJni.VERSION_SYSTEM_INFO)

    override fun initFromFile(
        modelPath: String,
        backendKind: Int,
        gpuDevice: Int,
        flashAttention: Boolean,
    ): RawHandleResult {
        val out = LongArray(1)
        val status = KggmlWhisperJni.nativeInitFromFile(modelPath, backendKind, gpuDevice, if (flashAttention) 1 else 0, out)
        return RawHandleResult(status, if (status == WhisperStatus.OK) out[0] else 0L)
    }

    override fun free(handle: Long) = KggmlWhisperJni.nativeFree(handle)

    override fun effectiveBackend(handle: Long): RawIntResult = intResult { KggmlWhisperJni.nativeGetBackend(handle, it) }

    override fun usesGpu(handle: Long): RawIntResult = intResult { KggmlWhisperJni.nativeUsesGpu(handle, it) }

    override fun requestAbort(handle: Long) = KggmlWhisperJni.nativeRequestAbort(handle)

    override fun clearAbort(handle: Long) = KggmlWhisperJni.nativeClearAbort(handle)

    override fun abortRequested(handle: Long): Boolean = KggmlWhisperJni.nativeAbortRequested(handle) != 0

    override fun transcribe(handle: Long, samples: FloatArray, count: Int, params: RawTranscribeParams): Int =
        KggmlWhisperJni.nativeTranscribe(
            handle, samples, count, params.language, params.initialPrompt, transcribeInts(params),
            params.temperature, params.temperatureIncrement,
        )

    private fun transcribeInts(params: RawTranscribeParams): IntArray {
        val ints = IntArray(KggmlWhisperJni.TRANSCRIBE_INT_COUNT)
        ints[0] = params.threads
        ints[1] = params.translate.toInt()
        ints[2] = params.noContext.toInt()
        ints[3] = params.singleSegment.toInt()
        ints[4] = params.noTimestamps.toInt()
        ints[5] = params.maxTokens
        return ints
    }

    override fun segmentCount(handle: Long): RawIntResult = intResult { KggmlWhisperJni.nativeSegmentCount(handle, it) }

    override fun segment(handle: Long, index: Int): RawSegmentResult {
        val times = LongArray(2)
        val probs = FloatArray(1)
        val text = arrayOfNulls<ByteArray>(1)
        val status = KggmlWhisperJni.nativeGetSegment(handle, index, times, probs, text)
        if (status != WhisperStatus.OK) return RawSegmentResult(status)
        return RawSegmentResult(status, times[0], times[1], probs[0], text[0] ?: ByteArray(0))
    }

    override fun stateCreate(handle: Long): RawHandleResult {
        val out = LongArray(1)
        val status = KggmlWhisperJni.nativeStateCreate(handle, out)
        return RawHandleResult(status, if (status == WhisperStatus.OK) out[0] else 0L)
    }

    override fun stateFree(state: Long) = KggmlWhisperJni.nativeStateFree(state)

    override fun maxPromptTokens(handle: Long): RawIntResult = intResult { KggmlWhisperJni.nativeMaxPromptTokens(handle, it) }

    override fun transcribeWithState(
        handle: Long,
        state: Long,
        samples: FloatArray,
        count: Int,
        params: RawTranscribeParams,
        promptTokens: IntArray,
        promptCount: Int,
        listener: WhisperSegmentListener?,
    ): Int = KggmlWhisperJni.nativeTranscribeWithState(
        handle, state, samples, count, params.language, params.initialPrompt, transcribeInts(params),
        params.temperature, params.temperatureIncrement, promptTokens, promptCount, listener,
    )

    override fun stateSegmentCount(state: Long): RawIntResult = intResult { KggmlWhisperJni.nativeStateSegmentCount(state, it) }

    override fun stateSegment(state: Long, index: Int): RawSegmentResult {
        val times = LongArray(2)
        val probs = FloatArray(1)
        val text = arrayOfNulls<ByteArray>(1)
        val status = KggmlWhisperJni.nativeStateGetSegment(state, index, times, probs, text)
        if (status != WhisperStatus.OK) return RawSegmentResult(status)
        return RawSegmentResult(status, times[0], times[1], probs[0], text[0] ?: ByteArray(0))
    }

    override fun stateSegmentTokens(state: Long, index: Int): RawTokensResult {
        val out = arrayOfNulls<IntArray>(1)
        val status = KggmlWhisperJni.nativeStateSegmentTokens(state, index, out)
        return RawTokensResult(status, if (status == WhisperStatus.OK) out[0] ?: IntArray(0) else IntArray(0))
    }

    override fun resamplerCreate(inputRate: Int, channels: Int, format: Int): RawHandleResult {
        val out = LongArray(1)
        val status = KggmlWhisperJni.nativeResamplerCreate(inputRate, channels, format, RESAMPLE_SINC, out)
        return RawHandleResult(status, if (status == WhisperStatus.OK) out[0] else 0L)
    }

    override fun resamplerFree(resampler: Long) = KggmlWhisperJni.nativeResamplerFree(resampler)

    override fun resamplerMaxOutput(resampler: Long, frames: Int): Long =
        KggmlWhisperJni.nativeResamplerMaxOutput(resampler, frames.toLong())

    override fun resamplerProcessF32(
        resampler: Long,
        input: FloatArray,
        frames: Int,
        channels: Int,
        output: FloatArray,
    ): RawIntResult = intResult { KggmlWhisperJni.nativeResamplerProcessF32(resampler, input, frames, channels, output, it) }

    override fun resamplerProcessS16(
        resampler: Long,
        input: ShortArray,
        frames: Int,
        channels: Int,
        output: FloatArray,
    ): RawIntResult = intResult { KggmlWhisperJni.nativeResamplerProcessS16(resampler, input, frames, channels, output, it) }

    override fun resamplerFlush(resampler: Long, output: FloatArray): RawIntResult =
        intResult { KggmlWhisperJni.nativeResamplerFlush(resampler, output, it) }

    override fun resampledLength(frames: Int, sampleRate: Int): Long =
        KggmlWhisperJni.nativeResampledLength(frames.toLong(), sampleRate)

    override fun convertF32(
        input: FloatArray,
        frames: Int,
        channels: Int,
        sampleRate: Int,
        output: FloatArray,
    ): RawIntResult = intResult {
        KggmlWhisperJni.nativePcmF32ToWhisper(input, frames, channels, sampleRate, RESAMPLE_SINC, output, it)
    }

    override fun convertS16(
        input: ShortArray,
        frames: Int,
        channels: Int,
        sampleRate: Int,
        output: FloatArray,
    ): RawIntResult = intResult {
        KggmlWhisperJni.nativePcmS16ToWhisper(input, frames, channels, sampleRate, RESAMPLE_SINC, output, it)
    }

    override fun statusName(status: Int): String = KggmlWhisperJni.nativeStatusName(status) ?: "KGGML_STATUS_$status"

    override fun lastError(): String = KggmlWhisperJni.nativeLastError().orEmpty()

    private fun version(which: Int): String = KggmlWhisperJni.nativeVersionString(which).orEmpty()

    private inline fun intResult(call: (IntArray) -> Int): RawIntResult {
        val out = IntArray(1)
        val status = call(out)
        return RawIntResult(status, out[0])
    }

    private fun Boolean.toInt(): Int = if (this) 1 else 0

    private companion object {
        /** `KGGML_RESAMPLE_SINC`. */
        const val RESAMPLE_SINC: Int = 1
    }
}
