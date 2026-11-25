package com.donfuldauer.kggml.whisper

/**
 * `external` declarations implemented by `kggml-whisper/native/jni/kggml_whisper_jni.cpp`
 * (`Java_com_donfuldauer_kggml_whisper_KggmlWhisperJni_*`).
 *
 * Members are deliberately public inside this internal object: Kotlin mangles the JVM names of `internal`
 * members, which would break the JNI symbol names.
 */
internal object KggmlWhisperJni {
    const val VERSION_MODULE: Int = 0
    const val VERSION_UPSTREAM: Int = 1
    const val VERSION_UPSTREAM_COMMIT: Int = 2
    const val VERSION_SYSTEM_INFO: Int = 3

    /** Number of ints in the `ints` argument of [nativeTranscribe]. */
    const val TRANSCRIBE_INT_COUNT: Int = 6

    @JvmStatic external fun nativeVersionString(which: Int): String?

    @JvmStatic external fun nativeHasCoreml(): Int

    @JvmStatic external fun nativeStatusName(status: Int): String?

    @JvmStatic external fun nativeLastError(): String?

    @JvmStatic external fun nativeInitFromFile(
        modelPath: String,
        backend: Int,
        gpuDevice: Int,
        flashAttn: Int,
        outHandle: LongArray,
    ): Int

    @JvmStatic external fun nativeFree(handle: Long)

    @JvmStatic external fun nativeGetBackend(handle: Long, out: IntArray): Int

    @JvmStatic external fun nativeUsesGpu(handle: Long, out: IntArray): Int

    @JvmStatic external fun nativeRequestAbort(handle: Long)

    @JvmStatic external fun nativeClearAbort(handle: Long)

    @JvmStatic external fun nativeAbortRequested(handle: Long): Int

    /** `ints`: n_threads, translate, no_context, single_segment, no_timestamps, max_tokens. */
    @JvmStatic external fun nativeTranscribe(
        handle: Long,
        samples: FloatArray,
        count: Int,
        language: String?,
        initialPrompt: String?,
        ints: IntArray,
        temperature: Float,
        temperatureInc: Float,
    ): Int

    @JvmStatic external fun nativeSegmentCount(handle: Long, out: IntArray): Int

    /** `times` = t0_ms, t1_ms; `probs[0]` = no-speech probability; `text[0]` = raw UTF-8 bytes. */
    @JvmStatic external fun nativeGetSegment(
        handle: Long,
        index: Int,
        times: LongArray,
        probs: FloatArray,
        text: Array<ByteArray?>,
    ): Int

    @JvmStatic external fun nativeResampledLength(frames: Long, sampleRate: Int): Long

    @JvmStatic external fun nativePcmF32ToWhisper(
        input: FloatArray,
        frames: Int,
        channels: Int,
        sampleRate: Int,
        quality: Int,
        output: FloatArray,
        outFrames: IntArray,
    ): Int

    @JvmStatic external fun nativePcmS16ToWhisper(
        input: ShortArray,
        frames: Int,
        channels: Int,
        sampleRate: Int,
        quality: Int,
        output: FloatArray,
        outFrames: IntArray,
    ): Int

    /** `true` only in builds with `KGGML_WHISPER_TEST_HOOKS` (never in release libraries). */
    @JvmStatic external fun nativeTestHooksAvailable(): Boolean

    /** Test hook: `out` = use_gpu, coreml_allowed that init would pass for `backend`. */
    @JvmStatic external fun nativeTestResolveUseGpu(backend: Int, out: IntArray): Int

    /** Test hook: abort-callback polls during the last transcription, or -1 when unavailable. */
    @JvmStatic external fun nativeTestAbortPollCount(handle: Long): Long
}
