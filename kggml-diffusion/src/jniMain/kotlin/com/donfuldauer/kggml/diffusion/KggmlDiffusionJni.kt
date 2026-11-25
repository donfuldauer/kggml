package com.donfuldauer.kggml.diffusion

/**
 * `external` declarations implemented by `kggml-diffusion/native/jni/kggml_diffusion_jni.cpp`
 * (`Java_com_donfuldauer_kggml_diffusion_KggmlDiffusionJni_*`).
 *
 * Members are deliberately public inside this internal object: Kotlin mangles the JVM names of `internal`
 * members, which would break the JNI symbol names. Text crosses as UTF-8 `ByteArray`s.
 */
internal object KggmlDiffusionJni {
    const val VERSION_MODULE: Int = 0
    const val VERSION_UPSTREAM: Int = 1
    const val VERSION_UPSTREAM_COMMIT: Int = 2
    const val VERSION_SYSTEM_INFO: Int = 3

    /** model, diffusion_model, vae, taesd, clip_l, clip_g, t5xxl, llm. */
    const val MODEL_PATH_COUNT: Int = 8

    /**
     * backend, gpu_device, n_threads, use_mmap, flash_attn, offload_params_to_cpu, keep_clip_on_cpu, keep_vae_on_cpu,
     * vae_decode_only, weight_type.
     */
    const val CONTEXT_PARAM_COUNT: Int = 10

    /** backend, uses_gpu, n_threads, supports_image_generation, default_sample_method. */
    const val CONTEXT_INFO_COUNT: Int = 5

    /** width, height, sample_steps, sample_method, scheduler, clip_skip, vae_tiling. */
    const val GENERATE_INT_COUNT: Int = 7

    /** cfg_scale, distilled_guidance. */
    const val GENERATE_FLOAT_COUNT: Int = 2

    /** width, height, channels. */
    const val IMAGE_DIMS_COUNT: Int = 3

    @JvmStatic external fun nativeVersionString(which: Int): String?

    @JvmStatic external fun nativeStatusName(status: Int): String?

    @JvmStatic external fun nativeLastError(): String?

    @JvmStatic external fun nativeContextCreate(paths: Array<ByteArray?>, ints: IntArray, outHandle: LongArray): Int

    @JvmStatic external fun nativeContextFree(handle: Long)

    @JvmStatic external fun nativeContextInfo(handle: Long, out: IntArray): Int

    @JvmStatic external fun nativeRequestAbort(handle: Long)

    @JvmStatic external fun nativeClearAbort(handle: Long)

    @JvmStatic external fun nativeAbortRequested(handle: Long): Int

    /**
     * txt2img. [listener] (may be null) is called back on this thread for every progress point. On OK [outDims] holds
     * width, height, channels and `outPixels[0]` the RGB8 pixels; the native image buffer is always freed before
     * returning.
     */
    @JvmStatic external fun nativeGenerate(
        handle: Long,
        promptUtf8: ByteArray,
        negativePromptUtf8: ByteArray?,
        ints: IntArray,
        floats: FloatArray,
        seed: Long,
        listener: DiffusionProgressListener?,
        outDims: IntArray,
        outPixels: Array<ByteArray?>,
    ): Int

    /** `true` only in builds with `KGGML_DIFFUSION_TEST_HOOKS` (never in release libraries). */
    @JvmStatic external fun nativeTestHooksAvailable(): Boolean

    /** Test hook: live native diffusion contexts in this process, or -1 when unavailable. */
    @JvmStatic external fun nativeTestLiveObjects(): Long

    /** Test hook: creates a model-less stub context (abort flag + simulated generation only). */
    @JvmStatic external fun nativeTestContextCreateStub(outHandle: LongArray): Int

    /** Test hook: the real generation machinery with [steps] simulated steps and a deterministic RGB8 gradient. */
    @JvmStatic external fun nativeTestSimulateGenerate(
        handle: Long,
        steps: Int,
        width: Int,
        height: Int,
        listener: DiffusionProgressListener?,
        outDims: IntArray,
        outPixels: Array<ByteArray?>,
    ): Int
}
