package com.donfuldauer.kggml.diffusion

/**
 * [NativeDiffusionBridge] over the `kggml_diffusion_jni` JNI library (Android + JVM Desktop).
 *
 * The native side copies the generated RGB8 pixels into a new `ByteArray` and releases the native image buffer
 * (`kggml_diffusion_image_free`) on every path. Progress reaches the [DiffusionProgressListener] on the generating
 * thread.
 *
 * @param loadLibraries loads `kggml_diffusion_jni` and its dependencies; invoked once, before any native call.
 */
internal class JniDiffusionBridge(loadLibraries: () -> Unit) : NativeDiffusionBridge {
    init {
        loadLibraries()
    }

    override fun moduleVersion(): String = version(KggmlDiffusionJni.VERSION_MODULE)

    override fun upstreamVersion(): String = version(KggmlDiffusionJni.VERSION_UPSTREAM)

    override fun upstreamCommit(): String = version(KggmlDiffusionJni.VERSION_UPSTREAM_COMMIT)

    override fun systemInfo(): String = version(KggmlDiffusionJni.VERSION_SYSTEM_INFO)

    override fun statusName(status: Int): String = KggmlDiffusionJni.nativeStatusName(status) ?: "KGGML_STATUS_$status"

    override fun lastError(): String = KggmlDiffusionJni.nativeLastError().orEmpty()

    override fun contextCreate(params: RawContextParams): RawHandleResult {
        val paths = arrayOfNulls<ByteArray>(KggmlDiffusionJni.MODEL_PATH_COUNT)
        params.paths.forEachIndexed { i, path -> paths[i] = path?.encodeToByteArray() }
        val ints = IntArray(KggmlDiffusionJni.CONTEXT_PARAM_COUNT)
        ints[0] = params.backend
        ints[1] = params.gpuDevice
        ints[2] = params.threads
        ints[3] = params.useMmap.toInt()
        ints[4] = params.flashAttention.toInt()
        ints[5] = params.offloadParamsToCpu.toInt()
        ints[6] = params.keepClipOnCpu.toInt()
        ints[7] = params.keepVaeOnCpu.toInt()
        ints[8] = params.vaeDecodeOnly.toInt()
        ints[9] = params.weightType
        val out = LongArray(1)
        val status = KggmlDiffusionJni.nativeContextCreate(paths, ints, out)
        return RawHandleResult(status, if (status == DiffusionStatus.OK) out[0] else 0L)
    }

    override fun contextFree(context: Long) = KggmlDiffusionJni.nativeContextFree(context)

    override fun contextInfo(context: Long): RawContextInfo {
        val out = IntArray(KggmlDiffusionJni.CONTEXT_INFO_COUNT)
        val status = KggmlDiffusionJni.nativeContextInfo(context, out)
        if (status != DiffusionStatus.OK) return RawContextInfo(status)
        return RawContextInfo(
            status = status,
            backend = out[0],
            usesGpu = out[1] != 0,
            threads = out[2],
            supportsImageGeneration = out[3] != 0,
            defaultSampleMethod = out[4],
        )
    }

    override fun requestAbort(context: Long) = KggmlDiffusionJni.nativeRequestAbort(context)

    override fun clearAbort(context: Long) = KggmlDiffusionJni.nativeClearAbort(context)

    override fun abortRequested(context: Long): Boolean = KggmlDiffusionJni.nativeAbortRequested(context) != 0

    override fun generate(context: Long, params: RawGenerateParams, listener: DiffusionProgressListener?): RawImageResult {
        val ints = IntArray(KggmlDiffusionJni.GENERATE_INT_COUNT)
        ints[0] = params.width
        ints[1] = params.height
        ints[2] = params.sampleSteps
        ints[3] = params.sampleMethod
        ints[4] = params.scheduler
        ints[5] = params.clipSkip
        ints[6] = params.vaeTiling.toInt()
        val floats = floatArrayOf(params.cfgScale, params.distilledGuidance)
        return imageResult { dims, pixels ->
            KggmlDiffusionJni.nativeGenerate(
                context, params.prompt.encodeToByteArray(), params.negativePrompt.encodeToByteArray(), ints, floats,
                params.seed, listener, dims, pixels,
            )
        }
    }

    private fun version(which: Int): String = KggmlDiffusionJni.nativeVersionString(which).orEmpty()

    private fun Boolean.toInt(): Int = if (this) 1 else 0

    internal companion object {
        /** Runs a JNI call that fills (dims, pixels) and wraps the outcome. */
        inline fun imageResult(call: (IntArray, Array<ByteArray?>) -> Int): RawImageResult {
            val dims = IntArray(KggmlDiffusionJni.IMAGE_DIMS_COUNT)
            val pixels = arrayOfNulls<ByteArray>(1)
            val status = call(dims, pixels)
            if (status != DiffusionStatus.OK) return RawImageResult(status)
            return RawImageResult(status, dims[0], dims[1], dims[2], pixels[0] ?: ByteArray(0))
        }
    }
}
