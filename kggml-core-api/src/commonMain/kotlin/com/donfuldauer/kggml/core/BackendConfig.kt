package com.donfuldauer.kggml.core

/**
 * Runtime compute configuration passed to every engine at construction time.
 *
 * Instances are validated on construction (including via [copy]); invalid combinations throw
 * [IllegalArgumentException].
 *
 * @property backend requested compute backend; resolved by the engine with [ComputeBackend.resolve].
 * @property gpuLayers number of model layers to offload to the GPU. [ALL_GPU_LAYERS] (`-1`) offloads every
 *   layer, `0` keeps everything on the CPU. Must be `>= -1`, and must be `0` or `-1` when [backend] is
 *   [ComputeBackend.CPU_ONLY].
 * @property cpuThreads number of CPU worker threads. [AUTO_THREADS] (`-1`) lets the SDK choose (see
 *   [resolveCpuThreads]); otherwise must be in `1..`[MAX_CPU_THREADS].
 * @property useFlashAttention enables flash attention kernels where the engine and backend support them.
 * @property useMmap memory-maps model weights instead of reading them into heap memory.
 */
public data class BackendConfig(
    val backend: ComputeBackend = ComputeBackend.AUTO,
    val gpuLayers: Int = ALL_GPU_LAYERS,
    val cpuThreads: Int = AUTO_THREADS,
    val useFlashAttention: Boolean = false,
    val useMmap: Boolean = true,
) {
    init {
        require(gpuLayers >= ALL_GPU_LAYERS) {
            "gpuLayers must be >= $ALL_GPU_LAYERS (use $ALL_GPU_LAYERS for all layers), was $gpuLayers"
        }
        require(cpuThreads == AUTO_THREADS || cpuThreads in 1..MAX_CPU_THREADS) {
            "cpuThreads must be $AUTO_THREADS (auto) or in 1..$MAX_CPU_THREADS, was $cpuThreads"
        }
        require(!(backend == ComputeBackend.CPU_ONLY && gpuLayers > 0)) {
            "gpuLayers must be 0 or $ALL_GPU_LAYERS when backend is CPU_ONLY, was $gpuLayers"
        }
    }

    /**
     * The number of GPU layers to request from native code once [backend] has been resolved to
     * [resolvedBackend]: always `0` for [ComputeBackend.CPU_ONLY], otherwise [gpuLayers].
     */
    public fun effectiveGpuLayers(resolvedBackend: ComputeBackend = backend): Int =
        if (resolvedBackend == ComputeBackend.CPU_ONLY) 0 else gpuLayers

    /**
     * The concrete thread count to pass to native code. Explicit values are returned unchanged;
     * [AUTO_THREADS] resolves to [availableProcessors] clamped to `1..`[MAX_AUTO_THREADS].
     */
    public fun resolveCpuThreads(availableProcessors: Int): Int =
        if (cpuThreads == AUTO_THREADS) availableProcessors.coerceIn(1, MAX_AUTO_THREADS) else cpuThreads

    public companion object {
        /** Sentinel for [gpuLayers]: offload all layers. */
        public const val ALL_GPU_LAYERS: Int = -1

        /** Sentinel for [cpuThreads]: choose automatically. */
        public const val AUTO_THREADS: Int = -1

        /** Upper bound accepted for an explicit [cpuThreads] value. */
        public const val MAX_CPU_THREADS: Int = 512

        /** Upper bound used when [AUTO_THREADS] is resolved (more threads rarely help memory-bound inference). */
        public const val MAX_AUTO_THREADS: Int = 8

        /** CPU-only configuration (no GPU offload). Safe for iOS background execution. */
        public val CPU: BackendConfig = BackendConfig(backend = ComputeBackend.CPU_ONLY, gpuLayers = 0)

        /** Default configuration: automatic backend, all layers offloaded, automatic thread count. */
        public val DEFAULT: BackendConfig = BackendConfig()
    }
}
