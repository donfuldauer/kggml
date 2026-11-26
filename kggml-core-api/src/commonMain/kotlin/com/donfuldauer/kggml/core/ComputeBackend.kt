package com.donfuldauer.kggml.core

/**
 * The compute backend an engine executes on.
 *
 * Engines receive the *requested* backend through [BackendConfig.backend] and resolve it against the
 * backends actually available on the current device/build with [resolve] before initializing native state.
 */
public enum class ComputeBackend {
    /** Let the engine pick the best available backend (see [AUTO_PREFERENCE_ORDER]). */
    AUTO,

    /**
     * CPU execution only (Apple Accelerate / ARM NEON / x86 SIMD). Never touches a GPU.
     *
     * On iOS this is required for background execution: Whisper engines must set
     * `whisper_context_params.use_gpu = false` so no Metal command buffers are submitted.
     */
    CPU_ONLY,

    /** Apple Metal GPU (iOS, macOS). */
    GPU_METAL,

    /** Vulkan GPU (Android, Linux, Windows). */
    GPU_VULKAN,

    /** Apple Neural Engine via Core ML (iOS Whisper encoder). Falls back to Metal, then CPU. */
    NPU_COREML,

    /**
     * OpenCL GPU (Android, from the optional `kggml-runtime-opencl` add-on; ggml's OpenCL backend is tuned for
     * Qualcomm Adreno). Falls back to the CPU.
     */
    GPU_OPENCL,
    ;

    /** `true` if this backend offloads work to a GPU or NPU (i.e. it is neither [CPU_ONLY] nor [AUTO]). */
    public val isAccelerated: Boolean
        get() = this == GPU_METAL || this == GPU_VULKAN || this == GPU_OPENCL || this == NPU_COREML

    /**
     * The backend to try next when this one is unavailable, or `null` for [CPU_ONLY] (which is always
     * available) and [AUTO] (which is resolved by preference order instead).
     *
     * - [NPU_COREML] → [GPU_METAL]
     * - [GPU_METAL] → [CPU_ONLY]
     * - [GPU_VULKAN] → [CPU_ONLY]
     * - [GPU_OPENCL] → [CPU_ONLY]
     */
    public val fallback: ComputeBackend?
        get() = when (this) {
            NPU_COREML -> GPU_METAL
            GPU_METAL, GPU_VULKAN, GPU_OPENCL -> CPU_ONLY
            CPU_ONLY, AUTO -> null
        }

    public companion object {
        /**
         * Order in which [AUTO] picks a concrete backend: the first one present in the available set wins. OpenCL
         * comes before Vulkan: it is only ever available when the app ships the OpenCL add-on and the device's vendor
         * exposes OpenCL (in practice Qualcomm Adreno, for which ggml's OpenCL kernels are tuned); everywhere else
         * AUTO continues with Vulkan.
         */
        public val AUTO_PREFERENCE_ORDER: List<ComputeBackend> =
            listOf(NPU_COREML, GPU_METAL, GPU_OPENCL, GPU_VULKAN, CPU_ONLY)

        /**
         * Resolves a [requested] backend to a concrete backend (never [AUTO]) given the set of backends the
         * engine can actually use on this device. [CPU_ONLY] is always treated as available.
         *
         * Rules:
         * 1. [AUTO] resolves to the first entry of [AUTO_PREFERENCE_ORDER] contained in [available].
         * 2. A concrete backend contained in [available] resolves to itself.
         * 3. Otherwise, if [allowFallback] is `true`, the [fallback] chain is walked until an available
         *    backend is found (ending at [CPU_ONLY]).
         * 4. If [allowFallback] is `false`, a [BackendUnavailableException] is thrown.
         *
         * @param available backends supported by the engine on this device; [AUTO] entries are ignored.
         * @throws BackendUnavailableException if the requested backend is unavailable and fallback is disabled.
         */
        public fun resolve(
            requested: ComputeBackend,
            available: Set<ComputeBackend>,
            allowFallback: Boolean = true,
        ): ComputeBackend {
            val usable = available - AUTO + CPU_ONLY
            if (requested == AUTO) return AUTO_PREFERENCE_ORDER.first { it in usable }
            if (requested in usable) return requested
            if (!allowFallback) {
                throw BackendUnavailableException(
                    backend = requested,
                    message = "Compute backend $requested is not available (available: ${usable.sorted()})",
                )
            }
            var candidate: ComputeBackend = requested
            while (candidate !in usable) {
                // The chain always terminates at CPU_ONLY, which is guaranteed to be in `usable`.
                candidate = candidate.fallback ?: CPU_ONLY
            }
            return candidate
        }
    }
}
