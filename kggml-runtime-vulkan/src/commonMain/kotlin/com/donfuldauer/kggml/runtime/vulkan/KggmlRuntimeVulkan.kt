package com.donfuldauer.kggml.runtime.vulkan

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.runtime.KggmlRuntime

/**
 * The GPU add-on: depending on `com.donfuldauer:kggml-runtime-vulkan` packages ggml's Vulkan backend into the app
 * (Android: `libggml-vulkan.so` in the APK; JVM desktop on Linux x86_64: a module in the jar, extracted on first use),
 * and `:kggml-runtime` registers it when it loads, before any engine runs. Nothing needs to be called:
 * [ComputeBackend.GPU_VULKAN] and [ComputeBackend.AUTO] then use the GPU when its Vulkan driver is usable (Android 9+
 * with Vulkan 1.1+; on desktop any installed Vulkan driver); otherwise the engines stay on the CPU.
 *
 * On macOS and iOS (Metal) the add-on adds nothing.
 */
public object KggmlRuntimeVulkan {
    /** Whether the runtime has a Vulkan backend registered (on Android: the add-on was found and loaded). */
    public val isRegistered: Boolean
        get() = KggmlRuntime.backends().any { it.backend == ComputeBackend.GPU_VULKAN && it.compiled }

    /** Whether a Vulkan device is usable right now ([isRegistered] and the driver reported at least one GPU). */
    public val isAvailable: Boolean
        get() = ComputeBackend.GPU_VULKAN in KggmlRuntime.availableBackends()
}
