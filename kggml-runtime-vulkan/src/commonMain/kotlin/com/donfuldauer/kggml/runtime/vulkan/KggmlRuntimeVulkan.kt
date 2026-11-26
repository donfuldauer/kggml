package com.donfuldauer.kggml.runtime.vulkan

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.runtime.KggmlRuntime

/**
 * The Android GPU add-on: depending on `com.donfuldauer:kggml-runtime-vulkan` packages ggml's Vulkan backend
 * (`libggml-vulkan.so`) into the app, and `:kggml-runtime` registers it when it loads, before any engine runs. Nothing
 * needs to be called: [ComputeBackend.GPU_VULKAN] and [ComputeBackend.AUTO] then use the phone's GPU when its Vulkan
 * driver is usable (Vulkan 1.1+, Android 9+); otherwise the engines stay on the CPU.
 *
 * On JVM desktop (Vulkan is linked into the runtime by `KGGML_ENABLE_VULKAN`) and iOS (Metal) the add-on adds nothing.
 */
public object KggmlRuntimeVulkan {
    /** Whether the runtime has a Vulkan backend registered (on Android: the add-on was found and loaded). */
    public val isRegistered: Boolean
        get() = KggmlRuntime.backends().any { it.backend == ComputeBackend.GPU_VULKAN && it.compiled }

    /** Whether a Vulkan device is usable right now ([isRegistered] and the driver reported at least one GPU). */
    public val isAvailable: Boolean
        get() = ComputeBackend.GPU_VULKAN in KggmlRuntime.availableBackends()
}
