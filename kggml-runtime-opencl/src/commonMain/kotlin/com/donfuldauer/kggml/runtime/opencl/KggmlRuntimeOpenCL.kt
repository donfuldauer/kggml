package com.donfuldauer.kggml.runtime.opencl

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.runtime.KggmlRuntime

/**
 * The Android OpenCL add-on: depending on `com.donfuldauer:kggml-runtime-opencl` packages ggml's OpenCL backend
 * (`libggml-opencl.so`, with the kernels tuned for Qualcomm Adreno) into the app and declares the vendor's
 * `libOpenCL.so` in its manifest. `:kggml-runtime` registers the backend when it loads, before any engine runs, so
 * [ComputeBackend.GPU_OPENCL] and [ComputeBackend.AUTO] (which prefers it over Vulkan) can use the GPU. Devices without
 * OpenCL (e.g. Pixel phones, emulators) cannot load the module: the engines keep running on Vulkan (with that add-on)
 * or the CPU, and logcat (`kggml`) says why.
 *
 * On JVM desktop and iOS the add-on adds nothing.
 */
public object KggmlRuntimeOpenCL {
    /** Whether the runtime has an OpenCL backend registered (on Android: the add-on was found and the device has OpenCL). */
    public val isRegistered: Boolean
        get() = KggmlRuntime.backends().any { it.backend == ComputeBackend.GPU_OPENCL && it.compiled }

    /** Whether an OpenCL device is usable right now ([isRegistered] and the driver reported at least one GPU). */
    public val isAvailable: Boolean
        get() = ComputeBackend.GPU_OPENCL in KggmlRuntime.availableBackends()
}
