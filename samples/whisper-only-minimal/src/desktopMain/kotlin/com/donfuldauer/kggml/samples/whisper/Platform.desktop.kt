package com.donfuldauer.kggml.samples.whisper

import com.donfuldauer.kggml.core.ComputeBackend

/** The desktop GPU: Metal on macOS, Vulkan on Linux / Windows (when the runtime is built with KGGML_ENABLE_VULKAN). */
actual val platformGpuBackend: ComputeBackend =
    if (System.getProperty("os.name").orEmpty().lowercase().startsWith("mac")) ComputeBackend.GPU_METAL else ComputeBackend.GPU_VULKAN
