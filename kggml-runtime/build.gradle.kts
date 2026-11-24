plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml shared dynamic ggml runtime (CPU, Metal, Vulkan backends)."

android {
    namespace = "com.donfuldauer.kggml.runtime"
}
