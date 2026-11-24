plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml stable-diffusion.cpp image generation engine."

android {
    namespace = "com.donfuldauer.kggml.diffusion"
}
