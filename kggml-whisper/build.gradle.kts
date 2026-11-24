plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml whisper.cpp speech-to-text engine."

android {
    namespace = "com.donfuldauer.kggml.whisper"
}
