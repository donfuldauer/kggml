plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml core API: InferenceEngine, BackendConfig, event and exception types (pure Kotlin, no native code)."

android {
    namespace = "com.donfuldauer.kggml.core"
}
