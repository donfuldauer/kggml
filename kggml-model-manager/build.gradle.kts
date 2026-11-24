plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml optional model manager: resumable downloads, SHA-256 verification, RAM estimation."

android {
    namespace = "com.donfuldauer.kggml.models"
}
