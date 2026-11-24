plugins {
    alias(libs.plugins.kggml.kmp.library)
}

description = "Kggml llama.cpp + libmtmd text and vision-language engine."

android {
    namespace = "com.donfuldauer.kggml.llama"
}
