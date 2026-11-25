package com.donfuldauer.kggml.llama

/**
 * Loads `libkggml_llama_jni.so` from the APK's native library directory. Its `DT_NEEDED` dependencies
 * (`libkggml_llama.so`, `libkggml_runtime.so`, `libggml*.so`, `libc++_shared.so`) are packaged alongside it as
 * jniLibs and resolved by the Android linker from the same directory.
 */
internal object AndroidLlamaLibraryLoader {
    const val JNI_LIBRARY: String = "kggml_llama_jni"

    @Volatile
    private var loaded = false

    fun load() {
        if (loaded) return
        synchronized(this) {
            if (loaded) return
            System.loadLibrary(JNI_LIBRARY)
            loaded = true
        }
    }
}

internal actual fun createPlatformLlamaBridge(): NativeLlamaBridge =
    JniLlamaBridge(AndroidLlamaLibraryLoader::load)
