package com.donfuldauer.kggml.runtime

/**
 * Loads `libkggml_runtime_jni.so` from the APK's native library directory. Its `DT_NEEDED` dependencies
 * (`libkggml_runtime.so`, `libggml*.so`, `libc++_shared.so`) are packaged alongside it as jniLibs and are
 * resolved by the Android linker from the same directory.
 */
internal object AndroidNativeLibraryLoader {
    const val JNI_LIBRARY: String = "kggml_runtime_jni"

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

internal actual fun createPlatformRuntimeBridge(): NativeRuntimeBridge =
    JniRuntimeBridge(AndroidNativeLibraryLoader::load)
