package com.donfuldauer.kggml.runtime

import android.util.Log
import dalvik.system.BaseDexClassLoader

/**
 * Loads `libkggml_runtime_jni.so` from the APK's native library directory. Its `DT_NEEDED` dependencies
 * (`libkggml_runtime.so`, `libggml*.so`, `libc++_shared.so`) are packaged alongside it as jniLibs and are
 * resolved by the Android linker from the same directory.
 *
 * Then registers the GPU backend add-ons packaged in the app (see [BACKEND_MODULES]), before the runtime is
 * initialized, so every engine sees their devices. An add-on that is absent, or fails to load, leaves the runtime on
 * the CPU.
 */
internal object AndroidNativeLibraryLoader {
    const val JNI_LIBRARY: String = "kggml_runtime_jni"

    /**
     * ggml backend modules shipped by the optional Android add-on artifacts, by library name (`lib<name>.so`):
     * `kggml-runtime-vulkan` packages `libggml-vulkan.so`.
     */
    val BACKEND_MODULES: List<String> = listOf("ggml-vulkan")

    private const val TAG = "kggml"

    @Volatile
    private var loaded = false

    fun load() {
        if (loaded) return
        synchronized(this) {
            if (loaded) return
            System.loadLibrary(JNI_LIBRARY)
            registerBackendModules(
                modules = BACKEND_MODULES,
                findLibrary = ::findPackagedLibrary,
                loadBackend = KggmlRuntimeJni::nativeLoadBackend,
                onFailure = { path, status ->
                    Log.w(TAG, "GPU backend add-on $path not loaded (${KggmlRuntimeJni.nativeStatusName(status)}: " +
                        "${KggmlRuntimeJni.nativeLastError()}); using the CPU")
                },
            )
            loaded = true
        }
    }

    /** Full path of `lib<name>.so` packaged in this app (also inside an uncompressed APK), or `null`. */
    private fun findPackagedLibrary(name: String): String? =
        (AndroidNativeLibraryLoader::class.java.classLoader as? BaseDexClassLoader)?.findLibrary(name)
}

/**
 * Registers each of [modules] that is packaged in the app ([findLibrary] returns its path) through [loadBackend]
 * (a `kggml_status`, 0 = OK). Absent modules are skipped; a failing one is reported to [onFailure] and skipped, so the
 * runtime keeps working on the CPU. Returns the paths that were registered.
 */
internal fun registerBackendModules(
    modules: List<String>,
    findLibrary: (String) -> String?,
    loadBackend: (String) -> Int,
    onFailure: (path: String, status: Int) -> Unit,
): List<String> = modules.mapNotNull { name ->
    val path = findLibrary(name) ?: return@mapNotNull null
    val status = loadBackend(path)
    if (status == 0) path else null.also { onFailure(path, status) }
}

internal actual fun createPlatformRuntimeBridge(): NativeRuntimeBridge =
    JniRuntimeBridge(AndroidNativeLibraryLoader::load)
