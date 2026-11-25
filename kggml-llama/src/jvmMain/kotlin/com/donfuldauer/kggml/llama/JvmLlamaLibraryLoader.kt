package com.donfuldauer.kggml.llama

import com.donfuldauer.kggml.runtime.KggmlRuntime
import java.io.File

/**
 * Loads the Kggml llama native libraries on JVM Desktop.
 *
 * The shared ggml runtime is loaded first through [KggmlRuntime] (see its loader for lookup rules), so that
 * `libkggml_llama`'s dynamic dependencies are already resident. Then, in order:
 *
 * 1. **Explicit directory** — the `kggml.llama.native.path` system property (or `KGGML_LLAMA_NATIVE_PATH`
 *    environment variable) naming a directory with `kggml_llama` and `kggml_llama_jni`
 *    (e.g. `build/native-llama/lib`).
 * 2. **Runtime directory** — the `kggml.native.path` directory, when it also contains the llama libraries.
 * 3. **`java.library.path`** — `System.loadLibrary("kggml_llama_jni")`.
 */
internal object JvmLlamaLibraryLoader {
    const val PATH_PROPERTY: String = "kggml.llama.native.path"
    const val PATH_ENV: String = "KGGML_LLAMA_NATIVE_PATH"
    const val RUNTIME_PATH_PROPERTY: String = "kggml.native.path"
    const val JNI_LIBRARY: String = "kggml_llama_jni"

    /** Library base names in load order. */
    val LOAD_ORDER: List<String> = listOf("kggml_llama", JNI_LIBRARY)

    @Volatile
    private var loadedFrom: String? = null

    /** Human-readable origin of the loaded libraries. */
    val origin: String?
        get() = loadedFrom

    fun load() {
        if (loadedFrom != null) return
        synchronized(this) {
            if (loadedFrom != null) return
            KggmlRuntime.info() // loads the shared ggml runtime (throws NativeFailureException if it cannot)
            loadedFrom = loadOnce()
        }
    }

    private fun loadOnce(): String {
        val explicit = System.getProperty(PATH_PROPERTY)?.takeIf { it.isNotBlank() }
            ?: System.getenv(PATH_ENV)?.takeIf { it.isNotBlank() }
        if (explicit != null) {
            loadFromDirectory(File(explicit))
            return explicit
        }
        val runtimeDir = System.getProperty(RUNTIME_PATH_PROPERTY)?.takeIf { it.isNotBlank() }?.let(::File)
        if (runtimeDir != null && File(runtimeDir, System.mapLibraryName(JNI_LIBRARY)).isFile) {
            loadFromDirectory(runtimeDir)
            return runtimeDir.path
        }
        System.loadLibrary(JNI_LIBRARY)
        return "java.library.path"
    }

    /** Loads every library of [LOAD_ORDER] from [dir], failing if one is missing. */
    fun loadFromDirectory(dir: File) {
        if (!dir.isDirectory) throw UnsatisfiedLinkError("'$dir' is not a directory")
        for (file in resolveDirectoryLibraries(dir)) System.load(file.absolutePath)
    }

    fun resolveDirectoryLibraries(dir: File): List<File> = LOAD_ORDER.map { name ->
        File(dir, System.mapLibraryName(name)).also {
            if (!it.isFile) throw UnsatisfiedLinkError("Required Kggml llama library '${it.name}' not found in '$dir'")
        }
    }
}

internal actual fun createPlatformLlamaBridge(): NativeLlamaBridge = JniLlamaBridge(JvmLlamaLibraryLoader::load)
