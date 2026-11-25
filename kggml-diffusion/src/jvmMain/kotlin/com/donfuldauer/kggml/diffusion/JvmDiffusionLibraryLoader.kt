package com.donfuldauer.kggml.diffusion

import com.donfuldauer.kggml.runtime.KggmlRuntime
import java.io.File

/**
 * Loads the Kggml diffusion native libraries on JVM Desktop.
 *
 * The shared ggml runtime is loaded first through [KggmlRuntime] (see its loader for lookup rules), so that
 * `libkggml_diffusion`'s dynamic dependencies are already resident. Then, in order:
 *
 * 1. **Explicit directory** — the `kggml.diffusion.native.path` system property (or `KGGML_DIFFUSION_NATIVE_PATH`
 *    environment variable) naming a directory with `kggml_diffusion` and `kggml_diffusion_jni`
 *    (e.g. `build/native-diffusion/lib`).
 * 2. **Runtime directory** — the `kggml.native.path` directory, when it also contains the diffusion libraries.
 * 3. **`java.library.path`** — `System.loadLibrary("kggml_diffusion_jni")`.
 */
internal object JvmDiffusionLibraryLoader {
    const val PATH_PROPERTY: String = "kggml.diffusion.native.path"
    const val PATH_ENV: String = "KGGML_DIFFUSION_NATIVE_PATH"
    const val RUNTIME_PATH_PROPERTY: String = "kggml.native.path"
    const val JNI_LIBRARY: String = "kggml_diffusion_jni"

    /** Library base names in load order. */
    val LOAD_ORDER: List<String> = listOf("kggml_diffusion", JNI_LIBRARY)

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
            if (!it.isFile) throw UnsatisfiedLinkError("Required Kggml diffusion library '${it.name}' not found in '$dir'")
        }
    }
}

internal actual fun createPlatformDiffusionBridge(): NativeDiffusionBridge = JniDiffusionBridge(JvmDiffusionLibraryLoader::load)
