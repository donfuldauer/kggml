package com.donfuldauer.kggml.runtime

import java.io.File
import java.io.IOException
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * Locates and loads the Kggml native runtime on JVM Desktop (macOS arm64, Linux x86_64, Windows x86_64).
 *
 * Lookup order:
 * 1. **Explicit directory** — the `kggml.native.path` system property (or `KGGML_NATIVE_PATH` environment
 *    variable) naming a directory that holds the shared libraries (e.g. `build/native/lib`).
 * 2. **Bundled in the jar** — `/kggml/native/<platform>/index.txt` lists the bundled file names in load order;
 *    they are extracted together into a private temporary directory (so `$ORIGIN` / `@loader_path` sibling
 *    lookups keep working) and loaded from there.
 * 3. **`java.library.path`** — `System.loadLibrary("kggml_runtime_jni")`.
 *
 * Libraries are loaded explicitly in dependency order (ggml first, the JNI bridge last), so each library's
 * dependencies are already resident when it is loaded, independent of RPATH.
 *
 * Then the GPU backend modules of the optional desktop add-ons ([BACKEND_MODULES]) are registered with the runtime
 * before it is initialized: from the `kggml.native.addons.path` directory (or `KGGML_NATIVE_ADDONS_PATH`), else from
 * `/kggml/native-addons/<platform>/<module>/` in an add-on jar (`kggml-runtime-vulkan`). A module that is absent or
 * fails to load (no Vulkan loader or driver) leaves the runtime on the CPU.
 */
internal object JvmNativeLibraryLoader {
    const val PATH_PROPERTY: String = "kggml.native.path"
    const val PATH_ENV: String = "KGGML_NATIVE_PATH"
    const val RESOURCE_ROOT: String = "/kggml/native"
    const val INDEX_FILE: String = "index.txt"
    const val JNI_LIBRARY: String = "kggml_runtime_jni"
    const val ADDONS_PATH_PROPERTY: String = "kggml.native.addons.path"
    const val ADDONS_PATH_ENV: String = "KGGML_NATIVE_ADDONS_PATH"
    const val ADDONS_RESOURCE_ROOT: String = "/kggml/native-addons"

    /**
     * ggml backend modules shipped by the optional desktop add-on jars, by library base name: `kggml-runtime-vulkan`
     * carries `ggml-vulkan` for Linux and Windows (macOS uses Metal, linked into the runtime).
     */
    val BACKEND_MODULES: List<String> = listOf("ggml-vulkan")

    /** Library base names in dependency (load) order, and whether each is required. */
    val LOAD_ORDER: List<Pair<String, Boolean>> = listOf(
        "ggml-base" to true,
        "ggml-cpu" to false,
        "ggml-blas" to false,
        "ggml-metal" to false,
        "ggml-vulkan" to false,
        "ggml" to true,
        "kggml_runtime" to true,
        JNI_LIBRARY to true,
    )

    @Volatile
    private var loadedFrom: String? = null

    /** Human-readable origin of the loaded libraries (directory, `jar:<platform>` or `java.library.path`). */
    val origin: String?
        get() = loadedFrom

    fun load() {
        if (loadedFrom != null) return
        synchronized(this) {
            if (loadedFrom != null) return
            loadedFrom = loadOnce()
        }
    }

    private fun loadOnce(): String {
        val (origin, loadedFiles) = loadRuntime()
        registerAddons(loadedFiles)
        return origin
    }

    /** Loads the runtime; returns its origin and the file names loaded (empty for `java.library.path`). */
    private fun loadRuntime(): Pair<String, List<String>> {
        val explicit = System.getProperty(PATH_PROPERTY)?.takeIf { it.isNotBlank() }
            ?: System.getenv(PATH_ENV)?.takeIf { it.isNotBlank() }
        if (explicit != null) {
            return explicit to loadFromDirectory(File(explicit))
        }
        val platform = platformId()
        val index = javaClass.getResource("$RESOURCE_ROOT/$platform/$INDEX_FILE")
        if (index != null) {
            val names = parseIndex(index.readText())
            loadFromResources(platform, names)
            return "jar:$platform" to names
        }
        System.loadLibrary(JNI_LIBRARY)
        return "java.library.path" to emptyList()
    }

    /** Loads every library of [LOAD_ORDER] present in [dir], failing if a required one is missing. */
    fun loadFromDirectory(dir: File): List<String> {
        if (!dir.isDirectory) {
            throw UnsatisfiedLinkError("$PATH_PROPERTY points to '$dir', which is not a directory")
        }
        val files = resolveDirectoryLibraries(dir)
        for (file in files) System.load(file.absolutePath)
        return files.map { it.name }
    }

    // ---- GPU add-ons ------------------------------------------------------------------------------------------

    /** Backend modules that are added on: those not already linked into the runtime (`KGGML_ENABLE_VULKAN` builds). */
    fun addonModules(runtimeFiles: List<String>): List<String> =
        BACKEND_MODULES.filter { module -> runtimeFiles.none { isLibraryFile(it, module) } }

    /** Whether [fileName] is the shared library `name` (`libname.so[.N]`, `libname[.N].dylib`, `name.dll`). */
    fun isLibraryFile(fileName: String, name: String): Boolean =
        fileName == System.mapLibraryName(name) || fileName == "$name.dll" ||
            fileName.startsWith("lib$name.so") || Regex("""^lib${Regex.escape(name)}(\.\d+)*\.dylib$""").matches(fileName)

    private fun registerAddons(runtimeFiles: List<String>) {
        val dir = System.getProperty(ADDONS_PATH_PROPERTY)?.takeIf { it.isNotBlank() }
            ?: System.getenv(ADDONS_PATH_ENV)?.takeIf { it.isNotBlank() }
        val platform = platformId()
        registerBackendModules(
            modules = addonModules(runtimeFiles),
            findLibrary = { module ->
                findAddon(module, dir?.let(::File), platform, javaClass::getResourceAsStream)?.absolutePath
            },
            loadBackend = KggmlRuntimeJni::nativeLoadBackend,
            onFailure = { path, status ->
                System.getLogger("com.donfuldauer.kggml").log(
                    System.Logger.Level.WARNING,
                    "GPU backend add-on $path not loaded (${KggmlRuntimeJni.nativeStatusName(status)}: " +
                        "${KggmlRuntimeJni.nativeLastError()}); using the CPU",
                )
            },
        )
    }

    /**
     * Locates the add-on library of [module]: in [explicitDir] when set (only there), else bundled at
     * `/kggml/native-addons/<platform>/<module>/<library file>` ([openResource]) and extracted to a private temporary
     * directory. `null` when the add-on is not present.
     */
    fun findAddon(module: String, explicitDir: File?, platform: String, openResource: (String) -> InputStream?): File? {
        val fileName = System.mapLibraryName(module)
        if (explicitDir != null) {
            return File(explicitDir, fileName).takeIf { it.isFile }
        }
        val stream = openResource("$ADDONS_RESOURCE_ROOT/$platform/$module/$fileName") ?: return null
        val dir = Files.createTempDirectory("kggml-addon-").toFile().apply { deleteOnExit() }
        val target = File(dir, fileName)
        try {
            stream.use { Files.copy(it, target.toPath(), StandardCopyOption.REPLACE_EXISTING) }
        } catch (e: IOException) {
            throw UnsatisfiedLinkError("Failed to extract '$fileName': ${e.message}").apply { initCause(e) }
        }
        target.deleteOnExit()
        return target
    }

    /** The files [loadFromDirectory] would load from [dir], in load order. */
    fun resolveDirectoryLibraries(dir: File): List<File> = LOAD_ORDER.mapNotNull { (name, required) ->
        val file = File(dir, System.mapLibraryName(name))
        when {
            file.isFile -> file
            required -> throw UnsatisfiedLinkError("Required Kggml native library '${file.name}' not found in '$dir'")
            else -> null
        }
    }

    private fun loadFromResources(platform: String, fileNames: List<String>) {
        if (fileNames.isEmpty()) throw UnsatisfiedLinkError("Empty Kggml native index for platform '$platform'")
        val dir = Files.createTempDirectory("kggml-native-").toFile().apply { deleteOnExit() }
        val extracted = fileNames.map { name ->
            val target = File(dir, name)
            val stream = javaClass.getResourceAsStream("$RESOURCE_ROOT/$platform/$name")
                ?: throw UnsatisfiedLinkError("Bundled Kggml native library '$name' missing for '$platform'")
            try {
                stream.use { Files.copy(it, target.toPath(), StandardCopyOption.REPLACE_EXISTING) }
            } catch (e: IOException) {
                throw UnsatisfiedLinkError("Failed to extract '$name': ${e.message}").apply { initCause(e) }
            }
            target.deleteOnExit()
            target
        }
        for (file in extracted) System.load(file.absolutePath)
    }

    /** Parses `index.txt`: one file name per line; blank lines and `#` comments are ignored. */
    fun parseIndex(text: String): List<String> = text.lineSequence()
        .map { it.trim() }
        .filter { it.isNotEmpty() && !it.startsWith("#") }
        .onEach { require('/' !in it && '\\' !in it && it != "..") { "Invalid native index entry '$it'" } }
        .toList()

    /** Platform directory name, e.g. `linux-x86_64`, `macos-aarch64`, `windows-x86_64`. */
    fun platformId(
        osName: String = System.getProperty("os.name").orEmpty(),
        osArch: String = System.getProperty("os.arch").orEmpty(),
    ): String {
        val os = osName.lowercase()
        val osPart = when {
            os.startsWith("mac") || os.startsWith("darwin") -> "macos"
            os.startsWith("windows") -> "windows"
            os.startsWith("linux") -> "linux"
            else -> os.replace(' ', '_')
        }
        val archPart = when (osArch.lowercase()) {
            "amd64", "x86_64", "x64" -> "x86_64"
            "aarch64", "arm64" -> "aarch64"
            else -> osArch.lowercase()
        }
        return "$osPart-$archPart"
    }
}

internal actual fun createPlatformRuntimeBridge(): NativeRuntimeBridge =
    JniRuntimeBridge(JvmNativeLibraryLoader::load)
