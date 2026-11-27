package com.donfuldauer.kggml.runtime

/**
 * Registers each of [modules] (ggml backend modules of the optional GPU add-ons, by library base name) that
 * [findLibrary] locates, through [loadBackend] (a `kggml_status`, 0 = OK). Absent modules are skipped; a failing one is
 * reported to [onFailure] and skipped, so the runtime keeps working on the CPU. Returns the paths that were registered.
 *
 * Shared by the Android loader (modules packaged in the app) and the JVM desktop loader (modules bundled in add-on
 * jars).
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
