package com.donfuldauer.kggml.runtime

import kotlin.test.Test
import kotlin.test.assertEquals

/** Discovery of the Android GPU backend add-ons packaged in an app ([registerBackendModules]). */
class BackendModuleDiscoveryTest {
    @Test
    fun theVulkanAddOnIsTheKnownModule() {
        assertEquals(listOf("ggml-vulkan"), AndroidNativeLibraryLoader.BACKEND_MODULES)
    }

    @Test
    fun packagedModulesAreLoadedAndAbsentOnesSkipped() {
        val loaded = mutableListOf<String>()
        val registered = registerBackendModules(
            modules = listOf("ggml-vulkan", "ggml-opencl"),
            findLibrary = { name -> if (name == "ggml-vulkan") "/data/app/x/lib/arm64/libggml-vulkan.so" else null },
            loadBackend = { path -> loaded += path; 0 },
            onFailure = { _, _ -> error("no failure expected") },
        )
        assertEquals(listOf("/data/app/x/lib/arm64/libggml-vulkan.so"), registered)
        assertEquals(registered, loaded, "an absent add-on is never passed to the native loader")
    }

    @Test
    fun aFailingModuleIsReportedAndSkippedSoTheRuntimeStaysOnTheCpu() {
        val failures = mutableListOf<Pair<String, Int>>()
        val registered = registerBackendModules(
            modules = listOf("ggml-vulkan", "ggml-opencl"),
            findLibrary = { name -> "/apk!/lib/arm64-v8a/lib$name.so" },
            loadBackend = { path -> if ("vulkan" in path) 3 else 0 },
            onFailure = { path, status -> failures += path to status },
        )
        assertEquals(listOf("/apk!/lib/arm64-v8a/libggml-opencl.so"), registered)
        assertEquals(listOf("/apk!/lib/arm64-v8a/libggml-vulkan.so" to 3), failures)
    }

    @Test
    fun noModulesPackagedMeansNothingToLoad() {
        val registered = registerBackendModules(
            modules = AndroidNativeLibraryLoader.BACKEND_MODULES,
            findLibrary = { null },
            loadBackend = { error("nothing to load") },
            onFailure = { _, _ -> error("no failure expected") },
        )
        assertEquals(emptyList(), registered)
    }
}
