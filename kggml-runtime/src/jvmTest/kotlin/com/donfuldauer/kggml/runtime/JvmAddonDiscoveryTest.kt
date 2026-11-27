package com.donfuldauer.kggml.runtime

import java.io.ByteArrayInputStream
import java.io.File
import java.nio.file.Files
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Desktop GPU add-on discovery in [JvmNativeLibraryLoader] (issue #92), without real native libraries. */
class JvmAddonDiscoveryTest {
    private val loader = JvmNativeLibraryLoader
    private val tempDir: File = Files.createTempDirectory("kggml-addon-test").toFile()
    private val vulkanFile = System.mapLibraryName("ggml-vulkan")

    @AfterTest
    fun cleanUp() {
        tempDir.deleteRecursively()
    }

    @Test
    fun vulkanIsTheDesktopAddon() {
        assertEquals(listOf("ggml-vulkan"), loader.BACKEND_MODULES)
    }

    @Test
    fun recognisesLibraryFileNamesOnEveryDesktopPlatform() {
        for (name in listOf("libggml-vulkan.so", "libggml-vulkan.so.0", "libggml-vulkan.0.dylib", "libggml-vulkan.dylib",
            "ggml-vulkan.dll")) {
            assertTrue(loader.isLibraryFile(name, "ggml-vulkan"), name)
        }
        for (name in listOf("libggml.so.0", "libggml-vulkan-shaders.so", "ggml.dll", "libggml-vulkanx.0.dylib")) {
            assertFalse(loader.isLibraryFile(name, "ggml-vulkan"), name)
        }
    }

    @Test
    fun aRuntimeWithVulkanLinkedInLoadsNoVulkanAddon() {
        assertEquals(emptyList(), loader.addonModules(listOf("libggml-base.so.0", "libggml-vulkan.so.0", "libggml.so.0")))
        assertEquals(listOf("ggml-vulkan"), loader.addonModules(listOf("libggml-base.so.0", "libggml.so.0")))
        assertEquals(listOf("ggml-vulkan"), loader.addonModules(emptyList()))
    }

    @Test
    fun anExplicitDirectoryIsTheOnlyPlaceLookedAt() {
        val resources = mutableListOf<String>()
        val open = { path: String -> resources += path; ByteArrayInputStream(byteArrayOf(1)) }
        assertNull(loader.findAddon("ggml-vulkan", tempDir, "linux-x86_64", open))
        assertEquals(emptyList(), resources, "no jar lookup when a directory is configured")
        File(tempDir, vulkanFile).writeBytes(byteArrayOf(1))
        assertEquals(File(tempDir, vulkanFile), loader.findAddon("ggml-vulkan", tempDir, "linux-x86_64", open))
    }

    @Test
    fun aBundledAddonIsExtractedFromItsPlatformDirectory() {
        val requested = mutableListOf<String>()
        val found = loader.findAddon("ggml-vulkan", null, "windows-x86_64") { path ->
            requested += path
            ByteArrayInputStream(byteArrayOf(7, 8, 9))
        }
        assertEquals(listOf("/kggml/native-addons/windows-x86_64/ggml-vulkan/$vulkanFile"), requested)
        assertEquals(vulkanFile, found!!.name)
        assertEquals(listOf<Byte>(7, 8, 9), found.readBytes().toList())
        found.parentFile.deleteRecursively()
    }

    @Test
    fun noBundledAddonMeansNone() {
        assertNull(loader.findAddon("ggml-vulkan", null, "linux-x86_64") { null })
    }

    @Test
    fun registrationReportsFailuresAndKeepsGoing() {
        val failures = mutableListOf<Pair<String, Int>>()
        val registered = registerBackendModules(
            modules = listOf("ggml-vulkan", "ggml-other"),
            findLibrary = { "/x/$it" },
            loadBackend = { if ("vulkan" in it) 3 else 0 },
            onFailure = { path, status -> failures += path to status },
        )
        assertEquals(listOf("/x/ggml-other"), registered)
        assertEquals(listOf("/x/ggml-vulkan" to 3), failures)
    }
}
