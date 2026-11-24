package com.donfuldauer.kggml.runtime

import java.io.File
import java.nio.file.Files
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class JvmNativeLibraryLoaderTest {
    private val tempDir: File = Files.createTempDirectory("kggml-loader-test").toFile()

    @AfterTest
    fun cleanUp() {
        tempDir.deleteRecursively()
    }

    private fun touch(name: String) = File(tempDir, System.mapLibraryName(name)).apply { writeText("") }

    @Test
    fun platformIdNormalizesOsAndArch() {
        assertEquals("linux-x86_64", JvmNativeLibraryLoader.platformId("Linux", "amd64"))
        assertEquals("linux-aarch64", JvmNativeLibraryLoader.platformId("Linux", "aarch64"))
        assertEquals("macos-aarch64", JvmNativeLibraryLoader.platformId("Mac OS X", "aarch64"))
        assertEquals("macos-aarch64", JvmNativeLibraryLoader.platformId("Darwin", "arm64"))
        assertEquals("windows-x86_64", JvmNativeLibraryLoader.platformId("Windows 11", "amd64"))
        assertEquals("windows-x86_64", JvmNativeLibraryLoader.platformId("Windows Server 2022", "x86_64"))
    }

    @Test
    fun indexParsingSkipsBlanksAndComments() {
        val parsed = JvmNativeLibraryLoader.parseIndex(
            "# comment\n\nlibggml-base.so.0\n  libggml.so.0  \r\nlibkggml_runtime_jni.so\n",
        )
        assertEquals(listOf("libggml-base.so.0", "libggml.so.0", "libkggml_runtime_jni.so"), parsed)
    }

    @Test
    fun indexRejectsPathTraversal() {
        assertFailsWith<IllegalArgumentException> { JvmNativeLibraryLoader.parseIndex("../evil.so") }
        assertFailsWith<IllegalArgumentException> { JvmNativeLibraryLoader.parseIndex("sub\\evil.dll") }
        assertFailsWith<IllegalArgumentException> { JvmNativeLibraryLoader.parseIndex("..") }
    }

    @Test
    fun directoryResolutionFollowsDependencyOrderAndSkipsOptionalLibraries() {
        listOf("kggml_runtime_jni", "ggml", "kggml_runtime", "ggml-base", "ggml-cpu").forEach(::touch)
        val resolved = JvmNativeLibraryLoader.resolveDirectoryLibraries(tempDir).map { it.name }
        assertEquals(
            listOf("ggml-base", "ggml-cpu", "ggml", "kggml_runtime", "kggml_runtime_jni").map(System::mapLibraryName),
            resolved,
        )
    }

    @Test
    fun missingRequiredLibraryFailsWithUnsatisfiedLinkError() {
        listOf("ggml-base", "ggml", "kggml_runtime").forEach(::touch)
        val e = assertFailsWith<UnsatisfiedLinkError> { JvmNativeLibraryLoader.resolveDirectoryLibraries(tempDir) }
        assertTrue(e.message!!.contains(System.mapLibraryName("kggml_runtime_jni")), e.message)
    }

    @Test
    fun nonDirectoryPathFailsWithUnsatisfiedLinkError() {
        assertFailsWith<UnsatisfiedLinkError> { JvmNativeLibraryLoader.loadFromDirectory(File(tempDir, "missing")) }
    }

    @Test
    fun loadOrderEndsWithTheJniBridge() {
        assertEquals(JvmNativeLibraryLoader.JNI_LIBRARY, JvmNativeLibraryLoader.LOAD_ORDER.last().first)
        assertTrue(JvmNativeLibraryLoader.LOAD_ORDER.filter { it.second }.map { it.first }
            .containsAll(listOf("ggml-base", "ggml", "kggml_runtime", "kggml_runtime_jni")))
    }
}
