package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.NativeFailureException
import java.io.File
import java.lang.reflect.InvocationTargetException
import java.nio.file.Files
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * Lookup rules of [JvmWhisperLibraryLoader] without real native libraries. The "libraries" written here are plain
 * bytes, so a `System.load` of them fails; the JVM reports that as `<canonical path>: <reason>`, which shows which
 * file the loader tried first. The lookup step (`loadOnce`) is private and is reached reflectively, because the
 * public [JvmWhisperLibraryLoader.load] first loads the shared runtime, which needs the real native build.
 */
class JvmWhisperLibraryLoaderTest {
    private val loader = JvmWhisperLibraryLoader
    private val dirName = "whisper"
    private val coreLibrary = "kggml_whisper"
    private val explicitEnv = "KGGML_WHISPER_NATIVE_PATH"

    private val tempDir: File = Files.createTempDirectory("kggml-$dirName-loader-test").toFile()
    private val savedProperties = mutableMapOf<String, String?>()

    @BeforeTest
    fun isolateFromConfiguredPaths() {
        assumeTrue(System.getenv(explicitEnv) == null, "$explicitEnv is set in the environment")
        for (key in listOf(loader.PATH_PROPERTY, loader.RUNTIME_PATH_PROPERTY)) {
            savedProperties[key] = System.getProperty(key)
            System.clearProperty(key)
        }
    }

    @AfterTest
    fun restore() {
        for ((key, value) in savedProperties) if (value == null) System.clearProperty(key) else System.setProperty(key, value)
        tempDir.deleteRecursively()
    }

    private fun dir(name: String) = File(tempDir, name).apply { mkdirs() }

    private fun File.withFakeLibraries(vararg names: String) = apply {
        for (name in names) File(this, System.mapLibraryName(name)).writeText("not a real library: $name")
    }

    private fun loadOnce() {
        val method = loader.javaClass.getDeclaredMethod("loadOnce").apply { isAccessible = true }
        try {
            method.invoke(loader)
        } catch (e: InvocationTargetException) {
            throw e.targetException
        }
    }

    private fun attemptedPath(e: Throwable) = e.message!!.substringBefore(": ")

    @Test
    fun constantsNameTheEngineLibraries() {
        assertEquals("kggml.$dirName.native.path", loader.PATH_PROPERTY)
        assertEquals("kggml.native.path", loader.RUNTIME_PATH_PROPERTY)
        assertEquals("${coreLibrary}_jni", loader.JNI_LIBRARY)
        // The engine library is loaded before the JNI bridge that links against it.
        assertEquals(listOf(coreLibrary, "${coreLibrary}_jni"), loader.LOAD_ORDER)
    }

    @Test
    fun resolveDirectoryLibrariesReturnsHostLibraryFilesInLoadOrder() {
        val libs = dir("libs").withFakeLibraries("${coreLibrary}_jni", coreLibrary)
        assertEquals(
            listOf(File(libs, System.mapLibraryName(coreLibrary)), File(libs, System.mapLibraryName("${coreLibrary}_jni"))),
            loader.resolveDirectoryLibraries(libs),
        )
    }

    @Test
    fun resolveDirectoryLibrariesRequiresEveryLibrary() {
        val libs = dir("libs").withFakeLibraries(coreLibrary)
        val e = assertFailsWith<UnsatisfiedLinkError> { loader.resolveDirectoryLibraries(libs) }
        assertEquals("Required Kggml $dirName library '${System.mapLibraryName("${coreLibrary}_jni")}' not found in '$libs'", e.message)
    }

    @Test
    fun loadFromDirectoryRejectsANonDirectory() {
        val file = File(tempDir, "plain-file").apply { writeText("") }
        val e = assertFailsWith<UnsatisfiedLinkError> { loader.loadFromDirectory(file) }
        assertEquals("'$file' is not a directory", e.message)
    }

    @Test
    fun loadFromDirectoryLoadsTheEngineLibraryFirst() {
        val libs = dir("libs").withFakeLibraries(coreLibrary, "${coreLibrary}_jni")
        val e = assertFailsWith<UnsatisfiedLinkError> { loader.loadFromDirectory(libs) }
        assertEquals(File(libs, System.mapLibraryName(coreLibrary)).canonicalPath, attemptedPath(e))
    }

    @Test
    fun explicitDirectoryTakesPrecedenceOverTheRuntimeDirectory() {
        val runtimeDir = dir("runtime").withFakeLibraries(coreLibrary, "${coreLibrary}_jni")
        val explicit = File(tempDir, "missing-explicit-dir")
        System.setProperty(loader.RUNTIME_PATH_PROPERTY, runtimeDir.path)
        System.setProperty(loader.PATH_PROPERTY, explicit.path)
        val e = assertFailsWith<UnsatisfiedLinkError> { loadOnce() }
        assertEquals("'$explicit' is not a directory", e.message)
    }

    @Test
    fun explicitDirectoryIsLoadedFrom() {
        val explicit = dir("explicit").withFakeLibraries(coreLibrary, "${coreLibrary}_jni")
        System.setProperty(loader.PATH_PROPERTY, explicit.path)
        val e = assertFailsWith<UnsatisfiedLinkError> { loadOnce() }
        assertEquals(File(explicit, System.mapLibraryName(coreLibrary)).canonicalPath, attemptedPath(e))
    }

    @Test
    fun blankExplicitDirectoryFallsBackToTheRuntimeDirectory() {
        val runtimeDir = dir("runtime").withFakeLibraries(coreLibrary, "${coreLibrary}_jni")
        System.setProperty(loader.PATH_PROPERTY, "  ")
        System.setProperty(loader.RUNTIME_PATH_PROPERTY, runtimeDir.path)
        val e = assertFailsWith<UnsatisfiedLinkError> { loadOnce() }
        assertEquals(File(runtimeDir, System.mapLibraryName(coreLibrary)).canonicalPath, attemptedPath(e))
    }

    @Test
    fun runtimeDirectoryWithTheJniBridgeMustAlsoHoldTheEngineLibrary() {
        val runtimeDir = dir("runtime").withFakeLibraries("${coreLibrary}_jni")
        System.setProperty(loader.RUNTIME_PATH_PROPERTY, runtimeDir.path)
        val e = assertFailsWith<UnsatisfiedLinkError> { loadOnce() }
        assertEquals("Required Kggml $dirName library '${System.mapLibraryName(coreLibrary)}' not found in '$runtimeDir'", e.message)
    }

    @Test
    fun runtimeDirectoryWithoutTheJniBridgeIsSkippedForJavaLibraryPath() {
        // The runtime build tree holds only the shared ggml runtime: the engine is looked up on java.library.path.
        val runtimeDir = dir("runtime").withFakeLibraries(coreLibrary, "kggml_runtime_jni")
        System.setProperty(loader.RUNTIME_PATH_PROPERTY, runtimeDir.path)
        val e = assertFailsWith<UnsatisfiedLinkError> { loadOnce() }
        assertTrue(e.message!!.startsWith("no ${coreLibrary}_jni in java.library.path"), e.message)
    }

    @Test
    fun withoutAnyDirectoryTheJniBridgeIsLoadedFromJavaLibraryPath() {
        val e = assertFailsWith<UnsatisfiedLinkError> { loadOnce() }
        assertTrue(e.message!!.startsWith("no ${coreLibrary}_jni in java.library.path"), e.message)
    }

    @Test
    fun loadRecordsTheConfiguredEngineDirectoryAsOrigin() {
        val configured = savedProperties[loader.PATH_PROPERTY]
        assumeTrue(configured != null && savedProperties[loader.RUNTIME_PATH_PROPERTY] != null, "native build not configured")
        for ((key, value) in savedProperties) System.setProperty(key, value!!)
        loader.load()
        loader.load() // idempotent
        assertEquals(configured, loader.origin)
    }

    @Test
    fun loadFailsWithTheRuntimeErrorBeforeLookingForEngineLibrariesWhenTheRuntimeIsMissing() {
        assumeTrue(savedProperties[loader.RUNTIME_PATH_PROPERTY] == null, "native runtime configured")
        assumeTrue(System.getenv("KGGML_NATIVE_PATH") == null, "native runtime configured")
        val bundledPlatforms = listOf("macos-aarch64", "macos-x86_64", "linux-x86_64", "linux-aarch64", "windows-x86_64")
        assumeTrue(bundledPlatforms.none { javaClass.getResource("/kggml/native/$it/index.txt") != null }, "native runtime bundled")
        // Even with a complete engine directory configured, nothing is loaded without the shared runtime.
        System.setProperty(loader.PATH_PROPERTY, dir("explicit").withFakeLibraries(coreLibrary, "${coreLibrary}_jni").path)
        val e = assertFailsWith<NativeFailureException> { loader.load() }
        assertTrue(e.message!!.startsWith("Failed to load the Kggml native runtime"), e.message)
        assertNull(loader.origin)
        // The JVM bridge factory goes through the same loader, so engines get the same typed error.
        assertFailsWith<NativeFailureException> { createPlatformWhisperBridge() }
        assertNull(loader.origin)
    }
}
