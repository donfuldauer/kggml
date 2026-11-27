package com.donfuldauer.kggml.samples.showcase

import com.donfuldauer.kggml.samples.showcase.whisper.DesktopAudioCapture
import com.donfuldauer.kggml.samples.showcase.whisper.defaultWhisperModelPath
import com.donfuldauer.kggml.samples.showcase.whisper.readAudioFile
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import java.io.File
import java.nio.file.Files
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * The desktop platform layer that needs no microphone or window: model / Whisper defaults from system properties
 * (falling back to the environment and the home directory), local file access, and the capture's call-order check.
 */
class PlatformDesktopTest {
    private val dir: File = Files.createTempDirectory("kggml-platform").toFile()
    private val savedProperties = listOf("kggml.models.dir", "kggml.whisper.model").associateWith { System.getProperty(it) }

    @AfterTest
    fun cleanUp() {
        savedProperties.forEach { (key, value) -> if (value == null) System.clearProperty(key) else System.setProperty(key, value) }
        dir.deleteRecursively()
    }

    @Test
    fun modelDirectoryComesFromThePropertyThenTheEnvironmentThenHome() {
        System.setProperty("kggml.models.dir", "/opt/models")
        assertEquals("/opt/models", defaultModelDirectory())

        // A blank property is ignored.
        System.setProperty("kggml.models.dir", "  ")
        val fromEnvironment = System.getenv("KGGML_MODELS_DIR")?.takeIf { it.isNotBlank() }
        val home = File(System.getProperty("user.home"), ".kggml/models").absolutePath
        assertEquals(fromEnvironment ?: home, defaultModelDirectory())
    }

    @Test
    fun whisperModelComesFromThePropertyThenTheEnvironment() {
        System.setProperty("kggml.whisper.model", "/opt/ggml-tiny.en.bin")
        assertEquals("/opt/ggml-tiny.en.bin", defaultWhisperModelPath())
        System.clearProperty("kggml.whisper.model")
        assertEquals(System.getenv("KGGML_WHISPER_MODEL") ?: "", defaultWhisperModelPath())
    }

    @Test
    fun exportsGoToPicturesInTheHomeDirectory() {
        assertEquals(File(System.getProperty("user.home"), "Pictures").absolutePath, defaultExportDirectory())
    }

    @Test
    fun writingCreatesParentDirectoriesAndReadingReturnsTheSameBytes() {
        val path = File(dir, "a/b/c.png").path
        val bytes = byteArrayOf(1, 2, 3, -1)
        writeLocalFile(path, bytes)
        assertContentEquals(bytes, readLocalFile(path))
        assertContentEquals(bytes, readAudioFile(path))
    }

    @Test
    fun framesBeforePrepareFailsWithoutOpeningTheMicrophone() {
        val e = assertFailsWith<IllegalStateException> { runBlocking { DesktopAudioCapture().frames().first() } }
        assertEquals("prepare() must be called before frames()", e.message)
    }
}
