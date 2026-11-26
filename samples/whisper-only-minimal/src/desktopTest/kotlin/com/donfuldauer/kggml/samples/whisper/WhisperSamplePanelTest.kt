package com.donfuldauer.kggml.samples.whisper

import com.donfuldauer.kggml.whisper.TinyWhisperModel
import java.awt.Component
import java.awt.Container
import java.io.File
import java.nio.file.Files
import javax.swing.AbstractButton
import javax.swing.SwingUtilities
import kotlin.math.PI
import kotlin.math.sin
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The Swing desktop screen of the minimal sample, driven headlessly on the event dispatch thread: model picker over
 * the folder's `.bin` models plus a custom path, rescan, and a real transcription (the synthetic tiny model). Like the
 * Android MainActivity tests, the transcription asserts the right outcome whether or not the native libraries load
 * (required under -Pkggml.requireNative=true).
 */
class WhisperSamplePanelTest {
    private val folder: File = Files.createTempDirectory("kggml-minimal-desktop").toFile()
    private var panel: WhisperSamplePanel? = null
    private val names = WhisperSamplePanel.Names

    private val nativeLoads: Boolean by lazy {
        val failure = nativeWhisperLoadFailure()
        if (failure != null && NativeTestEnvironment.requireNative) {
            throw AssertionError("native runtime / whisper are required but did not load: $failure", failure)
        }
        failure == null
    }

    @AfterTest
    fun tearDown() {
        panel?.let { p -> onEdt { p.close() } }
        folder.deleteRecursively()
    }

    private fun <T> onEdt(block: () -> T): T {
        var result: Result<T>? = null
        SwingUtilities.invokeAndWait { result = runCatching(block) }
        return result!!.getOrThrow()
    }

    private fun open(model: String? = null, audio: String? = null): WhisperSamplePanel =
        onEdt { WhisperSamplePanel(modelFolder = folder.path, requestedModel = model, requestedAudio = audio) }.also { panel = it }

    private fun file(name: String, bytes: ByteArray = byteArrayOf(0)) = File(folder, name).apply { writeBytes(bytes) }

    private fun Container.find(name: String): Component? {
        components.forEach { c ->
            if (c.name == name) return c
            if (c is Container) c.find(name)?.let { return it }
        }
        return null
    }

    private fun WhisperSamplePanel.button(name: String) = onEdt { find(name) as AbstractButton? }
    private fun WhisperSamplePanel.selected(name: String) = onEdt { (find(name) as AbstractButton).isSelected }
    private fun WhisperSamplePanel.click(name: String) = onEdt { (find(name) as AbstractButton).doClick() }
    private fun WhisperSamplePanel.path() = onEdt { modelPath.text }

    private fun awaitText(timeoutMs: Long = 120_000, read: () -> String, done: (String) -> Boolean): String {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (true) {
            val text = onEdt(read)
            if (done(text)) return text
            check(System.currentTimeMillis() < deadline) { "timed out; last text: $text" }
            Thread.sleep(50)
        }
    }

    @Test
    fun theRuntimeStatusIsShown() {
        val p = open()
        val status = awaitText(read = { p.status.text }) { !it.startsWith("Loading") }
        if (nativeLoads) {
            assertTrue(status.startsWith("Kggml whisper-only minimal sample\nRuntime "), status)
        } else {
            assertTrue(status.startsWith("Kggml whisper-only minimal sample\nNative runtime unavailable: "), status)
        }
    }

    @Test
    fun anEmptyFolderShowsTheHintAndKeepsTheDefaultPathAsACustomPath() {
        val p = open()
        assertTrue(onEdt { p.noModels.isVisible })
        assertEquals(noModelsHint(folder.path), onEdt { p.noModels.text })
        assertTrue(p.selected(names.MODEL_CUSTOM))
        assertEquals(File(folder, "ggml-tiny.en.bin").path, p.path())
    }

    @Test
    fun folderModelsAreListedAndTheDefaultIsPreselected() {
        file("ggml-tiny.en.bin"); file("ggml-base.en.bin"); file("jfk.wav")
        val p = open()
        assertFalse(onEdt { p.noModels.isVisible })
        assertTrue(p.selected(names.modelChoice("ggml-tiny.en.bin")))
        assertFalse(p.selected(names.modelChoice("ggml-base.en.bin")))
        assertNull(p.button(names.modelChoice("jfk.wav")), "audio is not offered as a model")
        assertFalse(p.selected(names.MODEL_CUSTOM))
    }

    @Test
    fun withoutTheDefaultModelTheFirstFolderModelIsPreselected() {
        file("ggml-small.bin"); file("ggml-base.en.bin")
        val p = open()
        assertTrue(p.selected(names.modelChoice("ggml-base.en.bin")))
        assertEquals(File(folder, "ggml-base.en.bin").path, p.path())
    }

    @Test
    fun choosingTypingAndCustomPathKeepThePickerAndTheFieldInSync() {
        file("ggml-base.en.bin"); file("ggml-small.bin")
        val p = open(model = "/elsewhere/other.bin")
        assertTrue(p.selected(names.MODEL_CUSTOM), "a requested path outside the folder is a custom path")

        p.click(names.modelChoice("ggml-small.bin"))
        assertEquals(File(folder, "ggml-small.bin").path, p.path())
        assertTrue(p.selected(names.modelChoice("ggml-small.bin")))

        onEdt { p.modelPath.text = "/sdcard/mine.bin" }
        assertTrue(p.selected(names.MODEL_CUSTOM))
        onEdt { p.modelPath.text = File(folder, "ggml-base.en.bin").path }
        assertTrue(p.selected(names.modelChoice("ggml-base.en.bin")))

        // "Custom path" from a folder model starts from an empty field; on a custom path it keeps the text.
        p.click(names.MODEL_CUSTOM)
        assertEquals("", p.path())
        onEdt { p.modelPath.text = "/sdcard/mine.bin" }
        p.click(names.MODEL_CUSTOM)
        assertEquals("/sdcard/mine.bin", p.path())
    }

    @Test
    fun rescanPicksUpModelsCopiedInLater() {
        val p = open()
        assertNull(p.button(names.modelChoice("ggml-tiny.en.bin")))
        file("ggml-tiny.en.bin")
        p.click(names.RESCAN)
        assertTrue(p.selected(names.modelChoice("ggml-tiny.en.bin")), "the default path's model is now the selection")
        assertFalse(onEdt { p.noModels.isVisible })
    }

    @Test
    fun aChosenFolderModelIsTranscribedOnTheCpu() {
        file("ggml-tiny-synthetic.bin", TinyWhisperModel.bytes())
        val audio = File(folder, "tone.wav").apply { writeBytes(toneWav(seconds = 1)) }
        val p = open(model = "/nonexistent/kggml/model.bin", audio = audio.path)
        p.click(names.modelChoice("ggml-tiny-synthetic.bin"))
        p.click(names.backend(BackendChoice.CPU_ONLY))
        p.click(names.TRANSCRIBE)
        val report = awaitText(read = { p.output.text }) { !it.startsWith("Transcribing") }
        if (nativeLoads) {
            // The real engine loaded the folder model (the requested path does not exist) and ran the 1 s clip.
            assertTrue(report.startsWith("Requested CPU_ONLY, ran on CPU_ONLY (use_gpu=false)\nAudio 1000 ms, "), report)
        } else {
            assertTrue(report.startsWith("Failed: NativeFailureException: "), report)
        }
        assertTrue(onEdt { p.transcribeButton.isEnabled }, "ready for another run")
    }

    @Test
    fun aMissingAudioFileFailsVisiblyInsteadOfCrashing() {
        val p = open(audio = "/nonexistent/kggml/audio.wav")
        p.click(names.TRANSCRIBE)
        val report = awaitText(read = { p.output.text }) { !it.startsWith("Transcribing") }
        // The message is the OS's own (Windows: "\\nonexistent\\kggml\\audio.wav (The system cannot find the path...)").
        assertTrue(report.startsWith("Failed: FileNotFoundException: "), report)
        assertTrue("audio.wav" in report, report)
    }

    /** A 330 Hz tone as a canonical 16 kHz mono 16-bit WAV. */
    private fun toneWav(seconds: Int): ByteArray {
        val samples = seconds * 16_000
        val out = ByteArray(44 + samples * 2)
        fun ascii(at: Int, s: String) = s.forEachIndexed { i, c -> out[at + i] = c.code.toByte() }
        fun le16(at: Int, v: Int) { out[at] = v.toByte(); out[at + 1] = (v shr 8).toByte() }
        fun le32(at: Int, v: Int) { le16(at, v and 0xFFFF); le16(at + 2, v ushr 16) }
        ascii(0, "RIFF"); le32(4, 36 + samples * 2); ascii(8, "WAVE")
        ascii(12, "fmt "); le32(16, 16); le16(20, 1); le16(22, 1); le32(24, 16_000); le32(28, 32_000); le16(32, 2); le16(34, 16)
        ascii(36, "data"); le32(40, samples * 2)
        for (i in 0 until samples) le16(44 + i * 2, (sin(2 * PI * 330 * i / 16_000) * 6_000).toInt())
        return out
    }
}
