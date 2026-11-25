package com.donfuldauer.kggml.samples.showcase.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.TranscriptionEvent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

@OptIn(ExperimentalCoroutinesApi::class)
class WhisperStudioControllerTest {
    private class Harness(
        scope: TestScope,
        val factory: FakeTranscriberFactory = FakeTranscriberFactory(),
        val capture: FakeAudioCapture? = FakeAudioCapture(),
        files: Map<String, ByteArray> = emptyMap(),
        modelPath: String = "/models/ggml-tiny.en.bin",
        backgroundStressSupported: Boolean = false,
    ) {
        val controller = WhisperStudioController(
            // Not backgroundScope: advanceUntilIdle() only drives foreground tasks. Work left running at the end of
            // a test (an open stream) is simply abandoned with this unparented scope.
            scope = CoroutineScope(StandardTestDispatcher(scope.testScheduler) + Job()),
            factory = factory,
            capture = capture,
            readFile = { path -> files[path] ?: throw IllegalArgumentException("No such file: $path") },
            ioDispatcher = StandardTestDispatcher(scope.testScheduler),
            initialModelPath = modelPath,
            backgroundStressSupported = backgroundStressSupported,
            maxLevels = 4,
        )
        val state get() = controller.state.value
        val transcriber get() = factory.opened.last()
    }

    @Test
    fun initialStateReflectsMicrophoneAndModelPath() = runTest {
        val h = Harness(this, modelPath = "")
        assertTrue(h.state.micAvailable)
        assertEquals(StudioActivity.IDLE, h.state.activity)
        assertFalse(h.state.canStartLive, "no model path yet")
        h.controller.setModelPath("/m.bin")
        assertTrue(h.state.canStartLive)
        assertFalse(h.state.canTranscribeFile, "no file path yet")
        h.controller.setFilePath("/a.wav")
        assertTrue(h.state.canTranscribeFile)
        assertEquals(ComputeBackend.AUTO, h.state.selectedBackend)
        assertNull(h.state.activeBackend)
    }

    @Test
    fun withoutMicrophoneLiveDictationIsDisabled() = runTest {
        val h = Harness(this, capture = null)
        assertFalse(h.state.micAvailable)
        assertFalse(h.state.canStartLive)
        h.controller.startLive()
        advanceUntilIdle()
        assertTrue(h.factory.opened.isEmpty())
    }

    @Test
    fun liveDictationStreamsPartialsThenCommitsOnStop() = runTest {
        val h = Harness(this)
        val mic = h.capture!!
        h.controller.selectBackend(ComputeBackend.GPU_METAL)
        h.controller.startLive()
        advanceUntilIdle()

        assertEquals(StudioActivity.LISTENING, h.state.activity)
        assertEquals(BackendConfig(backend = ComputeBackend.GPU_METAL), h.transcriber.config)
        assertEquals(ComputeBackend.GPU_METAL, h.state.activeBackend)
        assertTrue(h.state.usesGpu)
        assertEquals(CaptureFormat(48_000, 1), h.state.captureFormat)
        assertEquals(TranscriptSource.LIVE, h.state.source)
        assertTrue(h.state.canStop)

        mic.sendMs(100, amplitude = 0.5f)
        mic.sendMs(100, amplitude = 0.5f)
        advanceUntilIdle()
        assertEquals(200, h.state.capturedMs)
        assertEquals(listOf(0.5f, 0.5f), h.state.levels)
        val partial = assertNotNull(h.state.partial)
        assertEquals("word1 word2", partial.text)
        assertEquals(200, partial.endMs)
        assertEquals(0, partial.lagMs)
        assertTrue(h.state.committed.isEmpty())

        repeat(4) { mic.sendMs(100, amplitude = 0.1f) }
        advanceUntilIdle()
        assertEquals(4, h.state.levels.size, "waveform keeps only the most recent levels")
        assertEquals(0.1f, h.state.levels.last(), 1e-6f)

        h.controller.stop()
        advanceUntilIdle()
        assertEquals(StudioActivity.IDLE, h.state.activity)
        assertNull(h.state.partial)
        assertEquals(
            listOf(TranscriptLine(0, "word1 word2 word3 word4 word5 word6.", 0, 600)),
            h.state.committed,
        )
        assertEquals("0:00.0 → 0:00.6", h.state.committed.single().timeLabel)
        assertEquals("Completed: 1 segment(s), 0.6 s of audio", h.state.status)
        assertNull(h.state.error)
        assertFalse(h.transcriber.closed, "the model stays loaded for the next session")
    }

    @Test
    fun secondSessionReusesTheLoadedModelAndClearsTheTranscript() = runTest {
        val h = Harness(this)
        h.controller.startLive()
        advanceUntilIdle()
        h.capture!!.sendMs(100)
        advanceUntilIdle()
        h.controller.stop()
        advanceUntilIdle()
        assertEquals(1, h.state.committed.size)

        h.controller.startLive()
        advanceUntilIdle()
        assertEquals(1, h.factory.opened.size)
        assertTrue(h.state.committed.isEmpty())
        assertNull(h.state.status)
    }

    @Test
    fun inBandStreamErrorIsShown() = runTest {
        val h = Harness(this, factory = FakeTranscriberFactory { it.streamError = InferenceCancelledException("aborted") })
        h.controller.startLive()
        advanceUntilIdle()
        h.capture!!.sendMs(100)
        advanceUntilIdle()
        h.controller.stop()
        advanceUntilIdle()
        assertEquals(StudioActivity.IDLE, h.state.activity)
        assertEquals("InferenceCancelledException: aborted", h.state.error)
        assertNull(h.state.partial)
    }

    @Test
    fun modelLoadFailureIsShownAndReturnsToIdle() = runTest {
        val factory = FakeTranscriberFactory(failure = ModelLoadException("not a whisper model"))
        val h = Harness(this, factory = factory)
        h.controller.startLive()
        advanceUntilIdle()
        assertEquals(StudioActivity.IDLE, h.state.activity)
        assertContains(h.state.error!!, "not a whisper model")
        assertNull(h.state.activeBackend)
        assertEquals(0, h.capture!!.prepareCount)
    }

    @Test
    fun microphoneFailureIsShown() = runTest {
        val h = Harness(this)
        h.capture!!.prepareFailure = IllegalStateException("Microphone permission denied")
        h.controller.startLive()
        advanceUntilIdle()
        assertEquals(StudioActivity.IDLE, h.state.activity)
        assertEquals("IllegalStateException: Microphone permission denied", h.state.error)
    }

    @Test
    fun cancelAbandonsTheStream() = runTest {
        val h = Harness(this)
        h.controller.startLive()
        advanceUntilIdle()
        h.capture!!.sendMs(100)
        advanceUntilIdle()
        assertTrue(h.state.canCancel)
        h.controller.cancel()
        advanceUntilIdle()
        assertEquals(StudioActivity.IDLE, h.state.activity)
        assertEquals("Cancelled", h.state.status)
        assertNull(h.state.partial)
        assertTrue(h.state.committed.isEmpty())
    }

    @Test
    fun captureEndingCompletesTheStream() = runTest {
        val h = Harness(this)
        h.controller.startLive()
        advanceUntilIdle()
        h.capture!!.sendMs(100)
        h.capture.finish()
        advanceUntilIdle()
        assertEquals(StudioActivity.IDLE, h.state.activity)
        assertEquals(1, h.state.committed.size)
    }

    @Test
    fun selectingABackendReloadsAnIdleLoadedModel() = runTest {
        val h = Harness(this)
        h.controller.startLive()
        advanceUntilIdle()
        h.controller.stop()
        advanceUntilIdle()
        assertEquals(ComputeBackend.GPU_METAL, h.state.activeBackend, "AUTO resolves to Metal on the fake")

        h.controller.selectBackend(ComputeBackend.CPU_ONLY)
        advanceUntilIdle()
        assertEquals(listOf(BackendConfig.CPU), h.transcriber.reinitialized)
        assertEquals(ComputeBackend.CPU_ONLY, h.state.activeBackend)
        assertFalse(h.state.usesGpu)
        assertEquals(StudioActivity.IDLE, h.state.activity)

        // Unavailable backends fall back (Vulkan → CPU) and the same selection is a no-op.
        h.controller.selectBackend(ComputeBackend.GPU_VULKAN)
        h.controller.selectBackend(ComputeBackend.GPU_VULKAN)
        advanceUntilIdle()
        assertEquals(2, h.transcriber.reinitialized.size)
        assertEquals(ComputeBackend.GPU_VULKAN, h.state.selectedBackend)
        assertEquals(ComputeBackend.CPU_ONLY, h.state.activeBackend)
    }

    @Test
    fun backendCannotChangeWhileBusy() = runTest {
        val h = Harness(this)
        h.controller.startLive()
        advanceUntilIdle()
        h.controller.selectBackend(ComputeBackend.CPU_ONLY)
        assertEquals(ComputeBackend.AUTO, h.state.selectedBackend)
        assertTrue(h.transcriber.reinitialized.isEmpty())
    }

    @Test
    fun selectingABackendBeforeLoadingDoesNotOpenTheModel() = runTest {
        val h = Harness(this)
        h.controller.selectBackend(ComputeBackend.CPU_ONLY)
        advanceUntilIdle()
        assertTrue(h.factory.opened.isEmpty())
        assertEquals(ComputeBackend.CPU_ONLY, h.state.selectedBackend)
    }

    @Test
    fun changingTheModelPathOpensANewModel() = runTest {
        val h = Harness(this)
        h.controller.startLive()
        advanceUntilIdle()
        h.controller.stop()
        advanceUntilIdle()
        val first = h.transcriber
        h.controller.setModelPath("/models/other.bin")
        h.controller.startLive()
        advanceUntilIdle()
        assertTrue(first.closed)
        assertEquals("/models/other.bin", h.transcriber.modelPath)
    }

    @Test
    fun fileTranscriptionDecodesThePcmFileInMemory() = runTest {
        val segments = listOf(
            TranscriptionEvent.CommittedSegment(0, " Hello there.", 0, 900),
            TranscriptionEvent.CommittedSegment(1, " General Kenobi.", 900, 1_500),
        )
        val h = Harness(
            this,
            factory = FakeTranscriberFactory { it.fileSegments = segments },
            files = mapOf("/audio/hello.wav" to wavBytes(ShortArray(24_000) { 1000 })),
        )
        h.controller.setFilePath("/audio/hello.wav")
        h.controller.transcribeFile()
        advanceUntilIdle()

        assertEquals(StudioActivity.IDLE, h.state.activity)
        assertEquals(TranscriptSource.FILE, h.state.source)
        val audio = assertIs<PcmAudio.S16>(h.transcriber.transcribedFiles.single())
        assertEquals(16_000, audio.sampleRate)
        assertEquals(24_000, audio.samples.size)
        assertEquals(listOf("Hello there.", "General Kenobi."), h.state.committed.map { it.text })
        assertEquals("Transcribed 1.5 s of audio (16000 Hz, 1 ch): 2 segment(s)", h.state.status)
    }

    @Test
    fun unreadableFileIsReported() = runTest {
        val h = Harness(this, files = mapOf("/audio/bad.wav" to "RIFF\u0000\u0000\u0000\u0000WAVE".encodeToByteArray()))
        h.controller.setFilePath("/audio/missing.wav")
        h.controller.transcribeFile()
        advanceUntilIdle()
        assertEquals("IllegalArgumentException: No such file: /audio/missing.wav", h.state.error)

        h.controller.setFilePath("/audio/bad.wav")
        h.controller.transcribeFile()
        advanceUntilIdle()
        assertEquals("UnsupportedAudioException: No fmt chunk", h.state.error)
        assertEquals(StudioActivity.IDLE, h.state.activity)
    }

    @Test
    fun backgroundFallbackIsInstalledUnlessStressModeIsOn() = runTest {
        val h = Harness(this, backgroundStressSupported = true)
        h.controller.startLive()
        advanceUntilIdle()
        h.controller.stop()
        advanceUntilIdle()
        assertEquals(1, h.transcriber.fallbackInstalls)

        h.controller.setBackgroundStress(true)
        assertTrue(h.state.backgroundStress)
        assertEquals(1, h.transcriber.fallbackCloses)

        h.controller.setBackgroundStress(false)
        assertEquals(2, h.transcriber.fallbackInstalls)
    }

    @Test
    fun backgroundStressIsIgnoredWhereUnsupported() = runTest {
        val h = Harness(this, backgroundStressSupported = false)
        h.controller.setBackgroundStress(true)
        assertFalse(h.state.backgroundStress)
    }

    @Test
    fun closeReleasesTheModelAndStopsWork() = runTest {
        val h = Harness(this, backgroundStressSupported = true)
        h.controller.startLive()
        advanceUntilIdle()
        h.controller.close()
        advanceUntilIdle()
        assertTrue(h.transcriber.closed)
        assertEquals(1, h.transcriber.fallbackCloses)
        assertEquals(StudioActivity.IDLE, h.state.activity)
        h.controller.startLive()
        advanceUntilIdle()
        assertEquals(1, h.factory.opened.size, "a closed controller starts nothing")
    }

    @Test
    fun cpuOnlySelectionRequestsNoGpuLayers() {
        assertEquals(BackendConfig.CPU, studioBackendConfig(ComputeBackend.CPU_ONLY))
        assertEquals(0, studioBackendConfig(ComputeBackend.CPU_ONLY).gpuLayers)
        assertEquals(ComputeBackend.NPU_COREML, studioBackendConfig(ComputeBackend.NPU_COREML).backend)
        assertEquals(ComputeBackend.entries.toSet(), STUDIO_BACKENDS.toSet())
    }

    @Test
    fun formattingHelpers() {
        assertEquals("0:00.0", formatTimestamp(0))
        assertEquals("0:00.0", formatTimestamp(-5))
        assertEquals("1:05.4", formatTimestamp(65_432))
        assertEquals("12.3 s", formatDuration(12_345))
        assertEquals(0f, rmsLevel(FloatArray(0)))
        assertEquals(0.5f, rmsLevel(FloatArray(10) { if (it % 2 == 0) 0.5f else -0.5f }), 1e-6f)
        assertEquals(1f, rmsLevel(floatArrayOf(4f, -4f)), "input is clamped to [-1, 1]")
        assertEquals(100, CaptureFormat(44_100, 2).durationMs(8_820))
    }
}
