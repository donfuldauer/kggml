package com.donfuldauer.kggml.samples.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.whisper.TranscriptionResult
import com.donfuldauer.kggml.whisper.WhisperEngine
import com.donfuldauer.kggml.whisper.WhisperSegment
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.TestTimeSource
import kotlinx.coroutines.test.runTest

/** Deterministic [SpeechEngine]: CPU_ONLY never uses the GPU; GPU backends are "available" except [unavailable]. */
class FakeSpeechEngine(
    val modelPath: String,
    config: BackendConfig,
    private val unavailable: Set<ComputeBackend> = emptySet(),
    private val timeSource: TestTimeSource? = null,
    var failNext: Throwable? = null,
) : SpeechEngine {
    val log = mutableListOf<String>()
    var closed = false
    override var activeBackend: ComputeBackend = resolve(config.backend)
    override val usesGpu: Boolean get() = activeBackend != ComputeBackend.CPU_ONLY
    override val whisperEngine: WhisperEngine? = null

    private fun resolve(backend: ComputeBackend) = when (backend) {
        ComputeBackend.AUTO -> ComputeBackend.GPU_VULKAN
        in unavailable -> ComputeBackend.CPU_ONLY
        else -> backend
    }

    override suspend fun reinitialize(config: BackendConfig): ComputeBackend {
        log += "reinitialize ${config.backend}"
        activeBackend = resolve(config.backend)
        return activeBackend
    }

    override suspend fun transcribe(audio: PcmAudio): TranscriptionResult {
        log += "transcribe ${audio.sampleCount}"
        failNext?.let { failNext = null; throw it }
        timeSource?.let { it += 120.milliseconds }
        return TranscriptionResult(
            segments = listOf(WhisperSegment(0, " And so", 0, 500, 0.1f), WhisperSegment(1, " my fellow", 500, 900, 0.1f)),
            audioDurationMs = audio.durationMs,
            backend = activeBackend,
        )
    }

    override fun close() {
        closed = true
    }
}

class WhisperSampleSessionTest {
    private val audio = PcmAudio.S16(ShortArray(16_000), 16_000, 1)

    @Test
    fun reusesEngineAndReinitializesOnBackendChange() = runTest {
        val opened = mutableListOf<FakeSpeechEngine>()
        val time = TestTimeSource()
        val events = mutableListOf<String>()
        val session = WhisperSampleSession(
            factory = { path, config -> FakeSpeechEngine(path, config, timeSource = time).also { opened += it } },
            timeSource = time,
            onEngineOpened = { events += "opened" },
            onEngineClosing = { events += "closing" },
        )

        val cpu = session.transcribe("a.bin", audio, ComputeBackend.CPU_ONLY)
        assertEquals(ComputeBackend.CPU_ONLY, cpu.requestedBackend)
        assertEquals(ComputeBackend.CPU_ONLY, cpu.activeBackend)
        assertFalse(cpu.usesGpu)
        assertEquals("And so my fellow", cpu.text)
        assertEquals(2, cpu.segmentCount)
        assertEquals(1_000, cpu.audioDurationMs)
        assertEquals(120.milliseconds, cpu.elapsed)

        session.transcribe("a.bin", audio, ComputeBackend.CPU_ONLY)
        val gpu = session.transcribe("a.bin", audio, ComputeBackend.GPU_VULKAN)
        assertTrue(gpu.usesGpu)
        assertEquals(1, opened.size)
        assertSame(opened.single(), session.currentEngine)
        assertEquals(
            listOf("transcribe 16000", "transcribe 16000", "reinitialize GPU_VULKAN", "transcribe 16000"),
            opened.single().log,
        )

        // A different model opens a new engine and closes the old one.
        session.transcribe("b.bin", audio, ComputeBackend.AUTO)
        assertEquals(listOf("a.bin", "b.bin"), opened.map { it.modelPath })
        assertTrue(opened[0].closed)
        assertFalse(opened[1].closed)
        assertEquals(listOf("opened", "closing", "opened"), events)

        session.close()
        session.close()
        assertTrue(opened[1].closed)
        assertNull(session.currentEngine)
        assertEquals(listOf("opened", "closing", "opened", "closing"), events)
        assertFailsWith<IllegalStateException> { session.transcribe("b.bin", audio, ComputeBackend.AUTO) }
    }

    @Test
    fun reportsFallbackWhenGpuIsUnavailable() = runTest {
        val session = WhisperSampleSession(
            factory = { path, config -> FakeSpeechEngine(path, config, unavailable = setOf(ComputeBackend.GPU_METAL)) },
        )
        val report = session.transcribe("a.bin", audio, ComputeBackend.GPU_METAL)
        assertEquals(ComputeBackend.GPU_METAL, report.requestedBackend)
        assertEquals(ComputeBackend.CPU_ONLY, report.activeBackend)
        assertFalse(report.usesGpu)
        assertTrue(formatReport(report).startsWith("Requested GPU_METAL, ran on CPU_ONLY (use_gpu=false)\n"))
    }

    @Test
    fun transcriptionErrorsPropagateAndKeepTheEngine() = runTest {
        var engine: FakeSpeechEngine? = null
        val session = WhisperSampleSession(factory = { path, config -> FakeSpeechEngine(path, config).also { engine = it } })
        session.transcribe("a.bin", audio, ComputeBackend.CPU_ONLY)
        engine!!.failNext = InvalidInputException("bad audio")
        assertFailsWith<InvalidInputException> { session.transcribe("a.bin", audio, ComputeBackend.CPU_ONLY) }
        assertFalse(engine.closed)
        session.transcribe("a.bin", audio, ComputeBackend.CPU_ONLY)
    }

    @Test
    fun factoryFailureLeavesNoEngine() = runTest {
        val session = WhisperSampleSession(factory = { _, _ -> throw IllegalArgumentException("missing model") })
        assertFailsWith<IllegalArgumentException> { session.transcribe("a.bin", audio, ComputeBackend.CPU_ONLY) }
        assertNull(session.currentEngine)
        session.close()
    }

    @Test
    fun formatsReports() {
        val report = TranscriptionReport(
            requestedBackend = ComputeBackend.AUTO,
            activeBackend = ComputeBackend.GPU_METAL,
            usesGpu = true,
            audioDurationMs = 11_000,
            elapsed = 840.milliseconds,
            text = "",
            segmentCount = 0,
        )
        assertEquals(
            "Requested AUTO, ran on GPU_METAL (use_gpu=true)\nAudio 11000 ms, 0 segment(s) in 840 ms\n(no speech)",
            formatReport(report),
        )
    }

    @Test
    fun backendChoicesMapToComputeBackends() {
        assertEquals(ComputeBackend.CPU_ONLY, BackendChoice.CPU_ONLY.toComputeBackend(ComputeBackend.GPU_METAL))
        assertEquals(ComputeBackend.GPU_METAL, BackendChoice.GPU.toComputeBackend(ComputeBackend.GPU_METAL))
        assertEquals(ComputeBackend.GPU_VULKAN, BackendChoice.GPU.toComputeBackend(ComputeBackend.GPU_VULKAN))
        assertEquals(ComputeBackend.AUTO, BackendChoice.AUTO.toComputeBackend())
        assertEquals("GPU (Metal)", BackendChoice.GPU.label(ComputeBackend.GPU_METAL))
        assertEquals("GPU (Vulkan)", BackendChoice.GPU.label(ComputeBackend.GPU_VULKAN))
        assertEquals(listOf("CPU only", "Auto"), listOf(BackendChoice.CPU_ONLY.label(), BackendChoice.AUTO.label()))
        assertTrue(platformGpuBackend == ComputeBackend.GPU_METAL || platformGpuBackend == ComputeBackend.GPU_VULKAN)
    }
}
