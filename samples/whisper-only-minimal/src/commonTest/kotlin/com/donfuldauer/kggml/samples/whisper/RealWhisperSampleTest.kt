package com.donfuldauer.kggml.samples.whisper

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.whisper.TinyWhisperModel
import kotlin.math.PI
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotSame
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext

/**
 * The sample's session logic against the REAL whisper engine ([WhisperSpeechEngine.Factory] -> `:kggml-whisper` ->
 * shared `:kggml-runtime`), driven by the synthetic [TinyWhisperModel] (random weights: the text is meaningless, so the
 * tests check backends, durations, engine reuse and error typing, not words).
 *
 * Runs on the iOS simulator and on Android devices / emulators (`-Pkggml.bundleNative=true`); skipped in Android local
 * unit tests, where the APK's native libraries cannot load.
 */
class RealWhisperSampleTest {
    private fun tone(seconds: Int, sampleRate: Int = 16_000): PcmAudio.S16 =
        PcmAudio.S16(ShortArray(seconds * sampleRate) { i -> (sin(2 * PI * 330 * i / sampleRate) * 6_000).toInt().toShort() }, sampleRate, 1)

    private fun nativeTest(name: String, block: suspend () -> Unit) = runTest(timeout = 180.seconds) {
        if (!nativeWhisperAvailable(name)) return@runTest
        withContext(Dispatchers.Default) { block() }
    }

    @Test
    fun describeRuntimeReportsTheLoadedRuntime() = nativeTest("describeRuntimeReportsTheLoadedRuntime") {
        val summary = readRuntimeSummary()
        assertTrue(ComputeBackend.CPU_ONLY in summary.availableBackends, "CPU must always be available: $summary")
        assertTrue(summary.recommendedThreadCount >= 1)
        val text = describeRuntime()
        assertEquals(
            listOf(
                "Kggml whisper-only minimal sample",
                "Runtime ${summary.runtimeVersion} (ggml ${summary.ggmlVersion})",
                "Backends: ${summary.availableBackends.joinToString { it.name }}",
                "Recommended threads: ${summary.recommendedThreadCount}",
            ),
            text.lines(),
        )
    }

    @Test
    fun cpuOnlyTranscriptionRunsWithoutTheGpu() = nativeTest("cpuOnlyTranscriptionRunsWithoutTheGpu") {
        val model = NativeTestEnvironment.writeTempFile("ggml-tiny-synthetic.bin", TinyWhisperModel.bytes())
        WhisperSampleSession().use { session ->
            val report = session.transcribe(model, tone(2), ComputeBackend.CPU_ONLY)
            assertEquals(ComputeBackend.CPU_ONLY, report.requestedBackend)
            assertEquals(ComputeBackend.CPU_ONLY, report.activeBackend)
            assertFalse(report.usesGpu)
            assertEquals(2_000, report.audioDurationMs)
            assertFalse(session.currentEngine!!.usesGpu)
            val rendered = formatReport(report)
            assertTrue(rendered.startsWith("Requested CPU_ONLY, ran on CPU_ONLY (use_gpu=false)\nAudio 2000 ms, "), rendered)
        }
    }

    @Test
    fun backendSwitchReusesTheEngineAndModelSwitchReplacesIt() = nativeTest("backendSwitchReusesTheEngineAndModelSwitchReplacesIt") {
        val bytes = TinyWhisperModel.bytes()
        val first = NativeTestEnvironment.writeTempFile("a.bin", bytes)
        val second = NativeTestEnvironment.writeTempFile("b.bin", bytes)
        WhisperSampleSession().use { session ->
            session.transcribe(first, tone(1), ComputeBackend.CPU_ONLY)
            val engine = session.currentEngine!!
            if (NativeTestEnvironment.gpuBackendsUsable) {
                // Same model, different backend: WhisperEngine.reinitialize on the same engine (the runtime toggle).
                val auto = session.transcribe(first, tone(1), ComputeBackend.AUTO)
                assertSame(engine, session.currentEngine)
                assertEquals(ComputeBackend.AUTO, auto.requestedBackend)
                assertTrue(auto.activeBackend != ComputeBackend.AUTO, "AUTO must resolve to a concrete backend")
                // Back to CPU_ONLY on the same engine.
                assertEquals(ComputeBackend.CPU_ONLY, session.transcribe(first, tone(1), ComputeBackend.CPU_ONLY).activeBackend)
                assertSame(engine, session.currentEngine)
            }
            // Same model, same backend: the engine is reused as is.
            session.transcribe(first, tone(1), ComputeBackend.CPU_ONLY)
            assertSame(engine, session.currentEngine)
            // A different model file opens a new engine.
            session.transcribe(second, tone(1), ComputeBackend.CPU_ONLY)
            assertNotSame(engine, session.currentEngine)
        }
    }

    @Test
    fun wavFileParsedBySampleTranscribesItsFullDuration() = nativeTest("wavFileParsedBySampleTranscribesItsFullDuration") {
        val model = NativeTestEnvironment.writeTempFile("ggml-tiny-synthetic.bin", TinyWhisperModel.bytes())
        // 44.1 kHz stereo WAV: the engine resamples / downmixes natively.
        val source = tone(1, sampleRate = 44_100).samples
        val stereo = ShortArray(source.size * 2) { source[it / 2] }
        val audio = parsePcmFile(wav(stereo, sampleRate = 44_100, channels = 2))
        assertEquals(44_100, audio.sampleRate)
        assertEquals(2, audio.channels)
        WhisperSampleSession().use { session ->
            val report = session.transcribe(model, audio, ComputeBackend.CPU_ONLY)
            assertEquals(1_000, report.audioDurationMs)
        }
    }

    @Test
    fun invalidModelFailsTypedAndTheSessionStaysUsable() = nativeTest("invalidModelFailsTypedAndTheSessionStaysUsable") {
        val garbage = NativeTestEnvironment.writeTempFile("garbage.bin", "not a whisper model".encodeToByteArray())
        val model = NativeTestEnvironment.writeTempFile("ggml-tiny-synthetic.bin", TinyWhisperModel.bytes())
        WhisperSampleSession().use { session ->
            assertFailsWith<ModelLoadException> { session.transcribe(garbage, tone(1), ComputeBackend.CPU_ONLY) }
            assertEquals(1_000, session.transcribe(model, tone(1), ComputeBackend.CPU_ONLY).audioDurationMs)
        }
    }

    /** A canonical 16-bit PCM RIFF/WAVE file. */
    private fun wav(samples: ShortArray, sampleRate: Int, channels: Int): ByteArray {
        val data = samples.size * 2
        val out = ByteArray(44 + data)
        fun ascii(at: Int, s: String) = s.forEachIndexed { i, c -> out[at + i] = c.code.toByte() }
        fun le16(at: Int, v: Int) { out[at] = v.toByte(); out[at + 1] = (v shr 8).toByte() }
        fun le32(at: Int, v: Int) { le16(at, v and 0xFFFF); le16(at + 2, v ushr 16) }
        ascii(0, "RIFF"); le32(4, 36 + data); ascii(8, "WAVE")
        ascii(12, "fmt "); le32(16, 16); le16(20, 1); le16(22, channels); le32(24, sampleRate)
        le32(28, sampleRate * channels * 2); le16(32, channels * 2); le16(34, 16)
        ascii(36, "data"); le32(40, data)
        samples.forEachIndexed { i, s -> le16(44 + i * 2, s.toInt()) }
        return out
    }
}
