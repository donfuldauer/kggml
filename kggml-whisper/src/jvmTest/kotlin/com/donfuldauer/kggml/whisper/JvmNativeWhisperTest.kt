package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.runtime.KggmlRuntime
import java.io.File
import java.security.MessageDigest
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * WhisperEngine against the real native library (`kggml_whisper_jni` -> `kggml_whisper` -> shared ggml) on the
 * JVM, driven by the synthetic [TinyWhisperModel] fixture and generated PCM arrays.
 *
 * Skipped unless Gradle found the host builds (`build/native/lib` + `build/native-whisper/lib`); with
 * `-Pkggml.requireNative=true` a missing library is a failure instead.
 */
class JvmNativeWhisperTest {
    private val requireNative = System.getProperty("kggml.test.requireNative").toBoolean()
    private lateinit var modelFile: File

    @BeforeTest
    fun setUp() {
        val present = System.getProperty("kggml.whisper.native.path") != null
        if (requireNative) {
            assertTrue(present, "-Pkggml.requireNative=true but the whisper host build was not found")
        }
        assumeTrue(present, "native whisper host build not found; run kggml-whisper/native/scripts/build-host.sh")
        modelFile = File.createTempFile("kggml-whisper-tiny-", ".bin").apply {
            deleteOnExit()
            writeBytes(TinyWhisperModel.bytes())
        }
    }

    @AfterTest
    fun tearDown() {
        if (::modelFile.isInitialized) modelFile.delete()
    }

    private fun noise(samples: Int, seed: Int = 1): FloatArray {
        val random = Random(seed)
        return FloatArray(samples) { random.nextFloat() * 0.2f - 0.1f }
    }

    private suspend fun cpuEngine(): WhisperEngine = WhisperEngine(modelFile.absolutePath, BackendConfig.CPU)

    @Test
    fun kotlinFixtureIsByteIdenticalToTheNativeTestModel() {
        val sha = MessageDigest.getInstance("SHA-256").digest(TinyWhisperModel.bytes()).joinToString("") { "%02x".format(it) }
        // SHA-256 of the model written by kggml-whisper/native/tests/tiny_model.cpp.
        assertEquals("0d2b7a99e27aff725a2ea6b4444c182762fa332ac756e0f2af4d9b1982d3d208", sha)
    }

    @Test
    fun nativeInfoMatchesThePinnedUpstream() {
        val info = WhisperEngine.nativeInfo()
        assertEquals("1.8.4", info.whisperCppVersion)
        assertEquals("9386f239401074690479731c1e41683fbbeac557", info.whisperCppCommit)
        assertFalse(info.hasCoreMl, "CoreML is only compiled into iOS builds")
        assertTrue(info.systemInfo.isNotEmpty())
    }

    @Test
    fun cpuOnlyPassesUseGpuFalseToWhisperInit() = runBlocking {
        if (KggmlWhisperJni.nativeTestHooksAvailable()) {
            val out = IntArray(2)
            assertEquals(WhisperStatus.OK, KggmlWhisperJni.nativeTestResolveUseGpu(WhisperBackendKind.CPU, out))
            assertEquals(0, out[0], "CPU_ONLY must resolve to use_gpu = false")
            assertEquals(0, out[1], "CPU_ONLY must never allow the CoreML encoder")
            assertEquals(WhisperStatus.OK, KggmlWhisperJni.nativeTestResolveUseGpu(WhisperBackendKind.VULKAN, out))
            assertEquals(1, out[0])
        }
        cpuEngine().use { engine ->
            assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
            assertFalse(engine.usesGpu, "whisper context was created with use_gpu = true")
        }
    }

    @Test
    fun autoFallsBackToAnAvailableBackend() = runBlocking {
        WhisperEngine(modelFile.absolutePath).use { engine ->
            assertTrue(engine.activeBackend in KggmlRuntime.availableBackends(), "${engine.activeBackend}")
            assertEquals(engine.activeBackend != ComputeBackend.CPU_ONLY, engine.usesGpu)
        }
    }

    @Test
    fun transcribesFloatShortResampledAndShortInput() = runBlocking {
        cpuEngine().use { engine ->
            val mono = engine.transcribe(noise(16_000))
            assertEquals(1000L, mono.audioDurationMs)
            assertEquals(ComputeBackend.CPU_ONLY, mono.backend)
            mono.segments.forEach { assertTrue(it.endMs >= it.startMs) }

            val stereo44k = noise(44_100 * 2)
            assertEquals(1000L, engine.transcribe(stereo44k, sampleRate = 44_100, channels = 2).audioDurationMs)

            val pcm16 = ShortArray(48_000) { ((it % 200) - 100).toShort() }
            assertEquals(1000L, engine.transcribe(pcm16, sampleRate = 48_000).audioDurationMs)

            // 200 ms and empty input are zero-padded natively to the 1000 ms floor.
            assertEquals(200L, engine.transcribe(noise(3_200)).audioDurationMs)
            assertEquals(0L, engine.transcribe(FloatArray(0)).audioDurationMs)
        }
    }

    @Test
    fun nativeErrorsAreTyped() = runBlocking {
        assertFailsWith<ModelLoadException> { WhisperEngine(File(modelFile.parentFile, "missing-${System.nanoTime()}.bin").path) }
        val garbage = File.createTempFile("kggml-whisper-garbage-", ".bin").apply { writeText("not a whisper model") }
        try {
            assertFailsWith<ModelLoadException> { WhisperEngine(garbage.path, BackendConfig.CPU) }
        } finally {
            garbage.delete()
        }
        cpuEngine().use { engine ->
            assertFailsWith<InvalidInputException> {
                engine.transcribe(noise(16_000), options = TranscriptionOptions(language = "xx-not-a-language"))
            }
            engine.transcribe(noise(16_000)) // still usable
        }
    }

    @Test
    fun cancellationAbortsNativeDecodingPromptlyAndTheEngineStaysUsable() = runBlocking {
        WhisperEngine(modelFile.absolutePath, BackendConfig.CPU.copy(cpuThreads = 2)).use { engine ->
            // Baseline: one full 30 s whisper window.
            val window = TimeSource.Monotonic.markNow()
            engine.transcribe(noise(16_000 * 30, seed = 3))
            val perWindow = window.elapsedNow()

            // Twenty windows (10 min of audio) would take ~20x the baseline; cancellation must stop far sooner.
            val long = noise(16_000 * 600, seed = 7)
            val running = async(Dispatchers.Default) { engine.transcribe(long) }
            withTimeout(30.seconds) { while (!engine.isTranscribing) delay(1) }
            delay(perWindow / 2)
            val mark = TimeSource.Monotonic.markNow()
            running.cancel()
            assertFailsWith<CancellationException> { running.await() }
            val elapsed = mark.elapsedNow()
            val budget = perWindow * 3 + 250.milliseconds
            assertTrue(elapsed < budget, "native decoding took $elapsed to stop (budget $budget, one window $perWindow)")
            assertFalse(engine.isTranscribing)

            assertEquals(1000L, engine.transcribe(noise(16_000)).audioDurationMs)
        }
    }

    @Test
    fun multipleInstancesCoexistAndTranscribeConcurrently() = runBlocking {
        // One decoder thread each so three engines fit on small CI runners without oversubscription.
        val engines = List(3) { WhisperEngine(modelFile.absolutePath, BackendConfig.CPU.copy(cpuThreads = 1)) }
        try {
            val results = engines.mapIndexed { i, engine ->
                async(Dispatchers.Default) { List(2) { engine.transcribe(noise(16_000, seed = i)) } }
            }.awaitAll()
            results.flatten().forEach { assertEquals(1000L, it.audioDurationMs) }
            // Same input on separate instances decodes identically: no state leaks between contexts.
            val a = engines[0].transcribe(noise(16_000, seed = 9))
            val b = engines[1].transcribe(noise(16_000, seed = 9))
            assertEquals(a.segments.map { it.text }, b.segments.map { it.text })
        } finally {
            engines.forEach { it.close() }
        }
    }

    @Test
    fun closedEnginesReleaseTheirRuntimeReferences() = runBlocking {
        repeat(10) { cpuEngine().use { it.transcribe(noise(1_600)) } }
        val engines = List(4) { cpuEngine() }
        assertTrue(KggmlRuntime.isInitialized(), "every live whisper context holds a runtime reference")
        engines.forEach { it.close() }
        // Native release runs asynchronously on each engine's thread after close().
        withTimeout(10.seconds) { while (KggmlRuntime.isInitialized()) delay(5) }
        assertFalse(KggmlRuntime.isInitialized())
    }
}
