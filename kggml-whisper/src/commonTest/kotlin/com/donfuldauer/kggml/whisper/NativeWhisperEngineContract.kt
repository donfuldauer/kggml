package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.TranscriptionEvent
import com.donfuldauer.kggml.core.TranscriptionEvent.CommittedSegment
import com.donfuldauer.kggml.core.TranscriptionEvent.Completed
import kotlin.concurrent.atomics.AtomicReference
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.TestResult
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

/**
 * [WhisperEngine] (batch and streaming) against the real native library, driven by the synthetic [TinyWhisperModel]
 * fixture and generated PCM. Shared by the platforms whose tests reach the native library through a platform binding
 * other than the JVM's: `iosSimulatorArm64Test` (cinterop, `usePinned` zero-copy, `staticCFunction` segment
 * callback) and Android instrumented tests (JNI on the device). The JVM has its own, larger suite
 * (`JvmNativeWhisperTest`, `JvmNativeWhisperStreamTest`).
 *
 * Every case runs on [ComputeBackend.CPU_ONLY]: simulators and emulators cannot reliably allocate GPU buffers, and the
 * GPU <-> CPU toggle policy is covered by the commonTest fakes.
 */
abstract class NativeWhisperEngineContract {
    /** Writes [bytes] to a fresh temporary file and returns its path. */
    protected abstract fun writeTempFile(prefix: String, bytes: ByteArray): String

    protected abstract fun deleteFile(path: String)

    /** Names of the files currently in the platform's temporary directory (for the no-temp-audio-files check). */
    protected abstract fun temporaryFileNames(): Set<String>

    private lateinit var modelPath: String

    @BeforeTest
    fun writeModel() {
        modelPath = writeTempFile("kggml-whisper-tiny-", TinyWhisperModel.bytes())
    }

    @AfterTest
    fun deleteModel() {
        if (::modelPath.isInitialized) deleteFile(modelPath)
    }

    private fun nativeTest(block: suspend CoroutineScope.() -> Unit): TestResult = runTest(timeout = 240.seconds) {
        withContext(Dispatchers.Default, block)
    }

    private fun noise(samples: Int, seed: Int = 1): FloatArray {
        val random = Random(seed)
        return FloatArray(samples) { random.nextFloat() * 0.2f - 0.1f }
    }

    private fun speech(ms: Int) = FloatArray(ms * 16) { 0.1f * ((it % 100) / 50f - 1f) }

    private fun silence(ms: Int) = FloatArray(ms * 16)

    private fun chunked(vararg parts: FloatArray): Flow<FloatArray> = flow {
        for (p in parts) for (i in p.indices step 640) emit(p.copyOfRange(i, minOf(p.size, i + 640)))
    }

    @Test
    fun cpuOnlyRunsWithoutTheGpu() = nativeTest {
        WhisperEngine(modelPath, BackendConfig.CPU).use { engine ->
            assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
            assertFalse(engine.usesGpu, "CPU_ONLY must create the whisper context with use_gpu = false")
            assertEquals(1000L, engine.transcribe(noise(16_000)).audioDurationMs)
            assertEquals(1000L, engine.transcribe(ShortArray(48_000), sampleRate = 48_000).audioDurationMs)
            assertEquals(1000L, engine.transcribe(noise(88_200), sampleRate = 44_100, channels = 2).audioDurationMs)
        }
    }

    @Test
    fun reinitializeFreesAndReloadsTheNativeContext() = nativeTest {
        WhisperEngine(modelPath, BackendConfig.CPU).use { engine ->
            assertEquals(ComputeBackend.CPU_ONLY, engine.reinitialize(BackendConfig.CPU.copy(cpuThreads = 1)))
            assertFalse(engine.usesGpu)
            assertEquals(1, engine.backendConfig.cpuThreads)
            assertEquals(ComputeBackend.CPU_ONLY, engine.transcribe(noise(16_000)).backend)
        }
    }

    @Test
    fun cancellationAndMultipleInstances() = nativeTest {
        val engines = List(2) { WhisperEngine(modelPath, BackendConfig.CPU.copy(cpuThreads = 1)) }
        try {
            engines.map { e -> async(Dispatchers.Default) { e.transcribe(noise(16_000)) } }.awaitAll()
            val running = async(Dispatchers.Default) { engines[0].transcribe(noise(16_000 * 600)) }
            withTimeout(30.seconds) { while (!engines[0].isTranscribing) delay(1) }
            running.cancel()
            assertFailsWith<CancellationException> { running.await() }
            assertEquals(1000L, engines[0].transcribe(noise(16_000)).audioDurationMs)
        } finally {
            engines.forEach { it.close() }
        }
    }

    @Test
    fun missingModelIsTyped() = nativeTest {
        assertFailsWith<ModelLoadException> { WhisperEngine("$modelPath.missing", BackendConfig.CPU) }
    }

    @Test
    fun streamsCommittedSegmentsInOrderWithoutTemporaryFiles() = nativeTest {
        val before = temporaryFileNames()
        WhisperEngine(modelPath, BackendConfig.CPU).use { engine ->
            val parts = buildList {
                repeat(4) {
                    add(speech(2_400))
                    add(silence(900))
                }
                add(speech(31_200)) // split at the 30 s window: prompt carry-over across windows
                add(silence(990))
            }.toTypedArray()
            val events = withTimeout(180.seconds) { engine.transcribeStream(chunked(*parts)).toList() }
            assertTrue(events.none { it is TranscriptionEvent.Error }, "$events")
            val completed = assertIs<Completed>(events.last())
            assertEquals(parts.sumOf { it.size } / 16L, completed.audioDurationMs)
            val committed = events.filterIsInstance<CommittedSegment>()
            assertEquals(completed.segmentCount, committed.size)
            assertEquals(committed.indices.toList(), committed.map { it.index })
            committed.zipWithNext().forEach { (a, b) -> assertTrue(a.endMs <= b.startMs, "$a then $b") }

            // int16 at 48 kHz stereo through the native streaming resampler.
            val pcm = ShortArray(48 * 2 * 2_000) { ((it / 2 % 300) - 150).toShort() }
            val events16 = engine.transcribeStream(
                flow<ShortArray> {
                    emit(pcm)
                    emit(ShortArray(48 * 2 * 900))
                },
                sampleRate = 48_000,
                channels = 2,
            ).toList()
            assertEquals(2_900L, assertIs<Completed>(events16.last()).audioDurationMs)
        }
        assertEquals(emptySet(), temporaryFileNames() - before, "no temporary audio files")
    }

    @Test
    fun cancellationAndCloseEndStreamsCleanly() = nativeTest {
        val engine = WhisperEngine(modelPath, BackendConfig.CPU.copy(cpuThreads = 1))
        val endless = flow { while (true) emit(speech(100)) }
        val first = AtomicReference<TranscriptionEvent?>(null)
        val job = launch(Dispatchers.Default) { engine.transcribeStream(endless).collect { first.compareAndSet(null, it) } }
        withTimeout(60.seconds) { while (first.load() == null) delay(1) }
        withTimeout(30.seconds) { job.cancelAndJoin() }

        // The engine is still usable; closing it ends a live stream with InferenceCancelledException.
        val last = AtomicReference<TranscriptionEvent?>(null)
        val second = launch(Dispatchers.Default) { engine.transcribeStream(endless).collect { last.store(it) } }
        withTimeout(60.seconds) { while (last.load() == null) delay(1) }
        engine.close()
        withTimeout(30.seconds) { second.join() }
        assertIs<InferenceCancelledException>(assertIs<TranscriptionEvent.Error>(last.load()).exception)
    }
}
