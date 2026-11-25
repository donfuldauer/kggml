package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.ModelLoadException
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.convert
import kotlinx.cinterop.usePinned
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import platform.Foundation.NSData
import platform.Foundation.NSFileManager
import platform.Foundation.NSProcessInfo
import platform.Foundation.NSTemporaryDirectory
import platform.Foundation.dataWithBytes
import platform.Foundation.writeToFile

/**
 * WhisperEngine against KggmlWhisper.framework on the iOS simulator (cinterop, `usePinned` zero-copy), driven by
 * the synthetic [TinyWhisperModel] fixture. Only built when KggmlWhisper.xcframework exists (macOS + Xcode).
 */
@OptIn(ExperimentalForeignApi::class)
class IosNativeWhisperTest {
    private lateinit var modelPath: String

    @BeforeTest
    fun writeModel() {
        val bytes = TinyWhisperModel.bytes()
        modelPath = NSTemporaryDirectory() + "kggml-whisper-tiny-" + NSProcessInfo.processInfo.globallyUniqueString + ".bin"
        val written = bytes.usePinned { NSData.dataWithBytes(it.addressOf(0), bytes.size.convert()).writeToFile(modelPath, true) }
        assertTrue(written, "could not write $modelPath")
    }

    @AfterTest
    fun deleteModel() {
        NSFileManager.defaultManager.removeItemAtPath(modelPath, null)
    }

    private fun noise(samples: Int, seed: Int = 1): FloatArray {
        val random = Random(seed)
        return FloatArray(samples) { random.nextFloat() * 0.2f - 0.1f }
    }

    @Test
    fun cpuOnlyRunsWithoutTheGpu() = runBlocking {
        WhisperEngine(modelPath, BackendConfig.CPU).use { engine ->
            assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
            assertFalse(engine.usesGpu, "CPU_ONLY must create the whisper context with use_gpu = false")
            assertEquals(1000L, engine.transcribe(noise(16_000)).audioDurationMs)
            assertEquals(1000L, engine.transcribe(ShortArray(48_000), sampleRate = 48_000).audioDurationMs)
            assertEquals(1000L, engine.transcribe(noise(88_200), sampleRate = 44_100, channels = 2).audioDurationMs)
        }
    }

    @Test
    fun reinitializeFreesAndReloadsTheNativeContext() = runBlocking {
        // CPU -> CPU: the hosted iOS simulator cannot allocate Metal buffers (upstream whisper.cpp crashes in
        // model load when that fails), so the GPU <-> CPU toggle policy is covered by the commonTest fakes and the
        // native free + reload path is exercised here on the CPU.
        WhisperEngine(modelPath, BackendConfig.CPU).use { engine ->
            assertEquals(ComputeBackend.CPU_ONLY, engine.reinitialize(BackendConfig.CPU.copy(cpuThreads = 1)))
            assertFalse(engine.usesGpu)
            assertEquals(1, engine.backendConfig.cpuThreads)
            assertEquals(ComputeBackend.CPU_ONLY, engine.transcribe(noise(16_000)).backend)
        }
    }

    @Test
    fun cancellationAndMultipleInstances() = runBlocking {
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
    fun missingModelIsTyped() = runBlocking {
        assertFailsWith<ModelLoadException> { WhisperEngine(modelPath + ".missing", BackendConfig.CPU) }
        Unit
    }
}
