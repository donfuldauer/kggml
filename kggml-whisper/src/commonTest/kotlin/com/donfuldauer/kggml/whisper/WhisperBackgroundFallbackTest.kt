package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

class WhisperBackgroundFallbackTest {
    private val available = setOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_METAL)
    private val metal = BackendConfig(backend = ComputeBackend.GPU_METAL, cpuThreads = 2, useFlashAttention = true)

    private suspend fun open(bridge: FakeWhisperBridge, config: BackendConfig) = WhisperEngine.open(
        "/models/m.bin", config, WhisperEngineOptions(), WhisperBridgeHolder { bridge }, { available }, SerialDispatcherFactory.Default,
    )

    private suspend fun awaitCondition(condition: () -> Boolean) = withContext(Dispatchers.Default) {
        withTimeout(10.seconds) { while (!condition()) delay(1) }
    }

    @Test
    fun backgroundSwitchesToCpuOnlyAndForegroundRestores() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge, metal)
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val fallback = WhisperBackgroundFallback(engine, scope)

        fallback.onEnterBackground()
        awaitCondition { !engine.usesGpu }
        assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
        assertEquals(WhisperBackendKind.CPU, bridge.inits.items.last().backendKind)
        // Threads and flash attention are kept; only the backend changes.
        assertEquals(WhisperBackgroundFallback.cpuFallbackOf(metal), engine.backendConfig)
        assertEquals(2, engine.backendConfig.cpuThreads)
        awaitCondition { fallback.isFallbackActive }

        fallback.onEnterForeground()
        awaitCondition { engine.usesGpu }
        assertEquals(metal, engine.backendConfig)
        assertEquals(ComputeBackend.GPU_METAL, engine.activeBackend)
        awaitCondition { !fallback.isFallbackActive }

        fallback.close()
        scope.cancel()
        engine.close()
    }

    @Test
    fun cpuEngineIsLeftAlone() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge, BackendConfig.CPU)
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val fallback = WhisperBackgroundFallback(engine, scope)
        fallback.onEnterBackground()
        fallback.onEnterForeground()
        delay(50)
        withContext(Dispatchers.Default) { delay(50) }
        assertEquals(1, bridge.inits.items.size)
        assertFalse(fallback.isFallbackActive)
        fallback.close()
        scope.cancel()
        engine.close()
    }

    @Test
    fun runningGpuTranscriptionIsAbortedOnBackground() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge, metal)
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val fallback = WhisperBackgroundFallback(engine, scope)
        bridge.blockTranscription.store(true)
        val running = async(Dispatchers.Default) { runCatching { engine.transcribe(FloatArray(16_000)) } }
        awaitCondition { bridge.transcriptionStarted.load() == 1 }

        fallback.onEnterBackground()
        assertIs<InferenceCancelledException>(running.await().exceptionOrNull())
        awaitCondition { !engine.usesGpu }
        bridge.blockTranscription.store(false)
        assertEquals(ComputeBackend.CPU_ONLY, engine.transcribe(FloatArray(16_000)).backend)

        fallback.close()
        scope.cancel()
        engine.close()
    }

    @Test
    fun reloadFailuresAreReportedAndTheEngineKeepsWorking() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge, metal)
        val errors = ConcurrentLog<Throwable>()
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val fallback = WhisperBackgroundFallback(engine, scope, onError = { errors.add(it) })
        bridge.initStatus = { if (it == WhisperBackendKind.CPU) WhisperStatus.OUT_OF_MEMORY else WhisperStatus.OK }

        fallback.onEnterBackground()
        awaitCondition { errors.items.size == 1 }
        assertIs<com.donfuldauer.kggml.core.OutOfMemoryException>(errors.items.single())
        assertEquals(metal, engine.backendConfig) // restored
        assertFalse(fallback.isFallbackActive)
        assertTrue(engine.transcribe(FloatArray(16_000)).segments.isNotEmpty())

        fallback.close()
        scope.cancel()
        engine.close()
    }

    @Test
    fun closedEngineIsIgnored() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge, metal)
        engine.close()
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        val fallback = WhisperBackgroundFallback(engine, scope)
        fallback.onEnterBackground()
        withContext(Dispatchers.Default) { delay(50) }
        assertEquals(1, bridge.inits.items.size)
        fallback.close()
        scope.cancel()
    }
}
