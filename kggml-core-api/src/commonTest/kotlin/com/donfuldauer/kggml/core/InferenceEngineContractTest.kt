package com.donfuldauer.kggml.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest

/** Minimal engine built the way real engines are: a [SerialExecutor] guarding a fake native handle. */
private class FakeEngine(
    override val backendConfig: BackendConfig,
    available: Set<ComputeBackend>,
    factory: SerialDispatcherFactory,
) : InferenceEngine {
    private val executor = SerialExecutor("fake-engine", factory)
    var nativeReleases = 0
        private set

    override val activeBackend: ComputeBackend = ComputeBackend.resolve(backendConfig.backend, available)
    override val isClosed: Boolean get() = executor.isClosed

    suspend fun infer(input: Int): Int = executor.execute {
        if (input < 0) InferenceException.checkNativeResult(NativeErrorCode.INVALID_INPUT.code, "negative input")
        input * 2
    }

    override fun close() = executor.close { nativeReleases++ }
}

class InferenceEngineContractTest {

    @Test
    fun closeIsIdempotentAndReleasesNativeStateOnce() = runTest {
        val engine = FakeEngine(BackendConfig(), setOf(ComputeBackend.GPU_METAL)) { StandardTestDispatcher(testScheduler) }
        assertEquals(ComputeBackend.GPU_METAL, engine.activeBackend)
        assertEquals(8, engine.infer(4))

        engine.close()
        engine.close()
        advanceUntilIdle()

        assertTrue(engine.isClosed)
        assertEquals(1, engine.nativeReleases)
        assertFailsWith<IllegalStateException> { engine.infer(1) }
    }

    @Test
    fun useBlockClosesEngine() = runTest {
        val engine = FakeEngine(BackendConfig.CPU, emptySet()) { StandardTestDispatcher(testScheduler) }
        val result = engine.use {
            assertEquals(ComputeBackend.CPU_ONLY, it.activeBackend)
            it.infer(5)
        }
        advanceUntilIdle()
        assertEquals(10, result)
        assertTrue(engine.isClosed)
        assertEquals(1, engine.nativeReleases)
    }

    @Test
    fun nativeErrorsSurfaceAsTypedExceptions() = runTest {
        FakeEngine(BackendConfig(), emptySet()) { StandardTestDispatcher(testScheduler) }.use { engine ->
            val e = assertFailsWith<InvalidInputException> { engine.infer(-1) }
            assertEquals("negative input", e.message)
        }
    }

    @Test
    fun multipleInstancesAreIndependent() = runTest {
        val a = FakeEngine(BackendConfig(), emptySet()) { StandardTestDispatcher(testScheduler) }
        val b = FakeEngine(BackendConfig(), emptySet()) { StandardTestDispatcher(testScheduler) }
        a.close()
        advanceUntilIdle()
        assertTrue(a.isClosed)
        assertEquals(6, b.infer(3))
        b.close()
    }
}
