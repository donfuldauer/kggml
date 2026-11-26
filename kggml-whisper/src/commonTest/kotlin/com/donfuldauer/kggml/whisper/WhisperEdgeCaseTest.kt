package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.BackendUnavailableException
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.NativeFailureException
import com.donfuldauer.kggml.core.OutOfMemoryException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import com.donfuldauer.kggml.core.TranscriptionEvent
import com.donfuldauer.kggml.whisper.streaming.SpeechTurn
import com.donfuldauer.kggml.whisper.streaming.TurnEndReason
import com.donfuldauer.kggml.whisper.streaming.VadConfig
import com.donfuldauer.kggml.whisper.streaming.WindowFloor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

/** Failure paths of [WhisperEngine] loading / reinitialization / resampling, stream slots and bridge helpers. */
class WhisperEdgeCaseTest {
    private suspend fun open(bridge: NativeWhisperBridge): WhisperEngine = WhisperEngine.open(
        modelPath = "/models/ggml-tiny.bin",
        backendConfig = BackendConfig.CPU,
        options = WhisperEngineOptions(),
        bridges = WhisperBridgeHolder { bridge },
        availableBackends = { setOf(ComputeBackend.CPU_ONLY) },
        dispatcherFactory = SerialDispatcherFactory.Default,
    )

    private suspend fun awaitCondition(condition: () -> Boolean) = withContext(Dispatchers.Default) {
        withTimeout(10.seconds) { while (!condition()) delay(1) }
    }

    // ---- loading & reinitialization ------------------------------------------------------------------

    @Test
    fun reinitializeWhoseRestoreAlsoFailsKeepsTheOriginalErrorAndLeavesNoModel() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        bridge.initStatus = { WhisperStatus.MODEL_LOAD_FAILED }
        val e = assertFailsWith<ModelLoadException> { engine.reinitialize(BackendConfig.CPU) }
        val restore = assertIs<ModelLoadException>(e.suppressedExceptions.single(), "the failed restore is attached")
        assertTrue(restore.message!!.contains("kggml_whisper_init_from_file"), restore.message)
        assertEquals(3, bridge.inits.items.size, "initial load, the new config, the restore attempt")
        assertEquals(bridge.createdHandles, bridge.freedHandles, "the old context was freed before reloading")

        val noModel = assertFailsWith<IllegalStateException> { engine.transcribe(FloatArray(16_000)) }
        assertEquals("WhisperEngine has no model loaded (a previous reinitialize failed)", noModel.message)
        // A stream cannot open on a model-less (but not closed) engine: the same error reaches the collector.
        val streamFailure = assertFailsWith<IllegalStateException> {
            withContext(Dispatchers.Default) { engine.transcribeStream(flowOf(FloatArray(16_000))).toList() }
        }
        assertEquals(noModel.message, streamFailure.message)

        // A later successful reinitialize recovers the engine.
        bridge.initStatus = { WhisperStatus.OK }
        assertEquals(ComputeBackend.CPU_ONLY, engine.reinitialize(BackendConfig.CPU))
        assertEquals("hello world", engine.transcribe(FloatArray(16_000)).text)
        engine.close()
        awaitCondition { bridge.freedHandles.size == bridge.createdHandles.size }
        assertEquals(bridge.createdHandles, bridge.freedHandles)
    }

    @Test
    fun failingBackendQueriesAfterInitFreeTheNewContext() = runTest {
        val fake = FakeWhisperBridge()
        val backendQueryFails = object : NativeWhisperBridge by fake {
            override fun effectiveBackend(handle: Long) = RawIntResult(WhisperStatus.NATIVE_FAILURE, 0)
        }
        val e = assertFailsWith<NativeFailureException> { open(backendQueryFails) }
        assertTrue(e.message!!.startsWith("kggml_whisper_get_backend failed"), e.message)
        assertEquals(1, fake.createdHandles.size)
        assertEquals(fake.createdHandles, fake.freedHandles, "the context is freed exactly once")

        val fake2 = FakeWhisperBridge()
        val gpuQueryFails = object : NativeWhisperBridge by fake2 {
            override fun usesGpu(handle: Long) = RawIntResult(WhisperStatus.OUT_OF_MEMORY, 0)
        }
        assertFailsWith<OutOfMemoryException> { open(gpuQueryFails) }
        assertEquals(fake2.createdHandles, fake2.freedHandles)
    }

    @Test
    fun anUnknownEffectiveBackendKindReportsTheRequestedBackend() = runTest {
        val fake = FakeWhisperBridge()
        val unknownKind = object : NativeWhisperBridge by fake {
            override fun effectiveBackend(handle: Long) = RawIntResult(WhisperStatus.OK, 42)
        }
        val engine = open(unknownKind)
        assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
        engine.close()
    }

    // ---- input conversion ----------------------------------------------------------------------------

    @Test
    fun sixteenKhzStereoIsDownmixedNatively() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        engine.transcribe(floatArrayOf(0.2f, 0.4f, -0.2f, 0f), channels = 2)
        assertEquals(listOf("f32:2:2:16000"), bridge.conversions.items)
        val call = bridge.transcribes.items.single()
        assertEquals(2, call.count)
        assertEquals(0.3f, call.samples[0], 1e-6f)
        assertEquals(-0.1f, call.samples[1], 1e-6f)
        engine.close()
    }

    @Test
    fun anUnrepresentableResampledLengthIsInvalidInput() = runTest {
        for (length in listOf(-1L, Int.MAX_VALUE.toLong() + 1)) {
            val fake = FakeWhisperBridge()
            val bridge = object : NativeWhisperBridge by fake {
                override fun resampledLength(frames: Int, sampleRate: Int): Long = length
            }
            val engine = open(bridge)
            val e = assertFailsWith<InvalidInputException> { engine.transcribe(FloatArray(441), sampleRate = 44_100) }
            assertEquals("Cannot resample 441 frames at 44100 Hz to 16000 Hz", e.message)
            assertTrue(fake.conversions.items.isEmpty(), "nothing is converted")
            assertTrue(fake.transcribes.items.isEmpty())
            engine.close()
        }
    }

    @Test
    fun streamWhoseResamplerBoundIsUnrepresentableEndsWithInvalidInput() = runTest {
        val fake = FakeWhisperBridge()
        val bridge = object : NativeWhisperBridge by fake {
            override fun resamplerMaxOutput(resampler: Long, frames: Int): Long = -1
        }
        val engine = open(bridge)
        val events = withContext(Dispatchers.Default) {
            withTimeout(10.seconds) { engine.transcribeStream(flowOf(FloatArray(4_800)), sampleRate = 48_000).toList() }
        }
        val error = assertIs<TranscriptionEvent.Error>(events.single())
        assertIs<InvalidInputException>(error.exception)
        assertEquals("cannot resample 4800 frames at 48000 Hz", error.exception.message)
        awaitCondition { fake.liveStateCount == 0 && fake.freedResamplers.items.size == 1 }
        assertEquals(fake.createdResamplers.items, fake.freedResamplers.items)
        engine.close()
    }

    @Test
    fun pcm16StreamOnAClosedEngineFailsImmediately() = runTest {
        val engine = open(FakeWhisperBridge())
        engine.close()
        assertFailsWith<IllegalStateException> { engine.transcribeStream(flowOf(ShortArray(10))) }
    }

    // ---- stream slots ---------------------------------------------------------------------------------

    @Test
    fun streamSlotCreatesItsStateLazilyOnceAndReleasesItIdempotently() {
        val bridge = FakeWhisperBridge()
        val context = bridge.initFromFile("/m.bin", WhisperBackendKind.CPU, 0, false).handle
        var closedCallbacks = 0
        val slot = StreamSlot(maxPromptTokens = 7) { closedCallbacks++ }
        assertEquals(7, slot.maxPromptTokens)
        assertFalse(slot.hasState)
        val state = slot.ensureState(bridge, context)
        assertTrue(slot.hasState)
        assertEquals(state, slot.ensureState(bridge, context), "an existing state is reused")
        assertEquals(listOf(state), bridge.createdStates)
        slot.release(bridge)
        assertFalse(slot.hasState)
        slot.release(bridge) // no double free (the fake would throw)
        assertEquals(listOf(state), bridge.freedStates)
        slot.onEngineClosed()
        assertEquals(1, closedCallbacks)
    }

    @Test
    fun streamSlotStateAllocationFailureIsTypedAndLeavesNoState() {
        val bridge = FakeWhisperBridge().apply { stateCreateStatus = WhisperStatus.OUT_OF_MEMORY }
        val context = bridge.initFromFile("/m.bin", WhisperBackendKind.CPU, 0, false).handle
        val slot = StreamSlot(1) {}
        val e = assertFailsWith<OutOfMemoryException> { slot.ensureState(bridge, context) }
        assertTrue(e.message!!.startsWith("kggml_whisper_state_create failed"), e.message)
        assertFalse(slot.hasState)
    }

    // ---- VAD value types ------------------------------------------------------------------------------

    @Test
    fun vadThresholdMustBeFiniteAndPositive() {
        for (bad in listOf(Float.POSITIVE_INFINITY, 0f, -0.1f)) {
            assertFailsWith<IllegalArgumentException>("$bad") { VadConfig(speechThreshold = bad) }
        }
    }

    @Test
    fun speechTurnsEnforceTheOneSecondFloorAndAudioRange() {
        val floor = WindowFloor.MIN_WINDOW_SAMPLES
        assertFailsWith<IllegalStateException> { SpeechTurn(FloatArray(floor - 1), 1, 0, TurnEndReason.PAUSE) }
        assertFailsWith<IllegalStateException> { SpeechTurn(FloatArray(floor), 0, 0, TurnEndReason.PAUSE) }
        assertFailsWith<IllegalStateException> { SpeechTurn(FloatArray(floor), floor + 1, 0, TurnEndReason.PAUSE) }
        val padded = SpeechTurn(FloatArray(floor), 8_000, 160, TurnEndReason.FLUSH)
        assertTrue(padded.isPadded)
        assertEquals(500L, padded.audioDurationMs)
        assertEquals("SpeechTurn(start=160, audioSamples=8000, size=$floor, reason=FLUSH)", padded.toString())
        assertFalse(SpeechTurn(FloatArray(floor), floor, 0, TurnEndReason.MAX_LENGTH).isPadded)
    }

    // ---- bridge helpers -------------------------------------------------------------------------------

    @Test
    fun backendKindsMapBothWays() {
        for (backend in listOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_METAL, ComputeBackend.GPU_VULKAN, ComputeBackend.NPU_COREML)) {
            assertEquals(backend, WhisperBackendKind.toComputeBackend(WhisperBackendKind.of(backend)))
        }
        assertEquals(WhisperBackendKind.COREML, WhisperBackendKind.of(ComputeBackend.NPU_COREML))
        assertNull(WhisperBackendKind.toComputeBackend(42))
        assertFailsWith<IllegalStateException> { WhisperBackendKind.of(ComputeBackend.AUTO) }
    }

    @Test
    fun nativeErrorsAreTypedAndCarryStatusNameAndDiagnostic() {
        val bridge = FakeWhisperBridge()
        val plain = assertIs<InvalidInputException>(bridge.nativeError(WhisperStatus.INVALID_ARGUMENT, "kggml_op"))
        assertEquals("kggml_op failed: KGGML_STATUS_2 (2)", plain.message)
        bridge.lastErrorMessage = "detail"
        val unavailable = assertIs<BackendUnavailableException>(bridge.nativeError(WhisperStatus.BACKEND_UNAVAILABLE, "kggml_op"))
        assertNull(unavailable.backend)
        assertEquals("kggml_op failed: KGGML_STATUS_3 (3): detail", unavailable.message)

        val broken = object : NativeWhisperBridge by FakeWhisperBridge() {
            override fun lastError(): String = throw IllegalStateException("no diagnostic")
            override fun statusName(status: Int): String = throw IllegalStateException("no name")
        }
        val failure = assertIs<NativeFailureException>(broken.nativeError(77, "kggml_op"), "unknown codes are native failures")
        assertEquals("kggml_op failed: KGGML_STATUS_77 (77)", failure.message)
    }

    @Test
    fun bridgeHolderCreatesOnceAndPassesTypedFailuresThrough() {
        var created = 0
        val bridge = FakeWhisperBridge()
        val holder = WhisperBridgeHolder { created++; bridge }
        assertSame(bridge, holder.get())
        assertSame(bridge, holder.get())
        assertEquals(1, created)

        val typed = ModelLoadException("already typed")
        assertSame(typed, assertFailsWith<ModelLoadException> { WhisperBridgeHolder { throw typed }.get() })

        val wrapped = assertFailsWith<NativeFailureException> { WhisperBridgeHolder { throw IllegalStateException() }.get() }
        assertEquals("Failed to load the Kggml whisper native library: IllegalStateException", wrapped.message)
        assertIs<IllegalStateException>(wrapped.cause)
    }
}
