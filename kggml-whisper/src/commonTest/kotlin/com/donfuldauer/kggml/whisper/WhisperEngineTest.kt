package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.BackendUnavailableException
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.NativeFailureException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNotEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

class WhisperEngineTest {
    private val gpuDevices = setOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_METAL, ComputeBackend.GPU_VULKAN)

    private suspend fun open(
        bridge: NativeWhisperBridge,
        config: BackendConfig = BackendConfig.CPU,
        available: Set<ComputeBackend> = setOf(ComputeBackend.CPU_ONLY),
        options: WhisperEngineOptions = WhisperEngineOptions(),
        path: String = "/models/ggml-tiny.bin",
    ): WhisperEngine = WhisperEngine.open(
        modelPath = path,
        backendConfig = config,
        options = options,
        bridges = WhisperBridgeHolder { bridge },
        availableBackends = { available },
        dispatcherFactory = SerialDispatcherFactory.Default,
    )

    /** Polls [condition] in real time (engines run on real dedicated threads). */
    private suspend fun awaitCondition(condition: () -> Boolean) = withContext(Dispatchers.Default) {
        withTimeout(10.seconds) { while (!condition()) delay(1) }
    }

    // ---- lifecycle ------------------------------------------------------------------------------------

    @Test
    fun loadsModelOnCreationAndFreesItExactlyOnceOnClose() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        assertEquals(listOf("/models/ggml-tiny.bin"), bridge.inits.items.map { it.modelPath })
        assertFalse(engine.isClosed)

        engine.close()
        engine.close()
        assertTrue(engine.isClosed)
        awaitCondition { bridge.freedHandles.isNotEmpty() }
        assertEquals(bridge.createdHandles, bridge.freedHandles)
    }

    @Test
    fun callsAfterCloseFailWithIllegalStateException() = runTest {
        val engine = open(FakeWhisperBridge())
        engine.close()
        assertFailsWith<IllegalStateException> { engine.transcribe(FloatArray(16_000)) }
        assertFailsWith<IllegalStateException> { engine.transcribe(ShortArray(16_000)) }
        assertFailsWith<IllegalStateException> { engine.reinitialize(BackendConfig.CPU) }
    }

    @Test
    fun failedModelLoadThrowsModelLoadExceptionWithNativeDiagnostic() = runTest {
        val bridge = FakeWhisperBridge(initStatus = { WhisperStatus.MODEL_LOAD_FAILED })
        val e = assertFailsWith<ModelLoadException> { open(bridge) }
        assertTrue(e.message!!.contains("kggml_whisper_init_from_file"), e.message)
        assertTrue(e.message!!.contains("fake init failure"), e.message)
        assertTrue(bridge.createdHandles.isEmpty())
    }

    @Test
    fun nativeLibraryLoadFailureSurfacesAsNativeFailureException() = runTest {
        val holder = WhisperBridgeHolder { throw UnsupportedOperationException("no lib") }
        val e = assertFailsWith<NativeFailureException> {
            WhisperEngine.open("/m.bin", BackendConfig.CPU, WhisperEngineOptions(), holder, { emptySet() }, SerialDispatcherFactory.Default)
        }
        assertIs<UnsupportedOperationException>(e.cause)
    }

    @Test
    fun emptyModelPathIsRejected() = runTest {
        assertFailsWith<IllegalArgumentException> { open(FakeWhisperBridge(), path = "") }
    }

    // ---- BackendConfig propagation ------------------------------------------------------------------

    @Test
    fun cpuOnlyIsPassedAsCpuBackendSoUseGpuIsFalse() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge, config = BackendConfig.CPU, available = gpuDevices)
        assertEquals(WhisperBackendKind.CPU, bridge.inits.items.single().backendKind)
        assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
        assertFalse(engine.usesGpu)
        engine.close()
    }

    @Test
    fun explicitGpuBackendsReachNativeCode() = runTest {
        for ((backend, kind) in listOf(ComputeBackend.GPU_METAL to WhisperBackendKind.METAL, ComputeBackend.GPU_VULKAN to WhisperBackendKind.VULKAN)) {
            val bridge = FakeWhisperBridge()
            val engine = open(bridge, config = BackendConfig(backend = backend), available = gpuDevices)
            assertEquals(kind, bridge.inits.items.single().backendKind)
            assertEquals(backend, engine.activeBackend)
            assertTrue(engine.usesGpu)
            engine.close()
        }
    }

    @Test
    fun autoResolvesToTheBestAvailableBackend() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge, config = BackendConfig.DEFAULT, available = setOf(ComputeBackend.GPU_VULKAN))
        assertEquals(WhisperBackendKind.VULKAN, bridge.inits.items.single().backendKind)
        assertEquals(ComputeBackend.GPU_VULKAN, engine.activeBackend)
        engine.close()
    }

    @Test
    fun unavailableBackendFallsBackToCpuByDefault() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge, config = BackendConfig(backend = ComputeBackend.GPU_METAL), available = setOf(ComputeBackend.CPU_ONLY))
        assertEquals(WhisperBackendKind.CPU, bridge.inits.items.single().backendKind)
        assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
        engine.close()
    }

    @Test
    fun unavailableBackendWithoutFallbackThrowsBackendUnavailable() = runTest {
        val bridge = FakeWhisperBridge()
        val e = assertFailsWith<BackendUnavailableException> {
            open(
                bridge,
                config = BackendConfig(backend = ComputeBackend.GPU_METAL),
                options = WhisperEngineOptions(allowBackendFallback = false),
            )
        }
        assertEquals(ComputeBackend.GPU_METAL, e.backend)
        assertTrue(bridge.inits.items.isEmpty())
    }

    @Test
    fun nativeBackendUnavailableWalksTheFallbackChain() = runTest {
        // The runtime lists Metal, but whisper cannot create it (e.g. the device disappeared): Metal -> CPU.
        val bridge = FakeWhisperBridge(initStatus = { if (it == WhisperBackendKind.CPU) WhisperStatus.OK else WhisperStatus.BACKEND_UNAVAILABLE })
        val engine = open(bridge, config = BackendConfig(backend = ComputeBackend.NPU_COREML), available = setOf(ComputeBackend.NPU_COREML, ComputeBackend.GPU_METAL))
        assertEquals(listOf(WhisperBackendKind.COREML, WhisperBackendKind.METAL, WhisperBackendKind.CPU), bridge.inits.items.map { it.backendKind })
        assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
        engine.close()
    }

    @Test
    fun nativeBackendUnavailableWithoutFallbackIsTyped() = runTest {
        val bridge = FakeWhisperBridge(initStatus = { WhisperStatus.BACKEND_UNAVAILABLE })
        val e = assertFailsWith<BackendUnavailableException> {
            open(bridge, config = BackendConfig(backend = ComputeBackend.GPU_VULKAN), available = gpuDevices, options = WhisperEngineOptions(allowBackendFallback = false))
        }
        assertEquals(ComputeBackend.GPU_VULKAN, e.backend)
    }

    @Test
    fun coreMlWithoutEncoderBundleReportsTheEffectiveBackend() = runTest {
        val bridge = FakeWhisperBridge(coreMl = true).apply { coreMlEncoderPresent = false }
        val engine = open(bridge, config = BackendConfig(backend = ComputeBackend.NPU_COREML), available = setOf(ComputeBackend.NPU_COREML, ComputeBackend.GPU_METAL))
        assertEquals(WhisperBackendKind.COREML, bridge.inits.items.single().backendKind)
        assertEquals(ComputeBackend.GPU_METAL, engine.activeBackend)
        engine.close()
    }

    @Test
    fun flashAttentionGpuDeviceAndThreadsArePropagated() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(
            bridge,
            config = BackendConfig(backend = ComputeBackend.GPU_VULKAN, cpuThreads = 3, useFlashAttention = true),
            available = gpuDevices,
            options = WhisperEngineOptions(gpuDevice = 2),
        )
        val init = bridge.inits.items.single()
        assertTrue(init.flashAttention)
        assertEquals(2, init.gpuDevice)
        engine.transcribe(FloatArray(16_000))
        assertEquals(3, bridge.transcribes.items.single().params.threads)
        engine.close()
    }

    @Test
    fun autoThreadsMapToNativeDefault() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge, config = BackendConfig(backend = ComputeBackend.CPU_ONLY, gpuLayers = 0))
        engine.transcribe(FloatArray(16_000))
        assertEquals(0, bridge.transcribes.items.single().params.threads)
        engine.close()
    }

    // ---- transcription ------------------------------------------------------------------------------

    @Test
    fun sixteenKhzMonoFloatInputCrossesTheBridgeZeroCopy() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val pcm = FloatArray(24_000) { 0.1f }
        val result = engine.transcribe(pcm)
        val call = bridge.transcribes.items.single()
        assertSame(pcm, call.samples)
        assertEquals(pcm.size, call.count)
        assertTrue(bridge.conversions.items.isEmpty())
        assertEquals(1500L, result.audioDurationMs)
        engine.close()
    }

    @Test
    fun otherRatesAndChannelLayoutsAreConvertedNatively() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val stereo = FloatArray(44_100 * 2) { if (it % 2 == 0) 0.5f else -0.5f }
        val result = engine.transcribe(stereo, sampleRate = 44_100, channels = 2)
        assertEquals(listOf("f32:44100:2:44100"), bridge.conversions.items)
        val call = bridge.transcribes.items.single()
        assertEquals(16_000, call.count)
        assertTrue(call.samples.all { it == 0f }, "stereo channels should be averaged")
        assertEquals(1000L, result.audioDurationMs)
        engine.close()
    }

    @Test
    fun shortArrayInputIsNormalizedNatively() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        engine.transcribe(ShortArray(8_000) { 16_384 }, sampleRate = 8_000)
        assertEquals(listOf("s16:8000:1:8000"), bridge.conversions.items)
        val call = bridge.transcribes.items.single()
        assertEquals(16_000, call.count)
        assertEquals(0.5f, call.samples[100])
        engine.close()
    }

    @Test
    fun segmentsAreDecodedAndMappedToEvents() = runTest {
        val bridge = FakeWhisperBridge(segments = listOf(Triple(0L, 900L, " Grüße"), Triple(900L, 1800L, " 👋")))
        val engine = open(bridge)
        val result = engine.transcribe(FloatArray(32_000))
        assertEquals(listOf(" Grüße", " 👋"), result.segments.map { it.text })
        assertEquals("Grüße 👋", result.text)
        assertEquals(listOf(0, 1), result.segments.map { it.index })
        assertEquals(ComputeBackend.CPU_ONLY, result.backend)
        assertEquals(3, result.toEvents().size)
        engine.close()
    }

    @Test
    fun transcriptionOptionsArePropagated() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val options = TranscriptionOptions(
            language = "de", translate = true, initialPrompt = "Kggml", noContext = false, singleSegment = true,
            noTimestamps = true, maxTokens = 32, temperature = 0.4f, temperatureIncrement = 0f,
        )
        engine.transcribe(FloatArray(16_000), options = options)
        val p = bridge.transcribes.items.single().params
        assertEquals(RawTranscribeParams("de", "Kggml", 0, true, false, true, true, 32, 0.4f, 0f), p)
        engine.close()
    }

    @Test
    fun malformedAudioIsRejectedBeforeReachingNativeCode() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        assertFailsWith<InvalidInputException> { engine.transcribe(FloatArray(3), channels = 2) }
        assertFailsWith<InvalidInputException> { engine.transcribe(FloatArray(4), channels = 0) }
        assertFailsWith<InvalidInputException> { engine.transcribe(FloatArray(4), channels = 9) }
        assertFailsWith<InvalidInputException> { engine.transcribe(FloatArray(4), sampleRate = 999) }
        assertFailsWith<InvalidInputException> { engine.transcribe(ShortArray(4), sampleRate = 400_000) }
        assertTrue(bridge.transcribes.items.isEmpty())
        engine.close()
    }

    @Test
    fun nativeTranscriptionErrorsAreTyped() = runTest {
        val bridge = FakeWhisperBridge(transcribeStatus = WhisperStatus.INVALID_ARGUMENT)
        val engine = open(bridge)
        val e = assertFailsWith<InvalidInputException> { engine.transcribe(FloatArray(16_000)) }
        assertTrue(e.message!!.contains("fake transcribe failure"), e.message)
        bridge.transcribeStatus = WhisperStatus.OK
        engine.transcribe(FloatArray(16_000)) // still usable
        engine.close()
    }

    @Test
    fun emptyInputIsForwardedForNativePadding() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val result = engine.transcribe(FloatArray(0))
        assertEquals(0, bridge.transcribes.items.single().count)
        assertEquals(0L, result.audioDurationMs)
        engine.close()
    }

    // ---- cancellation -------------------------------------------------------------------------------

    @Test
    fun cancellingTheCallerAbortsNativeDecodingAndKeepsTheEngineUsable() = runTest {
        val bridge = FakeWhisperBridge()
        bridge.blockTranscription.store(true)
        val engine = open(bridge)
        val job = launch(Dispatchers.Default) { engine.transcribe(FloatArray(16_000)) }
        awaitCondition { bridge.transcriptionStarted.load() == 1 }
        assertTrue(engine.isTranscribing)

        job.cancelAndJoin()
        assertTrue(job.isCancelled)
        assertEquals(bridge.createdHandles, bridge.abortRequests.items.distinct())
        assertFalse(engine.isTranscribing)

        // The latched abort is cleared before the next call.
        bridge.blockTranscription.store(false)
        val result = engine.transcribe(FloatArray(16_000))
        assertEquals(2, result.segments.size)
        engine.close()
    }

    @Test
    fun cancellationSurfacesAsCancellationException() = runTest {
        val bridge = FakeWhisperBridge()
        bridge.blockTranscription.store(true)
        val engine = open(bridge)
        val deferred = async(Dispatchers.Default) { engine.transcribe(FloatArray(16_000)) }
        awaitCondition { bridge.transcriptionStarted.load() == 1 }
        deferred.cancel()
        assertFailsWith<CancellationException> { deferred.await() }
        engine.close()
    }

    @Test
    fun abortActiveTranscriptionFailsTheRunningCallOnly() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        assertFalse(engine.abortActiveTranscription(), "nothing running")
        bridge.blockTranscription.store(true)
        val deferred = async(Dispatchers.Default) { runCatching { engine.transcribe(FloatArray(16_000)) } }
        awaitCondition { bridge.transcriptionStarted.load() == 1 }
        assertTrue(engine.abortActiveTranscription())
        assertIs<InferenceCancelledException>(deferred.await().exceptionOrNull())
        engine.close()
    }

    @Test
    fun closeDuringTranscriptionAbortsItAndFreesAfterwards() = runTest {
        val bridge = FakeWhisperBridge()
        bridge.blockTranscription.store(true)
        val engine = open(bridge)
        val deferred = async(Dispatchers.Default) { runCatching { engine.transcribe(FloatArray(16_000)) } }
        awaitCondition { bridge.transcriptionStarted.load() == 1 }
        engine.close()
        val e = assertIs<InferenceCancelledException>(deferred.await().exceptionOrNull())
        assertTrue(e.message!!.contains("closed"), e.message)
        awaitCondition { bridge.freedHandles.isNotEmpty() }
        assertEquals(bridge.createdHandles, bridge.freedHandles)
    }

    @Test
    fun alreadyCancelledCallerNeverStartsNativeWork() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val job = launch(Dispatchers.Default, start = CoroutineStart.LAZY) { engine.transcribe(FloatArray(16_000)) }
        job.cancel()
        job.join()
        assertTrue(bridge.transcribes.items.isEmpty())
        engine.close()
    }

    // ---- multiple instances -------------------------------------------------------------------------

    @Test
    fun independentInstancesTranscribeConcurrentlyWithSeparateContexts() = runTest {
        val bridge = FakeWhisperBridge()
        val engines = List(3) { open(bridge) }
        assertEquals(3, bridge.createdHandles.distinct().size)

        bridge.blockTranscription.store(true)
        val jobs = engines.map { engine -> async(Dispatchers.Default) { runCatching { engine.transcribe(FloatArray(16_000)) } } }
        // All three are inside native code at the same time: no shared lock or thread between engines.
        awaitCondition { bridge.transcriptionStarted.load() == 3 }
        assertEquals(3, bridge.maxConcurrentTranscriptions)
        engines.forEach { assertTrue(it.abortActiveTranscription()) }
        jobs.awaitAll().forEach { assertIs<InferenceCancelledException>(it.exceptionOrNull()) }

        // Cancelling or closing one engine does not affect another.
        bridge.blockTranscription.store(false)
        engines[0].close()
        assertEquals(2, engines[1].transcribe(FloatArray(16_000)).segments.size)
        engines.drop(1).forEach { it.close() }
        awaitCondition { bridge.freedHandles.size == 3 }
        assertEquals(bridge.createdHandles.toSet(), bridge.freedHandles.toSet())
    }

    @Test
    fun callsOnOneInstanceAreSerialized() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        List(8) { async(Dispatchers.Default) { engine.transcribe(FloatArray(16_000)) } }.awaitAll()
        assertEquals(1, bridge.maxConcurrentTranscriptions)
        assertEquals(8, bridge.transcribes.items.size)
        engine.close()
    }

    // ---- reinitialize -------------------------------------------------------------------------------

    @Test
    fun reinitializeFreesTheOldContextBeforeLoadingTheNewOne() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge, config = BackendConfig(backend = ComputeBackend.GPU_METAL), available = gpuDevices)
        val first = bridge.createdHandles.single()
        assertTrue(engine.usesGpu)

        assertEquals(ComputeBackend.CPU_ONLY, engine.reinitialize(BackendConfig.CPU))
        assertEquals(listOf(first), bridge.freedHandles)
        assertEquals(WhisperBackendKind.CPU, bridge.inits.items.last().backendKind)
        assertFalse(engine.usesGpu)
        assertEquals(BackendConfig.CPU, engine.backendConfig)
        engine.transcribe(FloatArray(16_000))
        assertNotEquals(first, bridge.transcribes.items.single().handle)
        engine.close()
    }

    @Test
    fun failedReinitializeRestoresThePreviousConfiguration() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge, config = BackendConfig.CPU, available = gpuDevices)
        bridge.initStatus = { if (it == WhisperBackendKind.VULKAN) WhisperStatus.OUT_OF_MEMORY else WhisperStatus.OK }
        assertFailsWith<com.donfuldauer.kggml.core.OutOfMemoryException> {
            engine.reinitialize(BackendConfig(backend = ComputeBackend.GPU_VULKAN))
        }
        assertEquals(BackendConfig.CPU, engine.backendConfig)
        assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
        engine.transcribe(FloatArray(16_000))
        engine.close()
    }

    @Test
    fun nativeInfoIsReadFromTheBridge() {
        val info = WhisperEngine.nativeInfo(FakeWhisperBridge(coreMl = true))
        assertEquals("1.9.6", info.whisperCppVersion)
        assertEquals(40, info.whisperCppCommit.length)
        assertTrue(info.hasCoreMl)
    }

    @Test
    fun optionsAreValidated() {
        assertFailsWith<IllegalArgumentException> { WhisperEngineOptions(gpuDevice = -1) }
        assertFailsWith<IllegalArgumentException> { TranscriptionOptions(maxTokens = -1) }
        assertFailsWith<IllegalArgumentException> { TranscriptionOptions(temperature = -0.1f) }
        assertFailsWith<IllegalArgumentException> { TranscriptionOptions(language = " ") }
    }
}
