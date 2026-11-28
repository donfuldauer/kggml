package com.donfuldauer.kggml.diffusion

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.BackendUnavailableException
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.NativeFailureException
import com.donfuldauer.kggml.core.OutOfMemoryException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.AtomicInt
import kotlin.coroutines.CoroutineContext
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

class DiffusionEngineTest {
    private val params = DiffusionGenerationParams(prompt = "a lighthouse at dusk", width = 64, height = 32, sampleSteps = 4)

    private suspend fun open(
        bridge: FakeDiffusionBridge,
        config: BackendConfig = BackendConfig.CPU,
        available: Set<ComputeBackend> = setOf(ComputeBackend.CPU_ONLY),
        options: DiffusionEngineOptions = DiffusionEngineOptions(),
        model: DiffusionModelFiles = DiffusionModelFiles(modelPath = "/models/sd15.gguf"),
        dispatcherFactory: SerialDispatcherFactory = SerialDispatcherFactory.Default,
    ): DiffusionEngine = DiffusionEngine.open(
        model = model,
        backendConfig = config,
        options = options,
        bridges = DiffusionBridgeHolder { bridge },
        availableBackends = { available },
        dispatcherFactory = dispatcherFactory,
    )

    /** Polls [condition] in real time (engines run on real dedicated threads). */
    private suspend fun awaitCondition(condition: () -> Boolean) = withContext(Dispatchers.Default) {
        withTimeout(10.seconds) { while (!condition()) delay(1) }
    }

    private suspend fun DiffusionEngine.events(p: DiffusionGenerationParams = params) =
        withContext(Dispatchers.Default) { generate(p).toList() }

    /**
     * Blocks the fake sampler at [step] until an abort is latched (like a long denoising step) and returns a signal
     * that becomes `true` once [step] has been entered (after its progress callback and after the pre-loop abort
     * check).
     */
    private fun FakeDiffusionBridge.blockAtStep(step: Int): AtomicBoolean {
        val entered = AtomicBoolean(false)
        onStep = { b, ctx, s ->
            if (s == step) {
                entered.store(true)
                while (!b.abortRequested(ctx)) { /* spin */ }
            }
        }
        return entered
    }

    // ---- lifecycle & configuration ----------------------------------------------------------------------

    @Test
    fun cpuConfigMapsThreadsFlagsAndModelPaths() = runTest {
        val bridge = FakeDiffusionBridge()
        val config = BackendConfig(ComputeBackend.CPU_ONLY, gpuLayers = 0, cpuThreads = 3, useFlashAttention = true, useMmap = false)
        val model = DiffusionModelFiles(diffusionModelPath = "/m/flux.gguf", vaePath = "/m/ae.safetensors", clipLPath = "/m/clip_l.safetensors", t5xxlPath = "/m/t5.gguf", llmPath = "")
        val options = DiffusionEngineOptions(gpuDevice = 2, keepTextEncoderOnCpu = true, keepVaeOnCpu = true, weightType = 1)
        open(bridge, config, options = options, model = model).use { engine ->
            val raw = bridge.creates.items.single()
            assertEquals(listOf(null, "/m/flux.gguf", "/m/ae.safetensors", null, "/m/clip_l.safetensors", null, "/m/t5.gguf", null), raw.paths)
            assertEquals(DiffusionBackendKind.CPU, raw.backend)
            assertEquals(2, raw.gpuDevice)
            assertEquals(3, raw.threads)
            assertFalse(raw.useMmap)
            assertTrue(raw.flashAttention)
            assertFalse(raw.offloadParamsToCpu, "offload is meaningless on the CPU backend")
            assertTrue(raw.keepClipOnCpu)
            assertTrue(raw.keepVaeOnCpu)
            assertTrue(raw.vaeDecodeOnly, "txt2img only")
            assertEquals(1, raw.weightType)
            assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
            assertFalse(engine.usesGpu)
            assertEquals(DiffusionModelInfo(true, DiffusionSampler.EULER_A, 4), engine.modelInfo)
            assertEquals(model, engine.model)
            assertEquals(config, engine.backendConfig)
        }
    }

    @Test
    fun autoThreadsAndGpuLayersMapToNativeDefaultsAndOffload() = runTest {
        val all = FakeDiffusionBridge()
        open(all, BackendConfig(ComputeBackend.GPU_VULKAN), available = setOf(ComputeBackend.GPU_VULKAN)).use { engine ->
            val raw = all.creates.items.single()
            assertEquals(DiffusionBackendKind.VULKAN, raw.backend)
            assertEquals(0, raw.threads, "AUTO_THREADS -> native picks performance cores")
            assertFalse(raw.offloadParamsToCpu)
            assertEquals(ComputeBackend.GPU_VULKAN, engine.activeBackend)
            assertTrue(engine.usesGpu)
        }
        val none = FakeDiffusionBridge()
        open(none, BackendConfig(ComputeBackend.GPU_METAL, gpuLayers = 0), available = setOf(ComputeBackend.GPU_METAL)).use {
            assertTrue(none.creates.items.single().offloadParamsToCpu, "gpuLayers = 0 keeps the weights in host memory")
        }
    }

    @Test
    fun autoResolvesToTheBestAvailableBackend() = runTest {
        val bridge = FakeDiffusionBridge()
        open(bridge, BackendConfig.DEFAULT, available = setOf(ComputeBackend.GPU_METAL, ComputeBackend.CPU_ONLY)).use { engine ->
            assertEquals(DiffusionBackendKind.METAL, bridge.creates.items.single().backend)
            assertEquals(ComputeBackend.GPU_METAL, engine.activeBackend)
        }
    }

    @Test
    fun unavailableNativeBackendFallsBackToCpuWhenAllowed() = runTest {
        val bridge = FakeDiffusionBridge(createStatus = { if (it == DiffusionBackendKind.CPU) DiffusionStatus.OK else DiffusionStatus.BACKEND_UNAVAILABLE })
        open(bridge, BackendConfig(ComputeBackend.GPU_VULKAN), available = setOf(ComputeBackend.GPU_VULKAN)).use { engine ->
            assertEquals(listOf(DiffusionBackendKind.VULKAN, DiffusionBackendKind.CPU), bridge.creates.items.map { it.backend })
            assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
        }
    }

    @Test
    fun coreMlWithoutFallbackIsBackendUnavailable() = runTest {
        val bridge = FakeDiffusionBridge()
        val e = assertFailsWith<BackendUnavailableException> {
            open(bridge, BackendConfig(ComputeBackend.NPU_COREML), options = DiffusionEngineOptions(allowBackendFallback = false))
        }
        assertEquals(ComputeBackend.NPU_COREML, e.backend)
        assertTrue(bridge.creates.items.isEmpty(), "rejected before any native call")
    }

    @Test
    fun nativeUnavailableWithoutFallbackIsTyped() = runTest {
        val bridge = FakeDiffusionBridge(createStatus = { DiffusionStatus.BACKEND_UNAVAILABLE })
        val e = assertFailsWith<BackendUnavailableException> {
            open(bridge, BackendConfig(ComputeBackend.GPU_VULKAN), available = setOf(ComputeBackend.GPU_VULKAN), options = DiffusionEngineOptions(allowBackendFallback = false))
        }
        assertEquals(ComputeBackend.GPU_VULKAN, e.backend)
        assertTrue(e.message!!.contains("kggml_diffusion_context_create"), e.message)
    }

    @Test
    fun loadFailuresMapToTypedExceptionsAndLeakNothing() = runTest {
        val missing = FakeDiffusionBridge(createStatus = { DiffusionStatus.MODEL_LOAD_FAILED })
        val e = assertFailsWith<ModelLoadException> { open(missing) }
        assertTrue(e.message!!.contains("create failed"), "native diagnostic attached: ${e.message}")
        assertFailsWith<OutOfMemoryException> { open(FakeDiffusionBridge(createStatus = { DiffusionStatus.OUT_OF_MEMORY })) }
        assertFailsWith<NativeFailureException> { open(FakeDiffusionBridge(createStatus = { 102 })) }

        val notTxt2Img = FakeDiffusionBridge(supportsImageGeneration = false)
        assertFailsWith<ModelLoadException> { open(notTxt2Img) }
        awaitCondition { notTxt2Img.live.isEmpty() }
        assertEquals(1, notTxt2Img.frees.items.size)
    }

    @Test
    fun bridgeLoadFailureIsANativeFailure() = runTest {
        val e = assertFailsWith<NativeFailureException> {
            DiffusionEngine.open(
                DiffusionModelFiles(modelPath = "/m.gguf"), BackendConfig.CPU, DiffusionEngineOptions(),
                DiffusionBridgeHolder { throw IllegalStateException("no libkggml_diffusion_jni") },
                { setOf(ComputeBackend.CPU_ONLY) }, SerialDispatcherFactory.Default,
            )
        }
        assertTrue(e.message!!.startsWith("Failed to load the Kggml diffusion native library"), e.message)
    }

    @Test
    fun closeFreesTheContextOnceAndIsIdempotent() = runTest {
        val bridge = FakeDiffusionBridge()
        val engine = open(bridge)
        engine.events()
        assertEquals(1, bridge.live.size)
        engine.close()
        engine.close()
        assertTrue(engine.isClosed)
        awaitCondition { bridge.live.isEmpty() }
        assertEquals(1, bridge.frees.items.size)
        assertFailsWith<IllegalStateException> { engine.generate(params) }
        assertFailsWith<IllegalStateException> { engine.generateImage(params) }
        assertFalse(engine.abortActiveGeneration())
    }

    @Test
    fun enginesAreIndependentInstancesThatGenerateConcurrently() = runTest {
        val bridge = FakeDiffusionBridge()
        val engines = List(3) { open(bridge) }
        try {
            assertEquals(3, bridge.live.size)
            val images = withContext(Dispatchers.Default) {
                engines.mapIndexed { i, engine -> async { engine.generateImage(params.copy(seed = i.toLong())) } }.awaitAll()
            }
            images.forEachIndexed { i, image -> assertContentEquals(FakeDiffusionBridge.gradient(64, 32, 3, i.toLong()), image.pixels) }
            assertEquals(3, bridge.generates.items.map { it.context }.toSet().size, "each engine used its own context")
            engines[0].close()
            assertEquals(params.width, engines[1].generateImage(params).width, "closing one engine leaves the others usable")
        } finally {
            engines.forEach { it.close() }
        }
        awaitCondition { bridge.live.isEmpty() }
    }

    // ---- generation -------------------------------------------------------------------------------------

    @Test
    fun generationStreamsProgressThenTheImageAndHonoursEveryParameter() = runTest {
        val bridge = FakeDiffusionBridge()
        val p = DiffusionGenerationParams(
            prompt = "a red fox, watercolor",
            negativePrompt = "blurry",
            width = 64,
            height = 48,
            sampleSteps = 3,
            cfgScale = 5.5f,
            distilledGuidance = 2.0f,
            seed = 42L,
            sampler = DiffusionSampler.DPMPP_2M,
            scheduler = DiffusionScheduler.KARRAS,
            clipSkip = 2,
            vaeTiling = true,
        )
        open(bridge).use { engine ->
            val events = engine.events(p)
            val raw = bridge.generates.items.single().params
            assertEquals(
                RawGenerateParams("a red fox, watercolor", "blurry", 64, 48, 3, 5.5f, 2.0f, 5, 1, 42L, 2, true),
                raw,
            )
            assertTrue(bridge.generates.items.single().hadListener)
            assertEquals(1, bridge.clears.load(), "the abort latch is cleared before generating")
            val progress = events.dropLast(1).map { assertIs<DiffusionEvent.Progress>(it) }
            assertEquals(listOf(0, 1, 2, 3), progress.map { it.step })
            assertTrue(progress.all { it.totalSteps == 3 && it.phase == DiffusionPhase.DENOISING && it.secondsPerStep == 0.25f })
            val image = assertIs<DiffusionEvent.Completed>(events.last()).image
            assertEquals(64, image.width)
            assertEquals(48, image.height)
            assertEquals(3, image.channels)
            assertContentEquals(FakeDiffusionBridge.gradient(64, 48, 3, 42L), image.pixels)
            assertEquals(1, bridge.imagesAllocated.load())
            assertEquals(1, bridge.imagesFreed.load())
            assertFalse(engine.isGenerating)
        }
    }

    @Test
    fun randomSeedAndModelDefaultsReachNativeAsSentinels() = runTest {
        val bridge = FakeDiffusionBridge()
        open(bridge).use { engine ->
            engine.generateImage(DiffusionGenerationParams(prompt = "x", width = 8, height = 8, sampleSteps = 1, seed = -1234L))
            val raw = bridge.generates.items.single().params
            assertEquals(-1L, raw.seed)
            assertEquals(-1, raw.sampleMethod)
            assertEquals(-1, raw.scheduler)
            assertEquals(-1, raw.clipSkip)
            assertEquals("", raw.negativePrompt)
        }
    }

    @Test
    fun tiledVaeProgressIsReportedAsItsOwnPhase() = runTest {
        val bridge = FakeDiffusionBridge(vaeTiles = 2)
        open(bridge).use { engine ->
            val progress = engine.events(params.copy(sampleSteps = 2, vaeTiling = true)).filterIsInstance<DiffusionEvent.Progress>()
            assertEquals(
                listOf(DiffusionPhase.DENOISING to 0, DiffusionPhase.DENOISING to 1, DiffusionPhase.DENOISING to 2, DiffusionPhase.VAE_TILES to 0, DiffusionPhase.VAE_TILES to 1, DiffusionPhase.VAE_TILES to 2),
                progress.map { it.phase to it.step },
            )
        }
    }

    @Test
    fun generateImageReportsProgressAndReturnsTheImage() = runTest {
        val bridge = FakeDiffusionBridge()
        open(bridge).use { engine ->
            val steps = mutableListOf<Int>()
            val image = withContext(Dispatchers.Default) { engine.generateImage(params) { steps += it.step } }
            assertEquals(listOf(0, 1, 2, 3, 4), steps)
            assertEquals(DiffusionImage(64, 32, 3, FakeDiffusionBridge.gradient(64, 32, 3, -1L)), image)
        }
    }

    @Test
    fun nativeGenerationFailuresEndTheFlowWithATypedErrorAndTheEngineStaysUsable() = runTest {
        val bridge = FakeDiffusionBridge(generateStatus = DiffusionStatus.NATIVE_FAILURE)
        open(bridge).use { engine ->
            val error = assertIs<DiffusionEvent.Error>(engine.events().last())
            val e = assertIs<NativeFailureException>(error.exception)
            assertTrue(e.message!!.contains("kggml_diffusion_generate") && e.message!!.contains("forced failure"), e.message)
            bridge.generateStatus = DiffusionStatus.OUT_OF_MEMORY
            assertFailsWith<OutOfMemoryException> { engine.generateImage(params) }
            bridge.generateStatus = DiffusionStatus.OK
            assertEquals(64, engine.generateImage(params).width)
        }
    }

    @Test
    fun malformedNativeImagesAreRejectedAsNativeFailures() = runTest {
        val bridge = FakeDiffusionBridge(imageDims = { Triple(it.width, it.height, 4) })
        open(bridge).use { engine ->
            assertFailsWith<NativeFailureException> { engine.generateImage(params) }
            bridge.imageDims = { Triple(it.width * 2, it.height, 3) }
            assertFailsWith<NativeFailureException> { engine.generateImage(params) }
            assertEquals(bridge.imagesAllocated.load(), bridge.imagesFreed.load())
        }
    }

    // ---- cancellation -------------------------------------------------------------------------------------

    @Test
    fun cancellingTheCollectorAbortsAtTheNextStepBoundary() = runTest {
        val bridge = FakeDiffusionBridge()
        val entered = bridge.blockAtStep(2)
        open(bridge).use { engine ->
            val received = ConcurrentLog<DiffusionEvent>()
            val job = launch(Dispatchers.Default) { engine.generate(params).collect { received.add(it) } }
            awaitCondition { engine.isGenerating && entered.load() && received.items.size == 3 }
            job.cancelAndJoin()
            assertEquals(listOf(0, 1, 2), received.items.map { assertIs<DiffusionEvent.Progress>(it).step })
            assertEquals(listOf(2), bridge.cancelledAtStep.items, "stopped right after the step that was computing")
            assertFalse(engine.isGenerating)
            assertEquals(0, bridge.imagesAllocated.load(), "no image is produced for an aborted generation")
            // The latched abort is cleared for the next generation; the engine stays usable.
            bridge.onStep = { _, _, _ -> }
            assertEquals(64, engine.generateImage(params).width)
            assertEquals(1, bridge.imagesFreed.load())
        }
    }

    @Test
    fun cancellingGenerateImageThrowsCancellationException() = runTest {
        val bridge = FakeDiffusionBridge()
        val entered = bridge.blockAtStep(1)
        open(bridge).use { engine ->
            val result = async(Dispatchers.Default) { engine.generateImage(params) }
            awaitCondition { engine.isGenerating && entered.load() }
            result.cancel()
            assertFailsWith<CancellationException> { result.await() }
            awaitCondition { !engine.isGenerating && bridge.cancelledAtStep.items == listOf(1) }
        }
    }

    @Test
    fun abortActiveGenerationEndsTheFlowWithACancelledError() = runTest {
        val bridge = FakeDiffusionBridge()
        val entered = bridge.blockAtStep(0)
        open(bridge).use { engine ->
            assertFalse(engine.abortActiveGeneration(), "nothing to abort while idle")
            val events = async(Dispatchers.Default) { engine.generate(params).toList() }
            awaitCondition { engine.isGenerating && entered.load() }
            assertTrue(engine.abortActiveGeneration())
            val all = events.await()
            assertEquals(DiffusionEvent.Progress(DiffusionPhase.DENOISING, 0, 4, 0.25f), all.first())
            val error = assertIs<DiffusionEvent.Error>(all.last())
            assertIs<InferenceCancelledException>(error.exception)
            assertTrue(error.exception.message!!.contains("aborted at progress step 0/4"), error.exception.message)
        }
    }

    @Test
    fun abortRequestedBeforeTheCollectorStartsIsNotInherited() = runTest {
        val bridge = FakeDiffusionBridge()
        open(bridge).use { engine ->
            // A stale latch (e.g. from a native abort that arrived after the previous generation returned).
            bridge.requestAbort(bridge.live.single())
            assertEquals(64, engine.generateImage(params).width)
        }
    }

    @Test
    fun closeDuringGenerationEndsTheFlowAndFreesTheContext() = runTest {
        val bridge = FakeDiffusionBridge()
        val entered = bridge.blockAtStep(1)
        val engine = open(bridge)
        val events = async(Dispatchers.Default) { engine.generate(params).toList() }
        awaitCondition { engine.isGenerating && entered.load() }
        engine.close()
        val error = assertIs<DiffusionEvent.Error>(events.await().last())
        val e = assertIs<InferenceCancelledException>(error.exception)
        assertTrue(e.message!!.contains("closed"), e.message)
        awaitCondition { bridge.live.isEmpty() }
        assertEquals(0, bridge.imagesAllocated.load())
    }

    @Test
    fun generationsQueuedBehindARunningOneFailOnceTheEngineCloses() = runTest {
        val bridge = FakeDiffusionBridge()
        val entered = bridge.blockAtStep(1)
        val engine = open(bridge)
        val running = async(Dispatchers.Default) { engine.generate(params).toList() }
        awaitCondition { engine.isGenerating && entered.load() }
        val queued = async(Dispatchers.Default) { runCatching { engine.generate(params).toList() } }
        delay(50)
        engine.close()
        assertIs<DiffusionEvent.Error>(running.await().last())
        assertIs<IllegalStateException>(queued.await().exceptionOrNull())
        assertEquals(1, bridge.generates.items.size, "the queued generation never reached native code")
        awaitCondition { bridge.live.isEmpty() }
    }

    @Test
    fun firstProgressEventCancelsTheProducerWithoutLeaking() = runTest {
        val bridge = FakeDiffusionBridge()
        open(bridge).use { engine ->
            val first = withContext(Dispatchers.Default) { engine.generate(params.copy(sampleSteps = 200)).first() }
            assertEquals(0, assertIs<DiffusionEvent.Progress>(first).step)
            awaitCondition { !engine.isGenerating }
            assertEquals(bridge.imagesAllocated.load(), bridge.imagesFreed.load())
            assertEquals(64, engine.generateImage(params).width)
        }
    }

    /**
     * Single-threaded dispatcher that, like Kotlin/Native's `newSingleThreadContext`, cannot run anything once closed:
     * every dispatch after close() is recorded as a violation (and still executed elsewhere so the test cannot hang).
     */
    private class NativeLikeDispatcher : CoroutineDispatcher(), AutoCloseable {
        private val delegate = Dispatchers.Default.limitedParallelism(1)
        val closed = AtomicInt(0)
        val lateDispatches = AtomicInt(0)

        override fun dispatch(context: CoroutineContext, block: Runnable) {
            if (closed.load() != 0) lateDispatches.addAndFetch(1)
            delegate.dispatch(context, block)
        }

        override fun close() {
            closed.store(1)
        }
    }

    @Test
    fun closeFromASlowCollectorNeverStrandsTheGenerationOnAClosedEngineThread() = runTest {
        val bridge = FakeDiffusionBridge()
        val dispatchers = ConcurrentLog<NativeLikeDispatcher>()
        val engine = open(bridge, dispatcherFactory = SerialDispatcherFactory { NativeLikeDispatcher().also(dispatchers::add) })
        // Rendezvous downstream: the final send() of the image may suspend on the engine thread.
        val events = withContext(Dispatchers.Default) {
            engine.generate(params)
                .buffer(Channel.RENDEZVOUS)
                .onEach { if (it is DiffusionEvent.Completed && !engine.isClosed) engine.close() }
                .toList()
        }
        assertIs<DiffusionEvent.Completed>(events.last())
        val dispatcher = dispatchers.items.single()
        awaitCondition { bridge.live.isEmpty() && dispatcher.closed.load() != 0 }
        assertEquals(0, dispatcher.lateDispatches.load(), "the engine thread was shut down under a suspended generation")
        assertFailsWith<IllegalStateException> { engine.generate(params) }
    }

    @Test
    fun nativeInfoComesFromTheBridge() {
        val info = DiffusionEngine.nativeInfo(FakeDiffusionBridge())
        assertEquals(DiffusionNativeInfo("1.0.0", "fake-sd", "0".repeat(40), "fake"), info)
    }
}
