package com.donfuldauer.kggml.samples.showcase.diffusion

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.ManualMemoryPressureObserver
import com.donfuldauer.kggml.core.MemoryPressureLevel
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.OutOfMemoryException
import com.donfuldauer.kggml.samples.showcase.models.ResidentModels
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

@OptIn(ExperimentalCoroutinesApi::class)
class DiffusionStudioControllerTest {
    private class Harness(
        scope: TestScope,
        val factory: FakeImageGeneratorFactory = FakeImageGeneratorFactory(),
        val observer: ManualMemoryPressureObserver = ManualMemoryPressureObserver(),
        val writeFailure: Throwable? = null,
        modelPath: String = "/models/sd15.gguf",
    ) {
        val written = LinkedHashMap<String, ByteArray>()
        val residents = ResidentModels(1L shl 40, observer)
        val controller = DiffusionStudioController(
            scope = CoroutineScope(StandardTestDispatcher(scope.testScheduler) + Job()),
            factory = factory,
            residents = residents,
            ioDispatcher = StandardTestDispatcher(scope.testScheduler),
            writeFile = { path, bytes ->
                writeFailure?.let { throw it }
                written[path] = bytes
            },
            exportDirectory = "/exports",
            initialModelPath = modelPath,
            random = Random(7),
        )
        val state get() = controller.state.value
        val generator get() = factory.opened.last()
    }

    @Test
    fun parametersAreClampedAndValidated() = runTest {
        val h = Harness(this, modelPath = "")
        assertFalse(h.state.canGenerate, "no model")
        h.controller.setModelPath("/m.gguf")
        assertTrue(h.state.canGenerate)
        h.controller.setSteps(0)
        assertEquals(1, h.state.steps)
        h.controller.setSteps(500)
        assertEquals(DiffusionStudioController.MAX_STEPS, h.state.steps)
        h.controller.setCfgScale(7.3f)
        assertEquals(7.5f, h.state.cfgScale)
        h.controller.setCfgScale(0f)
        assertEquals(1f, h.state.cfgScale)
        h.controller.setSeedText("12a3")
        assertEquals("123", h.state.seedText)
        h.controller.setSeedText("")
        assertFalse(h.state.canGenerate, "no seed")
        h.controller.setRandomSeed(true)
        assertTrue(h.state.canGenerate, "random seed needs no seed text")
        h.controller.setPrompt("   ")
        assertFalse(h.state.canGenerate)
    }

    @Test
    fun generationStreamsProgressThenShowsTheImage() = runTest {
        val gate = Channel<Unit>(Channel.UNLIMITED)
        val h = Harness(this, FakeImageGeneratorFactory { it.stepGate = gate })
        h.controller.setSteps(4)
        h.controller.setSize(256)
        h.controller.setCfgScale(5f)
        h.controller.setNegativePrompt("ugly")
        h.controller.generate()
        assertEquals(DiffusionActivity.LOADING_MODEL, h.state.activity)
        advanceUntilIdle()
        assertEquals(DiffusionActivity.GENERATING, h.state.activity)
        assertEquals(0, h.state.progress!!.step)
        gate.send(Unit)
        gate.send(Unit)
        advanceUntilIdle()
        assertEquals(GenerationProgress(com.donfuldauer.kggml.diffusion.DiffusionPhase.DENOISING, 2, 4, 0.5f), h.state.progress)
        assertEquals(0.5f, h.state.progress!!.fraction)
        assertNull(h.state.result)
        repeat(2) { gate.send(Unit) }
        advanceUntilIdle()
        assertEquals(DiffusionActivity.IDLE, h.state.activity)
        val result = assertNotNull(h.state.result)
        assertEquals(256, result.image.width)
        assertEquals(42, result.seed)
        val params = h.generator.requests.single()
        assertEquals(4, params.sampleSteps)
        assertEquals(5f, params.cfgScale)
        assertEquals("ugly", params.negativePrompt)
        assertEquals(42, params.seed)
        assertEquals("/exports/kggml-diffusion-42.png", h.state.exportPath)
        assertContains(h.state.status!!, "Generated 256x256 image (seed 42)")
        assertEquals(ComputeBackend.GPU_METAL, h.state.loadedBackend)
    }

    @Test
    fun sameSeedReusesTheEngineAndRandomSeedIsShown() = runTest {
        val h = Harness(this)
        h.controller.setSteps(2)
        h.controller.generate()
        advanceUntilIdle()
        h.controller.generate()
        advanceUntilIdle()
        assertEquals(1, h.factory.opened.size)
        assertEquals(listOf(42L, 42L), h.generator.requests.map { it.seed })
        assertContentEquals(h.state.result!!.image.pixels, fakeImage(512, 512, 42).pixels)

        h.controller.setRandomSeed(true)
        h.controller.generate()
        advanceUntilIdle()
        val seed = h.generator.requests.last().seed
        assertEquals(Random(7).nextLong(0, DiffusionStudioController.MAX_SEED + 1), seed)
        assertEquals(seed.toString(), h.state.seedText, "the chosen seed is shown for reproduction")
        h.controller.randomizeSeed()
        assertTrue(h.state.seedText != seed.toString())
    }

    @Test
    fun cancelStopsMidGeneration() = runTest {
        val gate = Channel<Unit>(Channel.UNLIMITED)
        val h = Harness(this, FakeImageGeneratorFactory { it.stepGate = gate })
        h.controller.setSteps(10)
        h.controller.generate()
        advanceUntilIdle()
        gate.send(Unit)
        advanceUntilIdle()
        assertTrue(h.state.canCancel)
        h.controller.cancel()
        advanceUntilIdle()
        assertEquals(DiffusionActivity.IDLE, h.state.activity)
        assertEquals("Cancelled at denoising step 1 / 10", h.state.status)
        assertNull(h.state.result)
        assertFalse(h.generator.closed, "the engine stays loaded after a cancelled generation")
    }

    @Test
    fun inBandAndLoadErrorsAreShown() = runTest {
        val h = Harness(this, FakeImageGeneratorFactory { it.failAtStep = 1; it.failure = OutOfMemoryException("VRAM") })
        h.controller.setSteps(3)
        h.controller.generate()
        advanceUntilIdle()
        assertContains(h.state.error!!, "OutOfMemoryException: VRAM")
        assertNull(h.state.result)

        val failing = Harness(this, FakeImageGeneratorFactory(failure = ModelLoadException("missing file")))
        failing.controller.generate()
        advanceUntilIdle()
        assertContains(failing.state.error!!, "ModelLoadException: missing file")
        assertEquals(DiffusionActivity.IDLE, failing.state.activity)
    }

    @Test
    fun invalidDimensionsAreRejectedBeforeLoading() = runTest {
        val h = Harness(this)
        h.controller.setSize(100)
        h.controller.generate()
        advanceUntilIdle()
        assertContains(h.state.error!!, "Invalid parameters")
        assertTrue(h.factory.opened.isEmpty())
    }

    @Test
    fun backendChangeReloadsAndClosesTheOldEngine() = runTest {
        val h = Harness(this)
        h.controller.setSteps(1)
        h.controller.generate()
        advanceUntilIdle()
        val first = h.generator
        h.controller.setBackend(ComputeBackend.CPU_ONLY)
        h.controller.generate()
        advanceUntilIdle()
        assertTrue(first.closed)
        assertEquals(ComputeBackend.CPU_ONLY, h.generator.config.backend)
        assertEquals(ComputeBackend.CPU_ONLY, h.state.loadedBackend)
        assertFalse(h.state.usesGpu)
    }

    @Test
    fun lowMemoryEvictsTheIdleEngine() = runTest {
        val h = Harness(this)
        h.controller.setSteps(1)
        h.controller.generate()
        advanceUntilIdle()
        h.observer.emit(MemoryPressureLevel.CRITICAL)
        assertTrue(h.generator.closed)
        h.controller.generate()
        advanceUntilIdle()
        assertEquals(2, h.factory.opened.size)
    }

    @Test
    fun exportWritesAValidPng() = runTest {
        val h = Harness(this)
        h.controller.setSteps(1)
        h.controller.setSize(256)
        h.controller.generate()
        advanceUntilIdle()
        assertTrue(h.state.canExport)
        h.controller.setExportPath("/exports/out.png")
        h.controller.export()
        assertEquals(DiffusionActivity.EXPORTING, h.state.activity)
        advanceUntilIdle()
        val png = h.written.getValue("/exports/out.png")
        assertContentEquals(PngEncoderTest.SIGNATURE, png.copyOf(8))
        assertContentEquals(fakeImage(256, 256, 42).pixels, PngEncoderTest.decodeStoredPng(png).pixels)
        assertContains(h.state.status!!, "Exported PNG")
        assertEquals(DiffusionActivity.IDLE, h.state.activity)
    }

    @Test
    fun exportFailureIsReported() = runTest {
        val h = Harness(this, writeFailure = IllegalStateException("read-only"))
        h.controller.setSteps(1)
        h.controller.generate()
        advanceUntilIdle()
        h.controller.export()
        advanceUntilIdle()
        assertContains(h.state.error!!, "Export failed: IllegalStateException: read-only")
    }

    @Test
    fun closeReleasesTheEngine() = runTest {
        val h = Harness(this)
        h.controller.setSteps(1)
        h.controller.generate()
        advanceUntilIdle()
        h.controller.close()
        assertTrue(h.generator.closed)
        h.controller.generate()
        advanceUntilIdle()
        assertEquals(1, h.factory.opened.size, "a closed controller does nothing")
    }
}
