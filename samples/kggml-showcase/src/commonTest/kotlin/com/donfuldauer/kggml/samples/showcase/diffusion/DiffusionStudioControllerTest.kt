package com.donfuldauer.kggml.samples.showcase.diffusion

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.ManualMemoryPressureObserver
import com.donfuldauer.kggml.core.MemoryPressureLevel
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.OutOfMemoryException
import com.donfuldauer.kggml.diffusion.DiffusionPhase
import com.donfuldauer.kggml.samples.showcase.models.DiffusionPreset
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
    fun choosingAModelAppliesItsPresetAndKeepsTheSettingsWithoutOne() = runTest {
        val h = Harness(this, modelPath = "")
        h.controller.setSteps(30)
        h.controller.setCfgScale(9f)
        h.controller.setSize(384)
        h.controller.useModel("/models/sd15.safetensors")
        assertEquals("/models/sd15.safetensors", h.state.modelPath)
        assertEquals(Triple(30, 9f, 384), Triple(h.state.steps, h.state.cfgScale, h.state.size), "no preset: settings kept")

        h.controller.useModel("/models/sd_turbo.gguf", DiffusionPreset(steps = 4, cfgScale = 1f, size = 512))
        assertEquals("/models/sd_turbo.gguf", h.state.modelPath)
        assertEquals(Triple(4, 1f, 512), Triple(h.state.steps, h.state.cfgScale, h.state.size))
        assertEquals("Settings for this model: 4 steps, CFG 1, 512x512", h.state.status)

        // The preset reaches the engine as generation parameters.
        h.controller.generate()
        advanceUntilIdle()
        val params = h.generator.requests.single()
        assertEquals(4, params.sampleSteps)
        assertEquals(1f, params.cfgScale)
        assertEquals(512, params.width)

        // Out-of-range presets are clamped like manual input.
        h.controller.useModel("/models/x.gguf", DiffusionPreset(steps = 500, cfgScale = 99f, size = 256))
        assertEquals(Triple(DiffusionStudioController.MAX_STEPS, DiffusionStudioController.MAX_CFG, 256), Triple(h.state.steps, h.state.cfgScale, h.state.size))
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
    fun progressLabelsAndFractions() {
        assertEquals("Decoding VAE tile 2 / 4", GenerationProgress(DiffusionPhase.VAE_TILES, 2, 4, 0.1f).label)
        assertEquals("Denoising step 3 / 20", GenerationProgress(DiffusionPhase.DENOISING, 3, 20, 0.1f).label)
        assertEquals(0f, GenerationProgress(DiffusionPhase.DENOISING, 0, 0, 0f).fraction, "no division by zero steps")
        assertEquals(0.25f, GenerationProgress(DiffusionPhase.VAE_TILES, 1, 4, 0f).fraction)
    }

    @Test
    fun negativeSeedTextIsNotASeed() {
        assertNull(DiffusionStudioState(seedText = "-1").seed)
        assertNull(DiffusionStudioState(seedText = "abc").seed)
        assertEquals(7L, DiffusionStudioState(seedText = " 7 ").seed)
        assertFalse(DiffusionStudioState(modelPath = "/m.gguf", seedText = "-1").canGenerate)
    }

    @Test
    fun exportFileNameIsJoinedToAnyExportDirectory() = runTest {
        fun exportPathFor(directory: String): String {
            val controller = DiffusionStudioController(
                scope = CoroutineScope(StandardTestDispatcher(testScheduler) + Job()),
                factory = FakeImageGeneratorFactory(),
                residents = ResidentModels(1L shl 40, ManualMemoryPressureObserver()),
                ioDispatcher = StandardTestDispatcher(testScheduler),
                writeFile = { _, _ -> },
                exportDirectory = directory,
                initialModelPath = "/models/sd15.gguf",
            )
            controller.setSteps(1)
            controller.setSeedText("9")
            controller.generate()
            advanceUntilIdle()
            return controller.state.value.exportPath
        }
        assertEquals("kggml-diffusion-9.png", exportPathFor(""))
        assertEquals("/exports/kggml-diffusion-9.png", exportPathFor("/exports/"))
        assertEquals("C:\\Users\\me\\Pictures\\kggml-diffusion-9.png", exportPathFor("C:\\Users\\me\\Pictures\\"))
        assertEquals("/sdcard/Pictures/kggml-diffusion-9.png", exportPathFor("/sdcard/Pictures"))
    }

    @Test
    fun defaultsHaveNoModel() = runTest {
        val controller = DiffusionStudioController(
            scope = CoroutineScope(StandardTestDispatcher(testScheduler) + Job()),
            factory = FakeImageGeneratorFactory(),
            residents = ResidentModels(1L shl 40, ManualMemoryPressureObserver()),
            ioDispatcher = StandardTestDispatcher(testScheduler),
            writeFile = { _, _ -> },
            exportDirectory = "/exports",
        )
        assertEquals("", controller.state.value.modelPath)
        assertFalse(controller.state.value.canGenerate)
        controller.setRandomSeed(true)
        controller.setModelPath("/m.gguf")
        controller.setSteps(1)
        controller.generate()
        advanceUntilIdle()
        val seed = controller.state.value.result!!.seed
        assertTrue(seed in 0..DiffusionStudioController.MAX_SEED, "a random seed from the default Random is in range")
    }

    @Test
    fun cancelWithNothingRunningDoesNothing() = runTest {
        val h = Harness(this)
        h.controller.cancel()
        advanceUntilIdle()
        assertNull(h.state.status)
        assertEquals(DiffusionActivity.IDLE, h.state.activity)
        assertTrue(h.factory.opened.isEmpty())
    }

    @Test
    fun cancelWhileTheEngineIsOpeningClosesTheOpenedEngine() = runTest {
        lateinit var h: Harness
        h = Harness(this, FakeImageGeneratorFactory { h.controller.cancel() })
        h.controller.generate()
        advanceUntilIdle()
        assertTrue(h.generator.closed, "the engine loaded under a cancelled job is closed, not leaked")
        assertTrue(h.residents.state.value.entries.isEmpty())
        // No step ran yet, so the status names no step.
        assertEquals("Cancelled", h.state.status)
        assertEquals(DiffusionActivity.IDLE, h.state.activity)
        assertNull(h.state.loadedBackend)
    }

    @Test
    fun exportNeedsAnImageAndAPath() = runTest {
        val h = Harness(this)
        h.controller.export() // no image yet
        assertEquals(DiffusionActivity.IDLE, h.state.activity)
        h.controller.setSteps(1)
        h.controller.generate()
        advanceUntilIdle()
        h.controller.setExportPath("  ")
        assertFalse(h.state.canExport)
        h.controller.export()
        advanceUntilIdle()
        assertTrue(h.written.isEmpty())
        assertEquals(DiffusionActivity.IDLE, h.state.activity)
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
