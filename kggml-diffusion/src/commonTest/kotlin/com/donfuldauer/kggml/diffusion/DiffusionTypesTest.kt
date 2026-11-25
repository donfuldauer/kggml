package com.donfuldauer.kggml.diffusion

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class DiffusionTypesTest {
    private val ok = DiffusionGenerationParams(prompt = "a cat")

    @Test
    fun defaultsMatchTheNativeDefaults() {
        assertEquals(512, ok.width)
        assertEquals(512, ok.height)
        assertEquals(20, ok.sampleSteps)
        assertEquals(7.0f, ok.cfgScale)
        assertEquals(3.5f, ok.distilledGuidance)
        assertEquals(-1L, ok.seed)
        assertEquals(DiffusionSampler.MODEL_DEFAULT, ok.sampler)
        assertEquals(DiffusionScheduler.MODEL_DEFAULT, ok.scheduler)
        assertEquals(-1, ok.clipSkip)
        assertEquals("", ok.negativePrompt)
    }

    @Test
    fun promptMustNotBeBlank() {
        for (prompt in listOf("", "   ", "\n\t")) {
            assertFailsWith<IllegalArgumentException>(prompt) { DiffusionGenerationParams(prompt = prompt) }
        }
    }

    @Test
    fun dimensionsArePositiveMultiplesOfEightWithinTheNativeLimit() {
        for (bad in listOf(0, -8, 7, 12, 513, DiffusionGenerationParams.MAX_DIMENSION + 8)) {
            assertFailsWith<IllegalArgumentException>("width $bad") { ok.copy(width = bad) }
            assertFailsWith<IllegalArgumentException>("height $bad") { ok.copy(height = bad) }
        }
        for (good in listOf(8, 64, 768, 1024, DiffusionGenerationParams.MAX_DIMENSION)) {
            assertEquals(good, ok.copy(width = good, height = good).width)
        }
    }

    @Test
    fun stepsGuidanceAndClipSkipAreValidated() {
        assertFailsWith<IllegalArgumentException> { ok.copy(sampleSteps = 0) }
        assertFailsWith<IllegalArgumentException> { ok.copy(sampleSteps = -1) }
        assertFailsWith<IllegalArgumentException> { ok.copy(sampleSteps = DiffusionGenerationParams.MAX_SAMPLE_STEPS + 1) }
        assertEquals(1, ok.copy(sampleSteps = 1).sampleSteps)
        assertEquals(10000, ok.copy(sampleSteps = DiffusionGenerationParams.MAX_SAMPLE_STEPS).sampleSteps)
        for (bad in listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -0.5f)) {
            assertFailsWith<IllegalArgumentException>("cfg $bad") { ok.copy(cfgScale = bad) }
            assertFailsWith<IllegalArgumentException>("guidance $bad") { ok.copy(distilledGuidance = bad) }
        }
        assertEquals(0f, ok.copy(cfgScale = 0f).cfgScale)
        assertEquals(1f, ok.copy(cfgScale = 1f).cfgScale)
        assertFailsWith<IllegalArgumentException> { ok.copy(clipSkip = 0) }
        assertFailsWith<IllegalArgumentException> { ok.copy(clipSkip = -2) }
        assertEquals(2, ok.copy(clipSkip = 2).clipSkip)
    }

    @Test
    fun negativeSeedsAllMeanRandom() {
        assertEquals(-1L, ok.copy(seed = Long.MIN_VALUE).toRaw().seed)
        assertEquals(-1L, ok.copy(seed = -7L).toRaw().seed)
        assertEquals(0L, ok.copy(seed = 0L).toRaw().seed)
        assertEquals(Long.MAX_VALUE, ok.copy(seed = Long.MAX_VALUE).toRaw().seed)
    }

    @Test
    fun samplerAndSchedulerValuesMatchStableDiffusionCpp() {
        // sample_method_t / scheduler_t of the pinned stable-diffusion.cpp (include/stable-diffusion.h).
        assertEquals(17, DiffusionSampler.entries.size - 1, "SAMPLE_METHOD_COUNT")
        assertEquals(11, DiffusionScheduler.entries.size - 1, "SCHEDULER_COUNT")
        DiffusionSampler.entries.drop(1).forEachIndexed { i, s -> assertEquals(i, s.nativeValue, s.name) }
        DiffusionScheduler.entries.drop(1).forEachIndexed { i, s -> assertEquals(i, s.nativeValue, s.name) }
        assertEquals(1, DiffusionSampler.EULER_A.nativeValue)
        assertEquals(10, DiffusionScheduler.BONG_TANGENT.nativeValue)
        assertEquals(DiffusionSampler.LCM, DiffusionSampler.fromNative(9))
        assertNull(DiffusionSampler.fromNative(99))
    }

    @Test
    fun modelFilesNeedExactlyOneMainModel() {
        assertFailsWith<IllegalArgumentException> { DiffusionModelFiles() }
        assertFailsWith<IllegalArgumentException> { DiffusionModelFiles(modelPath = "", diffusionModelPath = "") }
        assertFailsWith<IllegalArgumentException> { DiffusionModelFiles(modelPath = "/a", diffusionModelPath = "/b") }
        assertFailsWith<IllegalArgumentException> { DiffusionModelFiles(vaePath = "/vae") }
        assertEquals(
            listOf("/sd.ckpt", null, "/vae", null, null, null, null, null),
            DiffusionModelFiles(modelPath = "/sd.ckpt", vaePath = "/vae", taesdPath = "").toRaw(),
        )
        assertEquals(RawContextParams.PATH_COUNT, DiffusionModelFiles(diffusionModelPath = "/f").toRaw().size)
    }

    @Test
    fun engineOptionsAreValidated() {
        assertFailsWith<IllegalArgumentException> { DiffusionEngineOptions(gpuDevice = -1) }
        assertFailsWith<IllegalArgumentException> { DiffusionEngineOptions(weightType = -2) }
        assertEquals(DiffusionEngineOptions.WEIGHT_TYPE_AS_STORED, DiffusionEngineOptions().weightType)
    }

    @Test
    fun imagesValidateTheirBufferAndCompareByContent() {
        val pixels = FakeDiffusionBridge.gradient(4, 2, 3, 0)
        val image = DiffusionImage(4, 2, 3, pixels)
        assertEquals(DiffusionImage(4, 2, 3, pixels.copyOf()), image)
        assertEquals(DiffusionImage(4, 2, 3, pixels.copyOf()).hashCode(), image.hashCode())
        assertNotEquals(DiffusionImage(2, 4, 3, pixels), image)
        assertEquals((1 * 4 + 3) * 3, image.offsetOf(3, 1))
        assertFailsWith<IllegalArgumentException> { image.offsetOf(4, 0) }
        assertFailsWith<IllegalArgumentException> { DiffusionImage(4, 2, 3, ByteArray(23)) }
        assertFailsWith<IllegalArgumentException> { DiffusionImage(0, 2, 3, ByteArray(0)) }
        assertFailsWith<IllegalArgumentException> { DiffusionImage(1, 1, 5, ByteArray(5)) }
        assertTrue(image.toString().contains("4x2x3"))
    }

    @Test
    fun progressEventsValidateTheirRange() {
        assertFailsWith<IllegalArgumentException> { DiffusionEvent.Progress(DiffusionPhase.DENOISING, 5, 4, 0f) }
        assertFailsWith<IllegalArgumentException> { DiffusionEvent.Progress(DiffusionPhase.DENOISING, -1, 4, 0f) }
        assertFailsWith<IllegalArgumentException> { DiffusionEvent.Progress(DiffusionPhase.DENOISING, 0, -1, 0f) }
    }

    @Test
    fun backendKindsRoundTrip() {
        for (backend in listOf(com.donfuldauer.kggml.core.ComputeBackend.CPU_ONLY, com.donfuldauer.kggml.core.ComputeBackend.GPU_METAL, com.donfuldauer.kggml.core.ComputeBackend.GPU_VULKAN, com.donfuldauer.kggml.core.ComputeBackend.NPU_COREML)) {
            assertEquals(backend, DiffusionBackendKind.toComputeBackend(DiffusionBackendKind.of(backend)))
        }
        assertNull(DiffusionBackendKind.toComputeBackend(42))
        assertFailsWith<IllegalStateException> { DiffusionBackendKind.of(com.donfuldauer.kggml.core.ComputeBackend.AUTO) }
    }
}
