package com.donfuldauer.kggml.diffusion

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.BackendUnavailableException
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.NativeFailureException
import com.donfuldauer.kggml.core.OutOfMemoryException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest

/** Native results [DiffusionEngine] must not trust, value-type edge cases and bridge helpers of kggml-diffusion. */
class DiffusionEdgeCaseTest {
    private val params = DiffusionGenerationParams(prompt = "a red barn", width = 64, height = 32, sampleSteps = 2)

    private suspend fun open(bridge: NativeDiffusionBridge): DiffusionEngine = DiffusionEngine.open(
        model = DiffusionModelFiles(modelPath = "/models/sd15.gguf"),
        backendConfig = BackendConfig.CPU,
        options = DiffusionEngineOptions(),
        bridges = DiffusionBridgeHolder { bridge },
        availableBackends = { setOf(ComputeBackend.CPU_ONLY) },
        dispatcherFactory = SerialDispatcherFactory.Default,
    )

    // ---- untrusted native results ---------------------------------------------------------------------

    @Test
    fun anImageThatCannotBeConstructedIsANativeFailureWithTheValidationCause() = runTest {
        val bridge = FakeDiffusionBridge(imageDims = { Triple(0, it.height, 3) })
        open(bridge).use { engine ->
            val e = assertFailsWith<NativeFailureException> { engine.generateImage(params) }
            assertEquals("kggml_diffusion_generate returned an invalid image: width / height must be > 0, was 0x32", e.message)
            assertIs<IllegalArgumentException>(e.cause)
            assertEquals(bridge.imagesAllocated.load(), bridge.imagesFreed.load())
            // The engine stays usable.
            bridge.imageDims = null
            assertEquals(64, engine.generateImage(params).width)
        }
    }

    @Test
    fun anImageWithTheWrongHeightIsRejected() = runTest {
        val bridge = FakeDiffusionBridge(imageDims = { Triple(it.width, it.height + 8, 3) })
        open(bridge).use { engine ->
            val e = assertFailsWith<NativeFailureException> { engine.generateImage(params) }
            assertEquals("kggml_diffusion_generate returned 64x40x3, expected 64x32x3", e.message)
        }
    }

    @Test
    fun unknownNativeSamplerAndBackendKindsFallBackToDefaults() = runTest {
        val fake = FakeDiffusionBridge()
        val bridge = object : NativeDiffusionBridge by fake {
            override fun contextInfo(context: Long): RawContextInfo =
                fake.contextInfo(context).copy(backend = 42, defaultSampleMethod = 999)
        }
        open(bridge).use { engine ->
            assertEquals(DiffusionSampler.MODEL_DEFAULT, engine.modelInfo.defaultSampler)
            assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend, "an unknown kind reports the requested backend")
        }
    }

    @Test
    fun aFailingContextInfoQueryFreesTheNewContext() = runTest {
        val fake = FakeDiffusionBridge()
        val bridge = object : NativeDiffusionBridge by fake {
            override fun contextInfo(context: Long): RawContextInfo = RawContextInfo(DiffusionStatus.OUT_OF_MEMORY)
        }
        val e = assertFailsWith<OutOfMemoryException> { open(bridge) }
        assertTrue(e.message!!.startsWith("kggml_diffusion_context_get_info failed"), e.message)
        assertEquals(1, fake.creates.items.size)
        assertEquals(1, fake.frees.items.size)
        assertTrue(fake.live.isEmpty())
    }

    // ---- value types ---------------------------------------------------------------------------------

    @Test
    fun imageEqualityConsidersEveryDimensionAndOffsetsAreBoundsChecked() {
        val image = DiffusionImage(2, 3, 1, ByteArray(6) { it.toByte() })
        assertNotEquals(DiffusionImage(3, 2, 1, image.pixels), image)
        assertNotEquals(DiffusionImage(2, 1, 3, image.pixels), image)
        assertNotEquals(DiffusionImage(2, 3, 1, ByteArray(6)), image)
        assertNotEquals<Any>(image, "image")
        assertEquals(5, image.offsetOf(1, 2))
        assertFailsWith<IllegalArgumentException> { image.offsetOf(-1, 0) }
        assertFailsWith<IllegalArgumentException> { image.offsetOf(0, 3) }
        assertFailsWith<IllegalArgumentException> { image.offsetOf(0, -1) }
        assertFailsWith<IllegalArgumentException> { DiffusionImage(2, 0, 1, ByteArray(0)) }
        assertFailsWith<IllegalArgumentException> { DiffusionImage(1, 1, 0, ByteArray(0)) }
        DiffusionImage(1, 1, 4, ByteArray(4)) // RGBA is the largest accepted layout
    }

    @Test
    fun contextParamsCarryExactlyEightModelPaths() {
        val paths = DiffusionModelFiles(modelPath = "/m.gguf").toRaw()
        assertEquals(RawContextParams.PATH_COUNT, paths.size)
        val e = assertFailsWith<IllegalArgumentException> {
            RawContextParams(paths.drop(1), 0, 0, 0, true, false, false, false, false, true, -1)
        }
        assertEquals("expected 8 paths, got 7", e.message)
    }

    // ---- bridge helpers ------------------------------------------------------------------------------

    @Test
    fun nativeErrorsAreTypedAndCarryStatusNameAndDiagnostic() {
        val bridge = FakeDiffusionBridge()
        val plain = assertIs<InvalidInputException>(bridge.nativeError(DiffusionStatus.INVALID_ARGUMENT, "kggml_op"))
        assertEquals("kggml_op failed: KGGML_STATUS_2 (2)", plain.message)
        bridge.lastErrorText = "detail"
        val unavailable = assertIs<BackendUnavailableException>(bridge.nativeError(DiffusionStatus.BACKEND_UNAVAILABLE, "kggml_op"))
        assertNull(unavailable.backend)
        assertEquals("kggml_op failed: KGGML_STATUS_3 (3): detail", unavailable.message)

        bridge.check(DiffusionStatus.OK, "kggml_op")
        val e = assertFailsWith<ModelLoadException> { bridge.check(DiffusionStatus.MODEL_LOAD_FAILED, "kggml_load") }
        assertEquals("kggml_load failed: KGGML_STATUS_1 (1): detail", e.message)

        val broken = object : NativeDiffusionBridge by FakeDiffusionBridge() {
            override fun lastError(): String = throw IllegalStateException("no diagnostic")
            override fun statusName(status: Int): String = throw IllegalStateException("no name")
        }
        val failure = assertIs<NativeFailureException>(broken.nativeError(55, "kggml_op"), "unknown codes are native failures")
        assertEquals("kggml_op failed: KGGML_STATUS_55 (55)", failure.message)
    }

    @Test
    fun bridgeHolderCreatesOnceAndPassesTypedFailuresThrough() {
        var created = 0
        val bridge = FakeDiffusionBridge()
        val holder = DiffusionBridgeHolder { created++; bridge }
        assertSame(bridge, holder.get())
        assertSame(bridge, holder.get())
        assertEquals(1, created)

        val typed = ModelLoadException("already typed")
        assertSame(typed, assertFailsWith<ModelLoadException> { DiffusionBridgeHolder { throw typed }.get() })

        val wrapped = assertFailsWith<NativeFailureException> { DiffusionBridgeHolder { throw IllegalStateException() }.get() }
        assertEquals("Failed to load the Kggml diffusion native library: IllegalStateException", wrapped.message)
        assertIs<IllegalStateException>(wrapped.cause)
    }
}
