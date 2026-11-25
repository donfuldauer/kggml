package com.donfuldauer.kggml.diffusion

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import java.io.File
import java.nio.file.Files
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeout
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * [NativeDiffusionEngineContract] on the JVM through the JNI bridge (`kggml_diffusion_jni` -> `kggml_diffusion` ->
 * shared ggml), plus JVM-only checks that drive a full [DiffusionEngine] through the real JNI bridge with the native
 * test hooks of the host build: a model-less stub context and the wrapper's simulated sampler, which runs the exact
 * native progress dispatch, latched abort, image hand-over and `kggml_diffusion_image_free` path of a real generation.
 *
 * Skipped unless Gradle found the host builds (`build/native/lib` + `build/native-diffusion/lib`); with
 * `-Pkggml.requireNative=true` a missing library is a failure instead.
 */
class JvmNativeDiffusionTest : NativeDiffusionEngineContract() {
    private val requireNative = System.getProperty("kggml.test.requireNative").toBoolean()
    private val tempDir: File = Files.createTempDirectory("kggml-diffusion-test").toFile().apply { deleteOnExit() }

    override fun writeTempFile(prefix: String, bytes: ByteArray): String =
        File.createTempFile(prefix, ".gguf", tempDir).apply {
            deleteOnExit()
            writeBytes(bytes)
        }.absolutePath

    override fun deleteFile(path: String) {
        File(path).delete()
    }

    override fun directoryPath(): String = tempDir.absolutePath

    @BeforeTest
    fun requireNativeLibrary() {
        val present = System.getProperty("kggml.diffusion.native.path") != null
        if (requireNative) assertTrue(present, "-Pkggml.requireNative=true but the diffusion host build was not found")
        assumeTrue(present, "native diffusion host build not found; run kggml-diffusion/native/scripts/build-host.sh")
    }

    private fun requireHooks() {
        val available = KggmlDiffusionJni.nativeTestHooksAvailable()
        if (requireNative) assertTrue(available, "-Pkggml.requireNative=true but the host build has no KGGML_DIFFUSION_TEST_HOOKS")
        assumeTrue(available, "host build without KGGML_DIFFUSION_TEST_HOOKS")
    }

    /**
     * The real JNI bridge, except that contexts are model-less stubs and generation runs the native simulated sampler
     * (both through the JNI test entry points). [blockAtStep] makes the progress listener wait (on the generating
     * thread, inside native code) until the context's abort flag is latched, like a long denoising step.
     */
    private class StubJniBridge(private val real: JniDiffusionBridge) : NativeDiffusionBridge by real {
        @Volatile var blockAtStep: Int = -1
        val contexts = java.util.concurrent.ConcurrentLinkedQueue<Long>()

        override fun contextCreate(params: RawContextParams): RawHandleResult {
            val out = LongArray(1)
            val status = KggmlDiffusionJni.nativeTestContextCreateStub(out)
            if (status == DiffusionStatus.OK) contexts += out[0]
            return RawHandleResult(status, out[0])
        }

        override fun contextInfo(context: Long): RawContextInfo {
            val info = real.contextInfo(context)
            check(info.status == DiffusionStatus.OK && !info.supportsImageGeneration) { "stub info: $info" }
            return info.copy(supportsImageGeneration = true)
        }

        override fun generate(context: Long, params: RawGenerateParams, listener: DiffusionProgressListener?): RawImageResult {
            val blocking = listener?.let { inner ->
                DiffusionProgressListener { step, total, sps ->
                    inner.onProgress(step, total, sps)
                    if (step == blockAtStep) {
                        val deadline = System.nanoTime() + 10_000_000_000L
                        while (!real.abortRequested(context) && System.nanoTime() < deadline) Thread.onSpinWait()
                    }
                }
            }
            return JniDiffusionBridge.imageResult { dims, pixels ->
                KggmlDiffusionJni.nativeTestSimulateGenerate(context, params.sampleSteps, params.width, params.height, blocking, dims, pixels)
            }
        }
    }

    private val bridge by lazy { StubJniBridge(createPlatformDiffusionBridge() as JniDiffusionBridge) }

    private suspend fun stubEngine(): DiffusionEngine = DiffusionEngine.open(
        DiffusionModelFiles(modelPath = "/stub.gguf"), BackendConfig.CPU, DiffusionEngineOptions(),
        DiffusionBridgeHolder { bridge }, { setOf(ComputeBackend.CPU_ONLY) }, SerialDispatcherFactory.Default,
    )

    private val params = DiffusionGenerationParams(prompt = "simulated", width = 40, height = 24, sampleSteps = 6)

    /** pixel (x, y) = {x & 255, y & 255, (x + y) & 255}: kggml_diffusion_test_simulate_generate()'s gradient. */
    private fun gradient(width: Int, height: Int) = ByteArray(width * height * 3).also { out ->
        for (y in 0 until height) for (x in 0 until width) {
            val i = (y * width + x) * 3
            out[i] = (x and 255).toByte()
            out[i + 1] = (y and 255).toByte()
            out[i + 2] = ((x + y) and 255).toByte()
        }
    }

    private suspend fun awaitLive(expected: Long) = withTimeout(10_000) {
        while (KggmlDiffusionJni.nativeTestLiveObjects() != expected) delay(5)
    }

    @Test
    fun progressAndTheRgbImageCrossTheJniBridge() = nativeTest {
        requireHooks()
        stubEngine().use { engine ->
            val events = engine.generate(params).toList()
            val progress = events.dropLast(1).map { assertIs<DiffusionEvent.Progress>(it) }
            assertEquals((0..6).toList(), progress.map { it.step })
            assertTrue(progress.all { it.totalSteps == 6 && it.phase == DiffusionPhase.DENOISING })
            val image = assertIs<DiffusionEvent.Completed>(events.last()).image
            assertEquals(40, image.width)
            assertEquals(24, image.height)
            assertEquals(3, image.channels)
            assertContentEquals(gradient(40, 24), image.pixels)
            // Large enough that a missing free would be obvious under a leak checker; repeated to exercise reuse.
            repeat(3) { assertEquals(gradient(512, 512).size, engine.generateImage(params.copy(width = 512, height = 512, sampleSteps = 1)).pixels.size) }
        }
    }

    @Test
    fun cancellingTheCollectorAbortsNativeGenerationAtTheNextStep() = nativeTest {
        requireHooks()
        val baseline = KggmlDiffusionJni.nativeTestLiveObjects()
        stubEngine().use { engine ->
            assertEquals(baseline + 1, KggmlDiffusionJni.nativeTestLiveObjects())
            bridge.blockAtStep = 2
            val received = java.util.concurrent.ConcurrentLinkedQueue<DiffusionEvent>()
            val job = launch(Dispatchers.Default) { engine.generate(params).collect { received += it } }
            withTimeout(10_000) { while (!(engine.isGenerating && received.size == 3)) delay(1) }
            job.cancelAndJoin()
            assertEquals(listOf(0, 1, 2), received.map { assertIs<DiffusionEvent.Progress>(it).step })
            assertFalse(engine.isGenerating)
            // The latched abort is cleared before the next generation; the context is reused.
            bridge.blockAtStep = -1
            assertContentEquals(gradient(40, 24), engine.generateImage(params).pixels)
        }
        awaitLive(baseline)
    }

    @Test
    fun abortActiveGenerationReportsTheNativeStepItStoppedAt() = nativeTest {
        requireHooks()
        stubEngine().use { engine ->
            bridge.blockAtStep = 3
            val events = async { engine.generate(params).toList() }
            withTimeout(10_000) { while (!engine.isGenerating) delay(1) }
            delay(20)
            assertTrue(engine.abortActiveGeneration())
            val all = events.await()
            assertEquals(listOf(0, 1, 2, 3), all.dropLast(1).map { assertIs<DiffusionEvent.Progress>(it).step })
            val error = assertIs<InferenceCancelledException>(assertIs<DiffusionEvent.Error>(all.last()).exception)
            assertTrue(error.message!!.contains("KGGML_STATUS_CANCELLED"), error.message)
            assertTrue(error.message!!.contains("step 3/6"), error.message)
        }
    }

    @Test
    fun closeDuringGenerationAbortsAndFreesTheNativeContext() = nativeTest {
        requireHooks()
        val baseline = KggmlDiffusionJni.nativeTestLiveObjects()
        val engine = stubEngine()
        bridge.blockAtStep = 1
        val events = async { engine.generate(params).toList() }
        withTimeout(10_000) { while (!engine.isGenerating) delay(1) }
        engine.close()
        engine.close()
        assertIs<InferenceCancelledException>(assertIs<DiffusionEvent.Error>(events.await().last()).exception)
        awaitLive(baseline)
    }

    @Test
    fun concurrentEnginesEachOwnAContextAndAllAreReleased() = nativeTest {
        requireHooks()
        val baseline = KggmlDiffusionJni.nativeTestLiveObjects()
        val engines = List(3) { stubEngine() }
        assertEquals(baseline + 3, KggmlDiffusionJni.nativeTestLiveObjects())
        val images = engines.map { engine -> async { engine.generateImage(params) } }.awaitAll()
        images.forEach { assertContentEquals(gradient(40, 24), it.pixels) }
        engines.forEach { it.close() }
        awaitLive(baseline)
    }

    @Test
    fun realGenerateOnAModelLessContextIsATypedInvalidInput() = nativeTest {
        requireHooks()
        val real = createPlatformDiffusionBridge()
        val out = LongArray(1)
        assertEquals(DiffusionStatus.OK, KggmlDiffusionJni.nativeTestContextCreateStub(out))
        try {
            val steps = java.util.concurrent.atomic.AtomicInteger()
            val result = real.generate(out[0], params.toRaw(), DiffusionProgressListener { _, _, _ -> steps.incrementAndGet() })
            assertEquals(DiffusionStatus.INVALID_ARGUMENT, result.status)
            assertTrue(real.lastError().contains("context has no model"), real.lastError())
            assertEquals(0, steps.get())
            assertIs<InvalidInputException>(real.nativeError(result.status, "kggml_diffusion_generate"))
            // Abort flag round-trip through JNI.
            assertFalse(real.abortRequested(out[0]))
            real.requestAbort(out[0])
            assertTrue(real.abortRequested(out[0]))
            real.clearAbort(out[0])
            assertFalse(real.abortRequested(out[0]))
        } finally {
            real.contextFree(out[0])
        }
        real.contextFree(0L) // NULL is a no-op
    }

    @Test
    fun aThrowingProgressListenerLatchesAnAbortInsteadOfCrossingJni() = nativeTest {
        requireHooks()
        val out = LongArray(1)
        assertEquals(DiffusionStatus.OK, KggmlDiffusionJni.nativeTestContextCreateStub(out))
        try {
            val calls = java.util.concurrent.atomic.AtomicInteger()
            val listener = DiffusionProgressListener { step, _, _ ->
                calls.incrementAndGet()
                if (step == 2) throw IllegalStateException("listener bug")
            }
            val result = JniDiffusionBridge.imageResult { dims, pixels ->
                KggmlDiffusionJni.nativeTestSimulateGenerate(out[0], 6, 16, 16, listener, dims, pixels)
            }
            assertEquals(DiffusionStatus.CANCELLED, result.status)
            assertEquals(3, calls.get(), "no progress point is delivered after the throwing one")
            assertEquals(0, result.pixels.size)
        } finally {
            KggmlDiffusionJni.nativeContextFree(out[0])
        }
    }

    @Test
    fun invalidJniArgumentsAreRejectedWithoutCrashing() {
        requireHooks()
        assertEquals(DiffusionStatus.INVALID_ARGUMENT, KggmlDiffusionJni.nativeContextCreate(arrayOfNulls(2), IntArray(10), LongArray(1)))
        assertEquals(DiffusionStatus.INVALID_ARGUMENT, KggmlDiffusionJni.nativeContextInfo(0L, IntArray(1)))
        assertEquals(DiffusionStatus.INVALID_ARGUMENT, KggmlDiffusionJni.nativeContextInfo(0L, IntArray(5)))
        assertEquals(
            DiffusionStatus.INVALID_ARGUMENT,
            KggmlDiffusionJni.nativeGenerate(0L, "x".encodeToByteArray(), null, IntArray(2), FloatArray(2), -1L, null, IntArray(3), arrayOfNulls(1)),
        )
        assertEquals(
            DiffusionStatus.INVALID_ARGUMENT,
            KggmlDiffusionJni.nativeGenerate(0L, "x".encodeToByteArray(), null, IntArray(7), FloatArray(2), -1L, null, IntArray(3), arrayOfNulls(1)),
            "NULL context",
        )
    }

    @Test
    fun failedNativeLoadsLeakNoContext() = nativeTest {
        requireHooks()
        val baseline = KggmlDiffusionJni.nativeTestLiveObjects()
        assertFailsWith<ModelLoadException> { DiffusionEngine(directoryPath() + "/missing.safetensors", BackendConfig.CPU) }
        assertEquals(baseline, KggmlDiffusionJni.nativeTestLiveObjects())
    }

    @Test
    fun jvmLoaderRequiresBothLibrariesInAnExplicitDirectory() {
        val e = assertFailsWith<UnsatisfiedLinkError> { JvmDiffusionLibraryLoader.resolveDirectoryLibraries(tempDir) }
        assertTrue(e.message!!.contains(System.mapLibraryName("kggml_diffusion")), e.message)
        assertEquals(listOf("kggml_diffusion", "kggml_diffusion_jni"), JvmDiffusionLibraryLoader.LOAD_ORDER)
    }
}
