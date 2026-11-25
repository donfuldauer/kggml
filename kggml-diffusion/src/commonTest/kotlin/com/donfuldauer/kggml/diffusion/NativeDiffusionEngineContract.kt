package com.donfuldauer.kggml.diffusion

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.BackendUnavailableException
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.ModelLoadException
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.test.TestResult
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext

/**
 * [DiffusionEngine] against the real native library (stable-diffusion.cpp + shared ggml). Shared by every platform:
 * `jvmTest` (JNI) and `iosSimulatorArm64Test` (cinterop) subclass it and supply file I/O and the decision whether
 * the native library is present.
 *
 * No diffusion model small enough to commit exists (the CTest suite has none either), so these cases cover what does
 * not need weights: the pinned native version, init-error mapping into typed exceptions and backend control. Real
 * generation is covered by the manual check in `kggml-diffusion/README.md`; progress, cancellation and image
 * hand-over through the real bridges are covered by the JVM test hooks (`JvmNativeDiffusionTest`).
 */
abstract class NativeDiffusionEngineContract {
    /** Writes [bytes] to a fresh temporary file and returns its path. */
    protected abstract fun writeTempFile(prefix: String, bytes: ByteArray): String

    protected abstract fun deleteFile(path: String)

    /** A path that exists but is a directory, not a model. */
    protected abstract fun directoryPath(): String

    private lateinit var garbagePath: String
    private lateinit var emptyPath: String

    @BeforeTest
    fun writeFixtures() {
        garbagePath = writeTempFile("kggml-diffusion-garbage-", "this is not a diffusion model".encodeToByteArray())
        emptyPath = writeTempFile("kggml-diffusion-empty-", ByteArray(0))
    }

    @AfterTest
    fun deleteFixtures() {
        if (::garbagePath.isInitialized) deleteFile(garbagePath)
        if (::emptyPath.isInitialized) deleteFile(emptyPath)
    }

    protected fun nativeTest(block: suspend CoroutineScope.() -> Unit): TestResult = runTest(timeout = 120.seconds) {
        withContext(Dispatchers.Default, block)
    }

    @Test
    fun nativeInfoMatchesThePinnedUpstream() {
        val info = DiffusionEngine.nativeInfo()
        assertEquals("1.0.0", info.moduleVersion)
        assertEquals("master-612-d7ecbe1-2-g3b4d26f", info.stableDiffusionCppVersion)
        assertEquals("3b4d26f3d92c8db0199d7326f66260f4c18f4b41", info.stableDiffusionCppCommit)
        assertTrue(info.systemInfo.isNotEmpty())
    }

    @Test
    fun unloadableModelsFailTypedNamingTheFile() = nativeTest {
        val missing = garbagePath + ".missing"
        for (path in listOf(missing, garbagePath, emptyPath, directoryPath())) {
            val e = assertFailsWith<ModelLoadException>(path) { DiffusionEngine(path, BackendConfig.CPU) }
            assertTrue(e.message!!.contains("kggml_diffusion_context_create"), e.message)
        }
        val e = assertFailsWith<ModelLoadException> { DiffusionEngine(missing, BackendConfig.CPU) }
        assertTrue(e.message!!.contains(missing), "the native diagnostic names the file: ${e.message}")
    }

    @Test
    fun missingComponentFilesFailTyped() = nativeTest {
        val model = DiffusionModelFiles(diffusionModelPath = garbagePath + ".missing", vaePath = garbagePath + ".vae")
        assertFailsWith<ModelLoadException> { DiffusionEngine(model, BackendConfig.CPU) }
    }

    @Test
    fun coreMlWithoutFallbackIsBackendUnavailable() = nativeTest {
        val e = assertFailsWith<BackendUnavailableException> {
            DiffusionEngine(garbagePath, BackendConfig(ComputeBackend.NPU_COREML), DiffusionEngineOptions(allowBackendFallback = false))
        }
        assertEquals(ComputeBackend.NPU_COREML, e.backend)
    }

    @Test
    fun repeatedFailedLoadsStayTyped() = nativeTest {
        repeat(5) {
            assertFailsWith<ModelLoadException> { DiffusionEngine(garbagePath, BackendConfig.CPU) }
        }
    }
}
