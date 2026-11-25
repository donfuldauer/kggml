package com.donfuldauer.kggml.llama

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.TokenEvent
import java.io.File
import java.nio.file.Files
import java.security.MessageDigest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * [NativeLlamaEngineContract] on the JVM through the JNI bridge (`kggml_llama_jni` -> `kggml_llama` -> shared ggml),
 * plus JVM-only checks that use the test hooks of the host build.
 *
 * Skipped unless Gradle found the host builds (`build/native/lib` + `build/native-llama/lib`); with
 * `-Pkggml.requireNative=true` a missing library is a failure instead.
 */
class JvmNativeLlamaTest : NativeLlamaEngineContract() {
    private val requireNative = System.getProperty("kggml.test.requireNative").toBoolean()
    private val tempDir: File = Files.createTempDirectory("kggml-llama-test").toFile().apply { deleteOnExit() }

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
        val present = System.getProperty("kggml.llama.native.path") != null
        if (requireNative) assertTrue(present, "-Pkggml.requireNative=true but the llama host build was not found")
        assumeTrue(present, "native llama host build not found; run kggml-llama/native/scripts/build-host.sh")
    }

    @Test
    fun kotlinFixtureIsByteIdenticalToTheNativeTestModel() {
        val sha = MessageDigest.getInstance("SHA-256").digest(TinyLlamaModel.bytes()).joinToString("") { "%02x".format(it) }
        // SHA-256 of the GGUF written by kggml-llama/native/tests/tiny_model.cpp through ggml's gguf_* API.
        assertEquals("202f373d8244ac6b885fb1717b4e5ade4b66c4f4ee85507594d8f21ace31adbb", sha)
    }

    @Test
    fun kotlinProjectorFixtureIsByteIdenticalToTheNativeTestProjector() {
        val sha = MessageDigest.getInstance("SHA-256").digest(TinyMmproj.bytes()).joinToString("") { "%02x".format(it) }
        // SHA-256 of the GGUF written by write_tiny_mmproj() in kggml-llama/native/tests/tiny_model.cpp.
        assertEquals("686242ef4925d6f5c9cc1a135f76f610ccdb4d90a4a2d1d60654b05a8b7bbc22", sha)
    }

    @Test
    fun imagePromptsWriteNoTemporaryFiles() = runBlocking {
        val tmp = File(System.getProperty("java.io.tmpdir"))
        visionEngine().use { engine ->
            // Fixtures of tests running in parallel (kggml-*) are ignored; nothing else may appear.
            fun entries() = tmp.list()!!.filterNot { it.startsWith("kggml-") }.toSet()
            val before = entries()
            val pixels = TinyMmproj.rgbPattern(45, 19, 5)
            engine.generateText("ab", listOf(LlamaImage.Encoded(TinyMmproj.bmp(pixels, 45, 19))), GenerationOptions.GREEDY.copy(maxTokens = 2))
            engine.generateText("ab", listOf(LlamaImage.Rgb(45, 19, pixels)), GenerationOptions.GREEDY.copy(maxTokens = 2))
            assertEquals(emptySet(), entries() - before, "images are decoded in memory, nothing is written to disk")
        }
    }

    @Test
    fun projectorAndImageChunksAreReleasedDeterministically() = runBlocking {
        assumeTrue(KggmlLlamaJni.nativeTestHooksAvailable(), "host build without KGGML_LLAMA_TEST_HOOKS")
        val baseline = KggmlLlamaJni.nativeTestLiveObjects()
        val engine = visionEngine()
        assertEquals(baseline + 3, KggmlLlamaJni.nativeTestLiveObjects(), "model + context + projector")
        val image = LlamaImage.Rgb(TinyMmproj.IMAGE_SIZE, TinyMmproj.IMAGE_SIZE, TinyMmproj.rgbPattern(TinyMmproj.IMAGE_SIZE, TinyMmproj.IMAGE_SIZE, 1))
        // Completion, invalid input, abort: the chunks (preprocessed images) and samplers never outlive the call.
        engine.generateText("ab", listOf(image), GenerationOptions.GREEDY.copy(maxTokens = 3))
        assertEquals(baseline + 3, KggmlLlamaJni.nativeTestLiveObjects())
        engine.generate("${LlamaEngine.MEDIA_MARKER}${LlamaEngine.MEDIA_MARKER}", listOf(image)).toList()
        engine.generate("ab", List(80) { image }).toList()
        assertEquals(baseline + 3, KggmlLlamaJni.nativeTestLiveObjects())
        val context = contextHandleOf(engine)
        assertEquals(LlamaStatus.OK, KggmlLlamaJni.nativeTestAbortAfterPolls(context, 1))
        val aborted = engine.generate("ab", listOf(image), GenerationOptions.GREEDY.copy(maxTokens = 50)).toList()
        assertIs<TokenEvent.Error>(aborted.last())
        assertEquals(baseline + 3, KggmlLlamaJni.nativeTestLiveObjects())
        engine.close()
        val deadline = System.nanoTime() + 10_000_000_000L
        while (KggmlLlamaJni.nativeTestLiveObjects() != baseline && System.nanoTime() < deadline) Thread.sleep(5)
        assertEquals(baseline, KggmlLlamaJni.nativeTestLiveObjects())
    }

    @Test
    fun abortCallbackIsPolledAndEveryNativeObjectIsReleased() = runBlocking {
        assumeTrue(KggmlLlamaJni.nativeTestHooksAvailable(), "host build without KGGML_LLAMA_TEST_HOOKS")
        val baseline = KggmlLlamaJni.nativeTestLiveObjects()
        val engine = LlamaEngine(modelPath, BackendConfig(ComputeBackend.CPU_ONLY, gpuLayers = 0, cpuThreads = 2), SMALL)
        engine.generateText("ab z", GenerationOptions.GREEDY.copy(maxTokens = 4))
        engine.embed("ab")
        assertEquals(baseline + 3, KggmlLlamaJni.nativeTestLiveObjects(), "model + generation context + embedding context")
        val context = contextHandleOf(engine)
        assertTrue(KggmlLlamaJni.nativeTestAbortPollCount(context) > 0, "llama_set_abort_callback is polled during decode")
        engine.close()
        val deadline = System.nanoTime() + 10_000_000_000L
        while (KggmlLlamaJni.nativeTestLiveObjects() != baseline && System.nanoTime() < deadline) Thread.sleep(5)
        assertEquals(baseline, KggmlLlamaJni.nativeTestLiveObjects())
    }

    private fun contextHandleOf(engine: LlamaEngine): Long =
        LlamaEngine::class.java.getDeclaredField("context").apply { isAccessible = true }.getLong(engine)
}
