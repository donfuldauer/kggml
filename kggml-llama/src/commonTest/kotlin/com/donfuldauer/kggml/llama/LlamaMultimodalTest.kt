package com.donfuldauer.kggml.llama

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import com.donfuldauer.kggml.core.StopReason
import com.donfuldauer.kggml.core.TokenEvent
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

/** Multimodal (libmtmd) behaviour of [LlamaEngine] against [FakeLlamaBridge]. */
class LlamaMultimodalTest {
    private val projector = "/models/mmproj.gguf"
    private val image = LlamaImage.Encoded(FakeLlamaBridge.IMAGE_MAGIC + byteArrayOf(1, 2, 3))
    private val rgb = LlamaImage.Rgb(2, 1, ByteArray(6) { it.toByte() })

    private suspend fun open(
        bridge: FakeLlamaBridge,
        config: BackendConfig = BackendConfig.CPU,
        options: LlamaEngineOptions = LlamaEngineOptions(projectorPath = projector),
        available: Set<ComputeBackend> = setOf(ComputeBackend.CPU_ONLY),
    ): LlamaEngine = LlamaEngine.open(
        modelPath = "/models/tiny.gguf",
        backendConfig = config,
        contextConfig = LlamaContextConfig(contextLength = 64, batchSize = 8),
        options = options,
        bridges = LlamaBridgeHolder { bridge },
        availableBackends = { available },
        dispatcherFactory = SerialDispatcherFactory.Default,
    )

    private suspend fun LlamaEngine.events(prompt: String, images: List<LlamaImage>, options: GenerationOptions = GenerationOptions.GREEDY) =
        withContext(Dispatchers.Default) { withTimeout(10.seconds) { generate(prompt, images, options).toList() } }

    /** close() releases native objects asynchronously on the engine thread. */
    private suspend fun awaitReleased(bridge: FakeLlamaBridge) = withContext(Dispatchers.Default) {
        withTimeout(10.seconds) { while (bridge.live.isNotEmpty()) delay(1) }
    }

    private fun List<TokenEvent>.text() = filterIsInstance<TokenEvent.Token>().joinToString("") { it.text }

    // ---- projector lifecycle -----------------------------------------------------------------------------

    @Test
    fun projectorIsLoadedOnTheModelBackendWithTheEngineConfig() = runTest {
        val bridge = FakeLlamaBridge()
        val config = BackendConfig(ComputeBackend.CPU_ONLY, gpuLayers = 0, cpuThreads = 3, useFlashAttention = true)
        open(bridge, config, LlamaEngineOptions(projectorPath = projector, imageMinTokens = 64, imageMaxTokens = 256)).use { engine ->
            assertTrue(engine.supportsVision)
            val (path, params) = bridge.projectors.items.single()
            assertEquals(projector, path)
            assertEquals(RawMtmdParams(LlamaBackendKind.CPU, 3, 1, 64, 256, warmup = false), params)
        }
        awaitReleased(bridge)
        assertEquals(listOf("context", "mtmd", "model"), bridge.frees.items, "projector freed after the context, before the model")
        assertTrue(bridge.live.isEmpty())
    }

    @Test
    fun projectorFollowsTheResolvedGpuBackendAndAutoThreads() = runTest {
        val bridge = FakeLlamaBridge()
        val config = BackendConfig(ComputeBackend.AUTO)
        open(bridge, config, available = setOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_VULKAN)).use {
            assertEquals(LlamaBackendKind.VULKAN, bridge.projectors.items.single().second.backend)
            assertEquals(0, bridge.projectors.items.single().second.threads)
        }
    }

    @Test
    fun noProjectorMeansTextOnly() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("ok"))
        open(bridge, options = LlamaEngineOptions()).use { engine ->
            assertFalse(engine.supportsVision)
            assertTrue(bridge.projectors.items.isEmpty())
            val events = engine.events("describe", listOf(image))
            val error = assertIs<TokenEvent.Error>(events.single())
            assertIs<InvalidInputException>(error.exception)
            assertTrue("projectorPath" in error.exception.message.orEmpty())
            assertEquals("ok", engine.generateText("hi", GenerationOptions.GREEDY).text)
        }
        awaitReleased(bridge)
    }

    @Test
    fun projectorWithoutVisionRejectsImages() = runTest {
        val bridge = FakeLlamaBridge(projectorVision = false)
        open(bridge).use { engine ->
            assertFalse(engine.supportsVision)
            val error = assertIs<TokenEvent.Error>(engine.events("x", listOf(image)).single())
            assertIs<InvalidInputException>(error.exception)
        }
    }

    @Test
    fun projectorLoadFailureFailsTheFactoryAndFreesEverything() = runTest {
        val bridge = FakeLlamaBridge()
        val error = assertFailsWith<ModelLoadException> {
            open(bridge, options = LlamaEngineOptions(projectorPath = "/models/missing-mmproj.gguf"))
        }
        assertTrue("kggml_llama_mtmd_init" in error.message.orEmpty())
        assertTrue("cannot open mmproj" in error.message.orEmpty())
        assertTrue(bridge.live.isEmpty(), "model and context are released: ${bridge.live}")
    }

    // ---- generation --------------------------------------------------------------------------------------

    @Test
    fun imagePromptStartsANewSessionAndGeneratesAfterTheImagePositions() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("cat"))
        open(bridge).use { engine ->
            engine.generateText("old text", GenerationOptions.GREEDY)
            val clearsBefore = bridge.clears.load()
            var chunksLiveDuringDecode = false
            bridge.onDecode = { b, _, _ ->
                chunksLiveDuringDecode = chunksLiveDuringDecode || "chunks" in b.live.values
                LlamaStatus.OK
            }

            val prompt = "USER: ${LlamaEngine.MEDIA_MARKER}\nwhat is this? ASSISTANT:"
            val events = engine.events(prompt, listOf(image))
            assertEquals("cat", events.text())
            assertEquals(StopReason.END_OF_GENERATION, assertIs<TokenEvent.Completed>(events.last()).stopReason)

            val call = bridge.tokenizes.items.single()
            assertEquals(prompt, call.prompt, "explicit markers are kept as written")
            assertSame(image.bytes, call.images.single().bytes, "image bytes are handed over without a copy")
            assertEquals(RawImage.FORMAT_ENCODED, call.images.single().format)
            assertTrue(call.addSpecial)
            assertTrue(bridge.clears.load() > clearsBefore, "the KV cache is cleared first")

            val eval = bridge.evals.items.single()
            assertEquals(0, eval.nPast)
            assertTrue(eval.logitsLast)
            val positions = prompt.replace(FakeLlamaBridge.MARKER, "").encodeToByteArray().size + 1 + FakeLlamaBridge.IMAGE_TOKENS
            // Generated tokens are decoded right after the image prompt.
            val generated = bridge.decodes.items.takeLast(3)
            assertEquals(listOf(positions, positions + 1, positions + 2), generated.map { it.pos0 })
            assertEquals(positions + 3, engine.sessionTokenCount)
            assertFalse(chunksLiveDuringDecode, "chunks are freed once evaluated")
            assertFalse("chunks" in bridge.live.values)
        }
        awaitReleased(bridge)
    }

    @Test
    fun missingMarkersArePrependedOnePerImage() = runTest {
        val bridge = FakeLlamaBridge()
        open(bridge).use { engine ->
            engine.events("compare", listOf(image, rgb))
            val call = bridge.tokenizes.items.single()
            assertEquals("${FakeLlamaBridge.MARKER}\n${FakeLlamaBridge.MARKER}\ncompare", call.prompt)
            assertEquals(listOf(RawImage.FORMAT_ENCODED, RawImage.FORMAT_RGB), call.images.map { it.format })
            assertEquals(2, call.images[1].width)
            assertEquals(1, call.images[1].height)
            assertSame(rgb.pixels, call.images[1].bytes)
        }
    }

    @Test
    fun specialTokenOptionsReachTheTokenizer() = runTest {
        val bridge = FakeLlamaBridge()
        open(bridge).use { engine ->
            engine.events("x", listOf(image), GenerationOptions.GREEDY.copy(addSpecialTokens = false, parseSpecialTokens = false))
            val call = bridge.tokenizes.items.single()
            assertFalse(call.addSpecial)
            assertFalse(call.parseSpecial)
        }
    }

    @Test
    fun markerCountMismatchAndUndecodableImagesAreInvalidInput() = runTest {
        val bridge = FakeLlamaBridge()
        open(bridge).use { engine ->
            val twoMarkers = "${LlamaEngine.MEDIA_MARKER} and ${LlamaEngine.MEDIA_MARKER}"
            val mismatch = assertIs<TokenEvent.Error>(engine.events(twoMarkers, listOf(image)).single())
            assertIs<InvalidInputException>(mismatch.exception)
            assertTrue("media markers" in mismatch.exception.message.orEmpty())

            val garbage = LlamaImage.Encoded(byteArrayOf(9, 9, 9, 9))
            val undecodable = assertIs<TokenEvent.Error>(engine.events("x", listOf(garbage)).single())
            assertIs<InvalidInputException>(undecodable.exception)
            assertTrue("could not decode" in undecodable.exception.message.orEmpty())

            assertTrue(bridge.evals.items.isEmpty())
            assertEquals(setOf("model", "context", "mtmd"), bridge.live.values.toSet(), "no chunks or sampler leak")
        }
    }

    @Test
    fun imagePromptLongerThanTheContextIsRejectedBeforeEvaluation() = runTest {
        val bridge = FakeLlamaBridge(contextSize = 16)
        open(bridge).use { engine ->
            val error = assertIs<TokenEvent.Error>(engine.events("a fairly long question", listOf(image)).single())
            assertIs<InvalidInputException>(error.exception)
            assertTrue("positions" in error.exception.message.orEmpty())
            assertTrue(bridge.evals.items.isEmpty())
            assertFalse("chunks" in bridge.live.values)
        }
    }

    @Test
    fun textFollowUpDoesNotReuseTheImageSession() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("a"))
        open(bridge).use { engine ->
            engine.events("x", listOf(image))
            val decodesBefore = bridge.decodes.items.size
            engine.generateText("x", GenerationOptions.GREEDY)
            val prompt = bridge.decodes.items[decodesBefore]
            assertEquals(0, prompt.pos0, "image positions are never a reusable prefix")
            assertEquals(listOf(FakeLlamaBridge.BOS) + FakeLlamaBridge.tokensOf("x"), prompt.tokens)
        }
    }

    @Test
    fun emptyImageListIsAPlainTextGeneration() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("hi"))
        open(bridge).use { engine ->
            assertEquals("hi", engine.generateText("x", emptyList(), GenerationOptions.GREEDY).text)
            assertTrue(bridge.tokenizes.items.isEmpty())
        }
    }

    @Test
    fun imageSessionCanBeSavedAndRestored() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("ok"))
        open(bridge).use { engine ->
            engine.generateText("x", listOf(image), GenerationOptions.GREEDY)
            val count = engine.sessionTokenCount
            val saved = engine.saveSession()
            engine.clearSession()
            assertEquals(0, engine.sessionTokenCount)
            engine.restoreSession(saved)
            assertEquals(count, engine.sessionTokenCount)
        }
    }

    @Test
    fun abortDuringImageEvaluationReportsCancellationAndClearsTheSession() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("never"))
        open(bridge).use { engine ->
            bridge.onEval = { b, context ->
                b.requestAbort(context) // as abortActiveGeneration() would, mid-encode
                LlamaStatus.OK
            }
            val error = assertIs<TokenEvent.Error>(engine.events("x", listOf(image)).single())
            assertIs<InferenceCancelledException>(error.exception)
            assertEquals(0, engine.sessionTokenCount)
            assertEquals(0, bridge.kvLength(bridge.evals.items.single().context), "partially evaluated chunks are dropped")
            assertEquals(setOf("model", "context", "mtmd"), bridge.live.values.toSet())

            bridge.onEval = { _, _ -> LlamaStatus.OK }
            assertEquals("never", engine.generateText("x", listOf(image), GenerationOptions.GREEDY).text, "engine stays usable")
        }
    }

    @Test
    fun nativeEvalFailureIsTypedAndLeaksNothing() = runTest {
        val bridge = FakeLlamaBridge()
        open(bridge).use { engine ->
            bridge.onEval = { _, _ -> LlamaStatus.NATIVE_FAILURE }
            val error = assertIs<TokenEvent.Error>(engine.events("x", listOf(image)).single())
            assertTrue("kggml_llama_mtmd_eval_chunks" in error.exception.message.orEmpty())
            assertEquals(0, engine.sessionTokenCount)
            assertEquals(setOf("model", "context", "mtmd"), bridge.live.values.toSet())
        }
        awaitReleased(bridge)
    }

    @Test
    fun generateWithImagesOnAClosedEngineThrows() = runTest {
        val bridge = FakeLlamaBridge()
        val engine = open(bridge)
        engine.close()
        assertFailsWith<IllegalStateException> { engine.generate("x", listOf(image)) }
    }

    // ---- value types -------------------------------------------------------------------------------------

    @Test
    fun imageAndOptionValidation() {
        assertFailsWith<IllegalArgumentException> { LlamaImage.Encoded(ByteArray(0)) }
        assertFailsWith<IllegalArgumentException> { LlamaImage.Rgb(0, 1, ByteArray(0)) }
        assertFailsWith<IllegalArgumentException> { LlamaImage.Rgb(2, 2, ByteArray(11)) }
        assertFailsWith<IllegalArgumentException> { LlamaImage.Rgb(-1, -1, ByteArray(3)) }
        val raw = LlamaImage.Rgb(2, 2, ByteArray(12)).toRaw()
        assertEquals(RawImage.FORMAT_RGB, raw.format)
        assertContentEquals(ByteArray(12), raw.bytes)
        assertEquals("LlamaImage.Rgb(2x2)", LlamaImage.Rgb(2, 2, ByteArray(12)).toString())
        assertEquals("LlamaImage.Encoded(4 bytes)", LlamaImage.Encoded(ByteArray(4)).toString())

        assertFailsWith<IllegalArgumentException> { LlamaEngineOptions(projectorPath = "") }
        assertFailsWith<IllegalArgumentException> { LlamaEngineOptions(imageMinTokens = -1) }
        assertFailsWith<IllegalArgumentException> { LlamaEngineOptions(imageMaxTokens = -1) }
        assertFailsWith<IllegalArgumentException> { LlamaEngineOptions(imageMinTokens = 10, imageMaxTokens = 5) }
        LlamaEngineOptions(imageMinTokens = 10, imageMaxTokens = 0) // 0 = projector default
        assertEquals("<__media__>", LlamaEngine.MEDIA_MARKER)
    }
}
