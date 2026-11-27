package com.donfuldauer.kggml.llama

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.BackendUnavailableException
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.NativeFailureException
import com.donfuldauer.kggml.core.OutOfMemoryException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import com.donfuldauer.kggml.core.StopReason
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext

/** Session-restore validation, KV-cache exhaustion mid-generation, value types and bridge helpers of kggml-llama. */
class LlamaEdgeCaseTest {
    private suspend fun open(
        bridge: NativeLlamaBridge,
        contextLength: Int = 64,
        options: LlamaEngineOptions = LlamaEngineOptions(),
    ): LlamaEngine = LlamaEngine.open(
        modelPath = "/models/tiny.gguf",
        backendConfig = BackendConfig.CPU,
        contextConfig = LlamaContextConfig(contextLength = contextLength, batchSize = 8),
        options = options,
        bridges = LlamaBridgeHolder { bridge },
        availableBackends = { setOf(ComputeBackend.CPU_ONLY) },
        dispatcherFactory = SerialDispatcherFactory.Default,
    )

    private val fakeFingerprint = SessionCodec.Fingerprint(300, 4, 1) // FakeLlamaBridge defaults

    // ---- sessions -----------------------------------------------------------------------------------------

    @Test
    fun sessionLongerThanTheContextIsRejectedAndLeavesTheCurrentSessionUntouched() = runTest {
        val saved = open(FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("ab"))).use { engine ->
            engine.generateText("hello", GenerationOptions.GREEDY)
            assertEquals(8, engine.sessionTokenCount) // BOS hello a b
            engine.saveSession()
        }
        val target = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("z"), contextSize = 8)
        open(target, contextLength = 8).use { engine ->
            engine.generateText("ab", GenerationOptions.GREEDY.copy(maxTokens = 1))
            val before = engine.sessionTokenCount
            assertEquals(4, before) // BOS a b z
            val clears = target.clears.load()
            val e = assertFailsWith<InvalidInputException> { engine.restoreSession(saved) }
            assertEquals("session holds 8 tokens; the context fits 8", e.message)
            assertEquals(before, engine.sessionTokenCount)
            assertEquals(clears, target.clears.load(), "rejected before reaching llama.cpp: the KV cache is not cleared")
            assertEquals(before, target.kvLength(target.decodes.items.first().context))
        }
    }

    @Test
    fun kvCacheThatDisagreesWithTheTokenHistoryIsRejectedAndEmptiesTheSession() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("a"))
        open(bridge).use { engine ->
            engine.generateText("abc", GenerationOptions.GREEDY.copy(maxTokens = 1))
            val context = bridge.decodes.items.first().context
            // The fake's state blob restores 5 KV positions, but the container claims a 3-token history.
            val inconsistent = SessionCodec.encode(fakeFingerprint, intArrayOf(1, 20, 21), byteArrayOf(0x51, 5))
            val e = assertFailsWith<InvalidInputException> { engine.restoreSession(inconsistent) }
            assertEquals("session KV cache holds 5 positions, token history 3", e.message)
            assertEquals(0, engine.sessionTokenCount)
            assertEquals(0, bridge.kvLength(context))
        }
    }

    // ---- generation -------------------------------------------------------------------------------------

    @Test
    fun kvCacheExhaustedWhileDecodingAGeneratedTokenCompletesWithContextFull() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("xyzw"))
        // llama.cpp reports a full KV cache (no slot for the batch) when the third generated token is decoded.
        bridge.onDecode = { _, _, pos0 -> if (pos0 == 6) LlamaStatus.BUFFER_TOO_SMALL else LlamaStatus.OK }
        open(bridge).use { engine ->
            val result = withContext(Dispatchers.Default) {
                engine.generateText("abc", GenerationOptions.GREEDY.copy(maxTokens = GenerationOptions.UNLIMITED))
            }
            assertEquals(StopReason.CONTEXT_FULL, result.stopReason)
            assertEquals("xyz", result.text, "the token whose decode failed was already emitted")
            assertEquals(FakeLlamaBridge.tokensOf("xyz"), result.tokens.toList())
            // Session: BOS a b c x y; the partially decoded z was dropped from the KV cache again.
            assertEquals(6, engine.sessionTokenCount)
            val context = bridge.decodes.items.first().context
            assertEquals(6, bridge.kvLength(context))

            // The engine stays usable and the cached prefix is reused: only "z" is evaluated.
            bridge.onDecode = { _, _, _ -> LlamaStatus.OK }
            val before = bridge.decodes.items.size
            engine.generateText("abcxyz", GenerationOptions.GREEDY.copy(maxTokens = 1))
            val prompt = bridge.decodes.items[before]
            assertEquals(6, prompt.pos0)
            assertEquals(FakeLlamaBridge.tokensOf("z"), prompt.tokens)
        }
    }

    @Test
    fun multimodalGenerateTextUsesTheDefaultOptions() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("ok"))
        open(bridge, options = LlamaEngineOptions(projectorPath = "/models/mmproj.gguf")).use { engine ->
            val image = LlamaImage.Encoded(FakeLlamaBridge.IMAGE_MAGIC + byteArrayOf(9))
            val result = withContext(Dispatchers.Default) {
                engine.generateText("look: ${LlamaEngine.MEDIA_MARKER}", listOf(image))
            }
            assertEquals("ok", result.text)
            assertEquals(StopReason.END_OF_GENERATION, result.stopReason)
            assertEquals(GenerationOptions.DEFAULT.toRaw(), bridge.samplers.items.single())
            assertEquals(-1, bridge.samplers.items.single().seed, "random seed sentinel 0xFFFFFFFF")
            assertEquals(1, bridge.evals.items.size)
        }
    }

    // ---- value types --------------------------------------------------------------------------------------

    @Test
    fun generationResultComparesTokensByContent() {
        val a = GenerationResult("hi", intArrayOf(1, 2), StopReason.MAX_TOKENS)
        val b = GenerationResult("hi", intArrayOf(1, 2), StopReason.MAX_TOKENS)
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
        assertNotEquals(a, GenerationResult("hi", intArrayOf(1, 3), StopReason.MAX_TOKENS))
        assertNotEquals(a, GenerationResult("ho", intArrayOf(1, 2), StopReason.MAX_TOKENS))
        assertNotEquals(a, GenerationResult("hi", intArrayOf(1, 2), StopReason.END_OF_GENERATION))
        assertNotEquals<Any>(a, "hi")
    }

    @Test
    fun grammarEqualityHashAndToString() {
        val g = LlamaGrammar.gbnf("start ::= \"x\"", root = "start")
        assertEquals(LlamaGrammar.gbnf("start ::= \"x\"", root = "start").hashCode(), g.hashCode())
        assertNotEquals(LlamaGrammar.gbnf("start ::= \"x\"", root = "Start-2"), g)
        assertNotEquals<Any>(g, "start ::= \"x\"")
        assertEquals("LlamaGrammar(root=start, gbnf=13 chars)", g.toString())
        assertFailsWith<IllegalArgumentException> { LlamaGrammar.gbnf("r ::= \"x\"", root = "") }
        assertFailsWith<IllegalArgumentException> { LlamaGrammar.gbnf("r ::= \"x\"", root = "a.b") }
    }

    @Test
    fun generationOptionsRejectNaNAndAcceptTheBoundaries() {
        for (bad in listOf<() -> Unit>(
            { GenerationOptions(topP = Float.NaN) },
            { GenerationOptions(minP = Float.NaN) },
            { GenerationOptions(repeatPenalty = Float.NaN) },
            { GenerationOptions(repeatPenalty = -1f) },
            { GenerationOptions(maxTokens = -2) },
        )) {
            assertFailsWith<IllegalArgumentException> { bad() }
        }
        val edge = GenerationOptions(seed = GenerationOptions.MAX_SEED, topP = 0f, minP = 1f, repeatLastN = 0)
        assertEquals(0xFFFFFFFE.toInt(), edge.toRaw().seed)
        assertEquals(42, GenerationOptions(seed = 42).toRaw().seed)
        assertNull(GenerationOptions().toRaw().grammar)
        val json = LlamaGrammar.json()
        val raw = GenerationOptions(grammar = json).toRaw()
        assertEquals(json.gbnf, raw.grammar)
        assertEquals("root", raw.grammarRoot)
    }

    @Test
    fun imagesValidateTheirShapeAndMapToRawFormats() {
        assertFailsWith<IllegalArgumentException> { LlamaImage.Encoded(ByteArray(0)) }
        assertFailsWith<IllegalArgumentException> { LlamaImage.Rgb(0, 1, ByteArray(0)) }
        assertFailsWith<IllegalArgumentException> { LlamaImage.Rgb(1, -1, ByteArray(0)) }
        assertFailsWith<IllegalArgumentException> { LlamaImage.Rgb(2, 2, ByteArray(11)) }
        val rgb = LlamaImage.Rgb(2, 2, ByteArray(12))
        assertEquals("LlamaImage.Rgb(2x2)", rgb.toString())
        assertEquals(RawImage.FORMAT_RGB, rgb.toRaw().format)
        assertSame(rgb.pixels, rgb.toRaw().bytes, "pixels are handed over without a copy")
        val encoded = LlamaImage.Encoded(byteArrayOf(1, 2, 3))
        assertEquals("LlamaImage.Encoded(3 bytes)", encoded.toString())
        assertEquals(RawImage.FORMAT_ENCODED, encoded.toRaw().format)
        assertSame(encoded.bytes, encoded.toRaw().bytes)
    }

    // ---- bridge helpers -----------------------------------------------------------------------------------

    @Test
    fun backendKindsMapBothWays() {
        assertEquals(LlamaBackendKind.COREML, LlamaBackendKind.of(ComputeBackend.NPU_COREML))
        for (backend in listOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_METAL, ComputeBackend.GPU_VULKAN, ComputeBackend.NPU_COREML)) {
            assertEquals(backend, LlamaBackendKind.toComputeBackend(LlamaBackendKind.of(backend)))
        }
        assertNull(LlamaBackendKind.toComputeBackend(42))
        assertFailsWith<IllegalStateException> { LlamaBackendKind.of(ComputeBackend.AUTO) }
    }

    @Test
    fun nativeErrorsAreTypedAndCarryStatusNameAndDiagnostic() {
        val bridge = FakeLlamaBridge()
        val plain = bridge.nativeError(LlamaStatus.INVALID_ARGUMENT, "kggml_op")
        assertIs<InvalidInputException>(plain)
        assertEquals("kggml_op failed: STATUS_2 (2)", plain.message)

        bridge.lastErrorText = "bad tensor"
        assertEquals("kggml_op failed: STATUS_1 (1): bad tensor", assertIs<ModelLoadException>(bridge.nativeError(LlamaStatus.MODEL_LOAD_FAILED, "kggml_op")).message)

        val unavailable = assertIs<BackendUnavailableException>(bridge.nativeError(LlamaStatus.BACKEND_UNAVAILABLE, "kggml_op"))
        assertNull(unavailable.backend)
        val metal = assertIs<BackendUnavailableException>(bridge.nativeError(LlamaStatus.BACKEND_UNAVAILABLE, "kggml_op", ComputeBackend.GPU_METAL))
        assertEquals(ComputeBackend.GPU_METAL, metal.backend)

        bridge.check(LlamaStatus.OK, "kggml_op")
        val e = assertFailsWith<OutOfMemoryException> { bridge.check(LlamaStatus.OUT_OF_MEMORY, "kggml_alloc") }
        assertEquals("kggml_alloc failed: STATUS_4 (4): bad tensor", e.message)
    }

    @Test
    fun nativeErrorSurvivesABridgeWhoseDiagnosticsThrow() {
        val broken = object : NativeLlamaBridge by FakeLlamaBridge() {
            override fun lastError(): String = throw IllegalStateException("no diagnostic")
            override fun statusName(status: Int): String = throw IllegalStateException("no name")
        }
        val e = broken.nativeError(99, "kggml_op")
        val failure = assertIs<NativeFailureException>(e, "unknown codes are native failures")
        assertEquals("kggml_op failed: KGGML_STATUS_99 (99)", failure.message)
    }

    @Test
    fun bridgeHolderCreatesOnceAndPassesTypedFailuresThrough() {
        var created = 0
        val bridge = FakeLlamaBridge()
        val holder = LlamaBridgeHolder { created++; bridge }
        assertSame(bridge, holder.get())
        assertSame(bridge, holder.get())
        assertEquals(1, created)

        val typed = ModelLoadException("already typed")
        assertSame(typed, assertFailsWith<ModelLoadException> { LlamaBridgeHolder { throw typed }.get() })

        val anonymous = assertFailsWith<NativeFailureException> { LlamaBridgeHolder { throw IllegalStateException() }.get() }
        assertEquals("Failed to load the Kggml llama native library: IllegalStateException", anonymous.message)
        assertIs<IllegalStateException>(anonymous.cause)
    }
}
