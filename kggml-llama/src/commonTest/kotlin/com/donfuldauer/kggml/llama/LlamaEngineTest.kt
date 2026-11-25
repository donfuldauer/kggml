package com.donfuldauer.kggml.llama

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.BackendUnavailableException
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.NativeFailureException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import com.donfuldauer.kggml.core.StopReason
import com.donfuldauer.kggml.core.TokenEvent
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.Dispatchers
import kotlin.concurrent.atomics.AtomicInt
import kotlin.coroutines.CoroutineContext
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

class LlamaEngineTest {
    private suspend fun open(
        bridge: FakeLlamaBridge,
        config: BackendConfig = BackendConfig.CPU,
        available: Set<ComputeBackend> = setOf(ComputeBackend.CPU_ONLY),
        contextConfig: LlamaContextConfig = LlamaContextConfig(contextLength = 64, batchSize = 8),
        options: LlamaEngineOptions = LlamaEngineOptions(),
    ): LlamaEngine = LlamaEngine.open(
        modelPath = "/models/tiny.gguf",
        backendConfig = config,
        contextConfig = contextConfig,
        options = options,
        bridges = LlamaBridgeHolder { bridge },
        availableBackends = { available },
        dispatcherFactory = SerialDispatcherFactory.Default,
    )

    /** Polls [condition] in real time (engines run on real dedicated threads). */
    private suspend fun awaitCondition(condition: () -> Boolean) = withContext(Dispatchers.Default) {
        withTimeout(10.seconds) { while (!condition()) delay(1) }
    }

    private suspend fun LlamaEngine.events(prompt: String, options: GenerationOptions = GenerationOptions.GREEDY) =
        withContext(Dispatchers.Default) { generate(prompt, options).toList() }

    private fun List<TokenEvent>.text() = filterIsInstance<TokenEvent.Token>().joinToString("") { it.text }

    // ---- lifecycle & configuration ----------------------------------------------------------------------

    @Test
    fun cpuConfigHandsNoGpuLayersAndMapsThreadsAndFlags() = runTest {
        val bridge = FakeLlamaBridge()
        val config = BackendConfig(ComputeBackend.CPU_ONLY, gpuLayers = 0, cpuThreads = 3, useFlashAttention = true, useMmap = false)
        open(bridge, config, options = LlamaEngineOptions(useMlock = true)).use { engine ->
            val (path, params) = bridge.loads.items.single()
            assertEquals("/models/tiny.gguf", path)
            assertEquals(RawModelParams(LlamaBackendKind.CPU, 0, 0, useMmap = false, useMlock = true), params)
            val ctx = bridge.contexts.items.single()
            assertEquals(64, ctx.contextLength)
            assertEquals(8, ctx.batchSize)
            assertEquals(3, ctx.threads)
            assertEquals(3, ctx.batchThreads)
            assertEquals(1, ctx.flashAttention)
            assertFalse(ctx.embeddings)
            assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
            assertFalse(engine.usesGpu)
            assertEquals(64, engine.contextLength)
            assertEquals("fake 1B F32", engine.modelInfo.description)
            assertEquals(300, engine.modelInfo.vocabularySize)
        }
    }

    @Test
    fun gpuConfigOffloadsTheRequestedLayersOnTheChosenDevice() = runTest {
        val bridge = FakeLlamaBridge()
        val config = BackendConfig(ComputeBackend.GPU_METAL, gpuLayers = 12)
        open(bridge, config, available = setOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_METAL), options = LlamaEngineOptions(gpuDevice = 1)).use { engine ->
            assertEquals(RawModelParams(LlamaBackendKind.METAL, 12, 1, useMmap = true, useMlock = false), bridge.loads.items.single().second)
            assertEquals(0, bridge.contexts.items.single().threads, "AUTO_THREADS defers to the native performance-core count")
            assertEquals(0, bridge.contexts.items.single().flashAttention)
            assertEquals(ComputeBackend.GPU_METAL, engine.activeBackend)
            assertTrue(engine.usesGpu)
        }
    }

    @Test
    fun autoPrefersTheGpuAndCoreMlFallsBackBecauseLlamaCppHasNoCoreMlPath() = runTest {
        val bridge = FakeLlamaBridge()
        open(bridge, BackendConfig.DEFAULT, available = setOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_VULKAN)).use {
            assertEquals(ComputeBackend.GPU_VULKAN, it.activeBackend)
        }
        val coreMl = FakeLlamaBridge()
        open(coreMl, BackendConfig(ComputeBackend.NPU_COREML), available = setOf(ComputeBackend.CPU_ONLY)).use {
            assertEquals(ComputeBackend.CPU_ONLY, it.activeBackend)
        }
    }

    @Test
    fun nativeBackendUnavailableFallsBackOrFailsTyped() = runTest {
        val metalBroken = { kind: Int -> if (kind == LlamaBackendKind.METAL) LlamaStatus.BACKEND_UNAVAILABLE else LlamaStatus.OK }
        val available = setOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_METAL)
        val bridge = FakeLlamaBridge(loadStatus = metalBroken)
        open(bridge, BackendConfig(ComputeBackend.GPU_METAL), available).use { engine ->
            assertEquals(listOf(LlamaBackendKind.METAL, LlamaBackendKind.CPU), bridge.loads.items.map { it.second.backend })
            assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
            assertEquals(0, bridge.loads.items.last().second.gpuLayers, "the CPU fallback must not offload layers")
        }
        val strict = FakeLlamaBridge(loadStatus = metalBroken)
        val e = assertFailsWith<BackendUnavailableException> {
            open(strict, BackendConfig(ComputeBackend.GPU_METAL), available, options = LlamaEngineOptions(allowBackendFallback = false))
        }
        assertEquals(ComputeBackend.GPU_METAL, e.backend)
    }

    @Test
    fun loadFailureIsTypedAndLeaksNothing() = runTest {
        val bridge = FakeLlamaBridge(loadStatus = { LlamaStatus.MODEL_LOAD_FAILED })
        val e = assertFailsWith<ModelLoadException> { open(bridge) }
        assertTrue(e.message!!.contains("cannot load /models/tiny.gguf"), e.message)
        assertTrue(bridge.live.isEmpty())
    }

    @Test
    fun closeFreesEverythingOnceChildrenFirstAndIsIdempotent() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("hi"))
        val engine = open(bridge)
        engine.embed("abc")
        assertEquals("hi", engine.events("x").text())
        assertEquals(3, bridge.live.size) // model + generation context + embedding context
        engine.close()
        engine.close()
        assertTrue(engine.isClosed)
        awaitCondition { bridge.live.isEmpty() }
        assertEquals(listOf("context", "context", "model"), bridge.frees.items.filter { it != "sampler" })
        assertFailsWith<IllegalStateException> { engine.generate("x") }
        assertFailsWith<IllegalStateException> { engine.tokenize("x") }
    }

    @Test
    fun enginesAreIndependentInstances() = runTest {
        val a = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("aaaa"))
        val b = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("bb"))
        val first = open(a)
        val second = open(b)
        val results = listOf(first, second).map { e -> async(Dispatchers.Default) { e.generateText("go", GenerationOptions.GREEDY) } }.awaitAll()
        assertEquals(listOf("aaaa", "bb"), results.map { it.text })
        first.close()
        assertEquals("bb", second.generateText("again", GenerationOptions.GREEDY).text, "closing one engine leaves the other usable")
        second.close()
        awaitCondition { a.live.isEmpty() && b.live.isEmpty() }
    }

    // ---- generation ---------------------------------------------------------------------------------------

    @Test
    fun streamsTokensThenCompletesAtEndOfGeneration() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("ok!"))
        open(bridge).use { engine ->
            val events = engine.events("hi")
            assertEquals(4, events.size)
            assertEquals(TokenEvent.Token(0, FakeLlamaBridge.tokensOf("o").single(), "o"), events[0])
            assertEquals(TokenEvent.Completed(3, StopReason.END_OF_GENERATION), events.last())
            assertEquals("ok!", events.text())
            // Prompt = BOS + "hi" in one batch with logits for the last token, then one decode per emitted token
            // (the end-of-generation token is never evaluated).
            val decodes = bridge.decodes.items
            assertEquals(listOf(FakeLlamaBridge.BOS) + FakeLlamaBridge.tokensOf("hi"), decodes[0].tokens)
            assertEquals(0, decodes[0].pos0)
            assertTrue(decodes.all { it.logitsLast && it.seqId == 0 })
            assertEquals(listOf(3, 4, 5), decodes.drop(1).map { it.pos0 })
            assertEquals(6, engine.sessionTokenCount)
            awaitCondition { bridge.live.values.none { it == "sampler" } }
        }
    }

    @Test
    fun maxTokensStopsGeneration() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("abcdefgh"))
        open(bridge).use { engine ->
            val result = engine.generateText("p", GenerationOptions.GREEDY.copy(maxTokens = 3))
            assertEquals("abc", result.text)
            assertEquals(StopReason.MAX_TOKENS, result.stopReason)
            assertContentEquals(FakeLlamaBridge.tokensOf("abc").toIntArray(), result.tokens)
        }
    }

    @Test
    fun stopSequenceSpanningTokensStopsGeneration() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("one\n\nUser: two"))
        open(bridge).use { engine ->
            val result = engine.generateText("p", GenerationOptions.GREEDY.copy(stopSequences = listOf("User:", "zzz")))
            assertEquals(StopReason.STOP_SEQUENCE, result.stopReason)
            assertEquals("one\n\nUser:", result.text)
        }
    }

    @Test
    fun fullContextCompletesWithContextFull() = runTest {
        val bridge = FakeLlamaBridge(script = List(100) { FakeLlamaBridge.tokensOf("x").single() }, contextSize = 10)
        open(bridge, contextConfig = LlamaContextConfig(contextLength = 10)).use { engine ->
            val result = engine.generateText("abc", GenerationOptions.GREEDY.copy(maxTokens = GenerationOptions.UNLIMITED))
            assertEquals(StopReason.CONTEXT_FULL, result.stopReason)
            assertEquals(10 - 4 + 1, result.tokens.size) // 4 prompt tokens, 6 decoded outputs, 1 final emitted token
            assertEquals(10, engine.sessionTokenCount)
        }
    }

    @Test
    fun promptLongerThanTheContextIsInvalidInput() = runTest {
        val bridge = FakeLlamaBridge(contextSize = 8)
        open(bridge, contextConfig = LlamaContextConfig(contextLength = 8)).use { engine ->
            val error = assertIs<TokenEvent.Error>(engine.events("abcdefghij").single())
            assertIs<InvalidInputException>(error.exception)
            assertFailsWith<InvalidInputException> { engine.generateText("abcdefghij") }
            assertTrue(bridge.decodes.items.isEmpty())
        }
    }

    @Test
    fun multiByteCharactersSplitAcrossTokensAreHeldBackUntilComplete() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("é🙂"))
        open(bridge).use { engine ->
            val tokens = engine.events("p").filterIsInstance<TokenEvent.Token>()
            assertEquals(6, tokens.size)
            assertEquals(listOf("", "é", "", "", "", "🙂"), tokens.map { it.text })
        }
    }

    @Test
    fun followUpPromptsReuseTheCachedPrefix() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("ab"))
        open(bridge).use { engine ->
            engine.generateText("Q1", GenerationOptions.GREEDY)
            // Session: BOS Q 1 a b (the end-of-generation token is never evaluated).
            assertEquals(5, engine.sessionTokenCount)
            bridge.decodes.items.size.let { before ->
                engine.generateText("Q1aMORE", GenerationOptions.GREEDY)
                val prompt = bridge.decodes.items[before]
                assertEquals(4, prompt.pos0, "only the new suffix is evaluated")
                assertEquals(FakeLlamaBridge.tokensOf("MORE"), prompt.tokens)
            }
            bridge.decodes.items.size.let { before ->
                engine.generateText("Q2", GenerationOptions.GREEDY)
                val prompt = bridge.decodes.items[before]
                assertEquals(2, prompt.pos0, "the diverging tail is dropped from the KV cache")
                assertEquals(FakeLlamaBridge.tokensOf("2"), prompt.tokens)
            }
            bridge.decodes.items.size.let { before ->
                engine.generateText("Q2", GenerationOptions.GREEDY.copy(maxTokens = 1))
                val prompt = bridge.decodes.items[before]
                assertEquals(2, prompt.pos0, "an identical prompt still re-evaluates its last token for fresh logits")
            }
            bridge.decodes.items.size.let { before ->
                val clears = bridge.clears.load()
                engine.generateText("Q2", GenerationOptions.GREEDY.copy(reusePromptCache = false))
                assertEquals(0, bridge.decodes.items[before].pos0)
                assertEquals(clears + 1, bridge.clears.load())
            }
        }
    }

    @Test
    fun samplingOptionsAndGrammarReachTheNativeSampler() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("{}"))
        open(bridge).use { engine ->
            val grammar = LlamaGrammar.gbnf("root ::= \"{}\"")
            val options = GenerationOptions(
                temperature = 0.5f, topK = 7, topP = 0.9f, minP = 0.1f, repeatPenalty = 1.1f, repeatLastN = 2,
                seed = 0xFFFFFFFEL, grammar = grammar,
            )
            engine.generateText("abc", options)
            val raw = bridge.samplers.items.single()
            assertEquals(RawSamplerParams(0.5f, 7, 0.9f, 0.1f, 1.1f, 2, 0xFFFFFFFE.toInt(), "root ::= \"{}\"", "root"), raw)
            // With a repetition penalty the last repeatLastN prompt tokens seed the penalty history, outside the grammar.
            assertEquals(FakeLlamaBridge.tokensOf("bc").map { it to false }, bridge.accepted.items)
            engine.generateText("abc", GenerationOptions.DEFAULT)
            assertEquals(-1, bridge.samplers.items.last().seed, "null seed = llama.cpp's random-seed sentinel 0xFFFFFFFF")
            assertEquals(null, bridge.samplers.items.last().grammar)
        }
    }

    @Test
    fun rejectedGrammarIsATerminalInvalidInputError() = runTest {
        val bridge = FakeLlamaBridge(samplerStatus = LlamaStatus.INVALID_ARGUMENT)
        open(bridge).use { engine ->
            val error = assertIs<TokenEvent.Error>(engine.events("p", GenerationOptions(grammar = LlamaGrammar.gbnf("root ::= ("))).single())
            assertIs<InvalidInputException>(error.exception)
            assertTrue(error.exception.message!!.contains("grammar does not parse"), error.exception.message)
        }
    }

    @Test
    fun nativeDecodeFailureDropsThePartialBatchAndKeepsTheEngineUsable() = runTest {
        var fail = true
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("z"))
        bridge.onDecode = { _, _, _ -> if (fail) LlamaStatus.NATIVE_FAILURE else LlamaStatus.OK }
        open(bridge).use { engine ->
            val error = assertIs<TokenEvent.Error>(engine.events("abc").single())
            assertIs<NativeFailureException>(error.exception)
            assertEquals(0, engine.sessionTokenCount)
            assertEquals(0, bridge.kvLength(bridge.live.entries.first { it.value == "context" }.key))
            fail = false
            assertEquals("z", engine.generateText("abc", GenerationOptions.GREEDY).text)
        }
    }

    // ---- cancellation -------------------------------------------------------------------------------------

    @Test
    fun cancellingTheCollectorAbortsTheRunningNativeDecode() = runTest {
        val bridge = FakeLlamaBridge(script = List(50) { FakeLlamaBridge.tokensOf("y").single() })
        bridge.onDecode = { b, ctx, pos ->
            if (pos > 0) {
                // Block like a long llama_decode until the abort callback would fire.
                while (!b.abortRequested(ctx)) { /* spin */ }
                LlamaStatus.CANCELLED
            } else {
                LlamaStatus.OK
            }
        }
        open(bridge).use { engine ->
            val received = ConcurrentLog<TokenEvent>()
            val job = launch(Dispatchers.Default) { engine.generate("p", GenerationOptions.GREEDY).collect { received.add(it) } }
            awaitCondition { engine.isGenerating && bridge.decodes.items.size >= 2 && received.items.size == 1 }
            job.cancelAndJoin()
            assertEquals(1, received.items.size, "only the token sampled before the blocked decode")
            assertFalse(engine.isGenerating)
            // The cancelled token is not part of the session; the engine stays usable.
            assertEquals(2, engine.sessionTokenCount)
            bridge.onDecode = { _, _, _ -> LlamaStatus.OK }
            bridge.script = FakeLlamaBridge.tokensOf("k")
            assertEquals("k", engine.generateText("p", GenerationOptions.GREEDY).text)
            awaitCondition { bridge.live.values.none { it == "sampler" } }
        }
    }

    @Test
    fun abortActiveGenerationEndsTheFlowWithACancelledError() = runTest {
        val bridge = FakeLlamaBridge(script = List(50) { FakeLlamaBridge.tokensOf("y").single() })
        bridge.onDecode = { b, ctx, _ ->
            while (!b.abortRequested(ctx)) { /* spin */ }
            LlamaStatus.CANCELLED
        }
        open(bridge).use { engine ->
            assertFalse(engine.abortActiveGeneration(), "nothing to abort while idle")
            val events = async(Dispatchers.Default) { engine.generate("p").toList() }
            awaitCondition { engine.isGenerating }
            assertTrue(engine.abortActiveGeneration())
            val error = assertIs<TokenEvent.Error>(events.await().single())
            assertIs<InferenceCancelledException>(error.exception)
        }
    }

    @Test
    fun closeDuringGenerationEndsTheFlowAndFreesEverything() = runTest {
        val bridge = FakeLlamaBridge(script = List(50) { FakeLlamaBridge.tokensOf("y").single() })
        bridge.onDecode = { b, ctx, pos ->
            if (pos > 0) while (!b.abortRequested(ctx)) { /* spin */ }
            if (pos > 0) LlamaStatus.CANCELLED else LlamaStatus.OK
        }
        val engine = open(bridge)
        val events = async(Dispatchers.Default) { engine.generate("p").toList() }
        awaitCondition { bridge.decodes.items.size >= 2 }
        engine.close()
        val last = assertIs<TokenEvent.Error>(events.await().last())
        assertIs<InferenceCancelledException>(last.exception)
        awaitCondition { bridge.live.isEmpty() }
    }

    @Test
    fun collectorCancelledWhileSuspendedDoesNotLeakTheSampler() = runTest {
        val bridge = FakeLlamaBridge(script = List(500) { FakeLlamaBridge.tokensOf("y").single() }, contextSize = 1024)
        open(bridge, contextConfig = LlamaContextConfig(contextLength = 1024)).use { engine ->
            // first() cancels the producer as soon as one token arrived, typically while it is suspended in send().
            val first = withContext(Dispatchers.Default) { engine.generate("p", GenerationOptions.GREEDY.copy(maxTokens = 400)).first() }
            assertIs<TokenEvent.Token>(first)
            awaitCondition { bridge.live.values.none { it == "sampler" } }
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
        val bridge = FakeLlamaBridge(script = List(50) { FakeLlamaBridge.tokensOf("y").single() })
        val dispatchers = ConcurrentLog<NativeLikeDispatcher>()
        val engine = LlamaEngine.open(
            modelPath = "/models/tiny.gguf",
            backendConfig = BackendConfig.CPU,
            contextConfig = LlamaContextConfig(contextLength = 64, batchSize = 8),
            options = LlamaEngineOptions(),
            bridges = LlamaBridgeHolder { bridge },
            availableBackends = { setOf(ComputeBackend.CPU_ONLY) },
            dispatcherFactory = SerialDispatcherFactory { NativeLikeDispatcher().also(dispatchers::add) },
        )
        // Rendezvous: the generation is suspended in send() on the engine thread whenever the collector holds a token.
        val events = withContext(Dispatchers.Default) {
            engine.generate("p", GenerationOptions.GREEDY.copy(maxTokens = GenerationOptions.UNLIMITED))
                .buffer(Channel.RENDEZVOUS)
                .onEach { if (!engine.isClosed) engine.close() }
                .toList()
        }
        assertIs<InferenceCancelledException>(assertIs<TokenEvent.Error>(events.last()).exception)
        val dispatcher = dispatchers.items.single()
        awaitCondition { bridge.live.isEmpty() && dispatcher.closed.load() != 0 }
        assertEquals(0, dispatcher.lateDispatches.load(), "the engine thread was shut down under a suspended generation")
        assertFailsWith<IllegalStateException> { engine.generate("p") }
    }

    @Test
    fun callsQueuedBehindAGenerationFailOnceTheEngineCloses() = runTest {
        val bridge = FakeLlamaBridge(script = List(50) { FakeLlamaBridge.tokensOf("y").single() })
        bridge.onDecode = { b, ctx, pos ->
            if (pos > 0) while (!b.abortRequested(ctx)) { /* spin */ }
            if (pos > 0) LlamaStatus.CANCELLED else LlamaStatus.OK
        }
        val engine = open(bridge)
        val generation = async(Dispatchers.Default) { engine.generate("p").toList() }
        awaitCondition { bridge.decodes.items.size >= 2 }
        val queued = async(Dispatchers.Default) { runCatching { engine.embed("abc") } }
        delay(50)
        engine.close()
        assertIs<TokenEvent.Error>(generation.await().last())
        assertIs<IllegalStateException>(queued.await().exceptionOrNull())
        awaitCondition { bridge.live.isEmpty() }
    }

    // ---- sessions -----------------------------------------------------------------------------------------

    @Test
    fun sessionRoundTripsIntoAnotherEngineAndContinuesFromThePrefix() = runTest {
        val source = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("ab"))
        val saved = open(source).use { engine ->
            engine.generateText("hello", GenerationOptions.GREEDY)
            engine.saveSession()
        }
        val target = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("c"))
        open(target).use { engine ->
            engine.restoreSession(saved)
            assertEquals(8, engine.sessionTokenCount) // BOS hello a b
            engine.generateText("helloab, more", GenerationOptions.GREEDY)
            val prompt = target.decodes.items.first()
            assertEquals(8, prompt.pos0)
            assertEquals(FakeLlamaBridge.tokensOf(", more"), prompt.tokens)
            engine.clearSession()
            assertEquals(0, engine.sessionTokenCount)
        }
    }

    @Test
    fun corruptOrForeignSessionsAreRejected() = runTest {
        val other = FakeLlamaBridge(vocab = 999)
        val foreign = open(other).use { it.saveSession() }
        open(FakeLlamaBridge()).use { engine ->
            assertFailsWith<InvalidInputException> { engine.restoreSession(foreign) }
            assertFailsWith<InvalidInputException> { engine.restoreSession(byteArrayOf(1, 2, 3)) }
            // A payload llama.cpp rejects (here: the fake's state magic) empties the session.
            val bridge = FakeLlamaBridge()
            val rejected = SessionCodec.encode(SessionCodec.Fingerprint(300, 4, 1), intArrayOf(1, 2), byteArrayOf(0x00, 2))
            engine.generateText("abc", GenerationOptions.GREEDY)
            assertFailsWith<InvalidInputException> { engine.restoreSession(rejected) }
            assertEquals(0, engine.sessionTokenCount)
            assertTrue(bridge.live.isEmpty())
        }
    }

    // ---- embeddings & tokenizer ---------------------------------------------------------------------------

    @Test
    fun embeddingsUseADedicatedContextAndLeaveTheSessionAlone() = runTest {
        val bridge = FakeLlamaBridge(script = FakeLlamaBridge.tokensOf("a"))
        val config = LlamaContextConfig(contextLength = 64, embeddingPooling = LlamaPooling.MEAN, embeddingContextLength = 16)
        open(bridge, contextConfig = config).use { engine ->
            engine.generateText("abc", GenerationOptions.GREEDY)
            val session = engine.sessionTokenCount
            assertContentEquals(floatArrayOf(1f, 0f, 0f, 0f), engine.embed("hello"))
            assertContentEquals(floatArrayOf(6f, 0f, 0f, 0f), engine.embed("hello", normalize = false))
            val embCtx = bridge.contexts.items.last()
            assertTrue(embCtx.embeddings)
            assertEquals(LlamaPooling.MEAN.nativeValue, embCtx.pooling)
            assertEquals(16, embCtx.contextLength)
            assertEquals(16, embCtx.batchSize)
            assertEquals(16, embCtx.microBatchSize)
            assertEquals(2, bridge.contexts.items.size, "the embedding context is created once")
            assertEquals(session, engine.sessionTokenCount)
            assertFailsWith<InvalidInputException> { engine.embed("x".repeat(16)) }
        }
    }

    @Test
    fun tokenizeAndDetokenizeRoundTrip() = runTest {
        open(FakeLlamaBridge()).use { engine ->
            val tokens = engine.tokenize("héllo")
            assertEquals(FakeLlamaBridge.tokensOf("héllo"), tokens.toList())
            assertEquals(FakeLlamaBridge.BOS, engine.tokenize("x", addSpecial = true).first())
            assertEquals("héllo", engine.detokenize(tokens))
            assertEquals("<s>x", engine.detokenize(engine.tokenize("x", addSpecial = true), renderSpecial = true))
        }
    }

    @Test
    fun nativeInfoReportsThePinnedUpstream() {
        val info = LlamaEngine.nativeInfo(FakeLlamaBridge())
        assertEquals("b8598", info.llamaCppVersion)
        assertEquals("9281dd135d6f175ceacca7d3a3ec440012264cfe", info.llamaCppCommit)
    }
}
