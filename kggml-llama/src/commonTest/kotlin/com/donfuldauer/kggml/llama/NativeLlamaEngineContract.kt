package com.donfuldauer.kggml.llama

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.StopReason
import com.donfuldauer.kggml.core.TokenEvent
import com.donfuldauer.kggml.llama.grammar.Json
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.TestResult
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext

/**
 * [LlamaEngine] against the real native library (llama.cpp + shared ggml) driven by the synthetic [TinyLlamaModel]
 * fixture. Shared by every platform: `jvmTest` (JNI) and `iosSimulatorArm64Test` (cinterop) subclass it and supply
 * file I/O and the decision whether the native library is present.
 */
abstract class NativeLlamaEngineContract {
    /** Writes [bytes] to a fresh temporary file and returns its path. */
    protected abstract fun writeTempFile(prefix: String, bytes: ByteArray): String

    protected abstract fun deleteFile(path: String)

    /** A path that exists but is a directory, not a model. */
    protected abstract fun directoryPath(): String

    protected lateinit var modelPath: String
    protected lateinit var mmprojPath: String
    private lateinit var garbagePath: String

    @BeforeTest
    fun writeFixtures() {
        modelPath = writeTempFile("kggml-llama-tiny-", TinyLlamaModel.bytes())
        mmprojPath = writeTempFile("kggml-llama-tiny-mmproj-", TinyMmproj.bytes())
        garbagePath = writeTempFile("kggml-llama-garbage-", "this is not a GGUF model".encodeToByteArray())
    }

    @AfterTest
    fun deleteFixtures() {
        if (::modelPath.isInitialized) deleteFile(modelPath)
        if (::mmprojPath.isInitialized) deleteFile(mmprojPath)
        if (::garbagePath.isInitialized) deleteFile(garbagePath)
    }

    private fun nativeTest(block: suspend CoroutineScope.() -> Unit): TestResult = runTest(timeout = 120.seconds) {
        withContext(Dispatchers.Default, block)
    }

    protected suspend fun cpuEngine(contextConfig: LlamaContextConfig = SMALL): LlamaEngine =
        LlamaEngine(modelPath, BackendConfig(ComputeBackend.CPU_ONLY, gpuLayers = 0, cpuThreads = 2), contextConfig)

    /** CPU engine with the tiny vision projector. */
    protected suspend fun visionEngine(contextConfig: LlamaContextConfig = SMALL): LlamaEngine = LlamaEngine(
        modelPath,
        BackendConfig(ComputeBackend.CPU_ONLY, gpuLayers = 0, cpuThreads = 2),
        contextConfig,
        LlamaEngineOptions(projectorPath = mmprojPath),
    )

    private suspend fun LlamaEngine.greedy(prompt: String, maxTokens: Int = 8, reuse: Boolean = true): GenerationResult =
        generateText(prompt, GenerationOptions.GREEDY.copy(maxTokens = maxTokens, reusePromptCache = reuse))

    @Test
    fun nativeInfoMatchesThePinnedUpstream() {
        val info = LlamaEngine.nativeInfo()
        assertEquals("b8598", info.llamaCppVersion)
        assertEquals("9281dd135d6f175ceacca7d3a3ec440012264cfe", info.llamaCppCommit)
        assertTrue(info.systemInfo.isNotEmpty())
    }

    @Test
    fun cpuOnlyLoadsTheTinyModelWithoutAnyGpu() = nativeTest {
        cpuEngine().use { engine ->
            assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
            assertFalse(engine.usesGpu, "CPU_ONLY hands llama.cpp an empty device list")
            val info = engine.modelInfo
            assertEquals(TinyLlamaModel.VOCAB, info.vocabularySize)
            assertEquals(TinyLlamaModel.EMBD, info.embeddingSize)
            assertEquals(TinyLlamaModel.CTX_TRAIN, info.trainContextLength)
            assertEquals(1, info.layerCount)
            assertEquals(TinyLlamaModel.BOS, info.bosToken)
            assertEquals(TinyLlamaModel.EOS, info.eosToken)
            assertFalse(info.hasEncoder)
            assertTrue(info.description.isNotEmpty())
            assertEquals(256, engine.contextLength)
        }
    }

    @Test
    fun unloadableModelsFailTyped() = nativeTest {
        for (path in listOf(modelPath + ".missing", garbagePath, directoryPath())) {
            assertFailsWith<ModelLoadException>(path) { LlamaEngine(path, BackendConfig.CPU, SMALL) }
        }
    }

    @Test
    fun tokenizerRoundTripsThroughTheSentencePieceVocabulary() = nativeTest {
        cpuEngine().use { engine ->
            val tokens = engine.tokenize("ab z")
            assertContentEquals(
                intArrayOf(TinyLlamaModel.SPACE, TinyLlamaModel.LETTER_BASE, TinyLlamaModel.LETTER_BASE + 1, TinyLlamaModel.SPACE_Z),
                tokens,
            )
            assertEquals(TinyLlamaModel.BOS, engine.tokenize("ab", addSpecial = true).first())
            assertEquals(" ab z", engine.detokenize(tokens))
            // Non-ASCII text falls back to byte tokens and still round-trips.
            assertEquals(" héllo 🙂", engine.detokenize(engine.tokenize("héllo 🙂")))
        }
    }

    @Test
    fun greedyStreamingIsDeterministicAcrossRunsAndEngines() = nativeTest {
        val first = cpuEngine().use { engine ->
            val events = engine.generate("ab", GenerationOptions.GREEDY.copy(maxTokens = 12)).toList()
            val completed = assertIs<TokenEvent.Completed>(events.last())
            val tokens = events.filterIsInstance<TokenEvent.Token>()
            assertEquals(completed.tokenCount, tokens.size)
            assertEquals(tokens.indices.toList(), tokens.map { it.index })
            assertTrue(tokens.all { it.tokenId in 0 until TinyLlamaModel.VOCAB })
            assertEquals(engine.greedy("ab", 12, reuse = false), engine.greedy("ab", 12, reuse = false))
            engine.greedy("ab", 12)
        }
        val second = cpuEngine().use { it.greedy("ab", 12) }
        assertEquals(first, second)
    }

    @Test
    fun concurrentEnginesAreIndependent() = nativeTest {
        val expected = cpuEngine().use { it.greedy("ab z", 16) }
        val engines = List(3) { cpuEngine() }
        try {
            val results = engines.map { e -> async { e.greedy("ab z", 16) } }.awaitAll()
            results.forEach { assertEquals(expected, it) }
        } finally {
            engines.forEach { it.close() }
        }
    }

    @Test
    fun gbnfGrammarConstrainsEverySampledToken() = nativeTest {
        cpuEngine().use { engine ->
            val grammar = LlamaGrammar.gbnf("root ::= \"{\\\"\" [a-z]{1,6} \"\\\"}\"")
            for (temperature in listOf(0f, 0.9f)) {
                val options = GenerationOptions(temperature = temperature, seed = 42, maxTokens = 64, grammar = grammar)
                val result = engine.generateText("ab", options)
                assertEquals(StopReason.END_OF_GENERATION, result.stopReason, "a completed grammar admits only EOG")
                assertTrue(Regex("\\{\"[a-z]{1,6}\"\\}").matches(result.text), "'${result.text}' violates the grammar")
            }
        }
    }

    @Test
    fun jsonSchemaConstrainedOutputIsValidJson() = nativeTest {
        val schema = """
            {"type":"object","properties":{
              "ok":{"type":"boolean"},
              "n":{"type":"integer"},
              "tag":{"enum":["red","green"]},
              "word":{"type":"string","maxLength":5}
            },"required":["ok","n","tag","word"]}
        """
        cpuEngine().use { engine ->
            for (seed in 1L..3L) {
                val options = GenerationOptions(temperature = 1f, seed = seed, maxTokens = 400, grammar = LlamaGrammar.jsonSchema(schema))
                val result = engine.generateText("ab", options)
                assertEquals(StopReason.END_OF_GENERATION, result.stopReason)
                val doc = assertIs<Json.Obj>(Json.parse(result.text.trim()), result.text)
                assertEquals(listOf("ok", "n", "tag", "word"), doc.members.keys.toList())
                assertIs<Json.Bool>(doc["ok"])
                assertTrue(assertIs<Json.Num>(doc["n"]).raw.matches(Regex("-?(0|[1-9][0-9]*)")))
                assertTrue(doc["tag"] in listOf(Json.Str("red"), Json.Str("green")))
                // maxLength counts code points as llama.cpp's grammar decodes them (GBNF `char` = one code point or one
                // escape). The random-weight model also emits stray bytes, which Kotlin decodes as U+FFFD.
                val word = assertIs<Json.Str>(doc["word"]).value
                if ('\uFFFD' !in word) assertTrue(word.count { !it.isLowSurrogate() } <= 5, "'$word' is longer than 5 code points")
            }
        }
    }

    @Test
    fun savedSessionRestoresIntoAnotherEngineAndContinuesIdentically() = nativeTest {
        val prompt = "ab z ab"
        lateinit var followUp: String
        lateinit var continued: GenerationResult
        var savedCount = 0
        val saved = cpuEngine().use { engine ->
            followUp = prompt + engine.greedy(prompt, 6).text + " z"
            val blob = engine.saveSession()
            savedCount = engine.sessionTokenCount
            continued = engine.greedy(followUp, 6)
            blob
        }
        cpuEngine().use { engine ->
            engine.restoreSession(saved)
            assertEquals(savedCount, engine.sessionTokenCount)
            assertEquals(continued, engine.greedy(followUp, 6))
        }
        cpuEngine().use { fresh ->
            assertEquals(continued, fresh.greedy(followUp, 6, reuse = false), "prefix reuse must not change results")
        }
    }

    @Test
    fun corruptOrForeignSessionsAreRejectedWithoutTouchingTheSession() = nativeTest {
        cpuEngine().use { engine ->
            engine.greedy("ab", 4)
            val blob = engine.saveSession()
            val count = engine.sessionTokenCount
            val foreignModel = blob.copyOf().also { it[8] = (it[8] + 1).toByte() } // n_vocab of the fingerprint
            // A damaged KV payload is caught by the container checksum before llama.cpp parses its sizes.
            val damaged = blob.copyOf().also { it.fill(0x7F, it.size - 64, it.size) }
            for (bad in listOf(blob.copyOf(blob.size - 3), foreignModel, damaged)) {
                assertFailsWith<InvalidInputException> { engine.restoreSession(bad) }
                assertEquals(count, engine.sessionTokenCount, "a rejected container leaves the session untouched")
            }
            engine.restoreSession(blob)
            assertTrue(engine.sessionTokenCount > 0)
            engine.greedy("ab", 4) // still usable
        }
    }

    @Test
    fun embeddingsAreNormalizedDeterministicAndInputDependent() = nativeTest {
        cpuEngine(SMALL.copy(embeddingPooling = LlamaPooling.MEAN)).use { engine ->
            engine.greedy("ab", 4)
            val session = engine.sessionTokenCount
            val a = engine.embed("ab z")
            assertEquals(TinyLlamaModel.EMBD, a.size)
            assertTrue(abs(sqrt(a.sumOf { (it * it).toDouble() }) - 1.0) < 1e-4)
            assertContentEquals(a, engine.embed("ab z"))
            assertFalse(a.contentEquals(engine.embed("z ab ab")))
            val raw = engine.embed("ab z", normalize = false)
            assertFalse(abs(sqrt(raw.sumOf { (it * it).toDouble() }) - 1.0) < 1e-4)
            assertEquals(session, engine.sessionTokenCount, "embeddings never touch the generation session")
        }
        cpuEngine().use { lastToken -> assertEquals(TinyLlamaModel.EMBD, lastToken.embed("ab").size) }
    }

    @Test
    fun contextFullEndsGeneration() = nativeTest {
        cpuEngine(LlamaContextConfig(contextLength = 32, batchSize = 8)).use { engine ->
            // A grammar that can never complete makes end-of-generation inadmissible, so the outcome does not depend on
            // what the random-weight model would emit on this CPU's float path.
            val result = engine.generateText("ab z", ENDLESS)
            assertEquals(StopReason.CONTEXT_FULL, result.stopReason)
            assertEquals(engine.contextLength, engine.sessionTokenCount)
        }
    }

    @Test
    fun collectorCancellationStopsGenerationAndKeepsTheEngineUsable() = nativeTest {
        cpuEngine().use { engine ->
            val taken = engine.generate("ab", ENDLESS)
                .buffer(Channel.RENDEZVOUS)
                .take(3)
                .toList()
            assertEquals(3, taken.size)
            assertFalse(engine.isGenerating)
            // The prompt + the tokens decoded before the producer saw the cancellation (it runs at most one ahead).
            val prompt = engine.tokenize("ab", addSpecial = true).size
            assertTrue(engine.sessionTokenCount in prompt + 2..prompt + 3, "session holds ${engine.sessionTokenCount} tokens")
            assertEquals(engine.greedy("ab", 5, reuse = false), engine.greedy("ab", 5, reuse = false))
        }
    }

    @Test
    fun abortActiveGenerationEndsTheStreamWithACancelledError() = nativeTest {
        cpuEngine().use { engine ->
            var aborted = false
            val events = engine.generate("ab", ENDLESS)
                .buffer(Channel.RENDEZVOUS)
                .onEach { if (!aborted) aborted = engine.abortActiveGeneration() }
                .toList()
            assertTrue(aborted)
            val error = assertIs<TokenEvent.Error>(events.last())
            assertIs<InferenceCancelledException>(error.exception)
            engine.greedy("ab", 2) // still usable
        }
    }

    @Test
    fun closeDuringGenerationEndsTheStreamAndRejectsFurtherCalls() = nativeTest {
        val engine = cpuEngine()
        val events = engine.generate("ab", ENDLESS)
            .buffer(Channel.RENDEZVOUS)
            .onEach { if (!engine.isClosed) engine.close() }
            .toList()
        val error = assertIs<TokenEvent.Error>(events.last())
        assertIs<InferenceCancelledException>(error.exception)
        assertFailsWith<IllegalStateException> { engine.generate("ab").collect() }
        assertFailsWith<IllegalStateException> { engine.embed("ab") }
        engine.close()
    }

    // ---- multimodal (libmtmd) ----------------------------------------------------------------------------

    @Test
    fun visionProjectorLoadsNextToTheModel() = nativeTest {
        visionEngine().use { engine ->
            assertTrue(engine.supportsVision)
            assertEquals(ComputeBackend.CPU_ONLY, engine.activeBackend)
            assertFalse(engine.usesGpu)
        }
        cpuEngine().use { assertFalse(it.supportsVision) }
    }

    @Test
    fun unloadableProjectorsFailTyped() = nativeTest {
        for (path in listOf(garbagePath, modelPath, directoryPath(), "$garbagePath.missing")) {
            assertFailsWith<ModelLoadException>(path) {
                LlamaEngine(modelPath, BackendConfig.CPU, SMALL, LlamaEngineOptions(projectorPath = path))
            }
        }
    }

    @Test
    fun inMemoryImagePromptsGenerateDeterministically() = nativeTest {
        val pixels = TinyMmproj.rgbPattern(TinyMmproj.IMAGE_SIZE, TinyMmproj.IMAGE_SIZE, 1)
        val image = LlamaImage.Rgb(TinyMmproj.IMAGE_SIZE, TinyMmproj.IMAGE_SIZE, pixels)
        val prompt = "ab ${LlamaEngine.MEDIA_MARKER} z"
        val options = GenerationOptions.GREEDY.copy(maxTokens = 6, grammar = LETTERS)
        visionEngine().use { a ->
            visionEngine().use { b ->
                val first = a.generateText(prompt, listOf(image), options)
                assertEquals(6, first.tokens.size)
                assertTrue(first.text.isNotEmpty())
                val textTokens = a.tokenize("ab ", addSpecial = true).size
                assertTrue(a.sessionTokenCount >= textTokens + TinyMmproj.IMAGE_TOKENS + 6, "session ${a.sessionTokenCount}")
                assertEquals(first.text, a.generateText(prompt, listOf(image), options).text, "same engine, same output")
                assertEquals(first.text, b.generateText(prompt, listOf(image), options).text, "independent engines agree")
                // Without a marker one is prepended: same images, different text layout, still valid.
                assertEquals(6, a.generateText("ab z", listOf(image), options).tokens.size)
            }
        }
    }

    @Test
    fun encodedImagesDecodeInMemoryLikeTheirPixels() = nativeTest {
        val options = GenerationOptions.GREEDY.copy(maxTokens = 6, grammar = LETTERS)
        visionEngine().use { engine ->
            for ((w, h) in listOf(28 to 28, 45 to 19)) {
                val pixels = TinyMmproj.rgbPattern(w, h, 3)
                val raw = engine.generateText("ab", listOf(LlamaImage.Rgb(w, h, pixels)), options)
                val bmp = engine.generateText("ab", listOf(LlamaImage.Encoded(TinyMmproj.bmp(pixels, w, h))), options)
                assertEquals(raw.tokens.toList(), bmp.tokens.toList(), "${w}x$h: BMP is lossless")
            }
        }
    }

    @Test
    fun invalidImagePromptsFailTypedAndKeepTheEngineUsable() = nativeTest {
        val good = LlamaImage.Rgb(TinyMmproj.IMAGE_SIZE, TinyMmproj.IMAGE_SIZE, TinyMmproj.rgbPattern(TinyMmproj.IMAGE_SIZE, TinyMmproj.IMAGE_SIZE, 1))
        visionEngine().use { engine ->
            val undecodable = engine.generate("ab", listOf(LlamaImage.Encoded("not an image".encodeToByteArray()))).toList()
            assertIs<InvalidInputException>(assertIs<TokenEvent.Error>(undecodable.single()).exception)
            val mismatch = engine.generate("${LlamaEngine.MEDIA_MARKER} ${LlamaEngine.MEDIA_MARKER}", listOf(good)).toList()
            assertIs<InvalidInputException>(assertIs<TokenEvent.Error>(mismatch.single()).exception)
            val tooMany = engine.generate("ab", List(80) { good }).toList() // 80 * 4 embeddings > 256 positions
            assertIs<InvalidInputException>(assertIs<TokenEvent.Error>(tooMany.single()).exception)
            assertEquals(4, engine.generateText("ab", listOf(good), GenerationOptions.GREEDY.copy(maxTokens = 4, grammar = LETTERS)).tokens.size)
        }
        cpuEngine().use { engine ->
            val noProjector = engine.generate("ab", listOf(good)).toList()
            assertIs<InvalidInputException>(assertIs<TokenEvent.Error>(noProjector.single()).exception)
        }
    }

    @Test
    fun imagePromptGenerationIsCancellable() = nativeTest {
        val image = LlamaImage.Rgb(TinyMmproj.IMAGE_SIZE, TinyMmproj.IMAGE_SIZE, TinyMmproj.rgbPattern(TinyMmproj.IMAGE_SIZE, TinyMmproj.IMAGE_SIZE, 2))
        visionEngine().use { engine ->
            val taken = engine.generate("ab", listOf(image), ENDLESS).buffer(Channel.RENDEZVOUS).take(2).toList()
            assertEquals(2, taken.size)
            var aborted = false
            val events = engine.generate("ab", listOf(image), ENDLESS)
                .buffer(Channel.RENDEZVOUS)
                .onEach { if (!aborted) aborted = engine.abortActiveGeneration() }
                .toList()
            assertTrue(aborted)
            assertIs<InferenceCancelledException>(assertIs<TokenEvent.Error>(events.last()).exception)
            // The image session saves and restores like a text one.
            val saved = engine.saveSession()
            val count = engine.sessionTokenCount
            engine.clearSession()
            engine.restoreSession(saved)
            assertEquals(count, engine.sessionTokenCount)
        }
    }

    protected companion object {
        /**
         * Greedy, unlimited and constrained by a grammar that never completes, so end-of-generation is inadmissible:
         * generation only stops on cancellation or a full context, whatever the model would emit on this CPU.
         */
        val ENDLESS: GenerationOptions =
            GenerationOptions.GREEDY.copy(maxTokens = GenerationOptions.UNLIMITED, grammar = LlamaGrammar.gbnf("root ::= [a-z] root"))

        /** Letters only and never complete, so a `maxTokens` limit always produces exactly that many tokens. */
        val LETTERS: LlamaGrammar = LlamaGrammar.gbnf("root ::= [a-z] root")

        val SMALL: LlamaContextConfig = LlamaContextConfig(contextLength = 256, batchSize = 16, embeddingContextLength = 64)
    }
}
