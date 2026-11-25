package com.donfuldauer.kggml.samples.showcase.llm

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.ManualMemoryPressureObserver
import com.donfuldauer.kggml.core.MemoryPressureLevel
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.samples.showcase.models.ResidentModels
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

@OptIn(ExperimentalCoroutinesApi::class)
class LlmChatControllerTest {
    private class Harness(
        scope: TestScope,
        val factory: FakeChatModelFactory = FakeChatModelFactory(),
        val observer: ManualMemoryPressureObserver = ManualMemoryPressureObserver(),
        files: Map<String, ByteArray> = emptyMap(),
        val imageSource: ImageSource = FakeImageSource(),
        modelPath: String = "/models/qwen.gguf",
    ) {
        val residents = ResidentModels(1L shl 40, observer)
        val controller = LlmChatController(
            scope = CoroutineScope(StandardTestDispatcher(scope.testScheduler) + Job()),
            factory = factory,
            residents = residents,
            ioDispatcher = StandardTestDispatcher(scope.testScheduler),
            imageSource = imageSource,
            readFile = { path -> files[path] ?: throw IllegalArgumentException("No such file: $path") },
            initialModelPath = modelPath,
        )
        val state get() = controller.state.value
        val model get() = factory.opened.last()

        fun say(text: String) {
            controller.setDraft(text)
            controller.send()
        }
    }

    @Test
    fun initialStateNeedsAModelAndADraft() = runTest {
        val h = Harness(this, modelPath = "")
        assertFalse(h.state.canSend)
        h.controller.setDraft("hi")
        assertFalse(h.state.canSend, "no model path")
        h.controller.setModelPath("/m.gguf")
        assertTrue(h.state.canSend)
        assertNull(h.state.loaded)
        assertEquals(BackendConfig.AUTO_THREADS, h.state.cpuThreads)
        assertEquals(BackendConfig.ALL_GPU_LAYERS, h.state.gpuLayers)
    }

    @Test
    fun multiTurnChatStreamsRepliesAndReusesTheEngine() = runTest {
        val h = Harness(this)
        h.say("Hello there")
        assertTrue(h.state.isBusy)
        assertFalse(h.state.canSend)
        assertEquals("", h.state.draft)
        advanceUntilIdle()
        assertEquals(ChatActivity.IDLE, h.state.activity)
        assertEquals(
            listOf(ChatMessage(ChatRole.USER, "Hello there"), ChatMessage(ChatRole.ASSISTANT, "Echo: Hello there")),
            h.state.messages,
        )
        assertNull(h.state.streaming)
        assertContains(h.state.status!!, "3 token(s), stopped: end of generation")
        assertEquals(ComputeBackend.GPU_METAL, h.state.loaded!!.backend)

        h.say("How are you?")
        advanceUntilIdle()
        assertEquals(4, h.state.messages.size)
        assertEquals("Echo: How are you?", h.state.messages.last().text)
        assertEquals(1, h.factory.opened.size, "the resident engine is reused across turns")
        // The second prompt re-renders the whole conversation (so the engine reuses the cached prefix).
        val second = h.model.calls.last()
        assertTrue(second.prompt.startsWith(h.model.calls.first().prompt.removeSuffix("<|im_start|>assistant\n")))
        assertContains(second.prompt, "<|im_start|>assistant\nEcho: Hello there<|im_end|>")
        assertEquals(listOf("<|im_end|>"), second.options.stopSequences)
        assertEquals(256, second.options.maxTokens)
    }

    @Test
    fun streamingShowsPartialTextAndCancelKeepsIt() = runTest {
        val gate = Channel<Unit>(Channel.UNLIMITED)
        val h = Harness(this, FakeChatModelFactory { it.tokenGate = gate; it.reply = { "one two three four" } })
        h.say("count")
        advanceUntilIdle()
        assertEquals(ChatActivity.GENERATING, h.state.activity)
        assertEquals("", h.state.streaming)
        gate.send(Unit)
        gate.send(Unit)
        advanceUntilIdle()
        assertEquals("one two", h.state.streaming)
        h.controller.cancel()
        advanceUntilIdle()
        assertEquals(ChatActivity.IDLE, h.state.activity)
        assertNull(h.state.streaming)
        assertEquals("Cancelled", h.state.status)
        assertEquals(ChatMessage(ChatRole.ASSISTANT, "one two"), h.state.messages.last())
    }

    @Test
    fun inBandErrorIsShownAndPartialReplyKept() = runTest {
        val h = Harness(this, FakeChatModelFactory { it.streamError = InvalidInputException("prompt too long") })
        h.say("hi")
        advanceUntilIdle()
        assertContains(h.state.error!!, "InvalidInputException: prompt too long")
        assertEquals(ChatMessage(ChatRole.ASSISTANT, "Echo:"), h.state.messages.last())
    }

    @Test
    fun modelLoadFailureIsReported() = runTest {
        val h = Harness(this, FakeChatModelFactory(failure = ModelLoadException("not a GGUF file")))
        h.say("hi")
        advanceUntilIdle()
        assertContains(h.state.error!!, "ModelLoadException: not a GGUF file")
        assertNull(h.state.loaded)
        assertTrue(h.residents.state.value.entries.isEmpty())
    }

    @Test
    fun threadAndGpuLayerChangesReloadTheEngineWithTheNewConfig() = runTest {
        val h = Harness(this)
        h.controller.loadModel()
        advanceUntilIdle()
        val first = h.model
        assertEquals(BackendConfig(), first.spec.config)

        h.controller.setCpuThreads(4)
        h.controller.setGpuLayers(12)
        assertTrue(h.state.needsReload)
        h.controller.loadModel()
        advanceUntilIdle()
        assertTrue(first.closed, "the engine with the old configuration is closed")
        val second = h.model
        assertEquals(BackendConfig(cpuThreads = 4, gpuLayers = 12), second.spec.config)
        assertEquals(4, h.state.loaded!!.config.cpuThreads)
        assertFalse(h.state.needsReload)

        // CPU_ONLY always loads with 0 GPU layers.
        h.controller.setBackend(ComputeBackend.CPU_ONLY)
        h.say("hi")
        advanceUntilIdle()
        assertTrue(second.closed)
        assertEquals(BackendConfig(backend = ComputeBackend.CPU_ONLY, gpuLayers = 0, cpuThreads = 4), h.model.spec.config)
        assertEquals(ComputeBackend.CPU_ONLY, h.state.loaded!!.backend)
        assertEquals(listOf(h.model.spec.key), h.residents.state.value.entries.map { it.key })
    }

    @Test
    fun threadAndLayerSettersClampToValidValues() = runTest {
        val h = Harness(this)
        h.controller.setCpuThreads(0)
        assertEquals(BackendConfig.AUTO_THREADS, h.state.cpuThreads)
        h.controller.setCpuThreads(1000)
        assertEquals(LlmChatController.MAX_THREADS, h.state.cpuThreads)
        h.controller.setGpuLayers(-5)
        assertEquals(BackendConfig.ALL_GPU_LAYERS, h.state.gpuLayers)
        h.controller.setMaxTokens(1)
        assertEquals(16, h.state.maxTokens)
    }

    @Test
    fun snapshotSaveAndRestoreBringBackKvCacheAndConversation() = runTest {
        val h = Harness(this)
        h.say("first")
        advanceUntilIdle()
        h.controller.saveSnapshot()
        advanceUntilIdle()
        val snapshot = h.state.snapshots.single()
        assertEquals(2, snapshot.messages.size)
        assertContentEquals("session:1".encodeToByteArray(), snapshot.bytes)
        assertContains(h.state.status!!, "Saved snapshot #1")

        h.say("second")
        advanceUntilIdle()
        assertEquals(4, h.state.messages.size)
        h.controller.restoreSnapshot(snapshot.id)
        advanceUntilIdle()
        assertEquals(2, h.state.messages.size)
        assertContentEquals(snapshot.bytes, h.model.restored.single())
        assertContains(h.state.status!!, "Restored snapshot #1")

        h.controller.deleteSnapshot(snapshot.id)
        assertTrue(h.state.snapshots.isEmpty())
    }

    @Test
    fun clearConversationClearsTheKvCache() = runTest {
        val h = Harness(this)
        h.say("first")
        advanceUntilIdle()
        h.controller.clearConversation()
        advanceUntilIdle()
        assertTrue(h.state.messages.isEmpty())
        assertEquals(1, h.model.sessionClears)
        assertFalse(h.state.canSaveSnapshot)
    }

    @Test
    fun evictedEngineIsReloadedOnNextUse() = runTest {
        val h = Harness(this)
        h.say("first")
        advanceUntilIdle()
        val first = h.model
        h.observer.emit(MemoryPressureLevel.CRITICAL)
        assertTrue(first.closed, "an idle engine is closed on an OS low-memory signal")
        assertTrue(h.residents.state.value.entries.isEmpty())

        h.controller.loadModel()
        advanceUntilIdle()
        assertEquals(2, h.factory.opened.size)
        assertEquals("Model was evicted from memory and has been reloaded", h.state.status)
        h.say("second")
        advanceUntilIdle()
        assertEquals(2, h.factory.opened.size)
        assertEquals("Echo: second", h.state.messages.last().text)
    }

    @Test
    fun leasedEngineSurvivesLowMemoryDuringGeneration() = runTest {
        val gate = Channel<Unit>(Channel.UNLIMITED)
        val h = Harness(this, FakeChatModelFactory { it.tokenGate = gate })
        h.say("hi")
        advanceUntilIdle()
        h.observer.emit(MemoryPressureLevel.CRITICAL)
        assertFalse(h.model.closed, "an engine in use is never evicted")
        repeat(3) { gate.send(Unit) }
        advanceUntilIdle()
        assertEquals("Echo: hi", h.state.messages.last().text)
    }

    @Test
    fun jsonSchemaExtractionUsesAGrammarAndValidatesLive() = runTest {
        val gate = Channel<Unit>(Channel.UNLIMITED)
        val h = Harness(this, FakeChatModelFactory { it.tokenGate = gate })
        h.controller.extract()
        advanceUntilIdle()
        assertEquals(ChatActivity.EXTRACTING, h.state.activity)
        gate.send(Unit)
        advanceUntilIdle()
        assertIs<JsonValidation.Incomplete>(h.state.validation, "a half-streamed object is incomplete, not invalid")
        repeat(10) { gate.send(Unit) }
        advanceUntilIdle()
        assertEquals("""{"name": "Maria", "age": 34, "city": "Lisbon"}""", h.state.extractionOutput)
        assertIs<JsonValidation.Valid>(h.state.validation)
        val call = h.model.calls.single()
        assertNotNull(call.options.grammar)
        assertEquals(0f, call.options.temperature)
        assertContains(call.prompt, "Maria is 34 years old")
        assertContains(h.state.status!!, "Valid JSON")
    }

    @Test
    fun extractionReportsSchemaViolationsAndInvalidJson() = runTest {
        val h = Harness(this, FakeChatModelFactory { it.jsonReply = """{"name": "Maria", "age": "old"}""" })
        h.controller.extract()
        advanceUntilIdle()
        val violation = assertIs<JsonValidation.SchemaViolation>(h.state.validation)
        assertEquals(listOf("$.city is required", "$.age must be an integer"), violation.problems)

        h.controller.setExtractionOutput("""{"name": "x",, }""")
        assertIs<JsonValidation.Invalid>(h.state.validation)
        h.controller.setExtractionOutput("""{"name": "x", "age": 1, "city": "y"}""")
        assertIs<JsonValidation.Valid>(h.state.validation)
    }

    @Test
    fun invalidSchemaBlocksExtraction() = runTest {
        val h = Harness(this)
        h.controller.setSchema("{ not json")
        assertNotNull(h.state.schemaError)
        h.controller.extract()
        advanceUntilIdle()
        assertContains(h.state.error!!, "Schema is not valid JSON")
        assertTrue(h.factory.opened.isEmpty())
    }

    @Test
    fun embeddingsSimilarityIsComputed() = runTest {
        val h = Harness(this)
        h.controller.setEmbedTextA("abc")
        h.controller.setEmbedTextB("abc")
        h.controller.compareEmbeddings()
        advanceUntilIdle()
        assertEquals(1f, h.state.similarity!!, 1e-5f)
        assertEquals(26, h.state.embeddingDimensions)
        h.controller.setEmbedTextB("xyz")
        assertNull(h.state.similarity, "editing a text clears the stale result")
        h.controller.compareEmbeddings()
        advanceUntilIdle()
        assertEquals(0f, h.state.similarity!!, 1e-5f)
    }

    @Test
    fun visionAnalysisPassesImageBytesInMemory() = runTest {
        val png = FakeImageSource.PNG_MAGIC + ByteArray(8)
        val h = Harness(this, files = mapOf("/photos/cat.png" to png))
        h.controller.setProjectorPath("/models/mmproj.gguf")
        h.controller.setImagePath("/photos/cat.png")
        h.controller.loadImageFromPath()
        advanceUntilIdle()
        assertEquals("PNG · 16 B · cat.png", h.state.image!!.label)
        h.controller.analyzeImage()
        advanceUntilIdle()
        val call = h.model.calls.single()
        assertContentEquals(png, call.images.single())
        assertContains(call.prompt, "<__media__>\nDescribe this image in one sentence.")
        assertEquals("An image of 1 picture(s), 16 bytes.", h.state.visionAnswer)
        assertTrue(h.state.loaded!!.supportsVision)
    }

    @Test
    fun visionWithoutProjectorFailsClearly() = runTest {
        val h = Harness(this)
        h.controller.attachImage(FakeImageSource.JPEG_MAGIC + ByteArray(4), "test")
        h.controller.analyzeImage()
        advanceUntilIdle()
        assertContains(h.state.error!!, "no vision projector")
    }

    @Test
    fun galleryAndCameraImagesAreAttached() = runTest {
        val h = Harness(this)
        assertTrue(h.controller.canPickFromGallery)
        assertTrue(h.controller.canCapture)
        h.controller.pickFromGallery()
        advanceUntilIdle()
        assertContains(h.state.image!!.label, "PNG")
        assertContains(h.state.image!!.label, "Photo library")
        h.controller.captureFromCamera()
        advanceUntilIdle()
        assertContains(h.state.image!!.label, "JPEG")
        h.controller.clearImage()
        assertNull(h.state.image)
    }

    @Test
    fun cancelledPickerLeavesNoImageAndUnreadableFileIsReported() = runTest {
        val h = Harness(this, imageSource = FakeImageSource(galleryImage = null))
        h.controller.pickFromGallery()
        advanceUntilIdle()
        assertNull(h.state.image)
        h.controller.setImagePath("/missing.png")
        h.controller.loadImageFromPath()
        advanceUntilIdle()
        assertContains(h.state.error!!, "Cannot read image")
    }

    @Test
    fun closeRemovesTheResidentEngine() = runTest {
        val h = Harness(this)
        h.controller.loadModel()
        advanceUntilIdle()
        h.controller.close()
        assertTrue(h.model.closed)
        assertTrue(h.residents.state.value.entries.isEmpty())
    }

    @Test
    fun cosineSimilarityHandlesZeroVectors() {
        assertEquals(0f, cosineSimilarity(FloatArray(3), floatArrayOf(1f, 0f, 0f)))
        assertEquals(-1f, cosineSimilarity(floatArrayOf(1f, 0f), floatArrayOf(-2f, 0f)), 1e-6f)
    }
}
