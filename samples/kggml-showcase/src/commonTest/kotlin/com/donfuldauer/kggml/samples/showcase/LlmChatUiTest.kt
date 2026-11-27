package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.assertTextEquals
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.runComposeUiTest
import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.samples.showcase.llm.FakeChatModelFactory
import com.donfuldauer.kggml.samples.showcase.llm.FakeImageSource
import com.donfuldauer.kggml.samples.showcase.llm.LlmChatTags
import com.donfuldauer.kggml.samples.showcase.llm.LlmPanel
import kotlinx.coroutines.channels.Channel
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Compose UI tests of the LLM Chat, Structured Output & Vision screen with a deterministic LLM (`LlmFakes.kt`):
 * multi-turn streaming, cancel, thread / GPU-layer reloads, KV snapshots, JSON-schema extraction with live
 * validation, embedding similarity and in-memory image analysis.
 */
@OptIn(ExperimentalTestApi::class)
class LlmChatUiTest : UiTestBase() {
    private fun androidx.compose.ui.test.ComposeUiTest.send(text: String) {
        typeInto(LlmChatTags.DRAFT, text)
        click(LlmChatTags.SEND)
    }

    @Test
    fun multiTurnChatRendersStreamedReplies() = runComposeUiTest {
        val fakes = FakeShowcase()
        showFakeShowcase(fakes, ShowcaseDestination.LLM_CHAT)
        onNodeWithTag(LlmChatTags.SEND).assertIsNotEnabled()
        send("Hello there")
        waitForTag(LlmChatTags.message(1))
        onNodeWithTag(LlmChatTags.message(0)).assertTextContains("Hello there", substring = true)
        onNodeWithTag(LlmChatTags.message(1)).assertTextContains("Echo: Hello there", substring = true)
        onNodeWithTag(LlmChatTags.LOADED).assertTextContains("fake-llm 0.5B Q4_K_M · Metal (GPU on)", substring = true)
        onNodeWithTag(LlmChatTags.STATUS).assertTextContains("3 token(s), stopped: end of generation")

        send("Second turn")
        waitForTag(LlmChatTags.message(3))
        onNodeWithTag(LlmChatTags.message(3)).assertTextContains("Echo: Second turn", substring = true)
        assertEquals(1, fakes.chatFactory.opened.size)
    }

    @Test
    fun streamingTextIsShownAndCancelStopsIt() = runComposeUiTest {
        val gate = Channel<Unit>(Channel.UNLIMITED)
        val fakes = FakeShowcase(chatFactory = FakeChatModelFactory { it.tokenGate = gate; it.reply = { "one two three" } })
        showFakeShowcase(fakes, ShowcaseDestination.LLM_CHAT)
        send("count")
        waitForTag(LlmChatTags.STREAMING)
        gate.trySend(Unit)
        waitForText(LlmChatTags.STREAMING, "one")
        onNodeWithTag(LlmChatTags.ACTIVITY).assertTextEquals("Generating…")
        onNodeWithTag(LlmChatTags.SEND).assertIsNotEnabled()
        click(LlmChatTags.CANCEL)
        waitForText(LlmChatTags.STATUS, "Cancelled")
        onNodeWithTag(LlmChatTags.STREAMING).assertDoesNotExist()
        onNodeWithTag(LlmChatTags.message(1)).assertTextContains("one", substring = true)
        onNodeWithTag(LlmChatTags.ACTIVITY).assertTextEquals("Idle")
    }

    @Test
    fun threadsAndGpuLayersReloadTheModel() = runComposeUiTest {
        val fakes = FakeShowcase()
        showFakeShowcase(fakes, ShowcaseDestination.LLM_CHAT)
        click(LlmChatTags.LOAD)
        waitForText(LlmChatTags.LOADED, "threads auto · GPU layers all")
        onNodeWithTag(LlmChatTags.THREADS).assertTextEquals("Auto")
        click("${LlmChatTags.THREADS}-inc")
        click("${LlmChatTags.THREADS}-inc")
        onNodeWithTag(LlmChatTags.THREADS).assertTextEquals("2")
        click("${LlmChatTags.GPU_LAYERS}-dec")
        onNodeWithTag(LlmChatTags.GPU_LAYERS).assertTextEquals("32")
        waitForText(LlmChatTags.LOADED, "Settings changed: reloads on next use")
        click(LlmChatTags.LOAD)
        waitForText(LlmChatTags.LOADED, "threads 2 · GPU layers 32")
        assertEquals(2, fakes.chatFactory.opened.size)
        assertTrue(fakes.chatFactory.opened.first().closed)
        assertEquals(BackendConfig(cpuThreads = 2, gpuLayers = 32), fakes.chatFactory.opened.last().spec.config)

        click("${LlmChatTags.BACKEND}-cpu_only")
        onNodeWithTag(LlmChatTags.GPU_LAYERS).assertTextEquals("0 (CPU)")
        click(LlmChatTags.LOAD)
        waitForText(LlmChatTags.LOADED, "CPU (GPU off)")
        assertEquals(ComputeBackend.CPU_ONLY, fakes.chatFactory.opened.last().spec.config.backend)
    }

    @Test
    fun snapshotsSaveAndRestoreTheConversation() = runComposeUiTest {
        val fakes = FakeShowcase()
        showFakeShowcase(fakes, ShowcaseDestination.LLM_CHAT)
        onNodeWithTag(LlmChatTags.SAVE_SNAPSHOT).assertIsNotEnabled()
        send("first")
        waitForTag(LlmChatTags.message(1))
        click(LlmChatTags.SAVE_SNAPSHOT)
        waitForTag(LlmChatTags.snapshot(1))
        onNodeWithTag(LlmChatTags.snapshot(1)).assertTextEquals("#1 · 2 message(s) · 9 B")
        send("second")
        waitForTag(LlmChatTags.message(3))
        click(LlmChatTags.snapshotRestore(1))
        waitForText(LlmChatTags.STATUS, "Restored snapshot #1")
        onNodeWithTag(LlmChatTags.message(2)).assertDoesNotExist()
        assertEquals("session:1", fakes.chatFactory.opened.single().restored.single().decodeToString())
        click(LlmChatTags.CLEAR)
        waitForText(LlmChatTags.STATUS, "Conversation cleared")
        onNodeWithTag(LlmChatTags.message(0)).assertDoesNotExist()
    }

    @Test
    fun structuredExtractionValidatesLive() = runComposeUiTest {
        val fakes = FakeShowcase()
        showFakeShowcase(fakes, ShowcaseDestination.LLM_CHAT)
        click(LlmChatTags.panel(LlmPanel.STRUCTURED))
        onNodeWithTag(LlmChatTags.VALIDATION).assertTextEquals("No output yet")
        click(LlmChatTags.EXTRACT)
        waitForText(LlmChatTags.VALIDATION, "Valid JSON matching the schema")
        onNodeWithTag(LlmChatTags.EXTRACT_OUTPUT).assertTextContains("""{"name": "Maria", "age": 34, "city": "Lisbon"}""", substring = true)
        assertTrue(fakes.chatFactory.opened.single().calls.single().options.grammar != null)

        typeInto(LlmChatTags.EXTRACT_OUTPUT, """{"name": "Maria", "age": "x", "city": "Lisbon"}""")
        waitForText(LlmChatTags.VALIDATION, "Schema violations: $.age must be an integer")
        typeInto(LlmChatTags.EXTRACT_OUTPUT, """{"name": "Maria",, }""")
        waitForText(LlmChatTags.VALIDATION, "Invalid JSON")
        typeInto(LlmChatTags.EXTRACT_OUTPUT, """{"name": "Ma""")
        waitForText(LlmChatTags.VALIDATION, "Incomplete JSON")

        typeInto(LlmChatTags.SCHEMA, "{ broken")
        waitForTag(LlmChatTags.SCHEMA_ERROR)
        onNodeWithTag(LlmChatTags.EXTRACT).assertIsNotEnabled()
    }

    @Test
    fun embeddingsShowCosineSimilarity() = runComposeUiTest {
        val fakes = FakeShowcase()
        showFakeShowcase(fakes, ShowcaseDestination.LLM_CHAT)
        click(LlmChatTags.panel(LlmPanel.EMBEDDINGS))
        typeInto(LlmChatTags.EMBED_A, "abc")
        typeInto(LlmChatTags.EMBED_B, "cab")
        click(LlmChatTags.EMBED)
        waitForTag(LlmChatTags.SIMILARITY)
        onNodeWithTag(LlmChatTags.SIMILARITY).assertTextEquals("Cosine similarity: 1.0000 (26 dimensions)")
    }

    @Test
    fun visionAnalyzesAPickedImage() = runComposeUiTest {
        val fakes = FakeShowcase()
        showFakeShowcase(fakes, ShowcaseDestination.LLM_CHAT)
        typeInto(LlmChatTags.PROJECTOR_PATH, "/models/mmproj.gguf")
        click(LlmChatTags.panel(LlmPanel.VISION))
        onNodeWithTag(LlmChatTags.ANALYZE).assertIsNotEnabled()
        click(LlmChatTags.CAMERA)
        waitForText(LlmChatTags.IMAGE, "JPEG · 24 B · Camera")
        click(LlmChatTags.GALLERY)
        waitForText(LlmChatTags.IMAGE, "PNG · 20 B · Photo library")
        onNodeWithTag(LlmChatTags.ANALYZE).assertIsEnabled()
        click(LlmChatTags.ANALYZE)
        waitForTag(LlmChatTags.VISION_ANSWER)
        onNodeWithTag(LlmChatTags.VISION_ANSWER).assertTextEquals("An image of 1 picture(s), 20 bytes.")
        val call = fakes.chatFactory.opened.single().calls.single()
        assertTrue(call.images.single().contentEquals(FakeImageSource.PNG_MAGIC + ByteArray(12)))
    }

    @Test
    fun visionLoadsAnImageFileIntoMemory() = runComposeUiTest {
        val fakes = FakeShowcase()
        fakes.files["/photos/cat.jpg"] = FakeImageSource.JPEG_MAGIC + ByteArray(6)
        showFakeShowcase(fakes, ShowcaseDestination.LLM_CHAT)
        click(LlmChatTags.panel(LlmPanel.VISION))
        typeInto(LlmChatTags.IMAGE_PATH, "/photos/cat.jpg")
        click(LlmChatTags.IMAGE_LOAD)
        waitForText(LlmChatTags.IMAGE, "JPEG · 10 B · cat.jpg")
        click(LlmChatTags.ANALYZE)
        waitForTag(LlmChatTags.ERROR)
        onNodeWithTag(LlmChatTags.ERROR).assertTextContains("no vision projector", substring = true)
    }
}
