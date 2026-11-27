package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.assertTextEquals
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.v2.runComposeUiTest
import com.donfuldauer.kggml.models.ChecksumMismatchException
import com.donfuldauer.kggml.models.DownloadEvent
import com.donfuldauer.kggml.samples.showcase.llm.LlmChatTags
import com.donfuldauer.kggml.samples.showcase.models.FakeModelStore
import com.donfuldauer.kggml.samples.showcase.models.ModelManagerTags
import com.donfuldauer.kggml.samples.showcase.models.TEST_METADATA
import com.donfuldauer.kggml.samples.showcase.models.modelRowTag
import kotlin.test.Test

/**
 * Compose UI tests of the Model Manager against an in-memory store with scripted downloads: progress, pause /
 * resume, SHA-256 status, header metadata with the RAM estimate, handing a model to the Chat screen, and resident
 * engine eviction (manual and simulated OS signal).
 */
@OptIn(ExperimentalTestApi::class)
class ModelManagerUiTest : UiTestBase() {
    private val llm = modelRowTag("llm")
    private val whisper = modelRowTag("whisper")

    @Test
    fun catalogShowsInstalledModelsWithMetadataAndRamEstimate() = runComposeUiTest {
        val store = FakeModelStore().apply {
            files["llm.gguf"] = 1000
            metadata["/data/models/llm.gguf"] = TEST_METADATA
            partials["ggml-tiny.bin"] = 200
        }
        showFakeShowcase(FakeShowcase(store = store), ShowcaseDestination.MODEL_MANAGER)
        onNodeWithTag(ShowcaseDestination.MODEL_MANAGER.navTag).assertIsSelected()
        waitForTag("$llm-metadata")
        onNodeWithTag(ModelManagerTags.DIRECTORY).assertTextContains("/data/models · 1000 B on disk", substring = true)
        onNodeWithTag("$llm-phase").assertTextEquals("Installed · 1000 B")
        onNodeWithTag("$llm-verification").assertTextContains("Not checked this session", substring = true)
        onNodeWithTag("$llm-metadata").assertTextContains("GGUF v3 · llama · Tiny Test LM", substring = true)
        onNodeWithTag("$llm-metadata").assertTextContains("494.0 M parameters · Q4_K_M · context 32768", substring = true)
        onNodeWithTag("$llm-metadata").assertTextContains("Estimated peak RAM: 812.0 MiB", substring = true)
        onNodeWithTag("$whisper-phase").assertTextContains("Paused (resumable): 200 B", substring = true)
        onNodeWithTag("$whisper-download").assertTextContains("Resume", substring = true)
        onNodeWithTag("$llm-download").assertDoesNotExist()
    }

    @Test
    fun downloadShowsProgressPausesResumesAndVerifies() = runComposeUiTest {
        val fakes = FakeShowcase()
        fakes.store.metadata["/data/models/llm.gguf"] = TEST_METADATA
        showFakeShowcase(fakes, ShowcaseDestination.MODEL_MANAGER)
        val script = fakes.store.script("llm.gguf")
        click("$llm-download")
        script.trySend(DownloadEvent.Started(0, 1000))
        script.trySend(DownloadEvent.Progress(250, 1000))
        waitForText("$llm-phase", "Downloading: 250 B / 1000 B (25%)")
        onNodeWithTag("$llm-progress").assertExists()

        click("$llm-cancel")
        waitForText("$llm-phase", "Paused (resumable): 250 B")
        onNodeWithTag(ModelManagerTags.STATUS).assertTextContains("Paused Test LLM", substring = true)

        click("$llm-download")
        script.trySend(DownloadEvent.Started(250, 1000))
        script.trySend(DownloadEvent.Verifying(1000))
        waitForText("$llm-phase", "Verifying SHA-256")
        script.trySend(DownloadEvent.Completed("/data/models/llm.gguf", 1000, "a".repeat(64), 250, false))
        waitForText("$llm-phase", "Installed")
        onNodeWithTag("$llm-verification").assertTextContains("SHA-256 verified · sha256 aaaaaaaaaaaaaaaa…")
        waitForTag("$llm-metadata")
        onNodeWithTag(ModelManagerTags.STATUS).assertTextContains("resumed at 250 B", substring = true)
    }

    @Test
    fun checksumMismatchIsShownAndCanBeRetried() = runComposeUiTest {
        val fakes = FakeShowcase()
        showFakeShowcase(fakes, ShowcaseDestination.MODEL_MANAGER)
        click("$llm-download")
        fakes.store.script("llm.gguf").trySend(ChecksumMismatchException("a".repeat(64), "c".repeat(64)))
        waitForTag("$llm-error")
        onNodeWithTag("$llm-error").assertTextContains("SHA-256 mismatch", substring = true)
        onNodeWithTag("$llm-verification").assertTextContains("SHA-256 mismatch", substring = true)
        onNodeWithTag("$llm-phase").assertTextEquals("Failed")
        onNodeWithTag("$llm-download").assertExists()
    }

    @Test
    fun unverifiedDownloadAndManualVerify() = runComposeUiTest {
        val fakes = FakeShowcase()
        fakes.store.digests["ggml-tiny.bin"] = "b".repeat(64)
        showFakeShowcase(fakes, ShowcaseDestination.MODEL_MANAGER)
        click("$whisper-download")
        fakes.store.script("ggml-tiny.bin").trySend(DownloadEvent.Completed("/data/models/ggml-tiny.bin", 500, "b".repeat(64), 0, false))
        waitForText("$whisper-verification", "Unverified (no published SHA-256)")
        waitForTag("$whisper-metadata-error")
        click("$whisper-verify")
        waitForText("$whisper-verification", "Unverified")
        click("$whisper-delete")
        waitForText("$whisper-phase", "Not downloaded")
    }

    @Test
    fun useSendsAnInstalledLlmToTheChatScreen() = runComposeUiTest {
        val fakes = FakeShowcase(store = FakeModelStore().apply { files["llm.gguf"] = 1000 }, llmModelPath = "")
        showFakeShowcase(fakes, ShowcaseDestination.MODEL_MANAGER)
        waitForTag("$llm-use")
        click("$llm-use")
        onNodeWithTag(ShowcaseDestination.LLM_CHAT.navTag).assertIsSelected()
        onNodeWithTag(LlmChatTags.MODEL_PATH).assertTextContains("/data/models/llm.gguf")
    }

    @Test
    fun inspectAnyPath() = runComposeUiTest {
        val fakes = FakeShowcase()
        fakes.store.metadata["/tmp/other.gguf"] = TEST_METADATA
        showFakeShowcase(fakes, ShowcaseDestination.MODEL_MANAGER)
        typeInto(ModelManagerTags.INSPECT_PATH, "/tmp/other.gguf")
        click(ModelManagerTags.INSPECT)
        waitForTag(ModelManagerTags.INSPECTED)
        onNodeWithTag(ModelManagerTags.INSPECTED).assertTextContains("Estimated peak RAM: 812.0 MiB", substring = true)
        typeInto(ModelManagerTags.INSPECT_PATH, "/tmp/bad.bin")
        click(ModelManagerTags.INSPECT)
        waitForTag(ModelManagerTags.ERROR)
        onNodeWithTag(ModelManagerTags.ERROR).assertTextContains("ModelFormatException", substring = true)
    }

    @Test
    fun simulatedLowMemoryEvictsTheIdleChatEngine() = runComposeUiTest {
        val fakes = FakeShowcase()
        showFakeShowcase(fakes, ShowcaseDestination.LLM_CHAT)
        click(LlmChatTags.LOAD)
        waitForText(LlmChatTags.LOADED, "Resident")

        onNodeWithTag(ShowcaseDestination.MODEL_MANAGER.navTag).performClick()
        waitForTag(ModelManagerTags.residentEntry(0))
        onNodeWithTag(ModelManagerTags.residentEntry(0)).assertTextContains("llm:/models/qwen.gguf", substring = true)
        onNodeWithTag(ModelManagerTags.RESIDENT_SUMMARY).assertTextContains("1 engine(s) resident · 1000 B of 9.77 KiB budget", substring = true)

        click(ModelManagerTags.SIMULATE_LOW_MEMORY)
        waitForTag(ModelManagerTags.EVICTION_LOG)
        onNodeWithTag(ModelManagerTags.residentEntry(0)).assertDoesNotExist()
        onNodeWithTag(ModelManagerTags.EVICTION_LOG).assertTextContains("OS memory-pressure signal: CRITICAL → evicted llm:", substring = true)
        onNodeWithTag(ModelManagerTags.STATUS).assertTextContains("Simulated CRITICAL memory pressure: released 1000 B", substring = true)
        check(fakes.chatFactory.opened.single().closed) { "the idle engine was closed" }

        onNodeWithTag(ShowcaseDestination.LLM_CHAT.navTag).performClick()
        waitForText(LlmChatTags.LOADED, "Evicted from memory")
    }

    @Test
    fun manualEvictionButtonsWork() = runComposeUiTest {
        val fakes = FakeShowcase()
        showFakeShowcase(fakes, ShowcaseDestination.LLM_CHAT)
        click(LlmChatTags.LOAD)
        waitForText(LlmChatTags.LOADED, "Resident")
        onNodeWithTag(ShowcaseDestination.MODEL_MANAGER.navTag).performClick()
        click(ModelManagerTags.EVICT_CRITICAL)
        waitForText(ModelManagerTags.STATUS, "Evicted 1 idle engine(s) (CRITICAL)")
        onNodeWithTag(ModelManagerTags.EVICTION_LOG).assertTextContains("Manual: CRITICAL", substring = true)
        onNodeWithText("No engine loaded. Chat and Diffusion engines appear here while resident.", substring = true).assertExists()
    }
}
