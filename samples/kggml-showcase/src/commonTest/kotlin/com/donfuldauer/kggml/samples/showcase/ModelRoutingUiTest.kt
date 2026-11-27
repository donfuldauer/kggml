package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.assert
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.v2.runComposeUiTest
import androidx.compose.ui.text.AnnotatedString
import com.donfuldauer.kggml.samples.showcase.diffusion.DiffusionStudioTags
import com.donfuldauer.kggml.samples.showcase.llm.LlmChatTags
import com.donfuldauer.kggml.samples.showcase.models.CatalogModel
import com.donfuldauer.kggml.samples.showcase.models.FakeModelStore
import com.donfuldauer.kggml.samples.showcase.models.ModelKind
import com.donfuldauer.kggml.samples.showcase.models.ModelManagerTags
import com.donfuldauer.kggml.samples.showcase.models.ModelMetadata
import com.donfuldauer.kggml.samples.showcase.models.RamBreakdown
import com.donfuldauer.kggml.samples.showcase.models.modelRowTag
import com.donfuldauer.kggml.samples.showcase.whisper.WhisperStudioTags
import kotlin.test.Test

/**
 * The navigation shell's routing of the Model Manager's "Use" button: every model kind lands on the screen that
 * runs it with the installed path filled in (a vision LLM together with its installed projector), plus Model Manager
 * rendering of files outside the catalog, headers without quantization / context, and an empty eviction.
 */
@OptIn(ExperimentalTestApi::class)
class ModelRoutingUiTest : UiTestBase() {
    private val catalog = listOf(
        model("whisper", ModelKind.WHISPER, "ggml-tiny.bin"),
        model("vlm", ModelKind.VISION_LLM, "vlm.gguf", projectorId = "mmproj"),
        model("mmproj", ModelKind.VISION_PROJECTOR, "mmproj.gguf"),
        model("sd", ModelKind.DIFFUSION, "sd.safetensors"),
    )

    private fun model(id: String, kind: ModelKind, fileName: String, projectorId: String? = null) = CatalogModel(
        id = id,
        displayName = "Test $id",
        kind = kind,
        url = "https://example.invalid/$fileName",
        fileName = fileName,
        approxSizeBytes = 1000,
        license = "MIT",
        projectorId = projectorId,
    )

    private fun installed(vararg fileNames: String) = FakeModelStore().apply { fileNames.forEach { files[it] = 1000 } }

    @Test
    fun whisperModelOpensTheWhisperStudioWithItsPath() = runComposeUiTest {
        showFakeShowcase(FakeShowcase(store = installed("ggml-tiny.bin"), catalog = catalog), ShowcaseDestination.MODEL_MANAGER)
        val use = "${modelRowTag("whisper")}-use"
        waitForTag(use)
        onNodeWithTag(use).assertTextContains("Use in Whisper")
        click(use)
        onNodeWithTag(ShowcaseDestination.WHISPER_STUDIO.navTag).assertIsSelected()
        onNodeWithTag(WhisperStudioTags.MODEL_PATH).assertTextContains("/data/models/ggml-tiny.bin")
    }

    @Test
    fun visionLlmOpensTheChatTogetherWithItsInstalledProjector() = runComposeUiTest {
        val fakes = FakeShowcase(store = installed("vlm.gguf", "mmproj.gguf"), catalog = catalog, llmModelPath = "")
        showFakeShowcase(fakes, ShowcaseDestination.MODEL_MANAGER)
        val use = "${modelRowTag("vlm")}-use"
        waitForTag(use)
        onNodeWithTag(use).assertTextContains("Use in Chat")
        click(use)
        onNodeWithTag(ShowcaseDestination.LLM_CHAT.navTag).assertIsSelected()
        onNodeWithTag(LlmChatTags.MODEL_PATH).assertTextContains("/data/models/vlm.gguf")
        onNodeWithTag(LlmChatTags.PROJECTOR_PATH).assertTextContains("/data/models/mmproj.gguf")
    }

    @Test
    fun visionLlmWithoutAnInstalledProjectorLeavesTheProjectorEmpty() = runComposeUiTest {
        val fakes = FakeShowcase(store = installed("vlm.gguf"), catalog = catalog, llmModelPath = "")
        showFakeShowcase(fakes, ShowcaseDestination.MODEL_MANAGER)
        val use = "${modelRowTag("vlm")}-use"
        waitForTag(use)
        click(use)
        onNodeWithTag(LlmChatTags.MODEL_PATH).assertTextContains("/data/models/vlm.gguf")
        onNodeWithTag(LlmChatTags.PROJECTOR_PATH).assert(SemanticsMatcher.expectValue(SemanticsProperties.EditableText, AnnotatedString("")))
    }

    @Test
    fun projectorIsSetWithoutReplacingTheChatModel() = runComposeUiTest {
        val fakes = FakeShowcase(store = installed("mmproj.gguf"), catalog = catalog, llmModelPath = "/models/qwen.gguf")
        showFakeShowcase(fakes, ShowcaseDestination.MODEL_MANAGER)
        val use = "${modelRowTag("mmproj")}-use"
        waitForTag(use)
        onNodeWithTag(use).assertTextContains("Use as projector")
        click(use)
        onNodeWithTag(ShowcaseDestination.LLM_CHAT.navTag).assertIsSelected()
        onNodeWithTag(LlmChatTags.PROJECTOR_PATH).assertTextContains("/data/models/mmproj.gguf")
        onNodeWithTag(LlmChatTags.MODEL_PATH).assertTextContains("/models/qwen.gguf")
    }

    @Test
    fun diffusionModelOpensTheDiffusionStudioWithItsPath() = runComposeUiTest {
        val fakes = FakeShowcase(store = installed("sd.safetensors"), catalog = catalog, diffusionModelPath = "")
        showFakeShowcase(fakes, ShowcaseDestination.MODEL_MANAGER)
        val use = "${modelRowTag("sd")}-use"
        waitForTag(use)
        onNodeWithTag(use).assertTextContains("Use in Diffusion")
        click(use)
        onNodeWithTag(ShowcaseDestination.DIFFUSION_STUDIO.navTag).assertIsSelected()
        onNodeWithTag(DiffusionStudioTags.MODEL_PATH).assertTextContains("/data/models/sd.safetensors")
    }

    @Test
    fun otherFilesHeadersWithoutDetailsAndEmptyEvictionsAreShown() = runComposeUiTest {
        val store = installed("ggml-tiny.bin").apply {
            files["notes.txt"] = 5
            metadata["/data/models/ggml-tiny.bin"] = ModelMetadata(
                format = "Whisper ggml",
                name = null,
                architecture = "whisper",
                parameterCount = 39_000_000,
                quantization = null,
                contextLength = null,
                estimatedPeakRam = RamBreakdown(weightsBytes = 1024, kvCacheBytes = 0, computeBytes = 0, overheadBytes = 0),
            )
        }
        showFakeShowcase(FakeShowcase(store = store, catalog = catalog), ShowcaseDestination.MODEL_MANAGER)
        val whisper = modelRowTag("whisper")
        waitForTag("$whisper-metadata")
        onNodeWithText("Other files in the model directory").assertExists()
        onNodeWithText("notes.txt · 5 B").assertExists()
        onNodeWithTag("$whisper-metadata").assertTextContains("Whisper ggml · whisper")
        onNodeWithTag("$whisper-metadata").assertTextContains("39.0 M parameters · unknown quantization")

        click(ModelManagerTags.EVICT_CRITICAL)
        waitForTag(ModelManagerTags.EVICTION_LOG)
        onNodeWithTag(ModelManagerTags.EVICTION_LOG).assertTextContains("Manual: CRITICAL → evicted nothing")
    }
}
