package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.assertIsNotSelected
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.assert
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.runComposeUiTest
import com.donfuldauer.kggml.models.DownloadEvent
import com.donfuldauer.kggml.samples.showcase.diffusion.DiffusionStudioTags
import com.donfuldauer.kggml.samples.showcase.llm.LlmChatTags
import com.donfuldauer.kggml.samples.showcase.models.CatalogModel
import com.donfuldauer.kggml.samples.showcase.models.DiffusionPreset
import com.donfuldauer.kggml.samples.showcase.models.FakeModelStore
import com.donfuldauer.kggml.samples.showcase.models.ModelKind
import com.donfuldauer.kggml.samples.showcase.models.ModelPickerTags
import com.donfuldauer.kggml.samples.showcase.models.modelRowTag
import com.donfuldauer.kggml.samples.showcase.whisper.WhisperStudioTags
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * The model picker of the Whisper, Chat and Diffusion screens: downloaded models that fit the screen (catalog models
 * and other files in the model folder) next to a custom path, kept in sync with the Model Manager.
 */
@OptIn(ExperimentalTestApi::class)
class ModelPickerUiTest : UiTestBase() {
    private val catalog = listOf(
        model("whisper", ModelKind.WHISPER, "ggml-tiny.bin"),
        model("vlm", ModelKind.VISION_LLM, "vlm.gguf", projectorId = "mmproj"),
        model("mmproj", ModelKind.VISION_PROJECTOR, "mmproj-vlm.gguf"),
        model("sd", ModelKind.DIFFUSION, "sd.safetensors"),
        model("turbo", ModelKind.DIFFUSION, "turbo.gguf", preset = DiffusionPreset(steps = 4, cfgScale = 1f, size = 512)),
    )

    private fun model(id: String, kind: ModelKind, fileName: String, projectorId: String? = null, preset: DiffusionPreset? = null) = CatalogModel(
        id = id,
        displayName = "Test $id",
        kind = kind,
        url = "https://example.invalid/$fileName",
        fileName = fileName,
        approxSizeBytes = 1000,
        license = "MIT",
        projectorId = projectorId,
        diffusionPreset = preset,
    )

    private fun installed(vararg fileNames: String) = FakeModelStore().apply { fileNames.forEach { files[it] = 1000 } }

    private fun fakes(vararg fileNames: String) = FakeShowcase(store = installed(*fileNames), catalog = catalog)

    /** Asserts the text field's own value (not its label). */
    private fun androidx.compose.ui.test.ComposeUiTest.assertPath(fieldTag: String, path: String) {
        onNode(hasTestTag(fieldTag) and hasSetTextAction()).performScrollTo()
            .assert(SemanticsMatcher.expectValue(SemanticsProperties.EditableText, AnnotatedString(path)))
    }

    private fun androidx.compose.ui.test.ComposeUiTest.assertChosen(fieldTag: String, selectedId: String?, vararg otherIds: String) {
        val custom = onNodeWithTag(ModelPickerTags.custom(fieldTag)).performScrollTo()
        if (selectedId == null) custom.assertIsSelected() else custom.assertIsNotSelected()
        selectedId?.let { onNodeWithTag(ModelPickerTags.choice(fieldTag, it)).performScrollTo().assertIsSelected() }
        otherIds.forEach { onNodeWithTag(ModelPickerTags.choice(fieldTag, it)).performScrollTo().assertIsNotSelected() }
    }

    @Test
    fun whisperPickerOffersDownloadedWhisperModelsAndACustomPath() = runComposeUiTest {
        // vlm.gguf and sd.safetensors are installed too, but are not whisper models.
        showFakeShowcase(fakes("ggml-tiny.bin", "ggml-base.en.bin", "vlm.gguf", "sd.safetensors"), ShowcaseDestination.WHISPER_STUDIO)
        val field = WhisperStudioTags.MODEL_PATH
        waitForTag(ModelPickerTags.choice(field, "ggml-base.en.bin"))
        onNodeWithTag(ModelPickerTags.choice(field, "whisper")).assertTextContains("Test whisper")
        onNodeWithTag(ModelPickerTags.choice(field, "ggml-base.en.bin")).assertTextContains("Other file in the model folder")
        listOf("vlm", "sd", "vlm.gguf").forEach { assertEquals(0, onAllNodes(hasTestTag(ModelPickerTags.choice(field, it))).fetchSemanticsNodes().size, it) }
        onAllNodes(hasTestTag(ModelPickerTags.empty(field))).fetchSemanticsNodes().let { assertEquals(0, it.size) }
        // No model yet: "Custom path" is the selection.
        assertChosen(field, null, "whisper", "ggml-base.en.bin")

        click(ModelPickerTags.choice(field, "whisper"))
        assertPath(field, "/data/models/ggml-tiny.bin")
        assertChosen(field, "whisper", "ggml-base.en.bin")

        click(ModelPickerTags.choice(field, "ggml-base.en.bin"))
        assertPath(field, "/data/models/ggml-base.en.bin")
        assertChosen(field, "ggml-base.en.bin", "whisper")

        // Typing a path elsewhere selects "Custom path".
        typeInto(field, "/sdcard/ggml-small.bin")
        assertChosen(field, null, "whisper", "ggml-base.en.bin")

        // Switching from a downloaded model to "Custom path" starts from an empty field; on "Custom path" it is kept.
        click(ModelPickerTags.choice(field, "whisper"))
        click(ModelPickerTags.custom(field))
        assertPath(field, "")
        typeInto(field, "/sdcard/mine.bin")
        click(ModelPickerTags.custom(field))
        assertPath(field, "/sdcard/mine.bin")
    }

    @Test
    fun chatPickerPairsAVisionLlmWithItsProjectorAndLoadsTheChosenFiles() = runComposeUiTest {
        val fakes = fakes("vlm.gguf", "mmproj-vlm.gguf", "llama.gguf", "mmproj-other.gguf", "ggml-tiny.bin")
        showFakeShowcase(fakes, ShowcaseDestination.LLM_CHAT)
        val model = LlmChatTags.MODEL_PATH
        val projector = LlmChatTags.PROJECTOR_PATH
        waitForTag(ModelPickerTags.choice(model, "llama.gguf"))
        // LLMs only in the model picker, projectors only in the projector picker.
        listOf("mmproj", "mmproj-other.gguf", "whisper").forEach {
            assertEquals(0, onAllNodes(hasTestTag(ModelPickerTags.choice(model, it))).fetchSemanticsNodes().size, it)
        }
        listOf("vlm", "llama.gguf").forEach {
            assertEquals(0, onAllNodes(hasTestTag(ModelPickerTags.choice(projector, it))).fetchSemanticsNodes().size, it)
        }
        onNodeWithTag(ModelPickerTags.choice(model, "vlm")).assertTextContains("with its vision projector", substring = true)
        // The initial path is not a downloaded model.
        assertPath(model, "/models/qwen.gguf")
        assertChosen(model, null, "vlm", "llama.gguf")
        assertChosen(projector, null, "mmproj", "mmproj-other.gguf")

        click(ModelPickerTags.choice(model, "vlm"))
        assertPath(model, "/data/models/vlm.gguf")
        assertPath(projector, "/data/models/mmproj-vlm.gguf")
        assertChosen(model, "vlm", "llama.gguf")
        assertChosen(projector, "mmproj", "mmproj-other.gguf")

        click(ModelPickerTags.choice(projector, "mmproj-other.gguf"))
        assertChosen(projector, "mmproj-other.gguf", "mmproj")
        click(LlmChatTags.LOAD)
        waitUntil(timeoutMillis = 5_000) { fakes.chatFactory.opened.isNotEmpty() }
        val spec = fakes.chatFactory.opened.single().spec
        assertEquals("/data/models/vlm.gguf", spec.modelPath)
        assertEquals("/data/models/mmproj-other.gguf", spec.projectorPath)

        // A projector belongs to its model: picking another model clears it (libmtmd would reject the pair).
        click(ModelPickerTags.choice(model, "llama.gguf"))
        assertPath(model, "/data/models/llama.gguf")
        assertPath(projector, "")
        assertChosen(projector, null, "mmproj", "mmproj-other.gguf")
        // Picking the vision LLM again brings its own projector back.
        click(ModelPickerTags.choice(model, "vlm"))
        assertPath(projector, "/data/models/mmproj-vlm.gguf")
    }

    @Test
    fun diffusionPickerOffersDownloadedDiffusionModels() = runComposeUiTest {
        showFakeShowcase(fakes("sd.safetensors", "turbo.ckpt", "vlm.gguf"), ShowcaseDestination.DIFFUSION_STUDIO)
        val field = DiffusionStudioTags.MODEL_PATH
        waitForTag(ModelPickerTags.choice(field, "turbo.ckpt"))
        assertEquals(0, onAllNodes(hasTestTag(ModelPickerTags.choice(field, "vlm.gguf"))).fetchSemanticsNodes().size)
        assertChosen(field, null, "sd", "turbo.ckpt")
        click(ModelPickerTags.choice(field, "sd"))
        assertPath(field, "/data/models/sd.safetensors")
        assertChosen(field, "sd", "turbo.ckpt")
    }

    @Test
    fun choosingAFastDiffusionModelAppliesItsSettings() = runComposeUiTest {
        showFakeShowcase(fakes("sd.safetensors", "turbo.gguf"), ShowcaseDestination.DIFFUSION_STUDIO)
        val field = DiffusionStudioTags.MODEL_PATH
        waitForTag(ModelPickerTags.choice(field, "turbo"))
        onNodeWithTag(ModelPickerTags.choice(field, "turbo")).assertTextContains("uses 4 steps, CFG 1, 512x512", substring = true)
        // Defaults before: 20 steps, CFG 7.
        onNodeWithTag(DiffusionStudioTags.STEPS_LABEL).performScrollTo().assertTextContains("Sampling steps: 20")

        click(ModelPickerTags.choice(field, "turbo"))
        assertPath(field, "/data/models/turbo.gguf")
        onNodeWithTag(DiffusionStudioTags.STEPS_LABEL).performScrollTo().assertTextContains("Sampling steps: 4")
        onNodeWithTag(DiffusionStudioTags.CFG_LABEL).performScrollTo().assertTextContains("CFG scale: 1.0")
        onNodeWithTag(DiffusionStudioTags.size(512)).performScrollTo().assertIsSelected()

        // A model without a preset keeps what is set.
        click(ModelPickerTags.choice(field, "sd"))
        onNodeWithTag(DiffusionStudioTags.STEPS_LABEL).performScrollTo().assertTextContains("Sampling steps: 4")
    }

    @Test
    fun withNothingDownloadedEachScreenPointsToTheModelsTabAndKeepsTheCustomPath() = runComposeUiTest {
        showFakeShowcase(fakes(), ShowcaseDestination.WHISPER_STUDIO)
        waitForTag(ModelPickerTags.empty(WhisperStudioTags.MODEL_PATH))
        onNodeWithTag(ModelPickerTags.empty(WhisperStudioTags.MODEL_PATH)).assertTextContains("No downloaded speech models yet", substring = true)
        onNodeWithTag(ModelPickerTags.custom(WhisperStudioTags.MODEL_PATH)).assertIsSelected()
        typeInto(WhisperStudioTags.MODEL_PATH, "/sdcard/ggml-tiny.bin")
        assertPath(WhisperStudioTags.MODEL_PATH, "/sdcard/ggml-tiny.bin")

        onNodeWithTag(ShowcaseDestination.LLM_CHAT.navTag).performClick()
        waitForTag(ModelPickerTags.empty(LlmChatTags.MODEL_PATH))
        onNodeWithTag(ModelPickerTags.empty(LlmChatTags.MODEL_PATH)).assertTextContains("No downloaded LLM models yet", substring = true)
        onNodeWithTag(ModelPickerTags.empty(LlmChatTags.PROJECTOR_PATH)).assertTextContains("No downloaded vision projector models yet", substring = true)

        onNodeWithTag(ShowcaseDestination.DIFFUSION_STUDIO.navTag).performClick()
        waitForTag(ModelPickerTags.empty(DiffusionStudioTags.MODEL_PATH))
        onNodeWithTag(ModelPickerTags.empty(DiffusionStudioTags.MODEL_PATH)).assertTextContains("No downloaded image generation models yet", substring = true)
    }

    @Test
    fun aModelDownloadedInTheModelsTabAppearsInThePicker() = runComposeUiTest {
        val fakes = fakes()
        showFakeShowcase(fakes, ShowcaseDestination.MODEL_MANAGER)
        val row = modelRowTag("whisper")
        waitForTag("$row-download")
        click("$row-download")
        fakes.store.script("ggml-tiny.bin").trySend(DownloadEvent.Completed("/data/models/ggml-tiny.bin", 1000, "b".repeat(64), 0, false))
        waitForText("$row-phase", "Installed")

        onNodeWithTag(ShowcaseDestination.WHISPER_STUDIO.navTag).performClick()
        val field = WhisperStudioTags.MODEL_PATH
        waitForTag(ModelPickerTags.choice(field, "whisper"))
        assertEquals(0, onAllNodes(hasTestTag(ModelPickerTags.empty(field))).fetchSemanticsNodes().size)
        click(ModelPickerTags.choice(field, "whisper"))
        assertPath(field, "/data/models/ggml-tiny.bin")

        // Deleting it in the Models tab takes it out of the picker again.
        onNodeWithTag(ShowcaseDestination.MODEL_MANAGER.navTag).performClick()
        click("$row-delete")
        waitForText("$row-phase", "Not downloaded")
        onNodeWithTag(ShowcaseDestination.WHISPER_STUDIO.navTag).performClick()
        waitForTag(ModelPickerTags.empty(field))
        assertChosen(field, null)
    }
}
