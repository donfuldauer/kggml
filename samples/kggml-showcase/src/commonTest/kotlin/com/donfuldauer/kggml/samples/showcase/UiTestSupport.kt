package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.test.ComposeUiTest
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextReplacement
import com.donfuldauer.kggml.core.ManualMemoryPressureObserver
import com.donfuldauer.kggml.samples.showcase.diffusion.FakeImageGeneratorFactory
import com.donfuldauer.kggml.samples.showcase.llm.FakeChatModelFactory
import com.donfuldauer.kggml.samples.showcase.llm.FakeImageSource
import com.donfuldauer.kggml.samples.showcase.models.CatalogModel
import com.donfuldauer.kggml.samples.showcase.models.FakeModelStore
import com.donfuldauer.kggml.samples.showcase.models.TEST_CATALOG
import com.donfuldauer.kggml.samples.showcase.whisper.FakeAudioCapture
import com.donfuldauer.kggml.samples.showcase.whisper.FakeTranscriberFactory
import kotlinx.coroutines.Dispatchers
import kotlin.random.Random

/** Fakes for every screen of the showcase: no native libraries, no network, no file system. */
class FakeShowcase(
    val store: FakeModelStore = FakeModelStore(),
    val observer: ManualMemoryPressureObserver = ManualMemoryPressureObserver(),
    val chatFactory: FakeChatModelFactory = FakeChatModelFactory(),
    val imageSource: FakeImageSource = FakeImageSource(),
    val files: MutableMap<String, ByteArray> = mutableMapOf(),
    val generatorFactory: FakeImageGeneratorFactory = FakeImageGeneratorFactory(),
    llmModelPath: String = "/models/qwen.gguf",
    diffusionModelPath: String = "/models/sd15.gguf",
    catalog: List<CatalogModel> = TEST_CATALOG,
) {
    val written = LinkedHashMap<String, ByteArray>()
    val transcriberFactory = FakeTranscriberFactory()

    val whisper = WhisperStudioEnvironment(
        transcriberFactory = transcriberFactory,
        audioCapture = FakeAudioCapture(),
        readFile = { files.getValue(it) },
        initialModelPath = "",
        backgroundStressSupported = false,
    )
    val models = ModelManagerEnvironment(store = store, memoryObserver = observer, residentBudgetBytes = 10_000, catalog = catalog)
    val llm = LlmChatEnvironment(
        modelFactory = chatFactory,
        imageSource = imageSource,
        readFile = { files[it] ?: throw IllegalArgumentException("No such file: $it") },
        initialModelPath = llmModelPath,
    )
    val diffusion = DiffusionStudioEnvironment(
        generatorFactory = generatorFactory,
        writeFile = { path, bytes -> written[path] = bytes },
        exportDirectory = "/exports",
        initialModelPath = diffusionModelPath,
        random = Random(7),
    )
}

@OptIn(ExperimentalTestApi::class)
fun ComposeUiTest.showFakeShowcase(fakes: FakeShowcase, destination: ShowcaseDestination) {
    setContent {
        ShowcaseApp(
            inspector = { fakeSnapshot() },
            ioDispatcher = Dispatchers.Unconfined,
            initialDestination = destination,
            whisperEnvironment = fakes.whisper,
            modelEnvironment = fakes.models,
            llmEnvironment = fakes.llm,
            diffusionEnvironment = fakes.diffusion,
        )
    }
}

@OptIn(ExperimentalTestApi::class)
fun ComposeUiTest.click(tag: String) {
    onNodeWithTag(tag).performScrollTo().performClick()
}

@OptIn(ExperimentalTestApi::class)
fun ComposeUiTest.typeInto(tag: String, text: String) {
    onNodeWithTag(tag).performScrollTo().performTextReplacement(text)
}

@OptIn(ExperimentalTestApi::class)
fun ComposeUiTest.waitForTag(tag: String) =
    waitUntil(timeoutMillis = 5_000) { onAllNodes(hasTestTag(tag)).fetchSemanticsNodes().isNotEmpty() }

@OptIn(ExperimentalTestApi::class)
fun ComposeUiTest.waitForText(tag: String, text: String) = waitUntil(timeoutMillis = 5_000) {
    onAllNodes(hasTestTag(tag) and hasText(text, substring = true)).fetchSemanticsNodes().isNotEmpty()
}
