package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.test.ComposeUiTest
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.SemanticsNodeInteraction
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.hasAnyDescendant
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.v2.runComposeUiTest
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.ManualMemoryPressureObserver
import com.donfuldauer.kggml.llama.LlamaContextConfig
import com.donfuldauer.kggml.llama.TinyLlamaModel
import com.donfuldauer.kggml.samples.showcase.diffusion.DiffusionImageGeneratorFactory
import com.donfuldauer.kggml.samples.showcase.diffusion.DiffusionStudioTags
import com.donfuldauer.kggml.samples.showcase.llm.LlamaChatModelFactory
import com.donfuldauer.kggml.samples.showcase.llm.LlmChatTags
import com.donfuldauer.kggml.samples.showcase.llm.LlmPanel
import com.donfuldauer.kggml.samples.showcase.llm.NoImageSource
import com.donfuldauer.kggml.samples.showcase.models.DiskModelStore
import com.donfuldauer.kggml.samples.showcase.models.ModelManagerTags
import com.donfuldauer.kggml.samples.showcase.models.TEST_CATALOG
import com.donfuldauer.kggml.samples.showcase.whisper.FakeAudioCapture
import com.donfuldauer.kggml.samples.showcase.whisper.CaptureFormat
import com.donfuldauer.kggml.samples.showcase.whisper.WhisperStudioTags
import com.donfuldauer.kggml.samples.showcase.whisper.WhisperTranscriberFactory
import com.donfuldauer.kggml.samples.showcase.whisper.wavBytes
import com.donfuldauer.kggml.whisper.TinyWhisperModel
import kotlinx.coroutines.Dispatchers
import kotlin.math.PI
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail
import kotlin.time.Duration.Companion.minutes

/**
 * End-to-end UI tests of the showcase against the REAL SDK: the production engine factories ([WhisperTranscriberFactory],
 * [LlamaChatModelFactory], [DiffusionImageGeneratorFactory]), [KggmlRuntimeInspector] and [DiskModelStore], driven
 * through the Compose UI exactly as a user would.
 *
 * Models are the engines' synthetic fixtures (`TinyWhisperModel`, `TinyLlamaModel`, compiled in from the engine
 * modules' tests; random weights, so their text output is meaningless): the tests verify that the Whisper Studio and
 * the LLM screen load the model through the native library, run inference to completion and render the result, not
 * what the model says. No diffusion model is small enough to generate with (see `NativeDiffusionEngineContract`), so
 * the Diffusion Studio is covered up to the native load: an invalid file must surface as a typed error. GGUF header
 * inspection in the Model Manager is pure Kotlin and runs everywhere.
 *
 * Runs wherever the native libraries load: desktop with the host builds (`-Pkggml.requireNative=true` in CI), the iOS
 * simulator, and Android devices / emulators with `-Pkggml.bundleNative=true`. Skipped (with a message) under
 * Robolectric and on desktop without host builds. Everything runs on [ComputeBackend.CPU_ONLY] so results do not
 * depend on the host GPU.
 */
@OptIn(ExperimentalTestApi::class)
class RealEngineEndToEndUiTest : UiTestBase() {
    private class Files(dir: String) {
        val whisperModel = "$dir/ggml-tiny-synthetic.bin".also { writeLocalFile(it, TinyWhisperModel.bytes()) }
        val llamaModel = "$dir/tiny-llama-synthetic.gguf".also { writeLocalFile(it, TinyLlamaModel.bytes()) }
        val notADiffusionModel = "$dir/not-a-model.gguf".also { writeLocalFile(it, "not a diffusion model".encodeToByteArray()) }

        /** 2 s of a 440 Hz tone plus noise, 16 kHz mono 16-bit WAV. */
        val wav = "$dir/tone.wav".also { path ->
            val random = Random(7)
            val samples = ShortArray(32_000) { i ->
                (sin(2 * PI * 440 * i / 16_000) * 8_000 + random.nextInt(-500, 500)).toInt().toShort()
            }
            writeLocalFile(path, wavBytes(samples))
        }
        val modelsDir = "$dir/models"
        val exportDir = "$dir/exports"
    }

    private fun ComposeUiTest.showRealShowcase(files: Files, destination: ShowcaseDestination, mic: FakeAudioCapture? = null) {
        setContent {
            ShowcaseApp(
                inspector = KggmlRuntimeInspector,
                ioDispatcher = Dispatchers.Default,
                initialDestination = destination,
                whisperEnvironment = WhisperStudioEnvironment(
                    transcriberFactory = WhisperTranscriberFactory,
                    audioCapture = mic,
                    readFile = ::readLocalFile,
                    initialModelPath = files.whisperModel,
                    backgroundStressSupported = false,
                ),
                modelEnvironment = ModelManagerEnvironment(
                    store = DiskModelStore(files.modelsDir),
                    memoryObserver = ManualMemoryPressureObserver(),
                    residentBudgetBytes = 1L shl 32,
                    catalog = TEST_CATALOG,
                ),
                llmEnvironment = LlmChatEnvironment(
                    modelFactory = LlamaChatModelFactory,
                    imageSource = NoImageSource,
                    readFile = ::readLocalFile,
                    initialModelPath = files.llamaModel,
                    // Pinned sampler seed: the reply of the random-weight model is deterministic (non-empty).
                    chatSeed = 42,
                ),
                diffusionEnvironment = DiffusionStudioEnvironment(
                    generatorFactory = DiffusionImageGeneratorFactory,
                    writeFile = ::writeLocalFile,
                    exportDirectory = files.exportDir,
                    initialModelPath = files.notADiffusionModel,
                    random = Random(1),
                ),
            )
        }
    }

    /** Native inference can take a while on emulators and the iOS simulator. */
    private fun ComposeUiTest.awaitText(tag: String, text: String, timeoutMillis: Long = 120_000) =
        waitUntil(timeoutMillis = timeoutMillis) {
            onAllNodes(hasTestTag(tag) and (hasText(text, substring = true) or hasAnyDescendant(hasText(text, substring = true))))
                .fetchSemanticsNodes().isNotEmpty()
        }

    private fun ComposeUiTest.failOnError(errorTag: String) {
        val errors = onAllNodes(hasTestTag(errorTag)).fetchSemanticsNodes()
        if (errors.isNotEmpty()) fail("Screen reported an error: " + errors.joinToString { it.config.texts() })
    }

    private fun androidx.compose.ui.semantics.SemanticsConfiguration.texts(): String =
        (getOrNull(SemanticsProperties.Text).orEmpty().map { it.text } +
            listOfNotNull(getOrNull(SemanticsProperties.EditableText)?.text)).joinToString(" ")

    private fun SemanticsNodeInteraction.text(): String = fetchSemanticsNode().config.texts()

    @Test
    fun backendInspectorReportsTheRealRuntime() = runComposeUiTest(testTimeout = 5.minutes) {
        if (!nativeEnginesAvailable("backendInspectorReportsTheRealRuntime")) return@runComposeUiTest
        showRealShowcase(Files(NativeTestEnvironment.newTempDirectory()), ShowcaseDestination.BACKEND_INSPECTOR)
        awaitText("inspector-metadata", "Kggml runtime:")
        onAllNodes(hasTestTag("inspector-error")).fetchSemanticsNodes().let { assertTrue(it.isEmpty(), "runtime failed to load") }
        // The CPU backend is compiled into every build and always has a device.
        onNode(hasTestTag("backend-row-cpu_only") and hasAnyDescendant(hasText("Available"))).assertExists()
        onNode(hasTestTag("inspector-metadata") and hasAnyDescendant(hasText("ggml:", substring = true))).assertExists()
    }

    @Test
    fun whisperStudioTranscribesAWavFileWithTheRealEngine() = runComposeUiTest(testTimeout = 5.minutes) {
        if (!nativeEnginesAvailable("whisperStudioTranscribesAWavFileWithTheRealEngine")) return@runComposeUiTest
        val files = Files(NativeTestEnvironment.newTempDirectory())
        showRealShowcase(files, ShowcaseDestination.WHISPER_STUDIO)
        click(WhisperStudioTags.backendChip(ComputeBackend.CPU_ONLY))
        typeInto(WhisperStudioTags.FILE_PATH, files.wav)
        click(WhisperStudioTags.TRANSCRIBE_FILE)
        waitUntil(timeoutMillis = 120_000) {
            onAllNodes(hasTestTag(WhisperStudioTags.STATUS) and hasText("Transcribed", substring = true)).fetchSemanticsNodes().isNotEmpty() ||
                onAllNodes(hasTestTag(WhisperStudioTags.ERROR)).fetchSemanticsNodes().isNotEmpty()
        }
        failOnError(WhisperStudioTags.ERROR)
        // The WAV header was parsed and all 2 s of audio went through the native engine on the requested backend.
        onNodeWithTag(WhisperStudioTags.STATUS).assertTextContains("Transcribed 2.0 s of audio (16000 Hz, 1 ch)", substring = true)
        onNodeWithTag(WhisperStudioTags.ACTIVE_BACKEND).assertTextContains("Active: CPU (use_gpu=false)")
        awaitText(WhisperStudioTags.ACTIVITY, "Idle")
    }

    @Test
    fun whisperStudioStreamsLiveAudioThroughTheRealEngine() = runComposeUiTest(testTimeout = 5.minutes) {
        if (!nativeEnginesAvailable("whisperStudioStreamsLiveAudioThroughTheRealEngine")) return@runComposeUiTest
        // 48 kHz stereo like a real microphone: the native resampler / downmixer has to convert it to 16 kHz mono.
        val mic = FakeAudioCapture(CaptureFormat(48_000, 2))
        showRealShowcase(Files(NativeTestEnvironment.newTempDirectory()), ShowcaseDestination.WHISPER_STUDIO, mic)
        click(WhisperStudioTags.backendChip(ComputeBackend.CPU_ONLY))
        click(WhisperStudioTags.START)
        awaitText(WhisperStudioTags.ACTIVITY, "Listening")
        repeat(15) { mic.sendMs(200, amplitude = if (it % 2 == 0) 0.4f else -0.4f) }
        mic.finish()
        waitUntil(timeoutMillis = 120_000) {
            onAllNodes(hasTestTag(WhisperStudioTags.STATUS) and hasText("Completed:", substring = true)).fetchSemanticsNodes().isNotEmpty() ||
                onAllNodes(hasTestTag(WhisperStudioTags.ERROR)).fetchSemanticsNodes().isNotEmpty()
        }
        failOnError(WhisperStudioTags.ERROR)
        // The engine's Completed event reports the audio it consumed: all 3 s of captured audio.
        onNodeWithTag(WhisperStudioTags.STATUS).assertTextContains("3.0 s of audio", substring = true)
        awaitText(WhisperStudioTags.ACTIVITY, "Idle")
    }

    @Test
    fun llmChatLoadsTheModelAndStreamsARealReply() = runComposeUiTest(testTimeout = 5.minutes) {
        if (!nativeEnginesAvailable("llmChatLoadsTheModelAndStreamsARealReply")) return@runComposeUiTest
        showRealShowcase(Files(NativeTestEnvironment.newTempDirectory()), ShowcaseDestination.LLM_CHAT)
        click("${LlmChatTags.BACKEND}-cpu_only")
        click(LlmChatTags.LOAD)
        awaitText(LlmChatTags.LOADED, "(GPU off)")
        failOnError(LlmChatTags.ERROR)
        // Reported by the native engine: the model's architecture from its GGUF metadata, CPU only (no GPU layers) and
        // the engine's default context window.
        onNodeWithTag(LlmChatTags.LOADED).assertTextContains("llama", substring = true)
        onNodeWithTag(LlmChatTags.LOADED).assertTextContains("GPU layers 0", substring = true)
        onNodeWithTag(LlmChatTags.LOADED).assertTextContains("context ${LlamaContextConfig().contextLength}", substring = true)

        typeInto(LlmChatTags.DRAFT, "ab ab")
        click(LlmChatTags.SEND)
        waitUntil(timeoutMillis = 120_000) {
            onAllNodes(hasTestTag(LlmChatTags.STATUS) and hasText("token(s), stopped:", substring = true)).fetchSemanticsNodes().isNotEmpty() ||
                onAllNodes(hasTestTag(LlmChatTags.ERROR)).fetchSemanticsNodes().isNotEmpty()
        }
        failOnError(LlmChatTags.ERROR)
        onNodeWithTag(LlmChatTags.message(0)).assertTextContains("ab ab", substring = true)
        onNodeWithTag(LlmChatTags.message(1)).assertExists()
        val status = onNodeWithTag(LlmChatTags.STATUS).text()
        val tokens = Regex("(\\d+) token\\(s\\)").find(status)?.groupValues?.get(1)?.toInt() ?: fail("no token count in '$status'")
        assertTrue(tokens in 1..256, "generated $tokens tokens (max 256): $status")

        // KV-cache snapshot of the real session.
        click(LlmChatTags.SAVE_SNAPSHOT)
        awaitText(LlmChatTags.STATUS, "Saved snapshot")
        failOnError(LlmChatTags.ERROR)
    }

    @Test
    fun llmStructuredOutputAndEmbeddingsRunOnTheRealEngine() = runComposeUiTest(testTimeout = 5.minutes) {
        if (!nativeEnginesAvailable("llmStructuredOutputAndEmbeddingsRunOnTheRealEngine")) return@runComposeUiTest
        showRealShowcase(Files(NativeTestEnvironment.newTempDirectory()), ShowcaseDestination.LLM_CHAT)
        click("${LlmChatTags.BACKEND}-cpu_only")

        click(LlmChatTags.panel(LlmPanel.EMBEDDINGS))
        typeInto(LlmChatTags.EMBED_A, "ab ab ab")
        typeInto(LlmChatTags.EMBED_B, "ab ab ab")
        click(LlmChatTags.EMBED)
        awaitText(LlmChatTags.SIMILARITY, "Cosine similarity:")
        failOnError(LlmChatTags.ERROR)
        // Embeddings are L2-normalized and deterministic, so identical texts have cosine similarity 1.
        onNodeWithTag(LlmChatTags.SIMILARITY).assertTextContains("Cosine similarity: 1.0000 (${TinyLlamaModel.EMBD} dimensions)")

        click(LlmChatTags.panel(LlmPanel.STRUCTURED))
        click(LlmChatTags.EXTRACT)
        waitUntil(timeoutMillis = 120_000) {
            onAllNodes(hasTestTag(LlmChatTags.STATUS) and hasText("Extraction finished", substring = true)).fetchSemanticsNodes().isNotEmpty() ||
                onAllNodes(hasTestTag(LlmChatTags.ERROR)).fetchSemanticsNodes().isNotEmpty()
        }
        failOnError(LlmChatTags.ERROR)
        // The JSON-schema grammar constrains every sampled token, so even random weights must emit JSON that matches the
        // schema, or a valid prefix of it when the token budget runs out first; never invalid JSON or schema violations.
        val output = onNodeWithTag(LlmChatTags.EXTRACT_OUTPUT).fetchSemanticsNode().config
            .getOrNull(SemanticsProperties.EditableText)?.text.orEmpty().trim()
        assertTrue(output.startsWith("{"), "grammar-constrained output must start a JSON object: '$output'")
        val validation = onNodeWithTag(LlmChatTags.VALIDATION).text()
        assertTrue(
            validation.startsWith("Valid JSON matching the schema") || validation.startsWith("Incomplete JSON"),
            "grammar-constrained output failed validation: '$validation' for '$output'",
        )
    }

    @Test
    fun modelManagerInspectsARealGgufHeader() = runComposeUiTest(testTimeout = 5.minutes) {
        // Pure-Kotlin header parsing (no native library): runs on every platform, Robolectric included.
        val files = Files(NativeTestEnvironment.newTempDirectory())
        showRealShowcase(files, ShowcaseDestination.MODEL_MANAGER)
        typeInto(ModelManagerTags.INSPECT_PATH, files.llamaModel)
        click(ModelManagerTags.INSPECT)
        awaitText(ModelManagerTags.INSPECTED, "Estimated peak RAM:")
        onNodeWithTag(ModelManagerTags.INSPECTED).assertTextContains("llama", substring = true)
        onNodeWithTag(ModelManagerTags.INSPECTED).assertTextContains("context ${TinyLlamaModel.CTX_TRAIN}", substring = true)
    }

    @Test
    fun diffusionStudioSurfacesATypedLoadErrorForAnInvalidModel() = runComposeUiTest(testTimeout = 5.minutes) {
        if (!nativeEnginesAvailable("diffusionStudioSurfacesATypedLoadErrorForAnInvalidModel")) return@runComposeUiTest
        showRealShowcase(Files(NativeTestEnvironment.newTempDirectory()), ShowcaseDestination.DIFFUSION_STUDIO)
        click("${DiffusionStudioTags.BACKEND}-cpu_only")
        typeInto(DiffusionStudioTags.PROMPT, "a red fox")
        click(DiffusionStudioTags.GENERATE)
        // stable-diffusion.cpp rejects the file; the native error crosses the bridge as a typed ModelLoadException.
        awaitText(DiffusionStudioTags.ERROR, "ModelLoadException")
        onNodeWithTag(DiffusionStudioTags.IMAGE).assertDoesNotExist()
        awaitText(DiffusionStudioTags.ACTIVITY, "Idle")
    }
}
