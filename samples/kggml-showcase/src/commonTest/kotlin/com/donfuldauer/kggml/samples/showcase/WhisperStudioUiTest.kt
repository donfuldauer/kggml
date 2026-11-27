package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.test.ComposeUiTest
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertIsOff
import androidx.compose.ui.test.assertIsOn
import androidx.compose.ui.test.assertIsSelected
import androidx.compose.ui.test.assertIsNotSelected
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextReplacement
import androidx.compose.ui.test.v2.runComposeUiTest
import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.TranscriptionEvent
import com.donfuldauer.kggml.samples.showcase.whisper.FakeAudioCapture
import com.donfuldauer.kggml.samples.showcase.whisper.FakeTranscriberFactory
import com.donfuldauer.kggml.samples.showcase.whisper.STUDIO_BACKENDS
import com.donfuldauer.kggml.samples.showcase.whisper.StudioActivity
import com.donfuldauer.kggml.samples.showcase.whisper.WhisperStudioContent
import com.donfuldauer.kggml.samples.showcase.whisper.WhisperStudioState
import com.donfuldauer.kggml.samples.showcase.whisper.WhisperStudioTags
import com.donfuldauer.kggml.samples.showcase.whisper.wavBytes
import kotlinx.coroutines.Dispatchers
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Compose UI tests of the Live Whisper Dictation & Audio Studio, driven by a deterministic microphone and
 * transcriber (see `WhisperFakes.kt`): stream start/stop/cancel controls, the backend switcher, partial vs.
 * committed rendering, file transcription, and error states.
 */
@OptIn(ExperimentalTestApi::class)
class WhisperStudioUiTest : UiTestBase() {
    private val mic = FakeAudioCapture()

    private fun ComposeUiTest.openStudio(
        factory: FakeTranscriberFactory = FakeTranscriberFactory(),
        files: Map<String, ByteArray> = emptyMap(),
        modelPath: String = "/models/ggml-tiny.en.bin",
        backgroundStressSupported: Boolean = false,
    ) {
        val environment = WhisperStudioEnvironment(
            transcriberFactory = factory,
            audioCapture = mic,
            readFile = { files.getValue(it) },
            initialModelPath = modelPath,
            backgroundStressSupported = backgroundStressSupported,
        )
        setContent {
            ShowcaseApp(
                inspector = { fakeSnapshot() },
                ioDispatcher = Dispatchers.Unconfined,
                initialDestination = ShowcaseDestination.WHISPER_STUDIO,
                whisperEnvironment = environment,
            )
        }
        onNodeWithTag(ShowcaseDestination.WHISPER_STUDIO.navTag).assertIsSelected()
        onNodeWithTag(ShowcaseDestination.WHISPER_STUDIO.screenTag).assertIsDisplayed()
    }

    private fun ComposeUiTest.waitForTag(tag: String) =
        waitUntil(timeoutMillis = 5_000) { onAllNodes(hasTestTag(tag)).fetchSemanticsNodes().isNotEmpty() }

    private fun ComposeUiTest.waitForText(tag: String, text: String) = waitUntil(timeoutMillis = 5_000) {
        onAllNodes(hasTestTag(tag) and hasText(text, substring = true)).fetchSemanticsNodes().isNotEmpty()
    }

    @Test
    fun navigatingToTheStudioShowsItsControls() = runComposeUiTest {
        setContent { ShowcaseApp(inspector = { fakeSnapshot() }, ioDispatcher = Dispatchers.Unconfined) }
        onNodeWithTag(ShowcaseDestination.WHISPER_STUDIO.navTag).performClick()
        onNodeWithTag(ShowcaseDestination.WHISPER_STUDIO.screenTag).assertIsDisplayed()
        onNodeWithTag(WhisperStudioTags.MODEL_PATH).assertExists()
        onNodeWithTag(WhisperStudioTags.START).assertExists()
        onNodeWithTag(WhisperStudioTags.WAVEFORM).assertExists()
        STUDIO_BACKENDS.forEach { onNodeWithTag(WhisperStudioTags.backendChip(it)).assertExists() }
    }

    @Test
    fun liveStreamingRendersPartialThenCommittedSegments() = runComposeUiTest {
        openStudio()
        onNodeWithTag(WhisperStudioTags.STOP).assertIsNotEnabled()
        onNodeWithTag(WhisperStudioTags.START).performScrollTo().assertIsEnabled().performClick()
        waitForText(WhisperStudioTags.ACTIVITY, "Listening")
        onNodeWithTag(WhisperStudioTags.ACTIVITY).assertTextContains("48000 Hz", substring = true)
        onNodeWithTag(WhisperStudioTags.START).assertIsNotEnabled()
        onNodeWithTag(WhisperStudioTags.STOP).assertIsEnabled()
        onNodeWithTag(WhisperStudioTags.backendChip(ComputeBackend.CPU_ONLY)).assertIsNotEnabled()
        onNodeWithTag(WhisperStudioTags.ACTIVE_BACKEND).assertTextContains("Active: Metal", substring = true)

        mic.sendMs(100)
        mic.sendMs(100)
        waitForTag(WhisperStudioTags.PARTIAL)
        waitForText(WhisperStudioTags.PARTIAL, "word1 word2")
        onNodeWithTag(WhisperStudioTags.PARTIAL).assertTextContains("lag 0 ms", substring = true)
        onNodeWithTag("committed-0").assertDoesNotExist()

        onNodeWithTag(WhisperStudioTags.STOP).performScrollTo().performClick()
        waitForTag("committed-0")
        onNodeWithTag("committed-0").assertTextContains("word1 word2.")
        onNodeWithTag("committed-0").assertTextContains("0:00.0 → 0:00.2")
        onNodeWithTag(WhisperStudioTags.PARTIAL).assertDoesNotExist()
        waitForText(WhisperStudioTags.ACTIVITY, "Idle")
        onNodeWithTag(WhisperStudioTags.STATUS).assertTextContains("Completed: 1 segment(s)", substring = true)
        onNodeWithTag(WhisperStudioTags.START).assertIsEnabled()
    }

    @Test
    fun cancelStopsTheStreamWithoutCommitting() = runComposeUiTest {
        openStudio()
        onNodeWithTag(WhisperStudioTags.START).performScrollTo().performClick()
        waitForText(WhisperStudioTags.ACTIVITY, "Listening")
        mic.sendMs(100)
        waitForTag(WhisperStudioTags.PARTIAL)
        onNodeWithTag(WhisperStudioTags.CANCEL).performScrollTo().assertIsEnabled().performClick()
        waitForText(WhisperStudioTags.ACTIVITY, "Idle")
        onNodeWithTag(WhisperStudioTags.STATUS).assertTextContains("Cancelled")
        onNodeWithTag(WhisperStudioTags.PARTIAL).assertDoesNotExist()
        onNodeWithTag("committed-0").assertDoesNotExist()
    }

    @Test
    fun backendSwitcherReloadsTheModelOnTheChosenBackend() = runComposeUiTest {
        val factory = FakeTranscriberFactory()
        openStudio(factory = factory)
        onNodeWithTag(WhisperStudioTags.ACTIVE_BACKEND).assertTextContains("Model not loaded")
        onNodeWithTag(WhisperStudioTags.backendChip(ComputeBackend.AUTO)).assertIsSelected()

        onNodeWithTag(WhisperStudioTags.backendChip(ComputeBackend.CPU_ONLY)).performScrollTo().performClick()
        onNodeWithTag(WhisperStudioTags.backendChip(ComputeBackend.CPU_ONLY)).assertIsSelected()
        onNodeWithTag(WhisperStudioTags.backendChip(ComputeBackend.AUTO)).assertIsNotSelected()
        onNodeWithTag(WhisperStudioTags.START).performScrollTo().performClick()
        waitForText(WhisperStudioTags.ACTIVE_BACKEND, "Active: CPU (use_gpu=false)")
        assertEquals(BackendConfig.CPU, factory.opened.single().config)
        onNodeWithTag(WhisperStudioTags.STOP).performScrollTo().performClick()
        waitForText(WhisperStudioTags.ACTIVITY, "Idle")

        // An idle loaded model is reloaded right away on the newly selected backend.
        onNodeWithTag(WhisperStudioTags.backendChip(ComputeBackend.GPU_METAL)).performScrollTo().performClick()
        waitForText(WhisperStudioTags.ACTIVE_BACKEND, "Active: Metal (use_gpu=true)")
        assertEquals(listOf(BackendConfig(backend = ComputeBackend.GPU_METAL)), factory.opened.single().reinitialized)
    }

    @Test
    fun streamErrorIsRendered() = runComposeUiTest {
        openStudio(factory = FakeTranscriberFactory { it.streamError = InferenceCancelledException("app entered background") })
        onNodeWithTag(WhisperStudioTags.START).performScrollTo().performClick()
        waitForText(WhisperStudioTags.ACTIVITY, "Listening")
        mic.sendMs(100)
        waitForTag(WhisperStudioTags.ERROR)
        onNodeWithTag(WhisperStudioTags.ERROR).assertTextContains("app entered background", substring = true)
    }

    @Test
    fun modelLoadErrorIsRendered() = runComposeUiTest {
        openStudio(factory = FakeTranscriberFactory(failure = ModelLoadException("not a whisper model")))
        onNodeWithTag(WhisperStudioTags.START).performScrollTo().performClick()
        waitForTag(WhisperStudioTags.ERROR)
        onNodeWithTag(WhisperStudioTags.ERROR).assertTextContains("ModelLoadException: not a whisper model")
        waitForText(WhisperStudioTags.ACTIVITY, "Idle")
    }

    @Test
    fun startIsDisabledWithoutAModelPath() = runComposeUiTest {
        openStudio(modelPath = "")
        onNodeWithTag(WhisperStudioTags.START).assertIsNotEnabled()
        onNodeWithTag(WhisperStudioTags.MODEL_PATH).performTextReplacement("/models/ggml-base.bin")
        onNodeWithTag(WhisperStudioTags.START).assertIsEnabled()
    }

    @Test
    fun fileTranscriptionRendersTimestampedSegments() = runComposeUiTest {
        val factory = FakeTranscriberFactory {
            it.fileSegments = listOf(
                TranscriptionEvent.CommittedSegment(0, " First sentence.", 0, 1_200),
                TranscriptionEvent.CommittedSegment(1, " Second sentence.", 1_200, 2_000),
            )
        }
        openStudio(factory = factory, files = mapOf("/audio/clip.wav" to wavBytes(ShortArray(32_000))))
        onNodeWithTag(WhisperStudioTags.TRANSCRIBE_FILE).performScrollTo().assertIsNotEnabled()
        onNodeWithTag(WhisperStudioTags.FILE_PATH).performScrollTo().performTextReplacement("/audio/clip.wav")
        onNodeWithTag(WhisperStudioTags.TRANSCRIBE_FILE).performScrollTo().assertIsEnabled().performClick()
        waitForTag("committed-1")
        onNodeWithTag("committed-0").assertTextContains("First sentence.")
        onNodeWithTag("committed-1").assertTextContains("0:01.2 → 0:02.0")
        onNodeWithTag(WhisperStudioTags.STATUS).assertTextContains("Transcribed 2.0 s of audio", substring = true)
        assertEquals(32_000, factory.opened.single().transcribedFiles.single().sampleCount)
    }

    @Test
    fun backgroundStressToggleIsOfferedOnlyWhereSupported() = runComposeUiTest {
        openStudio(backgroundStressSupported = true)
        onNodeWithTag(WhisperStudioTags.BACKGROUND_STRESS).performScrollTo().assertIsOff().performClick()
        onNodeWithTag(WhisperStudioTags.BACKGROUND_STRESS).assertIsOn()
    }

    @Test
    fun backgroundStressToggleIsHiddenOnDesktop() = runComposeUiTest {
        openStudio(backgroundStressSupported = false)
        onNodeWithTag(WhisperStudioTags.BACKGROUND_STRESS).assertDoesNotExist()
    }

    @Test
    fun missingMicrophoneIsExplained() = runComposeUiTest {
        setContent {
            WhisperStudioContent(state = WhisperStudioState(modelPath = "/m.bin", micAvailable = false))
        }
        onNodeWithTag(WhisperStudioTags.MIC_UNAVAILABLE).assertExists()
        onNodeWithTag(WhisperStudioTags.START).assertIsNotEnabled()
    }

    @Test
    fun loadingStateDisablesTheControls() = runComposeUiTest {
        setContent {
            WhisperStudioContent(
                state = WhisperStudioState(modelPath = "/m.bin", micAvailable = true, activity = StudioActivity.LOADING_MODEL),
            )
        }
        onNodeWithTag(WhisperStudioTags.ACTIVITY).assertTextContains("Loading model…")
        onNodeWithTag(WhisperStudioTags.START).assertIsNotEnabled()
        onNodeWithTag(WhisperStudioTags.CANCEL).assertIsEnabled()
        onNodeWithTag(WhisperStudioTags.MODEL_PATH).assertIsNotEnabled()
        STUDIO_BACKENDS.forEach { onNodeWithTag(WhisperStudioTags.backendChip(it)).assertIsNotEnabled() }
    }
}
