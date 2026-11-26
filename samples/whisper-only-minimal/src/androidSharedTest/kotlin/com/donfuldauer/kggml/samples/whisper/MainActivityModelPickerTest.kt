package com.donfuldauer.kggml.samples.whisper

import android.content.Context
import android.content.Intent
import android.view.View
import android.widget.EditText
import android.widget.RadioButton
import androidx.lifecycle.Lifecycle
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.NoMatchingViewException
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.action.ViewActions.replaceText
import androidx.test.espresso.action.ViewActions.scrollTo
import androidx.test.espresso.assertion.ViewAssertions.doesNotExist
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.Visibility
import androidx.test.espresso.matcher.ViewMatchers.isChecked
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.isNotChecked
import androidx.test.espresso.matcher.ViewMatchers.withEffectiveVisibility
import androidx.test.espresso.matcher.ViewMatchers.withTagValue
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.donfuldauer.kggml.whisper.TinyWhisperModel
import java.io.File
import org.hamcrest.Matcher
import org.hamcrest.Matchers.`is`
import org.hamcrest.Matchers.startsWith
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The model picker of the minimal sample's screen: the whisper models (`.bin`) already in the app's files directory
 * next to a custom path. Shared by Robolectric and device runs (the end-to-end transcription accepts either outcome,
 * like [MainActivityTest]).
 */
@RunWith(AndroidJUnit4::class)
class MainActivityModelPickerTest {
    private var scenario: ActivityScenario<MainActivity>? = null
    private val context: Context get() = ApplicationProvider.getApplicationContext()

    /** The folder the activity scans: its external files directory, or its internal one without external storage. */
    private val folder: File get() = context.getExternalFilesDir(null) ?: context.filesDir
    private val created = mutableListOf<File>()

    /** Models already in the folder (e.g. on a real phone) are moved here for the test and put back afterwards. */
    private val parked: File get() = File(context.cacheDir, "model-picker-test-parked")

    private val nativeLoads: Boolean by lazy {
        val failure = nativeWhisperLoadFailure()
        if (failure != null && NativeTestEnvironment.requireNative) {
            throw AssertionError("native runtime / whisper are required but did not load: $failure", failure)
        }
        failure == null
    }

    @Before
    fun startWithoutModels() {
        // Each test decides what is in the folder; models already there are kept safe and restored in tearDown.
        parked.mkdirs()
        folder.listFiles()?.filter { it.isFile && it.name.lowercase().endsWith(".bin") }?.forEach { model ->
            check(model.renameTo(File(parked, model.name))) { "cannot move ${model.path} aside" }
        }
    }

    @After
    fun tearDown() {
        scenario?.close()
        created.forEach { it.delete() }
        parked.listFiles()?.forEach { model -> check(model.renameTo(File(folder, model.name))) { "cannot restore ${model.name}" } }
    }

    private fun modelInFolder(name: String, bytes: ByteArray = byteArrayOf(0)): File =
        File(folder, name).apply { parentFile?.mkdirs(); writeBytes(bytes) }.also { created += it }

    private fun launch(configure: Intent.() -> Unit = {}): ActivityScenario<MainActivity> {
        val intent = Intent(context, MainActivity::class.java).apply(configure)
        return ActivityScenario.launch<MainActivity>(intent).also { scenario = it }
    }

    private fun tagged(tag: String): Matcher<View> = withTagValue(`is`(tag as Any))

    private fun choice(fileName: String) = tagged(MainActivityTags.modelChoice(fileName))

    private fun pathText(): String {
        var text = ""
        scenario!!.onActivity { text = it.window.decorView.findViewWithTag<EditText>(MainActivityTags.MODEL_PATH).text.toString() }
        return text
    }

    private fun awaitView(matcher: Matcher<View>, timeoutMs: Long = 120_000) {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (true) {
            try {
                onView(matcher).perform(scrollTo()).check(matches(isDisplayed()))
                return
            } catch (e: Throwable) {
                if (e !is NoMatchingViewException && e !is AssertionError) throw e
                if (System.currentTimeMillis() > deadline) throw e
                Thread.sleep(100)
            }
        }
    }

    @Test
    fun emptyFolderShowsTheHintAndKeepsTheDefaultPathAsACustomPath() {
        launch()
        awaitView(tagged(MainActivityTags.NO_MODELS))
        onView(tagged(MainActivityTags.NO_MODELS)).check(matches(withText(noModelsHint(folder.path))))
        onView(tagged(MainActivityTags.MODEL_CUSTOM)).check(matches(isChecked()))
        assertEquals(File(folder, "ggml-tiny.en.bin").path, pathText())
    }

    @Test
    fun folderModelsAreListedAndTheDefaultOneIsPreselected() {
        modelInFolder("ggml-tiny.en.bin")
        modelInFolder("ggml-base.en.bin")
        modelInFolder("jfk.wav")
        launch()
        onView(choice("ggml-base.en.bin")).check(matches(isNotChecked()))
        onView(choice("ggml-tiny.en.bin")).check(matches(isChecked()))
        onView(choice("jfk.wav")).check(doesNotExist())
        onView(tagged(MainActivityTags.MODEL_CUSTOM)).check(matches(isNotChecked()))
        onView(tagged(MainActivityTags.NO_MODELS)).check(matches(withEffectiveVisibility(Visibility.GONE)))
        assertEquals(File(folder, "ggml-tiny.en.bin").path, pathText())
    }

    @Test
    fun withoutTheDefaultModelTheFirstFolderModelIsPreselected() {
        modelInFolder("ggml-small.bin")
        modelInFolder("ggml-base.en.bin")
        launch()
        onView(choice("ggml-base.en.bin")).check(matches(isChecked()))
        assertEquals(File(folder, "ggml-base.en.bin").path, pathText())
    }

    @Test
    fun choosingAModelFillsThePathAndEditingThePathSelectsCustom() {
        modelInFolder("ggml-base.en.bin")
        modelInFolder("ggml-small.bin")
        launch { putExtra("model", "/data/local/tmp/other.bin") }
        // The intent's path is not in the folder: custom.
        onView(tagged(MainActivityTags.MODEL_CUSTOM)).check(matches(isChecked()))

        onView(choice("ggml-small.bin")).perform(scrollTo(), click())
        assertEquals(File(folder, "ggml-small.bin").path, pathText())
        onView(choice("ggml-small.bin")).check(matches(isChecked()))

        onView(choice("ggml-base.en.bin")).perform(scrollTo(), click())
        assertEquals(File(folder, "ggml-base.en.bin").path, pathText())
        onView(choice("ggml-small.bin")).check(matches(isNotChecked()))

        // Typing another path selects "Custom path"; typing a folder model's path selects that model again.
        onView(tagged(MainActivityTags.MODEL_PATH)).perform(scrollTo(), replaceText("/sdcard/mine.bin"))
        onView(tagged(MainActivityTags.MODEL_CUSTOM)).check(matches(isChecked()))
        onView(choice("ggml-base.en.bin")).check(matches(isNotChecked()))
        onView(tagged(MainActivityTags.MODEL_PATH)).perform(scrollTo(), replaceText(File(folder, "ggml-small.bin").path))
        onView(choice("ggml-small.bin")).check(matches(isChecked()))

        // "Custom path" from a folder model starts from an empty field.
        onView(tagged(MainActivityTags.MODEL_CUSTOM)).perform(scrollTo(), click())
        assertEquals("", pathText())
        onView(tagged(MainActivityTags.MODEL_CUSTOM)).check(matches(isChecked()))
    }

    @Test
    fun modelsCopiedInWhileTheAppIsAwayAppearOnReturn() {
        launch()
        awaitView(tagged(MainActivityTags.NO_MODELS))
        onView(choice("ggml-tiny.en.bin")).check(doesNotExist())

        scenario!!.moveToState(Lifecycle.State.CREATED)
        modelInFolder("ggml-tiny.en.bin")
        scenario!!.moveToState(Lifecycle.State.RESUMED)

        // The default path was already in the field, so its model is now the selection.
        awaitView(choice("ggml-tiny.en.bin"))
        onView(choice("ggml-tiny.en.bin")).check(matches(isChecked()))
        onView(tagged(MainActivityTags.NO_MODELS)).check(matches(withEffectiveVisibility(Visibility.GONE)))
    }

    @Test
    fun aChosenFolderModelIsTheOneTranscribedWith() {
        modelInFolder("ggml-tiny-synthetic.bin", TinyWhisperModel.bytes())
        val audio = NativeTestEnvironment.writeTempFile("tone.wav", wav16k(seconds = 1))
        launch {
            putExtra("model", "/nonexistent/kggml/model.bin")
            putExtra("audio", audio)
            putExtra("backend", "CPU_ONLY")
        }
        onView(choice("ggml-tiny-synthetic.bin")).perform(scrollTo(), click())
        onView(withText("Transcribe")).perform(scrollTo(), click())
        if (nativeLoads) {
            // The real engine loaded the folder model (the typed path does not exist) and ran the 1 s clip.
            awaitView(withText(startsWith("Requested CPU_ONLY, ran on CPU_ONLY (use_gpu=false)\nAudio 1000 ms, ")))
        } else {
            awaitView(withText(startsWith("Failed: NativeFailureException: ")))
        }
    }

    /** A 1 kHz-ish tone as a canonical 16 kHz mono 16-bit WAV. */
    private fun wav16k(seconds: Int): ByteArray {
        val samples = seconds * 16_000
        val out = ByteArray(44 + samples * 2)
        fun ascii(at: Int, s: String) = s.forEachIndexed { i, c -> out[at + i] = c.code.toByte() }
        fun le16(at: Int, v: Int) { out[at] = v.toByte(); out[at + 1] = (v shr 8).toByte() }
        fun le32(at: Int, v: Int) { le16(at, v and 0xFFFF); le16(at + 2, v ushr 16) }
        ascii(0, "RIFF"); le32(4, 36 + samples * 2); ascii(8, "WAVE")
        ascii(12, "fmt "); le32(16, 16); le16(20, 1); le16(22, 1); le32(24, 16_000); le32(28, 32_000); le16(32, 2); le16(34, 16)
        ascii(36, "data"); le32(40, samples * 2)
        for (i in 0 until samples) le16(44 + i * 2, if ((i / 8) % 2 == 0) 6_000 else -6_000)
        return out
    }
}
