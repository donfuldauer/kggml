package com.donfuldauer.kggml.samples.whisper

import android.content.Intent
import android.view.View
import android.widget.RadioButton
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.NoMatchingViewException
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.action.ViewActions.replaceText
import androidx.test.espresso.action.ViewActions.scrollTo
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isChecked
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.isEnabled
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.donfuldauer.kggml.whisper.TinyWhisperModel
import java.io.File
import kotlin.math.PI
import kotlin.math.sin
import org.hamcrest.Matcher
import org.hamcrest.Matchers.allOf
import org.hamcrest.Matchers.containsString
import org.hamcrest.Matchers.startsWith
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * UI tests of the minimal sample's plain-`Activity` screen. Shared by the local unit tests (Robolectric, where the
 * APK's native libraries cannot load) and the instrumented tests (device / emulator with
 * `-Pkggml.bundleNative=true`), with the expected outcome chosen by whether the native runtime loads:
 *
 * - native present: the status shows the real runtime and transcription runs on the real engine;
 * - native absent: the status explains the runtime is unavailable and transcription fails with a visible message,
 *   never a crash.
 */
@RunWith(AndroidJUnit4::class)
class MainActivityTest {
    private var scenario: ActivityScenario<MainActivity>? = null
    /** Probed, not skipped: these tests assert the right outcome for both cases (a required library must load). */
    private val nativeLoads: Boolean by lazy {
        val failure = nativeWhisperLoadFailure()
        if (failure != null && NativeTestEnvironment.requireNative) {
            throw AssertionError("native runtime / whisper are required but did not load: $failure", failure)
        }
        failure == null
    }

    @After
    fun tearDown() {
        scenario?.close()
    }

    private fun launch(configure: Intent.() -> Unit = {}): ActivityScenario<MainActivity> {
        val intent = Intent(ApplicationProvider.getApplicationContext(), MainActivity::class.java).apply(configure)
        return ActivityScenario.launch<MainActivity>(intent).also { scenario = it }
    }

    /**
     * Polls until a view matching [matcher] exists, scrolls it into view and checks it is displayed (the sample
     * updates views from background coroutines; the result text sits below the fold on small screens).
     */
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

    private fun toneWav(seconds: Int): ByteArray {
        val samples = seconds * 16_000
        val out = ByteArray(44 + samples * 2)
        fun ascii(at: Int, s: String) = s.forEachIndexed { i, c -> out[at + i] = c.code.toByte() }
        fun le16(at: Int, v: Int) { out[at] = v.toByte(); out[at + 1] = (v shr 8).toByte() }
        fun le32(at: Int, v: Int) { le16(at, v and 0xFFFF); le16(at + 2, v ushr 16) }
        ascii(0, "RIFF"); le32(4, 36 + samples * 2); ascii(8, "WAVE")
        ascii(12, "fmt "); le32(16, 16); le16(20, 1); le16(22, 1); le32(24, 16_000); le32(28, 32_000); le16(32, 2); le16(34, 16)
        ascii(36, "data"); le32(40, samples * 2)
        for (i in 0 until samples) le16(44 + i * 2, (sin(2 * PI * 330 * i / 16_000) * 6_000).toInt())
        return out
    }

    @Test
    fun screenShowsDefaultsAndTheRuntimeStatus() {
        launch()
        // Defaults point at ggml-tiny.en.bin / jfk.wav in the app's external files directory, CPU only selected.
        awaitView(withText(containsString("ggml-tiny.en.bin")))
        awaitView(withText(containsString("jfk.wav")))
        onView(withText(BackendChoice.CPU_ONLY.label())).check(matches(isChecked()))
        onView(withText("Transcribe")).check(matches(isEnabled()))
        if (nativeLoads) {
            awaitView(withText(allOf(startsWith("Kggml whisper-only minimal sample\nRuntime "), containsString("CPU_ONLY"))))
        } else {
            awaitView(withText(startsWith("Kggml whisper-only minimal sample\nNative runtime unavailable: ")))
        }
    }

    @Test
    fun intentExtrasPreselectTheBackendAndPaths() {
        launch {
            putExtra("model", "/data/local/tmp/model.bin")
            putExtra("audio", "/data/local/tmp/audio.wav")
            putExtra("backend", "AUTO")
        }
        awaitView(withText("/data/local/tmp/model.bin"))
        awaitView(withText("/data/local/tmp/audio.wav"))
        onView(withText(BackendChoice.AUTO.label())).check(matches(isChecked()))
        onView(withText(BackendChoice.CPU_ONLY.label())).check { view, _ -> assertEquals(false, (view as RadioButton).isChecked) }
    }

    @Test
    fun missingAudioFileShowsAFailureInsteadOfCrashing() {
        launch { putExtra("audio", "/nonexistent/kggml/audio.wav") }
        onView(withText("Transcribe")).perform(scrollTo(), click())
        awaitView(withText(startsWith("Failed: FileNotFoundException: /nonexistent/kggml/audio.wav")))
        // The button is re-enabled for another attempt.
        onView(withText("Transcribe")).check(matches(isEnabled()))
    }

    @Test
    fun autostartTranscribesWithTheSelectedBackend() {
        val dir = File(ApplicationProvider.getApplicationContext<android.content.Context>().cacheDir, "minimal-ui-test").apply { mkdirs() }
        val model = File(dir, "ggml-tiny-synthetic.bin").apply { writeBytes(TinyWhisperModel.bytes()) }
        val audio = File(dir, "tone.wav").apply { writeBytes(toneWav(2)) }
        launch {
            putExtra("model", model.absolutePath)
            putExtra("audio", audio.absolutePath)
            putExtra("backend", "CPU_ONLY")
            putExtra("autostart", true)
        }
        if (nativeLoads) {
            // The real engine ran the 2 s WAV on the CPU (use_gpu=false), as requested.
            awaitView(withText(startsWith("Requested CPU_ONLY, ran on CPU_ONLY (use_gpu=false)\nAudio 2000 ms, ")))
        } else {
            awaitView(withText(startsWith("Failed: NativeFailureException: ")))
        }
        onView(withText("Transcribe")).check(matches(isEnabled()))
    }

    @Test
    fun manualTranscribeAfterEditingThePaths() {
        val dir = File(ApplicationProvider.getApplicationContext<android.content.Context>().cacheDir, "minimal-ui-test-2").apply { mkdirs() }
        val model = File(dir, "ggml-tiny-synthetic.bin").apply { writeBytes(TinyWhisperModel.bytes()) }
        val audio = File(dir, "tone.wav").apply { writeBytes(toneWav(1)) }
        launch()
        awaitView(withText(containsString("ggml-tiny.en.bin")))
        onView(withText(containsString("ggml-tiny.en.bin"))).perform(replaceText(model.absolutePath))
        onView(withText(containsString("jfk.wav"))).perform(replaceText(audio.absolutePath))
        onView(withText(BackendChoice.CPU_ONLY.label())).perform(click())
        onView(withText("Transcribe")).perform(scrollTo(), click())
        if (nativeLoads) {
            awaitView(withText(startsWith("Requested CPU_ONLY, ran on CPU_ONLY (use_gpu=false)\nAudio 1000 ms, ")))
        } else {
            awaitView(withText(startsWith("Failed: NativeFailureException: ")))
        }
    }
}
