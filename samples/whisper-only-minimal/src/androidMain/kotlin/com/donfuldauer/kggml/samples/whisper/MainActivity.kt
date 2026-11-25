package com.donfuldauer.kggml.samples.whisper

import android.app.Activity
import android.os.Bundle
import android.text.InputType
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Plain platform `Activity` (no AndroidX / Compose) to keep the audited APK minimal.
 *
 * Transcribes a PCM file (16-bit / float WAV, or raw 16 kHz mono s16le `.pcm`) with a whisper.cpp model on the
 * selected backend (CPU only / Vulkan / Auto). Switching the backend reloads the same engine through
 * `WhisperEngine.reinitialize`.
 *
 * Defaults look for `ggml-tiny.en.bin` and `jfk.wav` in the app's external files directory:
 * ```
 * adb push ggml-tiny.en.bin jfk.wav /sdcard/Android/data/com.donfuldauer.kggml.samples.whisper/files/
 * ```
 * The same values can be passed as intent extras (`model`, `audio`, `backend` = CPU_ONLY | GPU | AUTO,
 * `autostart` = true) for scripted runs, e.g.
 * `adb shell am start -n com.donfuldauer.kggml.samples.whisper/.MainActivity --es backend CPU_ONLY --ez autostart true`.
 */
class MainActivity : Activity() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val session = WhisperSampleSession()
    private var running: Job? = null

    private lateinit var status: TextView
    private lateinit var modelPath: EditText
    private lateinit var audioPath: EditText
    private lateinit var backends: RadioGroup
    private lateinit var transcribeButton: Button
    private lateinit var output: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val filesDir = getExternalFilesDir(null) ?: filesDir
        val pad = (16 * resources.displayMetrics.density).toInt()

        status = TextView(this).apply { text = "Loading Kggml runtime…" }
        modelPath = pathField(intent.getStringExtra(EXTRA_MODEL) ?: File(filesDir, DEFAULT_MODEL).path)
        audioPath = pathField(intent.getStringExtra(EXTRA_AUDIO) ?: File(filesDir, DEFAULT_AUDIO).path)
        val initial = intent.getStringExtra(EXTRA_BACKEND)
            ?.let { name -> BackendChoice.entries.firstOrNull { it.name == name } } ?: BackendChoice.CPU_ONLY
        backends = RadioGroup(this).apply {
            orientation = RadioGroup.HORIZONTAL
            BackendChoice.entries.forEach { choice ->
                addView(RadioButton(this@MainActivity).apply {
                    id = View.generateViewId()
                    tag = choice
                    text = choice.label()
                    isChecked = choice == initial
                })
            }
        }
        transcribeButton = Button(this).apply {
            text = "Transcribe"
            setOnClickListener { transcribe() }
        }
        output = TextView(this).apply { setTextIsSelectable(true) }

        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
            addView(status)
            addView(label("Model (.bin)"))
            addView(modelPath)
            addView(label("Audio (WAV or 16 kHz mono s16le .pcm)"))
            addView(audioPath)
            addView(label("Compute backend"))
            addView(backends)
            addView(transcribeButton)
            addView(output)
        }
        setContentView(ScrollView(this).apply { addView(column) })

        scope.launch {
            val text = describeRuntime()
            runOnUiThread { if (!isDestroyed) status.text = text }
        }
        if (intent.getBooleanExtra(EXTRA_AUTOSTART, false)) transcribe()
    }

    override fun onDestroy() {
        scope.cancel()
        session.close()
        super.onDestroy()
    }

    private fun transcribe() {
        if (running?.isActive == true) return
        val model = modelPath.text.toString().trim()
        val audio = audioPath.text.toString().trim()
        val choice = backends.findViewById<RadioButton>(backends.checkedRadioButtonId)?.tag as? BackendChoice
            ?: BackendChoice.CPU_ONLY
        transcribeButton.isEnabled = false
        output.text = "Transcribing on ${choice.label()}…"
        running = scope.launch {
            val text = try {
                val pcm = parsePcmFile(File(audio).readBytes())
                formatReport(session.transcribe(model, pcm, choice.toComputeBackend()))
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                "Failed: ${t::class.simpleName}: ${t.message}"
            }
            runOnUiThread {
                if (!isDestroyed) {
                    output.text = text
                    transcribeButton.isEnabled = true
                }
            }
        }
    }

    private fun pathField(value: String) = EditText(this).apply {
        setText(value)
        isSingleLine = true
        inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
    }

    private fun label(text: String) = TextView(this).apply {
        this.text = text
        setPadding(0, (12 * resources.displayMetrics.density).toInt(), 0, 0)
    }

    private companion object {
        const val DEFAULT_MODEL = "ggml-tiny.en.bin"
        const val DEFAULT_AUDIO = "jfk.wav"
        const val EXTRA_MODEL = "model"
        const val EXTRA_AUDIO = "audio"
        const val EXTRA_BACKEND = "backend"
        const val EXTRA_AUTOSTART = "autostart"
    }
}
