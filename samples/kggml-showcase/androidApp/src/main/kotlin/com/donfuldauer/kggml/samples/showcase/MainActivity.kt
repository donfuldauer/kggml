package com.donfuldauer.kggml.samples.showcase

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.donfuldauer.kggml.samples.showcase.llm.AndroidImageSource

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Model directory, export directory and AndroidMemoryPressureObserver need the application context.
        ShowcaseAndroid.init(this)
        // Activity-result launchers must be registered before the activity is started.
        val llmEnvironment = LlmChatEnvironment(imageSource = AndroidImageSource(this))
        // The Whisper Studio's live dictation needs the microphone; capture fails with a visible error if denied.
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), RECORD_AUDIO_REQUEST)
        }
        setContent { ShowcaseApp(llmEnvironment = llmEnvironment) }
    }

    private companion object {
        const val RECORD_AUDIO_REQUEST = 1
    }
}
