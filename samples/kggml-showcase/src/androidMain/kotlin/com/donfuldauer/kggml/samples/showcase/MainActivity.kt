package com.donfuldauer.kggml.samples.showcase

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // The Whisper Studio's live dictation needs the microphone; capture fails with a visible error if denied.
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), RECORD_AUDIO_REQUEST)
        }
        setContent { ShowcaseApp() }
    }

    private companion object {
        const val RECORD_AUDIO_REQUEST = 1
    }
}
