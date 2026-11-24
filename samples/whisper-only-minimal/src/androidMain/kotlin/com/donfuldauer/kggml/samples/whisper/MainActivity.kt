package com.donfuldauer.kggml.samples.whisper

import android.app.Activity
import android.os.Bundle
import android.widget.TextView

/** Plain platform `Activity` (no AndroidX / Compose) to keep the audited APK minimal. */
class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val status = TextView(this).apply {
            setPadding(48, 48, 48, 48)
            text = "Loading Kggml runtime…"
        }
        setContentView(status)
        Thread {
            val text = describeRuntime()
            runOnUiThread { if (!isDestroyed) status.text = text }
        }.apply { name = "kggml-runtime-summary"; isDaemon = true }.start()
    }
}
