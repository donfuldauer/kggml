package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState

fun main() {
    // Before Compose loads Skiko: see preloadNativeRuntime (Linux libstdc++ clash).
    preloadNativeRuntime()
    application {
        Window(
            onCloseRequest = ::exitApplication,
            title = "Kggml Showcase",
            state = rememberWindowState(width = 960.dp, height = 720.dp),
        ) {
            ShowcaseApp()
        }
    }
}
