package com.donfuldauer.kggml.samples.showcase

actual abstract class UiTestBase actual constructor() {
    private companion object {
        // Every Compose UI test class extends this, so the native runtime (and the system libstdc++) is loaded before
        // the first test brings in Skiko, exactly as Main.kt does for the app. See preloadNativeRuntime.
        init {
            preloadNativeRuntime()
        }
    }
}
