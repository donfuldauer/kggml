package com.donfuldauer.kggml.samples.showcase

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.util.UUID

actual object NativeTestEnvironment {
    /**
     * The `kggml.requireNative` instrumentation argument (set from `-Pkggml.requireNative` by the build). Always
     * `false` under Robolectric, where the APK's arm64-v8a / x86_64 libraries cannot load.
     */
    actual val requireNative: Boolean
        get() = InstrumentationRegistry.getArguments().getString("kggml.requireNative").toBoolean()

    actual fun newTempDirectory(): String {
        val context = ApplicationProvider.getApplicationContext<Context>()
        return File(context.cacheDir, "kggml-showcase-e2e-${UUID.randomUUID()}").apply { mkdirs() }.absolutePath
    }
}
