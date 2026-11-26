package com.donfuldauer.kggml.samples.whisper

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.file.Files
import java.util.UUID

actual object NativeTestEnvironment {
    /** The `kggml.requireNative` instrumentation argument; `false` in local unit tests (with or without Robolectric). */
    actual val requireNative: Boolean
        get() = runCatching { InstrumentationRegistry.getArguments().getString("kggml.requireNative").toBoolean() }
            .getOrDefault(false)

    actual val gpuBackendsUsable: Boolean = true

    actual fun writeTempFile(name: String, bytes: ByteArray): String {
        // The app's cache directory on a device / Robolectric; a JVM temp directory in plain local unit tests.
        val base = runCatching { ApplicationProvider.getApplicationContext<Context>().cacheDir }.getOrNull()
            ?: Files.createTempDirectory("kggml-minimal").toFile()
        val dir = File(base, "kggml-minimal-${UUID.randomUUID()}").apply { mkdirs() }
        return File(dir, name).apply { writeBytes(bytes) }.absolutePath
    }

    actual fun skip(reason: String): Nothing = throw org.junit.AssumptionViolatedException(reason)
}
