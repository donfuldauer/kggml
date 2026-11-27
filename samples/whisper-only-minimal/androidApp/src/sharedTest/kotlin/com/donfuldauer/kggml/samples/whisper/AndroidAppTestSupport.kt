package com.donfuldauer.kggml.samples.whisper

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import com.donfuldauer.kggml.runtime.KggmlRuntime
import com.donfuldauer.kggml.whisper.WhisperEngine
import java.io.File
import java.nio.file.Files
import java.util.UUID
import kotlin.coroutines.cancellation.CancellationException

/** Test environment helper shared by Robolectric (`test`) and device (`androidTest`) runs of `androidApp`. */
internal object NativeTestEnvironment {
    val requireNative: Boolean
        get() = runCatching { InstrumentationRegistry.getArguments().getString("kggml.requireNative").toBoolean() }
            .getOrDefault(false)

    fun writeTempFile(name: String, bytes: ByteArray): String {
        val base = runCatching { ApplicationProvider.getApplicationContext<Context>().cacheDir }.getOrNull()
            ?: Files.createTempDirectory("kggml-minimal").toFile()
        val dir = File(base, "kggml-minimal-${UUID.randomUUID()}").apply { mkdirs() }
        return File(dir, name).apply { writeBytes(bytes) }.absolutePath
    }
}

/** Why the native runtime / whisper do not load here, or `null` when they do. */
internal fun nativeWhisperLoadFailure(): Throwable? = try {
    KggmlRuntime.info()
    WhisperEngine.nativeInfo()
    null
} catch (e: CancellationException) {
    throw e
} catch (t: Throwable) {
    t
}
