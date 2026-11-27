package com.donfuldauer.kggml.samples.showcase

import com.donfuldauer.kggml.diffusion.DiffusionEngine
import com.donfuldauer.kggml.llama.LlamaEngine
import com.donfuldauer.kggml.runtime.KggmlRuntime
import com.donfuldauer.kggml.whisper.WhisperEngine
import kotlin.coroutines.cancellation.CancellationException

/** Per-platform facts the real-engine end-to-end tests need. */
expect object NativeTestEnvironment {
    /**
     * `true` when the native engines MUST be present (`-Pkggml.requireNative=true` on desktop and for Android
     * instrumented runs; always on iOS, whose test binaries only build when the xcframeworks exist).
     */
    val requireNative: Boolean

    /** Ends the calling test as SKIPPED (a JUnit assumption failure on the JVM and Android), never as passed. */
    fun skip(reason: String): Nothing

    /** A fresh writable scratch directory for this test. */
    fun newTempDirectory(): String
}

/**
 * Whether the shared runtime and all three engines' native libraries load on this platform. When they do not and
 * [NativeTestEnvironment.requireNative] is false (no host build on desktop, Robolectric on Android), the calling
 * test is reported as skipped; when it is true, the missing library fails the test. Never returns `false`.
 */
fun nativeEnginesAvailable(testName: String): Boolean {
    val failure = try {
        KggmlRuntime.info()
        WhisperEngine.nativeInfo()
        LlamaEngine.nativeInfo()
        DiffusionEngine.nativeInfo()
        null
    } catch (e: CancellationException) {
        throw e
    } catch (t: Throwable) {
        t
    } ?: return true
    if (NativeTestEnvironment.requireNative) {
        throw AssertionError("$testName: native engines are required but did not load: $failure", failure)
    }
    NativeTestEnvironment.skip("$testName: native engines unavailable on this platform/build ($failure)")
}
