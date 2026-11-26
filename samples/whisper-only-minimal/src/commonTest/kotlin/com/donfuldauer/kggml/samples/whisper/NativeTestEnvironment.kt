package com.donfuldauer.kggml.samples.whisper

import com.donfuldauer.kggml.runtime.KggmlRuntime
import com.donfuldauer.kggml.whisper.WhisperEngine
import kotlin.coroutines.cancellation.CancellationException

/** Per-platform facts the real-engine tests need. */
expect object NativeTestEnvironment {
    /**
     * `true` when the native runtime and whisper MUST load (the `kggml.requireNative` instrumentation argument on an
     * Android device; always on iOS, whose test binaries only build when the xcframeworks exist).
     */
    val requireNative: Boolean

    /** Writes [bytes] to a new file named [name] in a fresh scratch directory and returns its path. */
    fun writeTempFile(name: String, bytes: ByteArray): String
}

/**
 * Whether the shared runtime and whisper native libraries load here. When they do not and
 * [NativeTestEnvironment.requireNative] is false (Android local unit tests on the host JVM), the calling test is
 * skipped with a message; otherwise a missing library fails it.
 */
fun nativeWhisperAvailable(testName: String): Boolean {
    val failure = try {
        KggmlRuntime.info()
        WhisperEngine.nativeInfo()
        null
    } catch (e: CancellationException) {
        throw e
    } catch (t: Throwable) {
        t
    } ?: return true
    if (NativeTestEnvironment.requireNative) {
        throw AssertionError("$testName: native runtime / whisper are required but did not load: $failure", failure)
    }
    println("SKIPPED $testName: native runtime / whisper unavailable on this platform ($failure)")
    return false
}
