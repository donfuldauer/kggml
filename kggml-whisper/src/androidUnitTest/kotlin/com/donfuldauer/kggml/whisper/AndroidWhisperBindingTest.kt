package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.NativeFailureException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest

/**
 * Android local unit tests run on the host JVM, where the APK's arm64-v8a / x86_64 jniLibs cannot be loaded. They
 * verify that the Android binding wiring degrades into a typed, catchable error rather than a crash (the native
 * path itself is exercised by jvmTest through the same JNI bridge, and on-device by instrumented tests).
 */
class AndroidWhisperBindingTest {
    @Test
    fun androidLoaderTargetsTheWhisperJniBridge() {
        assertEquals("kggml_whisper_jni", AndroidWhisperLibraryLoader.JNI_LIBRARY)
    }

    @Test
    fun missingNativeLibrarySurfacesAsNativeFailureException() = runTest {
        val e = assertFailsWith<NativeFailureException> { WhisperEngine("/data/local/tmp/ggml-tiny.bin", BackendConfig.CPU) }
        assertIs<UnsatisfiedLinkError>(e.cause)
        assertTrue(e.message!!.startsWith("Failed to load the Kggml whisper native library"), e.message)
    }

    @Test
    fun nativeInfoReportsTheSameTypedError() {
        assertFailsWith<NativeFailureException> { WhisperEngine.nativeInfo() }
    }
}
