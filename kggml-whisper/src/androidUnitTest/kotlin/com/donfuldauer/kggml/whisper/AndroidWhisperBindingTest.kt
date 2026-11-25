package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.NativeFailureException
import java.lang.reflect.Modifier
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlinx.coroutines.flow.emptyFlow
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

    @Test
    fun segmentListenerMatchesTheJniCallbackContract() {
        // kggml_whisper_jni.cpp looks the callback up as GetMethodID(cls, "onNewSegments", "(II)I") on the listener
        // object; a Kotlin lambda converted to the fun interface must expose exactly that method.
        val listener = WhisperSegmentListener { first, count -> first + count }
        val method = listener.javaClass.getMethod("onNewSegments", Int::class.javaPrimitiveType, Int::class.javaPrimitiveType)
        assertEquals(Int::class.javaPrimitiveType, method.returnType)
        assertEquals(5, method.invoke(listener, 2, 3))
        // The JNI symbols of the streaming entry points are bound by name (Java_..._KggmlWhisperJni_<name>).
        val natives = KggmlWhisperJni::class.java.declaredMethods.filter { Modifier.isNative(it.modifiers) }.map { it.name }.toSet()
        for (name in listOf("nativeStateCreate", "nativeStateFree", "nativeTranscribeWithState", "nativeStateSegmentTokens", "nativeResamplerProcessS16")) {
            assertTrue(name in natives, "missing native method $name")
        }
    }

    @Test
    fun streamingOnAndroidHostFailsWithTheSameTypedError() = runTest {
        // Without the native library the stream cannot even start; the failure is typed, not a crash.
        assertFailsWith<NativeFailureException> { WhisperEngine("/data/local/tmp/ggml-tiny.bin", BackendConfig.CPU).transcribeStream(emptyFlow<FloatArray>()) }
    }
}
