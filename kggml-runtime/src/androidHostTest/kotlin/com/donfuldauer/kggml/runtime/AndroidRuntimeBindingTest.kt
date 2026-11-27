package com.donfuldauer.kggml.runtime

import com.donfuldauer.kggml.core.NativeFailureException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * Android local unit tests run on the host JVM, where the APK's arm64-v8a / x86_64 jniLibs cannot be loaded.
 * They verify that the Android binding wiring degrades into a typed, catchable error rather than a crash, and
 * that the shared JNI bridge decodes native data correctly (exercised on-device by instrumented tests).
 */
class AndroidRuntimeBindingTest {
    @Test
    fun androidLoaderTargetsTheJniBridgeLibrary() {
        assertEquals("kggml_runtime_jni", AndroidNativeLibraryLoader.JNI_LIBRARY)
    }

    @Test
    fun missingNativeLibraryOnHostSurfacesAsNativeFailureException() {
        val client = RuntimeClient(::createPlatformRuntimeBridge)
        val e = assertFailsWith<NativeFailureException> { client.acquire() }
        assertIs<UnsatisfiedLinkError>(e.cause)
        assertTrue(e.message!!.startsWith("Failed to load the Kggml native runtime"), e.message)
    }

    @Test
    fun publicFacadeReportsTheSameTypedError() {
        assertFailsWith<NativeFailureException> { KggmlRuntime.availableBackends() }
        assertFailsWith<NativeFailureException> { KggmlRuntime.recommendedThreadCount() }
    }
}
