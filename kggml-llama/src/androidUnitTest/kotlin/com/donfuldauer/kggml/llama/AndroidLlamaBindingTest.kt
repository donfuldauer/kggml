package com.donfuldauer.kggml.llama

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.NativeFailureException
import java.lang.reflect.Modifier
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
class AndroidLlamaBindingTest {
    @Test
    fun androidLoaderTargetsTheLlamaJniBridge() {
        assertEquals("kggml_llama_jni", AndroidLlamaLibraryLoader.JNI_LIBRARY)
    }

    @Test
    fun missingNativeLibrarySurfacesAsNativeFailureException() = runTest {
        val e = assertFailsWith<NativeFailureException> { LlamaEngine("/data/local/tmp/tiny.gguf", BackendConfig.CPU) }
        assertIs<UnsatisfiedLinkError>(e.cause)
        assertTrue(e.message!!.startsWith("Failed to load the Kggml llama native library"), e.message)
        assertFailsWith<NativeFailureException> { LlamaEngine.nativeInfo() }
    }

    @Test
    fun jniDeclarationsMatchTheNativeSymbols() {
        // kggml_llama_jni.cpp defines Java_com_donfuldauer_kggml_llama_KggmlLlamaJni_<name>; the names must not drift.
        val natives = KggmlLlamaJni::class.java.declaredMethods.filter { Modifier.isNative(it.modifiers) }.map { it.name }.toSet()
        for (name in listOf("nativeModelLoad", "nativeDecode", "nativeSamplerCreate", "nativeStateSave", "nativeStateLoad", "nativeGetEmbeddings", "nativeRequestAbort")) {
            assertTrue(name in natives, "missing native method $name")
        }
        assertTrue(natives.all { Modifier.isStatic(KggmlLlamaJni::class.java.getDeclaredMethod(it, *KggmlLlamaJni::class.java.declaredMethods.first { m -> m.name == it }.parameterTypes).modifiers) })
    }
}
