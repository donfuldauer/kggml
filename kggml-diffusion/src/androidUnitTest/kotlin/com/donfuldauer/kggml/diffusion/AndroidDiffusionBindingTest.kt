package com.donfuldauer.kggml.diffusion

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
 * verify that the Android binding wiring degrades into a typed, catchable error rather than a crash, and that the
 * Kotlin JNI declarations and the C++ entry points / callback signature cannot drift apart (the native path itself is
 * exercised by jvmTest through the same JNI bridge).
 */
class AndroidDiffusionBindingTest {
    @Test
    fun androidLoaderTargetsTheDiffusionJniBridge() {
        assertEquals("kggml_diffusion_jni", AndroidDiffusionLibraryLoader.JNI_LIBRARY)
    }

    @Test
    fun missingNativeLibrarySurfacesAsNativeFailureException() = runTest {
        val e = assertFailsWith<NativeFailureException> { DiffusionEngine("/data/local/tmp/sd.gguf", BackendConfig.CPU) }
        assertIs<UnsatisfiedLinkError>(e.cause)
        assertTrue(e.message!!.startsWith("Failed to load the Kggml diffusion native library"), e.message)
        assertFailsWith<NativeFailureException> { DiffusionEngine.nativeInfo() }
    }

    @Test
    fun jniDeclarationsMatchTheNativeSymbols() {
        // kggml_diffusion_jni.cpp defines Java_com_donfuldauer_kggml_diffusion_KggmlDiffusionJni_<name>.
        val methods = KggmlDiffusionJni::class.java.declaredMethods.filter { Modifier.isNative(it.modifiers) }
        val natives = methods.map { it.name }.toSet()
        val cpp = java.io.File("native/jni/kggml_diffusion_jni.cpp").readText()
        val defined = Regex("""KGGML_JNI\(\w+, (native\w+)\)""").findAll(cpp).map { it.groupValues[1] }.toSet()
        assertTrue("nativeGenerate" in defined && "nativeContextCreate" in defined, "parsed ${defined.size} JNI entry points")
        assertEquals(defined, natives)
        assertTrue(methods.all { Modifier.isStatic(it.modifiers) })
    }

    @Test
    fun progressCallbackSignatureMatchesTheNativeLookup() {
        // GetMethodID(listener class, "onProgress", "(IIF)V") in kggml_diffusion_jni.cpp; consumer-rules.pro keeps it.
        val cpp = java.io.File("native/jni/kggml_diffusion_jni.cpp").readText()
        assertTrue(cpp.contains("\"onProgress\", \"(IIF)V\""))
        val method = DiffusionProgressListener::class.java.getMethod("onProgress", Int::class.java, Int::class.java, Float::class.java)
        assertEquals(Void.TYPE, method.returnType)
        val rules = java.io.File("consumer-rules.pro").readText()
        assertTrue(rules.contains("void onProgress(int, int, float);") && rules.contains("KggmlDiffusionJni"))
    }
}
