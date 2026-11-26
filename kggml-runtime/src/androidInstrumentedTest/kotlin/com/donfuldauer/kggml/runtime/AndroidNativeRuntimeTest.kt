package com.donfuldauer.kggml.runtime

import com.donfuldauer.kggml.core.ComputeBackend
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The JNI runtime bridge against the real `libkggml_runtime_jni.so` / `libggml*.so` packaged into the test APK
 * (`connectedDebugAndroidTest -Pkggml.bundleNative=true`) on an Android device or emulator: the on-device
 * counterpart of `AndroidRuntimeBindingTest` (host JVM, library absent) and `IosNativeRuntimeTest`.
 */
class AndroidNativeRuntimeTest {
    @AfterTest
    fun noLeakedReferences() {
        assertEquals(0, KggmlRuntime.client.refCount())
    }

    @Test
    fun repeatedInitTeardownIsSafe() {
        repeat(100) {
            KggmlRuntime.acquire().use { assertTrue(KggmlRuntime.isInitialized()) }
        }
        assertFalse(KggmlRuntime.isInitialized())
    }

    @Test
    fun handleCloseIsIdempotent() {
        val a = KggmlRuntime.acquire()
        val b = KggmlRuntime.acquire()
        a.close()
        a.close()
        assertEquals(1, KggmlRuntime.client.refCount())
        b.close()
    }

    @Test
    fun reportsTheAndroidBuildsBackends() {
        val backends = KggmlRuntime.backends()
        assertEquals(
            listOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_METAL, ComputeBackend.GPU_VULKAN, ComputeBackend.NPU_COREML),
            backends.map { it.backend },
        )
        val cpu = backends.first()
        assertTrue(cpu.compiled && cpu.available, "the CPU backend is always compiled and available: $cpu")
        // The Android presets (CMakePresets.json android-base) build the CPU backend only.
        assertFalse(backends[1].compiled, "Metal is Apple-only")
        assertFalse(backends[2].compiled, "Vulkan is off in the Android build (KGGML_ENABLE_VULKAN=OFF)")
        assertFalse(backends[3].compiled, "Core ML is Apple-only")
        assertEquals(setOf(ComputeBackend.CPU_ONLY), KggmlRuntime.availableBackends())
        assertFalse(KggmlRuntime.info().hasCoreMl)
    }

    @Test
    fun threadCountsAndVersionsAreReported() {
        val hardware = KggmlRuntime.hardwareThreadCount()
        assertEquals(Runtime.getRuntime().availableProcessors(), hardware)
        assertTrue(KggmlRuntime.recommendedThreadCount() in 1..hardware)
        val info = KggmlRuntime.info()
        assertEquals("1.0.0", info.runtimeVersion)
        assertEquals(40, info.ggmlCommit.length)
        assertTrue(info.ggmlVersion.isNotEmpty())
    }
}
