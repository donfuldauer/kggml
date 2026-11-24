package com.donfuldauer.kggml.runtime

import com.donfuldauer.kggml.core.ComputeBackend
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Exercises the cinterop bindings against the real GgmlRuntime.framework (iosSimulatorArm64Test). The test
 * binary only links when GgmlRuntime.xcframework has been built; otherwise Gradle skips the iOS test tasks.
 */
class IosNativeRuntimeTest {
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
    fun reportsCpuAndAppleBackends() {
        val backends = KggmlRuntime.backends()
        assertEquals(
            listOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_METAL, ComputeBackend.GPU_VULKAN, ComputeBackend.NPU_COREML),
            backends.map { it.backend },
        )
        assertTrue(backends.first().available)
        // The iOS framework is built with Metal and without Vulkan.
        assertTrue(backends[1].compiled, "Metal must be compiled into GgmlRuntime.framework")
        assertFalse(backends[2].compiled, "Vulkan is never built for iOS")
        val available = KggmlRuntime.availableBackends()
        assertTrue(ComputeBackend.CPU_ONLY in available)
        assertFalse(ComputeBackend.GPU_VULKAN in available)
        assertEquals(KggmlRuntime.info().hasCoreMl, backends[3].compiled)
    }

    @Test
    fun threadCountsAndVersionsAreReported() {
        val hardware = KggmlRuntime.hardwareThreadCount()
        assertTrue(KggmlRuntime.recommendedThreadCount() in 1..hardware)
        val info = KggmlRuntime.info()
        assertEquals("1.0.0", info.runtimeVersion)
        assertEquals(40, info.ggmlCommit.length)
        assertTrue(info.ggmlVersion.isNotEmpty())
    }
}
