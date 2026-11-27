package com.donfuldauer.kggml.runtime

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.NativeFailureException
import com.donfuldauer.kggml.core.OutOfMemoryException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.test.fail
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * End-to-end tests of [KggmlRuntime] through the real JNI bridge and native runtime from `build/native/lib`
 * (built by `./gradlew :kggml-runtime:buildNativeHost` or `kggml-runtime/native/scripts/build-host.sh`).
 *
 * Skipped when the native build is absent; `-Pkggml.requireNative=true` builds it first and turns a missing
 * library into a failure.
 */
class JvmNativeRuntimeTest {
    @BeforeTest
    fun requireNativeRuntime() {
        val required = System.getProperty("kggml.test.requireNative").toBoolean()
        val available = System.getProperty(JvmNativeLibraryLoader.PATH_PROPERTY) != null
        if (required && !available) fail("kggml.requireNative=true but build/native/lib has no kggml_runtime_jni")
        assumeTrue(available, "native runtime not built (run ./gradlew :kggml-runtime:buildNativeHost)")
        assertEquals(0, KggmlRuntime.client.refCount(), "a previous test leaked a runtime reference")
    }

    @AfterTest
    fun noLeakedReferences() {
        if (System.getProperty(JvmNativeLibraryLoader.PATH_PROPERTY) == null) return
        assertEquals(0, KggmlRuntime.client.refCount(), "test leaked a runtime reference")
    }

    @Test
    fun loadsFromTheConfiguredDirectory() {
        KggmlRuntime.hardwareThreadCount()
        assertEquals(System.getProperty(JvmNativeLibraryLoader.PATH_PROPERTY), JvmNativeLibraryLoader.origin)
    }

    @Test
    fun acquireAndReleaseDriveTheNativeReferenceCount() {
        assertFalse(KggmlRuntime.isInitialized())
        val a = KggmlRuntime.acquire()
        val b = KggmlRuntime.acquire()
        assertTrue(KggmlRuntime.isInitialized())
        assertEquals(2, KggmlRuntime.client.refCount())
        a.close()
        a.close() // idempotent
        assertEquals(1, KggmlRuntime.client.refCount())
        b.close()
        assertFalse(KggmlRuntime.isInitialized())
    }

    @Test
    fun repeatedInitTeardownIsSafe() {
        repeat(200) {
            KggmlRuntime.acquire().use { handle ->
                assertTrue(KggmlRuntime.isInitialized())
                assertFalse(handle.isClosed)
            }
        }
        assertFalse(KggmlRuntime.isInitialized())
    }

    @Test
    fun concurrentAcquireReleaseIsBalanced() {
        val threads = 8
        val pool = Executors.newFixedThreadPool(threads)
        val start = CountDownLatch(1)
        try {
            val futures = (0 until threads).map {
                pool.submit {
                    start.await()
                    repeat(200) {
                        KggmlRuntime.acquire().use { KggmlRuntime.availableBackends() }
                    }
                }
            }
            start.countDown()
            futures.forEach { it.get(60, TimeUnit.SECONDS) }
        } finally {
            pool.shutdownNow()
        }
        assertEquals(0, KggmlRuntime.client.refCount())
    }

    @Test
    fun reportsAllBackendKindsWithCpuAvailable() {
        val backends = KggmlRuntime.backends()
        assertEquals(
            listOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_METAL, ComputeBackend.GPU_VULKAN, ComputeBackend.NPU_COREML, ComputeBackend.GPU_OPENCL),
            backends.map { it.backend },
        )
        assertEquals(listOf("CPU", "Metal", "Vulkan", "CoreML", "OpenCL"), backends.map { it.name })
        val cpu = backends.first()
        assertTrue(cpu.compiled && cpu.available && cpu.deviceCount >= 1, "CPU must always be available: $cpu")
        // CoreML is never available on desktop builds, and nothing is available without being compiled in.
        assertFalse(backends.last().available)
        backends.forEach { assertTrue(it.compiled || !it.available, "available but not compiled: $it") }
    }

    @Test
    fun availableBackendsMatchTheHostPlatform() {
        val available = KggmlRuntime.availableBackends()
        assertTrue(ComputeBackend.CPU_ONLY in available)
        assertFalse(ComputeBackend.NPU_COREML in available)
        val os = System.getProperty("os.name").lowercase()
        if (!os.startsWith("mac")) assertFalse(ComputeBackend.GPU_METAL in available, "Metal outside Apple")
        assertEquals(
            KggmlRuntime.backends().filter { it.available }.map { it.backend }.toSet() + ComputeBackend.CPU_ONLY,
            available,
        )
        assertEquals(ComputeBackend.CPU_ONLY, KggmlRuntime.resolveBackend(ComputeBackend.CPU_ONLY))
    }

    /**
     * Issue #92: with the desktop Vulkan add-on configured (`-Pkggml.test.vulkanAddonDir=<dir>`, CI uses Mesa's
     * software Vulkan device lavapipe), the module registers, Vulkan is available and AUTO picks it over the CPU.
     */
    @Test
    fun theDesktopVulkanAddonRegistersWhenConfigured() {
        assumeTrue(System.getProperty("kggml.test.expectVulkanAddon").toBoolean(), "Vulkan add-on not configured")
        val vulkan = KggmlRuntime.backends().single { it.backend == ComputeBackend.GPU_VULKAN }
        assertTrue(
            vulkan.compiled,
            "the Vulkan backend module was not registered: $vulkan; add-ons: ${JvmNativeLibraryLoader.addonReport}",
        )
        assertTrue(vulkan.deviceCount >= 1, "no Vulkan device (driver missing?): $vulkan")
        assertTrue(ComputeBackend.GPU_VULKAN in KggmlRuntime.availableBackends())
        assertEquals(ComputeBackend.GPU_VULKAN, KggmlRuntime.resolveBackend(ComputeBackend.AUTO))
    }

    @Test
    fun threadCountsAreSane() {
        val hardware = KggmlRuntime.hardwareThreadCount()
        val recommended = KggmlRuntime.recommendedThreadCount()
        // std::thread::hardware_concurrency() ignores cgroup CPU quotas, so it is >= the JVM's view.
        assertTrue(hardware >= 1)
        assertTrue(recommended in 1..hardware, "recommended=$recommended hardware=$hardware")
    }

    @Test
    fun infoMatchesThePinnedVersions() {
        val info = KggmlRuntime.info()
        assertEquals(System.getProperty("kggml.test.expectedRuntimeVersion"), info.runtimeVersion)
        assertEquals(System.getProperty("kggml.test.expectedGgmlVersion"), info.ggmlVersion)
        assertEquals(System.getProperty("kggml.test.expectedGgmlCommit"), info.ggmlCommit)
        assertTrue(info.ggmlBuildCommit.isNotEmpty())
        assertFalse(info.hasCoreMl)
    }

    // ---- exception translation across JNI (requires a test-hooks build, i.e. build/native) ------------------

    private fun injectOrSkip(fault: Int) {
        KggmlRuntime.hardwareThreadCount() // ensure libraries are loaded before calling the raw JNI symbol
        assumeTrue(KggmlRuntimeJni.nativeTestInjectFault(fault), "runtime built without KGGML_RUNTIME_TEST_HOOKS")
    }

    @Test
    fun cppBadAllocBecomesOutOfMemoryException() {
        injectOrSkip(FAULT_BAD_ALLOC)
        val e = assertFailsWith<OutOfMemoryException> { KggmlRuntime.acquire() }
        assertTrue(e.message!!.contains("KGGML_STATUS_OUT_OF_MEMORY"), e.message)
        assertTrue(e.message!!.contains("kggml_runtime_init"), e.message)
        assertFalse(KggmlRuntime.isInitialized())
    }

    @Test
    fun cppStdExceptionBecomesNativeFailureException() {
        injectOrSkip(FAULT_RUNTIME_ERROR)
        val e = assertFailsWith<NativeFailureException> { KggmlRuntime.acquire() }
        assertEquals(RuntimeStatus.NATIVE_FAILURE, e.nativeCode)
        assertFalse(KggmlRuntime.isInitialized())
    }

    @Test
    fun nonStandardCppThrowBecomesNativeFailureWithUnknownCode() {
        injectOrSkip(FAULT_NON_STD)
        val e = assertFailsWith<NativeFailureException> { KggmlRuntime.acquire() }
        assertEquals(RuntimeStatus.UNKNOWN, e.nativeCode)
        assertTrue(e.message!!.contains("KGGML_STATUS_UNKNOWN"), e.message)
    }

    @Test
    fun faultDuringBackendQueryReleasesTheReference() {
        KggmlRuntime.acquire().use {
            injectOrSkip(FAULT_LOGIC_ERROR)
            // The fault fires inside kggml_runtime_init of the query's own reference.
            assertFailsWith<NativeFailureException> { KggmlRuntime.backends() }
            assertEquals(1, KggmlRuntime.client.refCount())
        }
    }

    @Test
    fun runtimeStaysUsableAfterTranslatedFaults() {
        injectOrSkip(FAULT_BAD_ARRAY_NEW)
        assertFailsWith<OutOfMemoryException> { KggmlRuntime.acquire() }
        KggmlRuntime.acquire().use { assertTrue(KggmlRuntime.isInitialized()) }
        assertTrue(ComputeBackend.CPU_ONLY in KggmlRuntime.availableBackends())
    }

    private companion object {
        // kggml_test_fault (native/src/kggml_runtime_testing.h)
        const val FAULT_BAD_ALLOC = 1
        const val FAULT_BAD_ARRAY_NEW = 2
        const val FAULT_RUNTIME_ERROR = 3
        const val FAULT_LOGIC_ERROR = 4
        const val FAULT_NON_STD = 5
    }
}
