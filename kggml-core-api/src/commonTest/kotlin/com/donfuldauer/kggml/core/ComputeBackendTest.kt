package com.donfuldauer.kggml.core

import com.donfuldauer.kggml.core.ComputeBackend.AUTO
import com.donfuldauer.kggml.core.ComputeBackend.CPU_ONLY
import com.donfuldauer.kggml.core.ComputeBackend.GPU_METAL
import com.donfuldauer.kggml.core.ComputeBackend.GPU_OPENCL
import com.donfuldauer.kggml.core.ComputeBackend.GPU_VULKAN
import com.donfuldauer.kggml.core.ComputeBackend.NPU_COREML
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ComputeBackendTest {

    @Test
    fun declaresExactlyTheSpecifiedBackends() {
        assertEquals(listOf(AUTO, CPU_ONLY, GPU_METAL, GPU_VULKAN, NPU_COREML, GPU_OPENCL), ComputeBackend.entries)
    }

    @Test
    fun isAcceleratedOnlyForGpuAndNpu() {
        assertFalse(AUTO.isAccelerated)
        assertFalse(CPU_ONLY.isAccelerated)
        assertTrue(GPU_METAL.isAccelerated)
        assertTrue(GPU_VULKAN.isAccelerated)
        assertTrue(GPU_OPENCL.isAccelerated)
        assertTrue(NPU_COREML.isAccelerated)
    }

    @Test
    fun fallbackChain() {
        assertEquals(GPU_METAL, NPU_COREML.fallback)
        assertEquals(CPU_ONLY, GPU_METAL.fallback)
        assertEquals(CPU_ONLY, GPU_VULKAN.fallback)
        assertEquals(CPU_ONLY, GPU_OPENCL.fallback)
        assertNull(CPU_ONLY.fallback)
        assertNull(AUTO.fallback)
    }

    @Test
    fun autoPicksFirstAvailableInPreferenceOrder() {
        assertEquals(NPU_COREML, ComputeBackend.resolve(AUTO, setOf(GPU_METAL, NPU_COREML)))
        assertEquals(GPU_METAL, ComputeBackend.resolve(AUTO, setOf(GPU_METAL, CPU_ONLY)))
        assertEquals(GPU_VULKAN, ComputeBackend.resolve(AUTO, setOf(GPU_VULKAN)))
        // OpenCL (only available with the add-on on an OpenCL device, i.e. Adreno) wins over Vulkan.
        assertEquals(GPU_OPENCL, ComputeBackend.resolve(AUTO, setOf(GPU_VULKAN, GPU_OPENCL)))
        assertEquals(GPU_OPENCL, ComputeBackend.resolve(AUTO, setOf(GPU_OPENCL)))
        assertEquals(listOf(NPU_COREML, GPU_METAL, GPU_OPENCL, GPU_VULKAN, CPU_ONLY), ComputeBackend.AUTO_PREFERENCE_ORDER)
        assertEquals(CPU_ONLY, ComputeBackend.resolve(AUTO, emptySet()))
    }

    @Test
    fun autoInAvailableSetIsIgnored() {
        assertEquals(CPU_ONLY, ComputeBackend.resolve(AUTO, setOf(AUTO)))
        assertEquals(CPU_ONLY, ComputeBackend.resolve(GPU_METAL, setOf(AUTO)))
    }

    @Test
    fun availableConcreteBackendResolvesToItself() {
        val all = setOf(CPU_ONLY, GPU_METAL, GPU_VULKAN, NPU_COREML, GPU_OPENCL)
        for (backend in all) assertEquals(backend, ComputeBackend.resolve(backend, all))
    }

    @Test
    fun cpuOnlyIsAlwaysAvailable() {
        assertEquals(CPU_ONLY, ComputeBackend.resolve(CPU_ONLY, emptySet()))
        assertEquals(CPU_ONLY, ComputeBackend.resolve(CPU_ONLY, setOf(GPU_METAL), allowFallback = false))
    }

    @Test
    fun unavailableBackendWalksFallbackChain() {
        assertEquals(GPU_METAL, ComputeBackend.resolve(NPU_COREML, setOf(GPU_METAL)))
        assertEquals(CPU_ONLY, ComputeBackend.resolve(NPU_COREML, emptySet()))
        assertEquals(CPU_ONLY, ComputeBackend.resolve(GPU_METAL, setOf(GPU_VULKAN)))
        assertEquals(CPU_ONLY, ComputeBackend.resolve(GPU_VULKAN, setOf(GPU_METAL, NPU_COREML)))
        // OpenCL never falls back to Vulkan: an explicit request means that backend or the CPU.
        assertEquals(CPU_ONLY, ComputeBackend.resolve(GPU_OPENCL, setOf(GPU_VULKAN)))
    }

    @Test
    fun unavailableBackendWithoutFallbackThrows() {
        val e = assertFailsWith<BackendUnavailableException> {
            ComputeBackend.resolve(GPU_VULKAN, setOf(GPU_METAL), allowFallback = false)
        }
        assertEquals(GPU_VULKAN, e.backend)
        assertEquals(NativeErrorCode.BACKEND_UNAVAILABLE, e.errorCode)
        assertTrue(e.message!!.contains("GPU_VULKAN"))
    }

    @Test
    fun autoWithoutFallbackNeverThrows() {
        assertEquals(CPU_ONLY, ComputeBackend.resolve(AUTO, emptySet(), allowFallback = false))
    }

    @Test
    fun resolveNeverReturnsAuto() {
        val subsets = listOf(emptySet(), setOf(GPU_METAL), setOf(GPU_VULKAN, NPU_COREML), ComputeBackend.entries.toSet())
        for (requested in ComputeBackend.entries) {
            for (available in subsets) {
                val resolved = ComputeBackend.resolve(requested, available)
                assertTrue(resolved != AUTO, "resolve($requested, $available) returned AUTO")
                assertTrue(resolved == CPU_ONLY || resolved in available)
            }
        }
    }
}
