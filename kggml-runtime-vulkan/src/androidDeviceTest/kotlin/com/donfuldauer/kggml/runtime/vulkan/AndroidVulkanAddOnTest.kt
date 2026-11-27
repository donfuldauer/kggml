package com.donfuldauer.kggml.runtime.vulkan

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.runtime.KggmlRuntime
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The add-on packaged in a test APK (-Pkggml.bundleNative=true): the runtime finds libggml-vulkan.so, registers it, and
 * reports Vulkan as compiled in. Whether a GPU device is usable depends on the device: emulators expose only software
 * Vulkan, which ggml-vulkan ignores, so there AUTO must still resolve to the CPU.
 */
@RunWith(AndroidJUnit4::class)
class AndroidVulkanAddOnTest {
    @Test
    fun theRuntimeRegistersTheVulkanBackendFromTheAddOn() {
        val vulkan = KggmlRuntime.backends().single { it.backend == ComputeBackend.GPU_VULKAN }
        assertTrue(vulkan.compiled, "libggml-vulkan.so is packaged, so the runtime must have registered it: $vulkan")
        assertTrue(KggmlRuntimeVulkan.isRegistered)
        assertEquals(vulkan.deviceCount > 0, vulkan.available)
        assertEquals(vulkan.available, KggmlRuntimeVulkan.isAvailable)
    }

    @Test
    fun autoResolvesToAUsableBackendAndTheCpuStaysAvailable() {
        val available = KggmlRuntime.availableBackends()
        assertTrue(ComputeBackend.CPU_ONLY in available)
        val resolved = KggmlRuntime.resolveBackend(ComputeBackend.AUTO)
        assertTrue(resolved in available, "AUTO resolved to $resolved, not one of $available")
        if (!KggmlRuntimeVulkan.isAvailable) {
            assertEquals(ComputeBackend.CPU_ONLY, resolved, "no usable Vulkan device: AUTO must fall back to the CPU")
        }
    }
}
