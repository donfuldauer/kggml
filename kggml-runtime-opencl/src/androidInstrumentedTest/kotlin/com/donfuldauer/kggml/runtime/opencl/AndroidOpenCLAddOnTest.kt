package com.donfuldauer.kggml.runtime.opencl

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.runtime.KggmlRuntime
import dalvik.system.BaseDexClassLoader
import java.io.File
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * The add-on packaged in a test APK (-Pkggml.bundleNative=true). Whether it registers depends on the device: with the
 * vendor's libOpenCL.so (Adreno phones) the runtime registers OpenCL; without it (emulators, Pixel) the module cannot
 * load and the runtime must keep working on the CPU, never crash.
 */
@RunWith(AndroidJUnit4::class)
class AndroidOpenCLAddOnTest {
    private val deviceHasOpenCL: Boolean =
        listOf("/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so", "/vendor/lib/libOpenCL.so")
            .any { File(it).exists() }

    @Test
    fun theAddOnIsPackagedInTheApp() {
        val path = (javaClass.classLoader as BaseDexClassLoader).findLibrary("ggml-opencl")
        assertNotNull(path, "libggml-opencl.so must be packaged by the add-on")
    }

    @Test
    fun theRuntimeRegistersOpenCLExactlyWhenTheDeviceHasIt() {
        val opencl = KggmlRuntime.backends().single { it.backend == ComputeBackend.GPU_OPENCL }
        assertEquals(KggmlRuntimeOpenCL.isRegistered, opencl.compiled)
        assertEquals(opencl.deviceCount > 0, opencl.available)
        if (!deviceHasOpenCL) {
            assertEquals(false, opencl.compiled, "no vendor libOpenCL.so: the module cannot load: $opencl")
        }
    }

    @Test
    fun autoResolvesToAUsableBackendAndTheCpuStaysAvailable() {
        val available = KggmlRuntime.availableBackends()
        assertTrue(ComputeBackend.CPU_ONLY in available)
        val resolved = KggmlRuntime.resolveBackend(ComputeBackend.AUTO)
        assertTrue(resolved in available, "AUTO resolved to $resolved, not one of $available")
        if (KggmlRuntimeOpenCL.isAvailable) assertEquals(ComputeBackend.GPU_OPENCL, resolved, "AUTO prefers OpenCL")
    }
}
