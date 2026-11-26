package com.donfuldauer.kggml.samples.showcase

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.NativeFailureException
import com.donfuldauer.kggml.runtime.BackendInfo
import kotlin.coroutines.cancellation.CancellationException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs

class BackendInspectorStateTest {
    @Test
    fun readyStateListsEveryBackendInDisplayOrder() {
        val state = loadInspectorState { fakeSnapshot() }
        assertIs<BackendInspectorState.Ready>(state)
        assertEquals(listOf("CPU", "Metal", "Vulkan", "OpenCL", "Core ML"), state.rows.map { it.label })
        assertEquals(INSPECTED_BACKENDS, state.rows.map { it.backend })
    }

    @Test
    fun rowStatusReflectsCompiledAndAvailableFlags() {
        val snapshot = fakeSnapshot(
            backends = listOf(
                BackendInfo(ComputeBackend.CPU_ONLY, "CPU", compiled = true, available = true, deviceCount = 1),
                BackendInfo(ComputeBackend.GPU_VULKAN, "Vulkan", compiled = true, available = false, deviceCount = 0),
            ),
        )
        val rows = snapshot.toInspectorState().rows.associateBy { it.backend }
        assertEquals(BackendStatus.AVAILABLE, rows.getValue(ComputeBackend.CPU_ONLY).status)
        assertEquals(1, rows.getValue(ComputeBackend.CPU_ONLY).deviceCount)
        assertEquals(BackendStatus.NO_DEVICE, rows.getValue(ComputeBackend.GPU_VULKAN).status)
        // Not reported by the runtime at all → still listed, as not compiled.
        assertEquals(BackendStatus.NOT_COMPILED, rows.getValue(ComputeBackend.GPU_METAL).status)
        assertEquals(0, rows.getValue(ComputeBackend.GPU_METAL).deviceCount)
    }

    @Test
    fun availableBackendSetMarksRowAvailableEvenIfBackendListOmitsIt() {
        // KggmlRuntime.availableBackends() always contains CPU_ONLY.
        val snapshot = fakeSnapshot(backends = emptyList(), availableBackends = setOf(ComputeBackend.CPU_ONLY))
        val rows = snapshot.toInspectorState().rows.associateBy { it.backend }
        assertEquals(BackendStatus.AVAILABLE, rows.getValue(ComputeBackend.CPU_ONLY).status)
    }

    @Test
    fun readyStateCarriesThreadCountsAndMetadata() {
        val state = fakeSnapshot().toInspectorState()
        assertEquals(4, state.recommendedThreadCount)
        assertEquals(10, state.hardwareThreadCount)
        assertEquals("1.0.0", state.info.runtimeVersion)
        assertEquals("0.9.9", state.info.ggmlVersion)
    }

    @Test
    fun runtimeLoadFailureBecomesFailedState() {
        val state = loadInspectorState {
            throw NativeFailureException(message = "Failed to load the Kggml native runtime: no kggml_runtime_jni")
        }
        assertIs<BackendInspectorState.Failed>(state)
        assertEquals("Failed to load the Kggml native runtime: no kggml_runtime_jni", state.message)
    }

    @Test
    fun errorWithoutMessageFallsBackToTypeName() {
        val state = loadInspectorState { throw IllegalStateException() }
        assertIs<BackendInspectorState.Failed>(state)
        assertEquals("IllegalStateException", state.message)
    }

    @Test
    fun cancellationIsRethrown() {
        assertFailsWith<CancellationException> { loadInspectorState { throw CancellationException("cancelled") } }
    }

    @Test
    fun rowTestTagsAreStable() {
        val tags = fakeSnapshot().toInspectorState().rows.map { it.testTag }
        assertEquals(
            listOf(
                "backend-row-cpu_only",
                "backend-row-gpu_metal",
                "backend-row-gpu_vulkan",
                "backend-row-gpu_opencl",
                "backend-row-npu_coreml",
            ),
            tags,
        )
    }
}
