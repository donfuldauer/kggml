package com.donfuldauer.kggml.samples.showcase

import com.donfuldauer.kggml.core.ComputeBackend
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Runs the production [KggmlRuntimeInspector] against the real JNI runtime. With a host build available
 * (`kggml.native.path`, set by Gradle when `build/native/lib` exists) the inspector must report a CPU backend;
 * without one it must degrade to [BackendInspectorState.Failed] instead of crashing the app.
 */
class KggmlRuntimeInspectorDesktopTest {
    private val nativeAvailable = System.getProperty("kggml.native.path") != null
    private val requireNative = System.getProperty("kggml.test.requireNative").toBoolean()

    @Test
    fun inspectorReflectsTheHostRuntime() {
        if (requireNative && !nativeAvailable) fail("-Pkggml.requireNative=true but no host runtime was found")
        val state = loadInspectorState(KggmlRuntimeInspector)
        if (nativeAvailable) {
            assertIs<BackendInspectorState.Ready>(state, "unexpected state: $state")
            val cpu = state.rows.single { it.backend == ComputeBackend.CPU_ONLY }
            assertEquals(BackendStatus.AVAILABLE, cpu.status)
            assertEquals(listOf("CPU", "Metal", "Vulkan", "OpenCL", "Core ML"), state.rows.map { it.label })
            assertTrue(state.recommendedThreadCount in 1..state.hardwareThreadCount)
            assertFalse(state.info.runtimeVersion.isBlank())
            assertEquals(40, state.info.ggmlCommit.length)
        } else {
            assertIs<BackendInspectorState.Failed>(state, "unexpected state: $state")
            assertTrue(state.message.isNotBlank())
        }
    }
}
