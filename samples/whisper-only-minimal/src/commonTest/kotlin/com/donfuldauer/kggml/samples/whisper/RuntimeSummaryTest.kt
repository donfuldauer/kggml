package com.donfuldauer.kggml.samples.whisper

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.NativeFailureException
import kotlin.test.Test
import kotlin.test.assertEquals

class RuntimeSummaryTest {
    @Test
    fun describesAvailableRuntime() {
        val text = describeRuntime {
            RuntimeSummary("1.0.0", "0.9.9", linkedSetOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_VULKAN), 4)
        }
        assertEquals(
            "Kggml whisper-only minimal sample\nRuntime 1.0.0 (ggml 0.9.9)\nBackends: CPU_ONLY, GPU_VULKAN\n" +
                "Recommended threads: 4",
            text,
        )
    }

    @Test
    fun describesRuntimeLoadFailure() {
        val text = describeRuntime { throw NativeFailureException(message = "no libkggml_runtime_jni.so") }
        assertEquals("Kggml whisper-only minimal sample\nNative runtime unavailable: no libkggml_runtime_jni.so", text)
    }
}
