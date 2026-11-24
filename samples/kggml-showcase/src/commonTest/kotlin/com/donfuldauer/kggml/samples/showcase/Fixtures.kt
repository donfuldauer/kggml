package com.donfuldauer.kggml.samples.showcase

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.runtime.BackendInfo
import com.donfuldauer.kggml.runtime.RuntimeInfo

/** Deterministic snapshot resembling an Apple-silicon desktop: CPU + Metal available, Vulkan not compiled. */
fun fakeSnapshot(
    backends: List<BackendInfo> = listOf(
        BackendInfo(ComputeBackend.CPU_ONLY, "CPU", compiled = true, available = true, deviceCount = 1),
        BackendInfo(ComputeBackend.GPU_METAL, "Metal", compiled = true, available = true, deviceCount = 1),
        BackendInfo(ComputeBackend.GPU_VULKAN, "Vulkan", compiled = false, available = false, deviceCount = 0),
        BackendInfo(ComputeBackend.NPU_COREML, "CoreML", compiled = false, available = false, deviceCount = 0),
    ),
    availableBackends: Set<ComputeBackend> = backends.filter { it.available }.mapTo(linkedSetOf()) { it.backend },
): RuntimeSnapshot = RuntimeSnapshot(
    backends = backends,
    availableBackends = availableBackends,
    recommendedThreadCount = 4,
    hardwareThreadCount = 10,
    info = RuntimeInfo(
        runtimeVersion = "1.0.0",
        ggmlVersion = "0.9.9",
        ggmlCommit = "0123456789abcdef0123456789abcdef01234567",
        ggmlBuildCommit = "0123456",
        hasCoreMl = false,
    ),
)
