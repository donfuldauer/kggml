package com.donfuldauer.kggml.runtime

/**
 * Deterministic in-memory [NativeRuntimeBridge] mirroring the semantics of `kggml_runtime.cpp`:
 * reference-counted init/shutdown (shutdown at zero is an OK no-op), backends requiring init, and a
 * "last error" message that is set by failing calls and cleared by successful ones.
 */
internal class FakeRuntimeBridge(
    var backends: List<RawBackendInfo> = listOf(
        RawBackendInfo(BackendKind.CPU, compiled = true, available = true, deviceCount = 1, name = "CPU"),
        RawBackendInfo(BackendKind.METAL, compiled = false, available = false, deviceCount = 0, name = "Metal"),
        RawBackendInfo(BackendKind.VULKAN, compiled = true, available = true, deviceCount = 2, name = "Vulkan"),
        RawBackendInfo(BackendKind.COREML, compiled = false, available = false, deviceCount = 0, name = "CoreML"),
    ),
    var performanceThreads: Int = 4,
    var hardwareThreads: Int = 8,
) : NativeRuntimeBridge {
    var refs: Int = 0
        private set
    var initCalls: Int = 0
        private set
    var shutdownCalls: Int = 0
        private set
    var getBackendsCapacities: MutableList<Int> = mutableListOf()
        private set

    /** Status the next call to the named operation returns instead of succeeding (one-shot). */
    var failInitWith: Int? = null
    var failShutdownWith: Int? = null
    var failBackendsWith: Int? = null
    var failThreadsWith: Int? = null
    var failureMessage: String = "injected failure"

    private var lastError = ""

    private fun fail(status: Int): Int {
        lastError = failureMessage
        return status
    }

    private fun ok(): Int {
        lastError = ""
        return RuntimeStatus.OK
    }

    override fun init(): Int {
        initCalls++
        failInitWith?.let { failInitWith = null; return fail(it) }
        refs++
        return ok()
    }

    override fun shutdown(): Int {
        shutdownCalls++
        failShutdownWith?.let { failShutdownWith = null; return fail(it) }
        if (refs > 0) refs--
        return ok()
    }

    override fun isInitialized(): Boolean = refs > 0

    override fun refCount(): Int = refs

    override fun getBackends(capacity: Int): RawBackendsResult {
        getBackendsCapacities += capacity
        failBackendsWith?.let { failBackendsWith = null; return RawBackendsResult(fail(it), backends.size, emptyList()) }
        if (refs == 0) {
            lastError = "kggml_runtime_get_backends: runtime not initialized"
            return RawBackendsResult(RuntimeStatus.NOT_INITIALIZED, 0, emptyList())
        }
        if (capacity < backends.size) {
            lastError = "buffer too small"
            return RawBackendsResult(RuntimeStatus.BUFFER_TOO_SMALL, backends.size, emptyList())
        }
        return RawBackendsResult(ok(), backends.size, backends)
    }

    override fun performanceThreadCount(): RawIntResult {
        failThreadsWith?.let { failThreadsWith = null; return RawIntResult(fail(it), 0) }
        return RawIntResult(ok(), performanceThreads)
    }

    override fun hardwareThreadCount(): Int = hardwareThreads

    override fun hasCoreMl(): Boolean = backends.any { it.kind == BackendKind.COREML && it.compiled }

    override fun runtimeVersion(): String = "1.0.0"

    override fun ggmlVersion(): String = "0.9.9"

    override fun ggmlCommit(): String = "a04eea0761a85d18f3f504d6ab970c5c9dce705f"

    override fun ggmlBuildCommit(): String = "unknown"

    override fun statusName(status: Int): String = when (status) {
        RuntimeStatus.OK -> "KGGML_STATUS_OK"
        RuntimeStatus.OUT_OF_MEMORY -> "KGGML_STATUS_OUT_OF_MEMORY"
        RuntimeStatus.NOT_INITIALIZED -> "KGGML_STATUS_NOT_INITIALIZED"
        else -> "KGGML_STATUS_<$status>"
    }

    override fun lastError(): String = lastError
}
