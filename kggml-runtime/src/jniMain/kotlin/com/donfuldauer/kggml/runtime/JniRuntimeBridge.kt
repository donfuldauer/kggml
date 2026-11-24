package com.donfuldauer.kggml.runtime

/**
 * [NativeRuntimeBridge] over the `kggml_runtime_jni` JNI library (Android + JVM Desktop).
 *
 * @param loadLibraries loads `kggml_runtime_jni` and its dependencies; invoked once, before any native call.
 */
internal class JniRuntimeBridge(loadLibraries: () -> Unit) : NativeRuntimeBridge {
    init {
        loadLibraries()
    }

    override fun init(): Int = KggmlRuntimeJni.nativeInit()

    override fun shutdown(): Int = KggmlRuntimeJni.nativeShutdown()

    override fun isInitialized(): Boolean = KggmlRuntimeJni.nativeIsInitialized() != 0

    override fun refCount(): Int = KggmlRuntimeJni.nativeRefCount()

    override fun getBackends(capacity: Int): RawBackendsResult {
        val safeCapacity = capacity.coerceAtLeast(0)
        val ints = IntArray(safeCapacity * KggmlRuntimeJni.BACKEND_INT_STRIDE)
        val names = arrayOfNulls<String>(safeCapacity)
        val count = IntArray(1)
        val status = KggmlRuntimeJni.nativeGetBackends(ints, names, count)
        if (status != RuntimeStatus.OK) return RawBackendsResult(status, count[0], emptyList())
        val entries = (0 until count[0].coerceIn(0, safeCapacity)).map { i ->
            val base = i * KggmlRuntimeJni.BACKEND_INT_STRIDE
            RawBackendInfo(
                kind = ints[base],
                compiled = ints[base + 1] != 0,
                available = ints[base + 2] != 0,
                deviceCount = ints[base + 3],
                name = names[i].orEmpty(),
            )
        }
        return RawBackendsResult(status, count[0], entries)
    }

    override fun performanceThreadCount(): RawIntResult {
        val out = IntArray(1)
        val status = KggmlRuntimeJni.nativePerformanceThreadCount(out)
        return RawIntResult(status, out[0])
    }

    override fun hardwareThreadCount(): Int = KggmlRuntimeJni.nativeHardwareThreadCount()

    override fun hasCoreMl(): Boolean = KggmlRuntimeJni.nativeHasCoreml() != 0

    override fun runtimeVersion(): String = version(KggmlRuntimeJni.VERSION_RUNTIME)

    override fun ggmlVersion(): String = version(KggmlRuntimeJni.VERSION_GGML)

    override fun ggmlCommit(): String = version(KggmlRuntimeJni.VERSION_GGML_COMMIT)

    override fun ggmlBuildCommit(): String = version(KggmlRuntimeJni.VERSION_GGML_BUILD_COMMIT)

    override fun statusName(status: Int): String =
        KggmlRuntimeJni.nativeStatusName(status) ?: "KGGML_STATUS_$status"

    override fun lastError(): String = KggmlRuntimeJni.nativeLastError().orEmpty()

    private fun version(which: Int): String = KggmlRuntimeJni.nativeVersionString(which).orEmpty()
}
