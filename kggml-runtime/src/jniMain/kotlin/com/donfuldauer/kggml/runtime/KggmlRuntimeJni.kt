package com.donfuldauer.kggml.runtime

/**
 * `external` declarations implemented by `kggml-runtime/native/jni/kggml_runtime_jni.cpp`
 * (`Java_com_donfuldauer_kggml_runtime_KggmlRuntimeJni_*`).
 *
 * Members are deliberately public inside this internal object: Kotlin mangles the JVM names of `internal`
 * members, which would break the JNI symbol names.
 */
internal object KggmlRuntimeJni {
    /** Ints per backend entry in [nativeGetBackends]: kind, compiled, available, device_count. */
    const val BACKEND_INT_STRIDE: Int = 4

    const val VERSION_RUNTIME: Int = 0
    const val VERSION_GGML: Int = 1
    const val VERSION_GGML_COMMIT: Int = 2
    const val VERSION_GGML_BUILD_COMMIT: Int = 3

    @JvmStatic external fun nativeInit(): Int

    @JvmStatic external fun nativeShutdown(): Int

    @JvmStatic external fun nativeIsInitialized(): Int

    @JvmStatic external fun nativeRefCount(): Int

    /**
     * Fills up to `names.size` entries ([ints] stride [BACKEND_INT_STRIDE]); `outCount[0]` receives the
     * required count. Returns a `kggml_status`.
     */
    @JvmStatic external fun nativeGetBackends(ints: IntArray, names: Array<String?>, outCount: IntArray): Int

    @JvmStatic external fun nativePerformanceThreadCount(out: IntArray): Int

    @JvmStatic external fun nativeHardwareThreadCount(): Int

    @JvmStatic external fun nativeHasCoreml(): Int

    @JvmStatic external fun nativeVersionString(which: Int): String?

    @JvmStatic external fun nativeStatusName(status: Int): String?

    @JvmStatic external fun nativeLastError(): String?

    @JvmStatic external fun nativeClearLastError()

    /** Test-only fault injection; returns `false` in builds without `KGGML_RUNTIME_TEST_HOOKS`. */
    @JvmStatic external fun nativeTestInjectFault(fault: Int): Boolean
}
