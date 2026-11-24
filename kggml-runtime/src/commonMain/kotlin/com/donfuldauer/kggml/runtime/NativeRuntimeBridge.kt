package com.donfuldauer.kggml.runtime

import com.donfuldauer.kggml.core.ComputeBackend
import kotlin.concurrent.atomics.AtomicBoolean

/**
 * Raw, 1:1 view of the `kggml_runtime.h` C ABI.
 *
 * Implemented by the JNI bridge (Android, JVM Desktop) and the Kotlin/Native cinterop bridge (iOS), and by
 * fakes in tests. Every method returns raw native values (status codes, 0/1 flags); translation into Kotlin
 * types and typed exceptions happens once, in [RuntimeClient].
 *
 * Contract: [lastError] must be called on the same thread, immediately after the failing call, because the
 * native diagnostic message is thread-local.
 */
internal interface NativeRuntimeBridge {
    fun init(): Int

    fun shutdown(): Int

    fun isInitialized(): Boolean

    fun refCount(): Int

    /** `kggml_runtime_get_backends()` with room for [capacity] entries. */
    fun getBackends(capacity: Int): RawBackendsResult

    /** `kggml_runtime_performance_thread_count()`. */
    fun performanceThreadCount(): RawIntResult

    fun hardwareThreadCount(): Int

    fun hasCoreMl(): Boolean

    fun runtimeVersion(): String

    fun ggmlVersion(): String

    fun ggmlCommit(): String

    fun ggmlBuildCommit(): String

    fun statusName(status: Int): String

    fun lastError(): String
}

/** One raw `kggml_backend_info` entry. */
internal data class RawBackendInfo(
    val kind: Int,
    val compiled: Boolean,
    val available: Boolean,
    val deviceCount: Int,
    val name: String,
)

/**
 * Result of `kggml_runtime_get_backends()`: the [status], the [requiredCount] reported through `out_count`
 * (always set, even on [RuntimeStatus.BUFFER_TOO_SMALL]) and the [entries] written on success.
 */
internal data class RawBackendsResult(
    val status: Int,
    val requiredCount: Int,
    val entries: List<RawBackendInfo>,
)

/** A status code plus one `int32_t` out-parameter. */
internal data class RawIntResult(val status: Int, val value: Int)

/** `kggml_status` values of `kggml_runtime.h`. Codes 0..6 equal `NativeErrorCode`. */
internal object RuntimeStatus {
    const val OK: Int = 0
    const val INVALID_ARGUMENT: Int = 2
    const val OUT_OF_MEMORY: Int = 4
    const val NATIVE_FAILURE: Int = 6
    const val NOT_INITIALIZED: Int = 100
    const val BUFFER_TOO_SMALL: Int = 101
    const val UNKNOWN: Int = 102
}

/** `kggml_backend_kind` values of `kggml_runtime.h`. */
internal object BackendKind {
    const val CPU: Int = 0
    const val METAL: Int = 1
    const val VULKAN: Int = 2
    const val COREML: Int = 3

    /** `KGGML_BACKEND_KIND_COUNT`. */
    const val COUNT: Int = 4

    /** Maps a native `kggml_backend_kind` onto [ComputeBackend]; unknown (future) kinds map to `null`. */
    fun toComputeBackend(kind: Int): ComputeBackend? = when (kind) {
        CPU -> ComputeBackend.CPU_ONLY
        METAL -> ComputeBackend.GPU_METAL
        VULKAN -> ComputeBackend.GPU_VULKAN
        COREML -> ComputeBackend.NPU_COREML
        else -> null
    }
}

/**
 * Creates the platform bridge, loading the native libraries if needed.
 *
 * @throws Throwable (typically `UnsatisfiedLinkError`) if the native runtime cannot be loaded; [RuntimeClient]
 *   translates this into a `NativeFailureException`.
 */
internal expect fun createPlatformRuntimeBridge(): NativeRuntimeBridge

/** A thread-safe flag that can be set exactly once. */
internal class OnceFlag {
    private val flag = AtomicBoolean(false)

    val isSet: Boolean
        get() = flag.load()

    /** Sets the flag; returns `true` only for the call that actually set it. */
    fun trySet(): Boolean = flag.compareAndSet(expectedValue = false, newValue = true)
}
