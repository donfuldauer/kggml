package com.donfuldauer.kggml.runtime

import com.donfuldauer.kggml.core.BackendUnavailableException
import com.donfuldauer.kggml.core.ComputeBackend

/**
 * Kotlin entry point to the shared native ggml runtime (`kggml_runtime.h`).
 *
 * The native runtime is inherently process-wide (one shared ggml per process, see AGENTS.md §2.1), so this
 * is a stateless facade over a reference-counted native lifecycle rather than an engine singleton: engines
 * remain ordinary instantiable classes that each [acquire] their own [RuntimeHandle] and close it with
 * themselves.
 *
 * - **Android / JVM Desktop:** calls go through the `kggml_runtime_jni` JNI bridge. On the JVM the native
 *   libraries are loaded from the directory named by the `kggml.native.path` system property, else
 *   extracted from the jar (`/kggml/native/<os>-<arch>/`), else from `java.library.path`.
 * - **iOS:** calls go straight to the C ABI through Kotlin/Native cinterop, linked against the dynamic
 *   `GgmlRuntime.framework` (`GgmlRuntime.xcframework`).
 *
 * Every native failure surfaces as a typed [com.donfuldauer.kggml.core.InferenceException]; status codes
 * `0..6` map onto their `NativeErrorCode` counterparts and runtime-specific codes (`100+`) onto
 * [com.donfuldauer.kggml.core.NativeFailureException] with the raw code preserved. A runtime that cannot be
 * loaded at all is reported as a `NativeFailureException` whose cause is the platform loading error.
 */
public object KggmlRuntime {
    internal val client: RuntimeClient = RuntimeClient(::createPlatformRuntimeBridge)

    /**
     * Takes a reference on the native runtime (initializing it on the first reference). Close the returned
     * handle to release it; the runtime shuts down when the last handle is closed.
     */
    public fun acquire(): RuntimeHandle = client.acquire()

    /** `true` while at least one runtime reference is held (by any caller in this process). */
    public fun isInitialized(): Boolean = client.isInitialized()

    /** Every backend kind known to the runtime, CPU first, with compile-time and device availability. */
    public fun backends(): List<BackendInfo> = client.backends()

    /**
     * The backends usable on this device with this build. Always contains [ComputeBackend.CPU_ONLY] and never
     * [ComputeBackend.AUTO].
     */
    public fun availableBackends(): Set<ComputeBackend> = client.availableBackends()

    /**
     * Resolves [requested] against [availableBackends] using [ComputeBackend.resolve].
     *
     * @throws BackendUnavailableException if [requested] is unavailable and [allowFallback] is `false`.
     */
    public fun resolveBackend(requested: ComputeBackend, allowFallback: Boolean = true): ComputeBackend =
        client.resolveBackend(requested, allowFallback)

    /**
     * Recommended inference thread count: the number of performance (P-) cores, always at least `1` and at
     * most [hardwareThreadCount]. Use it as the default for `BackendConfig.cpuThreads`.
     */
    public fun recommendedThreadCount(): Int = client.recommendedThreadCount()

    /** Number of logical hardware threads (at least `1`). */
    public fun hardwareThreadCount(): Int = client.hardwareThreadCount()

    /** Version metadata of the loaded native runtime and its pinned ggml. */
    public fun info(): RuntimeInfo = client.info()
}
