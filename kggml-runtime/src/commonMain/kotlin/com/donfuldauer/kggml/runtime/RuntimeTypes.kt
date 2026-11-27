package com.donfuldauer.kggml.runtime

import com.donfuldauer.kggml.core.ComputeBackend

/**
 * One hardware backend known to the native runtime, as reported by `kggml_runtime_get_backends()`.
 *
 * @property backend the [ComputeBackend] this entry describes (never [ComputeBackend.AUTO]).
 * @property name native display name, e.g. `"CPU"`, `"Metal"`, `"Vulkan"`, `"CoreML"`.
 * @property compiled `true` if support for this backend was compiled into the runtime build.
 * @property available `true` if the backend is usable on this device right now (compiled and a device exists).
 * @property deviceCount number of ggml devices backing this backend (always `0` for [ComputeBackend.NPU_COREML],
 *   which is a compile-time capability flag rather than a ggml device).
 */
public data class BackendInfo(
    public val backend: ComputeBackend,
    public val name: String,
    public val compiled: Boolean,
    public val available: Boolean,
    public val deviceCount: Int,
)

/**
 * Version metadata of the loaded native runtime.
 *
 * @property runtimeVersion Kggml runtime version, e.g. `"0.1.0"`.
 * @property ggmlVersion version reported by the linked ggml, e.g. `"0.9.9"`.
 * @property ggmlCommit full 40-character upstream ggml commit SHA the runtime is pinned to (see `NOTICE`).
 * @property ggmlBuildCommit commit string compiled into ggml itself (short SHA or `"unknown"`).
 * @property hasCoreMl `true` if this build advertises the Core ML capability (iOS builds with CoreML enabled).
 */
public data class RuntimeInfo(
    public val runtimeVersion: String,
    public val ggmlVersion: String,
    public val ggmlCommit: String,
    public val ggmlBuildCommit: String,
    public val hasCoreMl: Boolean,
)

/**
 * A reference on the process-wide native runtime obtained from [KggmlRuntime.acquire].
 *
 * The runtime stays initialized while at least one handle is open. [close] releases the reference and is
 * idempotent: only the first call has an effect, so a handle can never release more than it acquired.
 */
public class RuntimeHandle internal constructor(private val client: RuntimeClient) : AutoCloseable {
    private val closed = OnceFlag()

    /** `true` once [close] has been called. */
    public val isClosed: Boolean
        get() = closed.isSet

    /**
     * Releases this handle's runtime reference. Subsequent calls are no-ops.
     *
     * @throws com.donfuldauer.kggml.core.InferenceException if the native shutdown fails.
     */
    override fun close() {
        if (closed.trySet()) client.release()
    }
}
