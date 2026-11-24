package com.donfuldauer.kggml.core

/**
 * Common contract of every Kggml engine (`WhisperEngine`, `LlamaEngine`, `DiffusionEngine`, …).
 *
 * Engines are ordinary instantiable classes — never global singletons — and multiple instances may be
 * alive concurrently. Each instance serializes all of its native work (`init`, inference, `close`) on its
 * own [SerialExecutor].
 *
 * Closing contract:
 * - [close] releases all native resources held by the engine and is **idempotent**: calling it again (from
 *   any thread) is a no-op.
 * - After [close], [isClosed] returns `true` and any further inference call fails with
 *   [IllegalStateException].
 */
public interface InferenceEngine : AutoCloseable {
    /** The configuration this engine was constructed with. */
    public val backendConfig: BackendConfig

    /** The concrete backend selected for this engine after [ComputeBackend.resolve]; never [ComputeBackend.AUTO]. */
    public val activeBackend: ComputeBackend

    /** `true` once [close] has been called. */
    public val isClosed: Boolean

    /** Releases native resources. Idempotent. */
    override fun close()
}
