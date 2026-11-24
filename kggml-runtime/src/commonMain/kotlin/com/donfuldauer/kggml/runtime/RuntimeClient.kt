package com.donfuldauer.kggml.runtime

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceException
import com.donfuldauer.kggml.core.NativeFailureException
import kotlin.concurrent.atomics.AtomicReference

/**
 * Platform-independent logic behind [KggmlRuntime]: lazy bridge creation, reference-counted lifecycle,
 * native status → [InferenceException] translation, and mapping of raw native values onto
 * `:kggml-core-api` types. Constructed with a [bridgeFactory] so it can be tested with fakes.
 */
internal class RuntimeClient(private val bridgeFactory: () -> NativeRuntimeBridge) {
    private val bridgeRef = AtomicReference<NativeRuntimeBridge?>(null)

    /** The native bridge, created (and the native libraries loaded) on first use. */
    fun bridge(): NativeRuntimeBridge {
        bridgeRef.load()?.let { return it }
        val created = try {
            bridgeFactory()
        } catch (e: InferenceException) {
            throw e
        } catch (t: Throwable) {
            // UnsatisfiedLinkError, SecurityException, I/O failures while extracting bundled libraries, …
            throw NativeFailureException(
                message = "Failed to load the Kggml native runtime: ${t.message ?: t::class.simpleName}",
                cause = t,
            )
        }
        // Racing first calls may both create a bridge; they are stateless views of the same native library.
        bridgeRef.compareAndSet(null, created)
        return bridgeRef.load() ?: created
    }

    fun acquire(): RuntimeHandle {
        val bridge = bridge()
        check(bridge, bridge.init(), "kggml_runtime_init")
        return RuntimeHandle(this)
    }

    fun release() {
        val bridge = bridge()
        check(bridge, bridge.shutdown(), "kggml_runtime_shutdown")
    }

    /** Runs [block] while holding a runtime reference, releasing it afterwards. */
    inline fun <T> withRuntime(block: (NativeRuntimeBridge) -> T): T {
        val handle = acquire()
        var failure: Throwable? = null
        try {
            return block(bridge())
        } catch (t: Throwable) {
            failure = t
            throw t
        } finally {
            if (failure == null) {
                handle.close()
            } else {
                try {
                    handle.close()
                } catch (closeFailure: Throwable) {
                    failure.addSuppressed(closeFailure)
                }
            }
        }
    }

    fun isInitialized(): Boolean = bridge().isInitialized()

    fun refCount(): Int = bridge().refCount()

    fun backends(): List<BackendInfo> = withRuntime { bridge ->
        var capacity = BackendKind.COUNT
        var result = bridge.getBackends(capacity)
        // A newer runtime may report more backend kinds than this binding knows about; retry once at the
        // size it asked for.
        if (result.status == RuntimeStatus.BUFFER_TOO_SMALL && result.requiredCount > capacity) {
            capacity = result.requiredCount
            result = bridge.getBackends(capacity)
        }
        check(bridge, result.status, "kggml_runtime_get_backends")
        result.entries.mapNotNull { raw ->
            val backend = BackendKind.toComputeBackend(raw.kind) ?: return@mapNotNull null
            BackendInfo(
                backend = backend,
                name = raw.name,
                compiled = raw.compiled,
                available = raw.available,
                deviceCount = raw.deviceCount.coerceAtLeast(0),
            )
        }
    }

    fun availableBackends(): Set<ComputeBackend> =
        backends().filter { it.available }.mapTo(linkedSetOf(ComputeBackend.CPU_ONLY)) { it.backend }

    fun resolveBackend(requested: ComputeBackend, allowFallback: Boolean): ComputeBackend =
        ComputeBackend.resolve(requested, availableBackends(), allowFallback)

    fun recommendedThreadCount(): Int {
        val bridge = bridge()
        val result = bridge.performanceThreadCount()
        check(bridge, result.status, "kggml_runtime_performance_thread_count")
        return result.value.coerceAtLeast(1)
    }

    fun hardwareThreadCount(): Int = bridge().hardwareThreadCount().coerceAtLeast(1)

    fun info(): RuntimeInfo {
        val bridge = bridge()
        return RuntimeInfo(
            runtimeVersion = bridge.runtimeVersion(),
            ggmlVersion = bridge.ggmlVersion(),
            ggmlCommit = bridge.ggmlCommit(),
            ggmlBuildCommit = bridge.ggmlBuildCommit(),
            hasCoreMl = bridge.hasCoreMl(),
        )
    }

    /**
     * Throws the typed [InferenceException] for a non-OK native [status], with the native status name and
     * thread-local diagnostic message attached. Must be called right after the failing native call.
     */
    fun check(bridge: NativeRuntimeBridge, status: Int, operation: String, backend: ComputeBackend? = null) {
        if (status == RuntimeStatus.OK) return
        throw nativeError(bridge, status, operation, backend)
    }

    companion object {
        fun nativeError(
            bridge: NativeRuntimeBridge,
            status: Int,
            operation: String,
            backend: ComputeBackend? = null,
        ): InferenceException {
            val detail = runCatching { bridge.lastError() }.getOrDefault("")
            val name = runCatching { bridge.statusName(status) }.getOrDefault("KGGML_STATUS_$status")
            val message = buildString {
                append(operation).append(" failed: ").append(name).append(" (").append(status).append(')')
                if (detail.isNotEmpty()) append(": ").append(detail)
            }
            return InferenceException.fromNativeError(status, message, backend)
        }
    }
}
