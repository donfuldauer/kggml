package com.donfuldauer.kggml.core

import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.coroutines.EmptyCoroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Creates the dedicated dispatcher backing a [SerialExecutor].
 *
 * Production code uses [Default], which spawns one dedicated OS thread per executor. Tests may supply a
 * deterministic fake (e.g. one wrapping a `TestDispatcher`).
 */
public fun interface SerialDispatcherFactory {
    /**
     * Creates a new dispatcher named [name] that executes tasks one at a time, in submission order.
     * The [SerialExecutor] owns it: if it implements [AutoCloseable], it is closed by [SerialExecutor.close].
     */
    public fun create(name: String): CoroutineDispatcher

    public companion object {
        /** Backs each executor with its own dedicated single OS thread. */
        public val Default: SerialDispatcherFactory = SerialDispatcherFactory(::newDedicatedThreadDispatcher)
    }
}

/** Platform implementation of [SerialDispatcherFactory.Default]: a dispatcher backed by one dedicated thread. */
internal expect fun newDedicatedThreadDispatcher(name: String): CoroutineDispatcher

/**
 * Per-engine serialized native dispatch.
 *
 * `ggml` contexts are not safe for concurrent mutation, so every engine instance owns exactly one
 * [SerialExecutor] and routes all native calls (`init`, inference, `close`) through it:
 *
 * - All blocks run on a single dedicated thread ([dispatcher]).
 * - Blocks passed to [execute] are mutually exclusive **as a whole**, even if they suspend, and are
 *   admitted in FIFO order of invocation.
 * - [close] runs the engine's native release step on the same thread after all previously queued work,
 *   then shuts the thread down. It is idempotent.
 *
 * Instances are independent: two executors never share a thread, so separate engines run concurrently.
 *
 * @param name thread/dispatcher name, used for diagnostics (e.g. `"kggml-whisper-1"`).
 * @param factory creates the backing dispatcher; defaults to a dedicated OS thread.
 */
public class SerialExecutor(
    public val name: String,
    factory: SerialDispatcherFactory = SerialDispatcherFactory.Default,
) : AutoCloseable {
    private val ownedDispatcher: CoroutineDispatcher = factory.create(name)
    private val mutex = Mutex()
    private val closed = AtomicBoolean(false)

    /** The single-threaded dispatcher all native work runs on. */
    public val dispatcher: CoroutineDispatcher get() = ownedDispatcher

    /** `true` once [close] has been called. */
    public val isClosed: Boolean get() = closed.load()

    /**
     * Runs [block] on [dispatcher], exclusively with respect to every other [execute] call on this executor.
     *
     * @throws IllegalStateException if this executor has been closed.
     */
    public suspend fun <T> execute(block: suspend CoroutineScope.() -> T): T {
        checkOpen()
        return mutex.withLock {
            checkOpen()
            val started = AtomicBoolean(false)
            try {
                withContext(ownedDispatcher) {
                    started.store(true)
                    block()
                }
            } catch (e: CancellationException) {
                currentCoroutineContext().ensureActive()
                if (!started.load()) checkOpen()
                throw e
            }
        }
    }

    /**
     * Closes this executor without a native release step. Idempotent.
     *
     * @see close
     */
    override fun close() {
        close(release = {})
    }

    /**
     * Closes this executor, scheduling [release] (e.g. `whisper_free`) to run on [dispatcher] after all work
     * already queued on it, then shutting the dispatcher down. Only the first call has any effect; later
     * calls (with any [release]) are no-ops. New [execute] calls fail immediately afterwards.
     *
     * [release] must not throw: an exception it throws is caught and discarded so the dispatcher can still
     * shut down. Do not call this from inside an [execute] block if the platform dispatcher cannot be closed
     * from its own thread.
     */
    public fun close(release: () -> Unit) {
        if (!closed.compareAndSet(expectedValue = false, newValue = true)) return
        try {
            ownedDispatcher.dispatch(
                EmptyCoroutineContext,
                Runnable {
                    try {
                        release()
                    } catch (_: Throwable) {
                        // Swallowed by contract: the release step runs detached from any caller.
                    }
                },
            )
        } finally {
            (ownedDispatcher as? AutoCloseable)?.close()
        }
    }

    private fun checkOpen() {
        check(!isClosed) { "SerialExecutor '$name' is closed" }
    }
}
