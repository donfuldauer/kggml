package com.donfuldauer.kggml.models

import com.donfuldauer.kggml.core.MemoryPressureLevel
import com.donfuldauer.kggml.core.MemoryPressureListener
import com.donfuldauer.kggml.core.MemoryPressureObserver

/**
 * A borrowed resident model. While at least one lease is open the model is never evicted. Closing a lease twice is a
 * no-op.
 */
public interface ModelLease<out T : AutoCloseable> : AutoCloseable {
    /** Cache key. */
    public val key: String

    /** The loaded model (an engine, for example). */
    public val resource: T
}

/**
 * In-memory cache of loaded models (engines or other [AutoCloseable] resources) with a RAM budget and OS low-memory
 * eviction.
 *
 * - [put] / [getOrPut] register a model with its estimated resident size (for example
 *   [ModelHeaderInfo.estimatedPeakRam]); the least recently used *idle* models are closed when the budget is exceeded.
 * - [attachTo] subscribes to a [MemoryPressureObserver] (Android `AndroidMemoryPressureObserver`, iOS
 *   `IosMemoryPressureObserver`, or `ManualMemoryPressureObserver` on desktop):
 *   [MemoryPressureLevel.MODERATE] closes idle models until at most half the budget is resident,
 *   [MemoryPressureLevel.CRITICAL] closes every idle model.
 * - Leased (in-use) models are never closed by eviction.
 *
 * Resources are closed outside the internal lock, on the thread that triggered the eviction. Thread-safe.
 *
 * @param budgetBytes resident budget in bytes.
 */
public class ResidentModelCache(
    public val budgetBytes: Long = Long.MAX_VALUE,
) : AutoCloseable {
    private class Entry(val key: String, val resource: AutoCloseable, val sizeBytes: Long, var tick: Long, var leases: Int)

    private val lock = SpinLock()
    private val entries = LinkedHashMap<String, Entry>()
    private var tick = 0L
    private var closed = false

    init {
        require(budgetBytes >= 0) { "budgetBytes must be >= 0" }
    }

    /** Keys currently resident, least recently used first. */
    public val keys: List<String> get() = lock.withLock { entries.values.sortedBy { it.tick }.map { it.key } }

    /** Sum of the estimated sizes of resident models. */
    public val residentBytes: Long get() = lock.withLock { entries.values.sumOf { it.sizeBytes } }

    /**
     * Adds [resource] under [key] and returns a lease on it. A previous idle model under [key] is closed; if the
     * previous one is leased, [put] fails. Then idle models are evicted until the budget holds.
     */
    public fun <T : AutoCloseable> put(key: String, estimatedBytes: Long, resource: T): ModelLease<T> {
        require(estimatedBytes >= 0) { "estimatedBytes must be >= 0" }
        val toClose = ArrayList<AutoCloseable>()
        val lease = lock.withLock {
            check(!closed) { "ResidentModelCache is closed" }
            entries[key]?.let { old ->
                check(old.leases == 0) { "model '$key' is in use" }
                entries.remove(key)
                if (old.resource !== resource) toClose += old.resource
            }
            val entry = Entry(key, resource, estimatedBytes, ++tick, leases = 1)
            entries[key] = entry
            toClose += evictLocked(budgetBytes)
            Lease(entry, resource)
        }
        closeAll(toClose)
        return lease
    }

    /** Leases the model under [key], or returns `null` if it is not resident. */
    public fun <T : AutoCloseable> acquire(key: String): ModelLease<T>? = lock.withLock {
        val entry = entries[key] ?: return@withLock null
        entry.leases++
        entry.tick = ++tick
        @Suppress("UNCHECKED_CAST")
        Lease(entry, entry.resource as T)
    }

    /**
     * Leases the model under [key], loading it with [loader] if it is not resident. If another caller loaded the same
     * key concurrently, the extra copy is closed and the resident one is returned.
     */
    public suspend fun <T : AutoCloseable> getOrPut(key: String, estimatedBytes: Long, loader: suspend () -> T): ModelLease<T> {
        acquire<T>(key)?.let { return it }
        val loaded = loader()
        val existing = acquire<T>(key)
        if (existing != null) {
            loaded.close()
            return existing
        }
        return put(key, estimatedBytes, loaded)
    }

    /**
     * Closes idle models according to [level]: `NORMAL` enforces the budget, `MODERATE` trims to half the budget
     * (or half of what is resident if no budget is set), `CRITICAL` closes every idle model.
     *
     * @return the evicted keys.
     */
    public fun evict(level: MemoryPressureLevel): List<String> {
        val evicted = ArrayList<Entry>()
        lock.withLock {
            val target = when (level) {
                MemoryPressureLevel.NORMAL -> budgetBytes
                MemoryPressureLevel.MODERATE ->
                    if (budgetBytes == Long.MAX_VALUE) entries.values.sumOf { it.sizeBytes } / 2 else budgetBytes / 2
                MemoryPressureLevel.CRITICAL -> 0L
            }
            evicted += evictEntriesLocked(target)
        }
        closeAll(evicted.map { it.resource })
        return evicted.map { it.key }
    }

    /** Closes the idle model under [key]. Returns `false` if it is absent or leased. */
    public fun remove(key: String): Boolean {
        val entry = lock.withLock {
            val e = entries[key] ?: return@withLock null
            if (e.leases > 0) return@withLock null
            entries.remove(key)
        } ?: return false
        entry.resource.close()
        return true
    }

    /** Evicts on every signal from [observer]. Close the returned handle to unsubscribe. */
    public fun attachTo(observer: MemoryPressureObserver): AutoCloseable =
        observer.addListener(MemoryPressureListener { level -> if (level != MemoryPressureLevel.NORMAL) evict(level) })

    /** Closes every resident model, leased or not. Idempotent; the cache cannot be used afterwards. */
    override fun close() {
        val all = lock.withLock {
            if (closed) return
            closed = true
            entries.values.map { it.resource }.also { entries.clear() }
        }
        closeAll(all)
    }

    private fun evictLocked(target: Long): List<AutoCloseable> = evictEntriesLocked(target).map { it.resource }

    private fun evictEntriesLocked(target: Long): List<Entry> {
        var total = entries.values.sumOf { it.sizeBytes }
        val out = ArrayList<Entry>()
        for (e in entries.values.sortedBy { it.tick }) {
            if (total <= target) break
            if (e.leases > 0) continue
            entries.remove(e.key)
            total -= e.sizeBytes
            out += e
        }
        return out
    }

    private fun closeAll(resources: List<AutoCloseable>) {
        var first: Throwable? = null
        for (r in resources) {
            try {
                r.close()
            } catch (t: Throwable) {
                if (first == null) first = t else first.addSuppressed(t)
            }
        }
        first?.let { throw it }
    }

    private inner class Lease<T : AutoCloseable>(private val entry: Entry, override val resource: T) : ModelLease<T> {
        private var released = false
        override val key: String get() = entry.key

        override fun close() {
            lock.withLock {
                if (released) return
                released = true
                if (entries[entry.key] === entry) {
                    entry.leases--
                    entry.tick = ++tick
                }
            }
        }
    }
}
