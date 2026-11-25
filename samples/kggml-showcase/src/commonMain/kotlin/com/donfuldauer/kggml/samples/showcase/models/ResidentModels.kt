package com.donfuldauer.kggml.samples.showcase.models

import com.donfuldauer.kggml.core.ManualMemoryPressureObserver
import com.donfuldauer.kggml.core.MemoryPressureLevel
import com.donfuldauer.kggml.core.MemoryPressureListener
import com.donfuldauer.kggml.core.MemoryPressureObserver
import com.donfuldauer.kggml.models.ModelLease
import com.donfuldauer.kggml.models.ResidentModelCache
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlin.concurrent.atomics.AtomicReference
import kotlin.concurrent.atomics.ExperimentalAtomicApi

/** A loaded engine held by [ResidentModels]. */
data class ResidentEntry(val key: String, val estimatedBytes: Long)

/** Where an eviction came from. */
enum class EvictionSource(val label: String) { OS("OS memory-pressure signal"), MANUAL("Manual") }

/** One eviction pass, newest first in [ResidentState.evictions]. */
data class EvictionRecord(val level: MemoryPressureLevel, val source: EvictionSource, val evictedKeys: List<String>)

data class ResidentState(
    val budgetBytes: Long,
    /** Resident engines, least recently used first. */
    val entries: List<ResidentEntry> = emptyList(),
    val evictions: List<EvictionRecord> = emptyList(),
    /** The memory-pressure level last reported by [ResidentModels.observer]. */
    val lastLevel: MemoryPressureLevel = MemoryPressureLevel.NORMAL,
) {
    val residentBytes: Long get() = entries.sumOf { it.estimatedBytes }
}

/**
 * The engines the showcase keeps loaded (LLM chat, diffusion), in a `ResidentModelCache` with a RAM budget that is
 * subscribed to the platform's [MemoryPressureObserver] (Android `onTrimMemory`, iOS memory warnings): on a
 * `MODERATE` signal idle engines are trimmed to half the budget, on `CRITICAL` every idle engine is closed. Engines
 * in use (leased by a running generation) are never closed. Screens lease an engine for each operation and load it
 * again when it was evicted in the meantime.
 *
 * Thread-safe; OS signals may arrive on any thread.
 */
@OptIn(ExperimentalAtomicApi::class)
class ResidentModels(
    budgetBytes: Long,
    val observer: MemoryPressureObserver,
) : AutoCloseable {
    private val cache = ResidentModelCache(budgetBytes)
    private val sizes = AtomicReference<Map<String, Long>>(emptyMap())
    private val mutableState = MutableStateFlow(ResidentState(budgetBytes))
    val state: StateFlow<ResidentState> = mutableState.asStateFlow()

    // Registered after the cache's own listener, so the snapshot below already reflects the eviction.
    private val cacheSubscription = cache.attachTo(observer)
    private val stateSubscription = observer.addListener(
        MemoryPressureListener { level -> record(level, EvictionSource.OS, evictedKeys = null) },
    )

    /**
     * Leases the engine under [key], loading it with [load] (which returns the engine and its estimated resident
     * bytes) when it is not resident. Close the lease when the operation is done; the engine then stays resident
     * (idle) until it is evicted or [remove]d.
     */
    suspend fun <T : AutoCloseable> lease(key: String, load: suspend () -> Pair<T, Long>): ModelLease<T> {
        cache.acquire<T>(key)?.let { return it }
        val (resource, bytes) = load()
        val existing = cache.acquire<T>(key)
        if (existing != null) {
            resource.close()
            return existing
        }
        updateSizes { it + (key to bytes) }
        return cache.put(key, bytes, resource).also { refresh() }
    }

    /** Closes the idle engine under [key] (for example after its configuration changed). */
    fun remove(key: String): Boolean = cache.remove(key).also { refresh() }

    /** Evicts idle engines right away, as an OS signal of [level] would. */
    fun evict(level: MemoryPressureLevel): List<String> {
        val evicted = cache.evict(level)
        record(level, EvictionSource.MANUAL, evicted)
        return evicted
    }

    /**
     * Delivers a synthetic OS signal through [observer] (`emit` on the platform observer), exercising exactly the path
     * a real `onTrimMemory` / memory warning takes. Returns `false` if the observer cannot be driven manually.
     */
    fun simulateOsSignal(level: MemoryPressureLevel): Boolean {
        val manual = observer as? ManualMemoryPressureObserver ?: return false
        manual.emit(level)
        return true
    }

    /** Re-reads the resident set (after a lease was closed, for example). */
    fun refresh() {
        val keys = cache.keys
        val known = updateSizes { current -> current.filterKeys { it in keys } }
        val entries = keys.map { ResidentEntry(it, known[it] ?: 0L) }
        mutableState.update { it.copy(entries = entries, lastLevel = observer.currentLevel) }
    }

    override fun close() {
        stateSubscription.close()
        cacheSubscription.close()
        cache.close()
        refresh()
    }

    private fun record(level: MemoryPressureLevel, source: EvictionSource, evictedKeys: List<String>?) {
        val before = mutableState.value.entries.map { it.key }
        refresh()
        val after = mutableState.value.entries.map { it.key }.toSet()
        val evicted = evictedKeys ?: before.filterNot { it in after }
        mutableState.update {
            it.copy(evictions = (listOf(EvictionRecord(level, source, evicted)) + it.evictions).take(MAX_RECORDS))
        }
    }

    private inline fun updateSizes(transform: (Map<String, Long>) -> Map<String, Long>): Map<String, Long> {
        while (true) {
            val current = sizes.load()
            val next = transform(current)
            if (sizes.compareAndSet(current, next)) return next
        }
    }

    private companion object {
        const val MAX_RECORDS = 8
    }
}
