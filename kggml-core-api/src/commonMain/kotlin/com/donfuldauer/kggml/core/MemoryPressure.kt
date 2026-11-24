package com.donfuldauer.kggml.core

import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.AtomicReference

/** Severity of an OS memory-pressure signal. */
public enum class MemoryPressureLevel {
    /** No memory pressure reported. */
    NORMAL,

    /** The OS is running low on memory; release caches that are cheap to rebuild. */
    MODERATE,

    /** The OS is about to kill processes; release everything that is not strictly required. */
    CRITICAL,
}

/** Receives memory-pressure signals from a [MemoryPressureObserver]. */
public fun interface MemoryPressureListener {
    /** Called on the thread that delivered the OS signal. Implementations must be fast and non-blocking. */
    public fun onMemoryPressure(level: MemoryPressureLevel)
}

/**
 * Source of OS low-memory signals (Android `ComponentCallbacks2.onTrimMemory`, iOS
 * `UIApplicationDidReceiveMemoryWarningNotification`, …). Engines and the model manager subscribe to it to
 * evict KV caches, model weights, and download buffers.
 */
public interface MemoryPressureObserver {
    /** The most recently reported level, or [MemoryPressureLevel.NORMAL] if none has been reported. */
    public val currentLevel: MemoryPressureLevel

    /**
     * Registers [listener]. Closing the returned handle unregisters it; closing it more than once is a no-op.
     */
    public fun addListener(listener: MemoryPressureListener): AutoCloseable
}

/**
 * A [MemoryPressureObserver] whose signals are pushed with [emit].
 *
 * Used as the base of platform observers, on platforms without an OS signal (JVM desktop), and as a
 * deterministic fake in tests. Thread-safe: listeners may be added, removed, and notified concurrently.
 */
public open class ManualMemoryPressureObserver : MemoryPressureObserver {
    private val listeners = AtomicReference<List<MemoryPressureListener>>(emptyList())
    private val level = AtomicReference(MemoryPressureLevel.NORMAL)

    override val currentLevel: MemoryPressureLevel get() = level.load()

    override fun addListener(listener: MemoryPressureListener): AutoCloseable {
        // Wrap so the same listener instance may be registered (and removed) independently more than once.
        val registration = MemoryPressureListener { listener.onMemoryPressure(it) }
        updateListeners { it + registration }
        val removed = AtomicBoolean(false)
        return AutoCloseable {
            if (!removed.exchange(true)) updateListeners { current -> current.filterNot { it === registration } }
        }
    }

    /** Records [newLevel] as [currentLevel] and notifies every registered listener in registration order. */
    public fun emit(newLevel: MemoryPressureLevel) {
        level.store(newLevel)
        listeners.load().forEach { it.onMemoryPressure(newLevel) }
    }

    private inline fun updateListeners(transform: (List<MemoryPressureListener>) -> List<MemoryPressureListener>) {
        while (true) {
            val current = listeners.load()
            if (listeners.compareAndSet(current, transform(current))) return
        }
    }
}
