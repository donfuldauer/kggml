package com.donfuldauer.kggml.core

import android.content.ComponentCallbacks2
import android.content.Context
import android.content.res.Configuration

/**
 * [MemoryPressureObserver] driven by Android's `ComponentCallbacks2.onTrimMemory` / `onLowMemory`.
 *
 * Registers on the application context at construction; [close] unregisters it (idempotent).
 *
 * @param context any context; its application context is used.
 */
public class AndroidMemoryPressureObserver(context: Context) : ManualMemoryPressureObserver(), AutoCloseable {
    private val appContext: Context = context.applicationContext ?: context
    private var registered = true

    private val callbacks = object : ComponentCallbacks2 {
        override fun onTrimMemory(level: Int) {
            emit(memoryPressureLevelForTrim(level))
        }

        @Deprecated("Deprecated in Android API 34", ReplaceWith("onTrimMemory(level)"))
        override fun onLowMemory() {
            emit(MemoryPressureLevel.CRITICAL)
        }

        override fun onConfigurationChanged(newConfig: Configuration): Unit = Unit
    }

    init {
        appContext.registerComponentCallbacks(callbacks)
    }

    /** Unregisters from the application context. Idempotent. */
    @Synchronized
    override fun close() {
        if (!registered) return
        registered = false
        appContext.unregisterComponentCallbacks(callbacks)
    }
}

/**
 * Maps an `onTrimMemory` level to a [MemoryPressureLevel]:
 * - `RUNNING_CRITICAL` and `COMPLETE` (or above) → [MemoryPressureLevel.CRITICAL]
 * - `RUNNING_LOW`, `BACKGROUND`, `MODERATE` → [MemoryPressureLevel.MODERATE]
 * - `RUNNING_MODERATE`, `UI_HIDDEN`, unknown lower levels → [MemoryPressureLevel.NORMAL]
 */
@Suppress("DEPRECATION")
internal fun memoryPressureLevelForTrim(level: Int): MemoryPressureLevel = when {
    level >= ComponentCallbacks2.TRIM_MEMORY_COMPLETE -> MemoryPressureLevel.CRITICAL
    level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL -> MemoryPressureLevel.CRITICAL
    level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND -> MemoryPressureLevel.MODERATE
    level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW -> MemoryPressureLevel.MODERATE
    else -> MemoryPressureLevel.NORMAL
}
