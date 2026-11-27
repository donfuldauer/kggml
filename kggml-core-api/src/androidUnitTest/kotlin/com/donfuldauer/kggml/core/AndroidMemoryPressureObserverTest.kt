package com.donfuldauer.kggml.core

import android.content.ComponentCallbacks
import android.content.ComponentCallbacks2
import android.content.Context
import android.content.ContextWrapper
import android.content.res.Configuration
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Recording [Context]: the Android framework classes are the SDK's stub jar here (`isReturnDefaultValues`), so
 * this overrides exactly the methods [AndroidMemoryPressureObserver] calls.
 */
private class RecordingContext(private val application: Context? = null) : ContextWrapper(null) {
    val registered = mutableListOf<ComponentCallbacks>()
    val unregistered = mutableListOf<ComponentCallbacks>()

    override fun getApplicationContext(): Context? = application

    override fun registerComponentCallbacks(callback: ComponentCallbacks) {
        registered += callback
    }

    override fun unregisterComponentCallbacks(callback: ComponentCallbacks) {
        unregistered += callback
    }
}

class AndroidMemoryPressureObserverTest {

    @Test
    fun mapsTrimLevels() {
        // Raw values of ComponentCallbacks2.TRIM_MEMORY_* (stable since API 14/16).
        val expected = mapOf(
            0 to MemoryPressureLevel.NORMAL,
            5 to MemoryPressureLevel.NORMAL, // RUNNING_MODERATE
            10 to MemoryPressureLevel.MODERATE, // RUNNING_LOW
            15 to MemoryPressureLevel.CRITICAL, // RUNNING_CRITICAL
            20 to MemoryPressureLevel.NORMAL, // UI_HIDDEN
            40 to MemoryPressureLevel.MODERATE, // BACKGROUND
            60 to MemoryPressureLevel.MODERATE, // MODERATE
            80 to MemoryPressureLevel.CRITICAL, // COMPLETE
            100 to MemoryPressureLevel.CRITICAL,
        )
        for ((level, pressure) in expected) {
            assertEquals(pressure, memoryPressureLevelForTrim(level), "trim level $level")
        }
    }

    @Test
    fun registersOnTheApplicationContextNotTheGivenOne() {
        val application = RecordingContext()
        val activity = RecordingContext(application)
        AndroidMemoryPressureObserver(activity)
        assertEquals(1, application.registered.size)
        assertIs<ComponentCallbacks2>(application.registered.single())
        assertTrue(activity.registered.isEmpty(), "must not hold on to a short-lived context")
    }

    @Test
    fun contextWithoutApplicationContextIsUsedDirectly() {
        val context = RecordingContext(application = null)
        val observer = AndroidMemoryPressureObserver(context)
        assertEquals(1, context.registered.size)
        observer.close()
        assertSame(context.registered.single(), context.unregistered.single())
    }

    @Test
    fun trimAndLowMemorySignalsAreEmittedToListeners() {
        val context = RecordingContext()
        val observer = AndroidMemoryPressureObserver(context)
        val callbacks = context.registered.single() as ComponentCallbacks2
        val received = mutableListOf<MemoryPressureLevel>()
        observer.addListener { received += it }

        assertEquals(MemoryPressureLevel.NORMAL, observer.currentLevel)
        callbacks.onTrimMemory(10) // TRIM_MEMORY_RUNNING_LOW
        assertEquals(MemoryPressureLevel.MODERATE, observer.currentLevel)
        callbacks.onTrimMemory(15) // TRIM_MEMORY_RUNNING_CRITICAL
        callbacks.onTrimMemory(20) // TRIM_MEMORY_UI_HIDDEN
        @Suppress("DEPRECATION")
        callbacks.onLowMemory()
        callbacks.onConfigurationChanged(Configuration()) // not a memory signal

        assertEquals(
            listOf(MemoryPressureLevel.MODERATE, MemoryPressureLevel.CRITICAL, MemoryPressureLevel.NORMAL, MemoryPressureLevel.CRITICAL),
            received,
        )
        assertEquals(MemoryPressureLevel.CRITICAL, observer.currentLevel)
    }

    @Test
    fun closeUnregistersTheRegisteredCallbacksExactlyOnce() {
        val application = RecordingContext()
        val observer = AndroidMemoryPressureObserver(RecordingContext(application))
        observer.close()
        observer.close()
        assertEquals(1, application.unregistered.size, "close() is idempotent")
        assertSame(application.registered.single(), application.unregistered.single())
    }
}
