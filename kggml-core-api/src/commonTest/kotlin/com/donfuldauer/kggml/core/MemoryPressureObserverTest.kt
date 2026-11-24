package com.donfuldauer.kggml.core

import kotlin.test.Test
import kotlin.test.assertEquals

class MemoryPressureObserverTest {

    @Test
    fun startsAtNormal() {
        assertEquals(MemoryPressureLevel.NORMAL, ManualMemoryPressureObserver().currentLevel)
    }

    @Test
    fun emitUpdatesCurrentLevelAndNotifiesListenersInOrder() {
        val observer = ManualMemoryPressureObserver()
        val log = mutableListOf<String>()
        observer.addListener { log += "a:$it" }
        observer.addListener { log += "b:$it" }

        observer.emit(MemoryPressureLevel.MODERATE)
        observer.emit(MemoryPressureLevel.CRITICAL)

        assertEquals(MemoryPressureLevel.CRITICAL, observer.currentLevel)
        assertEquals(listOf("a:MODERATE", "b:MODERATE", "a:CRITICAL", "b:CRITICAL"), log)
    }

    @Test
    fun closingHandleUnregistersIdempotently() {
        val observer = ManualMemoryPressureObserver()
        val log = mutableListOf<MemoryPressureLevel>()
        val listener = MemoryPressureListener { log += it }
        val first = observer.addListener(listener)
        val second = observer.addListener(listener)

        observer.emit(MemoryPressureLevel.MODERATE)
        first.close()
        first.close()
        observer.emit(MemoryPressureLevel.CRITICAL)
        second.close()
        observer.emit(MemoryPressureLevel.NORMAL)

        assertEquals(
            listOf(MemoryPressureLevel.MODERATE, MemoryPressureLevel.MODERATE, MemoryPressureLevel.CRITICAL),
            log,
        )
    }

    @Test
    fun listenerMayUnregisterItselfDuringDispatch() {
        val observer = ManualMemoryPressureObserver()
        val log = mutableListOf<String>()
        lateinit var handle: AutoCloseable
        handle = observer.addListener {
            log += "self"
            handle.close()
        }
        observer.addListener { log += "other" }

        observer.emit(MemoryPressureLevel.CRITICAL)
        observer.emit(MemoryPressureLevel.CRITICAL)

        assertEquals(listOf("self", "other", "other"), log)
    }
}
