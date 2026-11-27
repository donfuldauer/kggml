package com.donfuldauer.kggml.core

import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * [ManualMemoryPressureObserver] is documented as thread-safe for concurrent add/remove/notify. Contended
 * registrations go through a compare-and-set retry loop; a lost update would drop or resurrect listeners.
 */
class ManualMemoryPressureObserverConcurrencyTest {

    @Test
    fun concurrentRegistrationsAndRemovalsLoseNoUpdates() {
        val threads = 8
        val perThread = 500
        val observer = ManualMemoryPressureObserver()
        val notified = AtomicInteger()
        val listener = MemoryPressureListener { notified.incrementAndGet() }
        val pool = Executors.newFixedThreadPool(threads)
        try {
            val start = CyclicBarrier(threads)
            val handles = (1..threads).map {
                pool.submit<List<AutoCloseable>> {
                    start.await(10, TimeUnit.SECONDS)
                    List(perThread) { observer.addListener(listener) }
                }
            }.flatMap { it.get(30, TimeUnit.SECONDS) }

            observer.emit(MemoryPressureLevel.MODERATE)
            assertEquals(threads * perThread, notified.get(), "every concurrent registration must be kept")

            // Remove all but the first registration of each thread concurrently, closing every handle twice.
            val keep = handles.chunked(perThread).map { it.first() }.toSet()
            val removeStart = CyclicBarrier(threads)
            handles.chunked(perThread).map { chunk ->
                pool.submit {
                    removeStart.await(10, TimeUnit.SECONDS)
                    chunk.filterNot { it in keep }.forEach { it.close(); it.close() }
                }
            }.forEach { it.get(30, TimeUnit.SECONDS) }

            notified.set(0)
            observer.emit(MemoryPressureLevel.CRITICAL)
            assertEquals(threads, notified.get(), "exactly the kept registrations remain")
            assertEquals(MemoryPressureLevel.CRITICAL, observer.currentLevel)
        } finally {
            pool.shutdownNow()
        }
    }
}
