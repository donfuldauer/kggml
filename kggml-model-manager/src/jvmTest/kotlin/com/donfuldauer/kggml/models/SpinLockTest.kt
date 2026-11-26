package com.donfuldauer.kggml.models

import java.util.concurrent.CountDownLatch
import kotlin.concurrent.thread
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class SpinLockTest {
    @Test
    fun contendedSectionsAreMutuallyExclusive() {
        val lock = SpinLock()
        val threads = 8
        val iterations = 20_000
        var counter = 0 // deliberately unsynchronized: only the lock protects it
        val start = CountDownLatch(1)
        val workers = List(threads) {
            thread {
                start.await()
                repeat(iterations) { lock.withLock { counter++ } }
            }
        }
        start.countDown()
        workers.forEach { it.join() }
        assertEquals(threads * iterations, counter)
        assertFalse(lock.locked.load())
    }

    @Test
    fun lockIsReleasedWhenTheBlockThrows() {
        val lock = SpinLock()
        assertFailsWith<IllegalStateException> { lock.withLock { error("boom") } }
        assertFalse(lock.locked.load())
        assertEquals(1, lock.withLock { 1 }, "the lock can be taken again")
    }
}
