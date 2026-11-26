package com.donfuldauer.kggml.runtime

import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

class RuntimeClientConcurrencyTest {

    @Test
    fun racingFirstCallsConvergeOnTheFirstPublishedBridge() {
        val threads = 8
        val created = mutableListOf<FakeRuntimeBridge>()
        val factoryEntered = CountDownLatch(threads)
        val calls = AtomicInteger()
        // Every caller is held inside the factory until all of them are, so each one creates its own bridge
        // and all but one lose the publication race.
        val client = RuntimeClient {
            calls.incrementAndGet()
            factoryEntered.countDown()
            check(factoryEntered.await(10, TimeUnit.SECONDS)) { "callers did not race" }
            FakeRuntimeBridge().also { synchronized(created) { created += it } }
        }
        val pool = Executors.newFixedThreadPool(threads)
        try {
            val results = (1..threads).map { pool.submit<NativeRuntimeBridge> { client.bridge() } }.map { it.get(20, TimeUnit.SECONDS) }

            assertEquals(threads, calls.get())
            val winner = results.first()
            assertTrue(results.all { it === winner }, "every racing caller must get the same bridge")
            assertTrue(created.any { it === winner })
            // Later calls keep returning the published bridge without invoking the factory again.
            assertSame(winner, client.bridge())
            assertEquals(threads, calls.get())
        } finally {
            pool.shutdownNow()
        }
    }
}
