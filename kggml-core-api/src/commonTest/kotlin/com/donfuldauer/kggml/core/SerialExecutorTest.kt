package com.donfuldauer.kggml.core

import kotlin.coroutines.ContinuationInterceptor
import kotlin.coroutines.CoroutineContext
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotSame
import kotlin.test.assertSame
import kotlin.test.assertTrue
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield

/** Deterministic stand-in for a dedicated thread: forwards to the test scheduler and records lifecycle. */
private class FakeSerialDispatcher(private val delegate: CoroutineDispatcher) : CoroutineDispatcher(), AutoCloseable {
    var closeCount = 0
    val isClosed: Boolean get() = closeCount > 0

    override fun dispatch(context: CoroutineContext, block: Runnable) {
        check(!isClosed) { "dispatch after close" }
        delegate.dispatch(context, block)
    }

    override fun close() {
        closeCount++
    }
}

/** Holds dispatched tasks until [drain] is called, so tests can observe queue order. */
private class QueueDispatcher : CoroutineDispatcher(), AutoCloseable {
    private val queue = ArrayDeque<Runnable>()
    var isClosed = false
        private set
    val size: Int get() = queue.size

    override fun dispatch(context: CoroutineContext, block: Runnable) {
        check(!isClosed) { "dispatch after close" }
        queue.addLast(block)
    }

    override fun close() {
        isClosed = true
    }

    fun drain() {
        while (queue.isNotEmpty()) queue.removeFirst().run()
    }
}

/** Rejects every dispatch (like a shut-down executor), counting the attempts. */
private open class RejectingDispatcher(private val rejection: Throwable) : CoroutineDispatcher() {
    var dispatchCount = 0
        private set

    override fun dispatch(context: CoroutineContext, block: Runnable) {
        dispatchCount++
        throw rejection
    }
}

/** A [RejectingDispatcher] the executor owns and must close. */
private class CloseableRejectingDispatcher(rejection: Throwable) : RejectingDispatcher(rejection), AutoCloseable {
    var closeCount = 0
        private set

    override fun close() {
        closeCount++
    }
}

private class FakeFactory(private val scope: TestScope) : SerialDispatcherFactory {
    val created = mutableListOf<Pair<String, FakeSerialDispatcher>>()

    override fun create(name: String): CoroutineDispatcher =
        FakeSerialDispatcher(StandardTestDispatcher(scope.testScheduler)).also { created += name to it }
}

class SerialExecutorTest {

    @Test
    fun factoryReceivesNameAndExecutorRunsOnItsDispatcher() = runTest {
        val factory = FakeFactory(this)
        val executor = SerialExecutor("kggml-test-1", factory)
        assertEquals("kggml-test-1", factory.created.single().first)
        assertSame(factory.created.single().second, executor.dispatcher)

        val interceptor = executor.execute { coroutineContext[ContinuationInterceptor] }
        assertSame(executor.dispatcher, interceptor)
    }

    @Test
    fun executeReturnsValueAndPropagatesExceptions() = runTest {
        val executor = SerialExecutor("x", FakeFactory(this))
        assertEquals(42, executor.execute { 42 })
        val e = assertFailsWith<InvalidInputException> { executor.execute { throw InvalidInputException("bad") } }
        assertEquals("bad", e.message)
        // The executor remains usable after a failed block.
        assertEquals("ok", executor.execute { "ok" })
    }

    @Test
    fun blocksRunInInvocationOrderAndExclusivelyEvenWhenSuspending() = runTest {
        val executor = SerialExecutor("x", FakeFactory(this))
        val log = mutableListOf<String>()
        var active = 0
        var maxActive = 0
        repeat(10) { i ->
            launch {
                executor.execute {
                    active++
                    maxActive = maxOf(maxActive, active)
                    log += "start$i"
                    // Earlier blocks suspend more often: without block-level exclusion, later blocks would
                    // interleave with and overtake them on the shared dispatcher.
                    repeat(10 - i) { yield() }
                    log += "end$i"
                    active--
                }
            }
        }
        advanceUntilIdle()
        assertEquals(1, maxActive)
        assertEquals((0 until 10).flatMap { listOf("start$it", "end$it") }, log)
    }

    @Test
    fun executeAfterCloseFails() = runTest {
        val executor = SerialExecutor("closed-exec", FakeFactory(this))
        assertFalse(executor.isClosed)
        executor.close()
        assertTrue(executor.isClosed)
        val e = assertFailsWith<IllegalStateException> { executor.execute { 1 } }
        assertTrue(e.message!!.contains("closed-exec"))
    }

    @Test
    fun releaseRunsOnDispatcherAfterQueuedWork() = runTest {
        val queue = QueueDispatcher()
        val executor = SerialExecutor("x") { queue }
        val log = mutableListOf<String>()
        launch { executor.execute { log += "work" } }
        runCurrent() // `work` is now queued on the executor's dispatcher but has not run yet.
        assertEquals(1, queue.size)

        executor.close { log += "release" }
        assertTrue(queue.isClosed)
        assertEquals(2, queue.size)
        queue.drain()
        advanceUntilIdle()
        assertEquals(listOf("work", "release"), log)
    }

    @Test
    fun closeIsIdempotent() = runTest {
        val factory = FakeFactory(this)
        val executor = SerialExecutor("x", factory)
        var releases = 0
        executor.close { releases++ }
        executor.close { releases++ }
        executor.close()
        advanceUntilIdle()
        assertEquals(1, releases)
        assertEquals(1, factory.created.single().second.closeCount)
    }

    @Test
    fun throwingReleaseIsContainedAndDispatcherStillClosed() = runTest {
        val factory = FakeFactory(this)
        val executor = SerialExecutor("x", factory)
        executor.close { throw IllegalStateException("boom") }
        advanceUntilIdle()
        assertTrue(executor.isClosed)
        assertTrue(factory.created.single().second.isClosed)
    }

    @Test
    fun dispatchFailureDuringClosePropagatesButTheDispatcherIsStillClosed() = runTest {
        val rejection = IllegalStateException("dispatcher rejected the release step")
        val dispatcher = CloseableRejectingDispatcher(rejection)
        val executor = SerialExecutor("rejecting") { dispatcher }
        var released = false

        val e = assertFailsWith<IllegalStateException> { executor.close { released = true } }

        assertSame(rejection, e)
        assertEquals(1, dispatcher.closeCount, "the dispatcher must be shut down even if scheduling release fails")
        assertFalse(released)
        assertTrue(executor.isClosed)
        executor.close() // idempotent: no second dispatch or shutdown attempt
        assertEquals(1, dispatcher.dispatchCount)
        assertEquals(1, dispatcher.closeCount)
        assertFailsWith<IllegalStateException> { executor.execute { } }
    }

    @Test
    fun dispatchFailureDuringCloseOfANonCloseableDispatcherPropagates() = runTest {
        val rejection = IllegalStateException("rejected")
        val dispatcher = RejectingDispatcher(rejection)
        val executor = SerialExecutor("rejecting") { dispatcher }
        assertSame(rejection, assertFailsWith<IllegalStateException> { executor.close() })
        assertTrue(executor.isClosed)
        assertEquals(1, dispatcher.dispatchCount)
    }

    @Test
    fun nonCloseableDispatcherIsAccepted() = runTest {
        val executor = SerialExecutor("x") { StandardTestDispatcher(testScheduler) }
        assertEquals(3, executor.execute { 3 })
        executor.close()
        assertTrue(executor.isClosed)
    }

    @Test
    fun defaultFactoryGivesEachExecutorItsOwnDispatcherAndSerializesCrossThreadCallers() = runTest {
        val first = SerialExecutor("kggml-serial-a")
        val second = SerialExecutor("kggml-serial-b")
        try {
            assertNotSame(first.dispatcher, second.dispatcher)
            var counter = 0 // Deliberately non-atomic: only safe if access is serialized.
            withContext(Dispatchers.Default) {
                (1..200).map {
                    async {
                        first.execute {
                            val read = counter
                            yield()
                            counter = read + 1
                        }
                    }
                }.awaitAll()
            }
            assertEquals(200, counter)
            assertEquals("b", second.execute { "b" })
        } finally {
            first.close()
            second.close()
        }
        assertFailsWith<IllegalStateException> { first.execute { } }
    }

    @Test
    fun closeRacingWithDispatchTranslatesDispatcherRejectionToIllegalStateException() = runTest {
        val delegate = StandardTestDispatcher(testScheduler)
        var executorRef: SerialExecutor? = null
        // Mimics JVM ExecutorCoroutineDispatcherImpl when close() shuts down the backing ExecutorService right
        // between SerialExecutor's checkOpen() and withContext(ownedDispatcher, block): dispatch() cancels the
        // dispatched context and delegates the cancellation completion to a fallback dispatcher.
        val racingDispatcher = object : CoroutineDispatcher(), AutoCloseable {
            private var closed = false
            private var triggered = false

            override fun dispatch(context: CoroutineContext, block: Runnable) {
                if (!triggered && !closed) {
                    triggered = true
                    executorRef!!.close()
                }
                if (closed) {
                    context.cancel(CancellationException("The task was rejected"))
                }
                delegate.dispatch(context, block)
            }

            override fun close() {
                closed = true
            }
        }
        val executor = SerialExecutor("racing-close") { racingDispatcher }
        executorRef = executor

        val e = assertFailsWith<IllegalStateException> { executor.execute { "never" } }
        assertTrue(e.message!!.contains("racing-close"))
        assertTrue(currentCoroutineContext().isActive, "caller coroutine must remain active after closed rejection")
    }
}
