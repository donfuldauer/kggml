package com.donfuldauer.kggml.whisper.streaming

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds

class PcmRingBufferTest {
    private fun seq(from: Int, count: Int) = FloatArray(count) { (from + it).toFloat() }

    @Test
    fun writeThenReadPreservesOrder() {
        val ring = PcmRingBuffer(8)
        assertEquals(5, ring.write(seq(0, 5)))
        assertEquals(5, ring.available)
        assertEquals(3, ring.remainingCapacity)
        val out = FloatArray(5)
        assertEquals(5, ring.read(out))
        assertContentEquals(seq(0, 5), out)
        assertEquals(0, ring.available)
        assertEquals(0, ring.read(out))
    }

    @Test
    fun wrapAroundAcrossTheEndOfTheBackingArray() {
        val ring = PcmRingBuffer(8)
        ring.write(seq(0, 6))
        ring.skip(5) // read position 5, one unread sample
        assertEquals(7, ring.write(seq(6, 7))) // fills slots 6,7 then wraps to 0..4
        assertEquals(8, ring.available)
        val out = FloatArray(8)
        assertEquals(8, ring.read(out))
        assertContentEquals(seq(5, 8), out)
        assertEquals(13L, ring.writePosition)
        assertEquals(13L, ring.readPosition)
    }

    @Test
    fun repeatedWrapAroundOverManyCycles() {
        val ring = PcmRingBuffer(7)
        val out = FloatArray(5)
        var next = 0
        var expected = 0
        repeat(1_000) {
            ring.write(seq(next, 5)); next += 5
            assertEquals(5, ring.read(out))
            assertContentEquals(seq(expected, 5), out); expected += 5
        }
    }

    @Test
    fun dropOldestOverwritesUnreadHead() {
        val ring = PcmRingBuffer(4, OverflowPolicy.DROP_OLDEST)
        ring.write(seq(0, 3))
        assertEquals(3, ring.write(seq(3, 3))) // 6 offered, capacity 4 -> samples 0,1 dropped
        assertEquals(2L, ring.droppedSamples)
        assertEquals(4, ring.available)
        val out = FloatArray(4)
        ring.read(out)
        assertContentEquals(seq(2, 4), out)
    }

    @Test
    fun dropOldestOversizedWriteKeepsOnlyTheNewestCapacity() {
        val ring = PcmRingBuffer(4, OverflowPolicy.DROP_OLDEST)
        ring.write(seq(100, 2))
        assertEquals(10, ring.write(seq(0, 10)))
        assertEquals(8L, ring.droppedSamples) // 6 from the oversized input + 2 unread
        val out = FloatArray(4)
        assertEquals(4, ring.read(out))
        assertContentEquals(seq(6, 4), out)
    }

    @Test
    fun dropNewestRejectsWhatDoesNotFit() {
        val ring = PcmRingBuffer(4, OverflowPolicy.DROP_NEWEST)
        assertEquals(3, ring.write(seq(0, 3)))
        assertEquals(1, ring.write(seq(3, 3)))
        assertEquals(2L, ring.droppedSamples)
        assertEquals(0, ring.write(seq(6, 1)))
        assertEquals(3L, ring.droppedSamples)
        val out = FloatArray(4)
        ring.read(out)
        assertContentEquals(seq(0, 4), out)
    }

    @Test
    fun peekAndSkipGiveSlidingWindows() {
        val ring = PcmRingBuffer(16)
        ring.write(seq(0, 10))
        val window = FloatArray(4)
        val windows = mutableListOf<List<Float>>()
        while (ring.peek(window) == window.size) {
            windows += window.toList()
            ring.skip(2) // hop
        }
        assertEquals(listOf(0f, 1f, 2f, 3f), windows.first())
        assertEquals(listOf(6f, 7f, 8f, 9f), windows.last())
        assertEquals(4, windows.size)
        assertEquals(2, ring.available)
    }

    @Test
    fun peekWithSkipOffsetsIntoUnreadDataWithoutConsuming() {
        val ring = PcmRingBuffer(8)
        ring.write(seq(0, 6))
        val out = FloatArray(10) { -1f }
        assertEquals(3, ring.peek(out, offset = 2, length = 3, skip = 2))
        assertContentEquals(floatArrayOf(-1f, -1f, 2f, 3f, 4f, -1f), out.copyOf(6))
        assertEquals(6, ring.available)
        assertEquals(0, ring.peek(out, skip = 6))
        assertEquals(0, ring.peek(out, skip = 100))
        assertEquals(2, ring.peek(out, length = 5, skip = 4))
    }

    @Test
    fun peekLatestReturnsTheNewestSamplesAcrossTheWrap() {
        val ring = PcmRingBuffer(5)
        ring.write(seq(0, 4))
        ring.skip(3)
        ring.write(seq(4, 4)) // unread 3..7, physically wrapped
        val out = FloatArray(3)
        assertEquals(3, ring.peekLatest(out))
        assertContentEquals(seq(5, 3), out)
        val all = FloatArray(10)
        assertEquals(5, ring.peekLatest(all))
        assertContentEquals(seq(3, 5), all.copyOf(5))
        assertEquals(5, ring.available)
    }

    @Test
    fun clearDiscardsEverythingUnread() {
        val ring = PcmRingBuffer(8)
        ring.write(seq(0, 5))
        assertEquals(5, ring.clear())
        assertEquals(0, ring.available)
        assertEquals(0, ring.clear())
        ring.write(seq(5, 2))
        val out = FloatArray(2)
        ring.read(out)
        assertContentEquals(seq(5, 2), out)
    }

    @Test
    fun skipIsBoundedByAvailable() {
        val ring = PcmRingBuffer(8)
        ring.write(seq(0, 3))
        assertEquals(0, ring.skip(0))
        assertEquals(3, ring.skip(10))
        assertEquals(0, ring.skip(1))
    }

    @Test
    fun zeroLengthOperationsAreNoOps() {
        val ring = PcmRingBuffer(4)
        assertEquals(0, ring.write(FloatArray(0)))
        assertEquals(0, ring.read(FloatArray(0)))
        assertEquals(0, ring.peek(FloatArray(0)))
        assertEquals(0, ring.peekLatest(FloatArray(0)))
        assertEquals(0L, ring.writePosition)
    }

    @Test
    fun writeHonoursOffsetAndLength() {
        val ring = PcmRingBuffer(8)
        ring.write(seq(0, 10), offset = 4, length = 3)
        val out = FloatArray(3)
        ring.read(out)
        assertContentEquals(seq(4, 3), out)
    }

    @Test
    fun rejectsInvalidArguments() {
        assertFailsWith<IllegalArgumentException> { PcmRingBuffer(0) }
        assertFailsWith<IllegalArgumentException> { PcmRingBuffer(-1) }
        val ring = PcmRingBuffer(4)
        assertFailsWith<IndexOutOfBoundsException> { ring.write(FloatArray(2), offset = 1, length = 2) }
        assertFailsWith<IndexOutOfBoundsException> { ring.write(FloatArray(2), offset = -1, length = 1) }
        assertFailsWith<IndexOutOfBoundsException> { ring.read(FloatArray(2), offset = 0, length = 3) }
        assertFailsWith<IndexOutOfBoundsException> { ring.peek(FloatArray(2), offset = 3, length = 0) }
        assertFailsWith<IndexOutOfBoundsException> { ring.peekLatest(FloatArray(2), length = -1) }
        assertFailsWith<IllegalArgumentException> { ring.peek(FloatArray(2), skip = -1) }
        assertFailsWith<IllegalArgumentException> { ring.skip(-1) }
    }

    @Test
    fun concurrentProducerConsumerIsLosslessWithDropNewest() = runTest(timeout = 60.seconds) {
        val total = 1_000_000
        val ring = PcmRingBuffer(1_024, OverflowPolicy.DROP_NEWEST)
        withContext(Dispatchers.Default) {
            coroutineScope {
                val producer = async {
                    val rnd = Random(1)
                    val chunk = FloatArray(700)
                    var next = 0
                    while (next < total) {
                        val n = minOf(rnd.nextInt(1, chunk.size + 1), total - next)
                        for (i in 0 until n) chunk[i] = (next + i).toFloat()
                        var done = 0
                        while (done < n) {
                            val w = ring.write(chunk, done, n - done)
                            done += w
                            if (w == 0) yield()
                        }
                        next += n
                    }
                }
                val consumer = async {
                    val rnd = Random(2)
                    val out = FloatArray(500)
                    var expected = 0
                    while (expected < total) {
                        val n = ring.read(out, 0, rnd.nextInt(1, out.size + 1))
                        if (n == 0) yield()
                        for (i in 0 until n) {
                            check(out[i] == expected.toFloat()) { "sample $expected read as ${out[i]}" }
                            expected++
                        }
                    }
                    expected
                }
                producer.await()
                assertEquals(total, consumer.await())
            }
        }
        // Rejected attempts are counted as drops, but every sample was eventually re-offered and delivered.
        assertEquals(total.toLong(), ring.writePosition)
        assertEquals(total.toLong(), ring.readPosition)
    }

    @Test
    fun concurrentDropOldestNeverDeliversTornOrReorderedData() = runTest(timeout = 60.seconds) {
        val total = 1_000_000
        val ring = PcmRingBuffer(256, OverflowPolicy.DROP_OLDEST)
        var delivered = 0L
        withContext(Dispatchers.Default) {
            coroutineScope {
                val producer = async {
                    val chunk = FloatArray(97)
                    var next = 0
                    while (next < total) {
                        val n = minOf(chunk.size, total - next)
                        for (i in 0 until n) chunk[i] = (next + i).toFloat()
                        ring.write(chunk, 0, n)
                        next += n
                    }
                }
                val consumer = async {
                    val out = FloatArray(64)
                    val window = FloatArray(32)
                    var last = -1f
                    var count = 0L
                    while (true) {
                        val done = producer.isCompleted
                        // Non-consuming peeks must always see one contiguous run too.
                        val p = ring.peek(window)
                        for (i in 1 until p) check(window[i] == window[i - 1] + 1f) { "torn peek" }
                        val n = ring.read(out)
                        if (n == 0) {
                            if (done) break
                            yield()
                            continue
                        }
                        check(out[0] > last) { "reordered: ${out[0]} after $last" }
                        for (i in 1 until n) check(out[i] == out[i - 1] + 1f) { "torn read at $i: ${out.toList()}" }
                        last = out[n - 1]
                        count += n
                    }
                    assertEquals((total - 1).toFloat(), last)
                    count
                }
                producer.await()
                delivered = consumer.await()
            }
        }
        assertEquals(total.toLong(), delivered + ring.droppedSamples)
        assertTrue(ring.droppedSamples >= 0)
    }
}
