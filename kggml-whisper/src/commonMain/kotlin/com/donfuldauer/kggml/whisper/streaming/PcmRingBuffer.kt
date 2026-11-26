package com.donfuldauer.kggml.whisper.streaming

import kotlin.concurrent.atomics.AtomicLong

/** What [PcmRingBuffer.write] does when the samples being written do not fit in the free space. */
public enum class OverflowPolicy {
    /**
     * Keep the newest audio: unread samples at the head of the buffer are discarded to make room (live
     * capture where a stalled consumer should resume with the latest audio). Every sample offered is accepted.
     */
    DROP_OLDEST,

    /**
     * Keep the oldest audio: only as many samples as fit are written and the rest of the call's input is
     * discarded (lossless prefix, the producer can see the shortfall from [PcmRingBuffer.write]'s result).
     */
    DROP_NEWEST,
}

/**
 * Fixed-capacity, lock-free circular buffer of mono Float32 PCM samples (16 kHz in the Whisper pipeline).
 *
 * **Threading:** one producer thread calls [write]; one consumer thread calls [read], [peek], [peekLatest],
 * [skip] and [clear]. The producer and the consumer may run concurrently without external locking; multiple
 * producers or multiple consumers must be serialized by the caller. Read-only properties may be read from
 * any thread (they are consistent snapshots of monotonically increasing positions).
 *
 * **Allocation:** the backing array is allocated once; no operation allocates afterwards.
 *
 * Positions are 64-bit, monotonically increasing sample counters (they never wrap in practice: 2^63 samples
 * at 16 kHz is ~18 million years), so a changed read position can always be detected (no ABA).
 *
 * ### Concurrency protocol
 * The producer publishes samples by storing [writePosition] after copying them. Under
 * [OverflowPolicy.DROP_OLDEST] it first advances [readPosition] past the samples it is about to overwrite
 * (with a compare-and-set, so it never moves the read position backwards), and only then overwrites them.
 * The consumer copies samples starting at a read position it loaded and commits (or, for peeks, validates)
 * with a compare-and-set on that same read position: success proves the producer had not yet claimed — and
 * therefore had not yet overwritten — any of the copied slots; failure means the copy may be torn and is
 * retried against the new read position.
 *
 * @property capacity maximum number of samples the buffer holds.
 * @property overflowPolicy what [write] does when the buffer is full.
 */
public class PcmRingBuffer(
    public val capacity: Int,
    public val overflowPolicy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
) {
    init {
        require(capacity > 0) { "capacity must be > 0, was $capacity" }
    }

    private val samples = FloatArray(capacity)
    private val readPos = AtomicLong(0L)
    private val writePos = AtomicLong(0L)
    private val dropped = AtomicLong(0L)

    /** Total samples ever accepted into the buffer (including ones later discarded by [OverflowPolicy.DROP_OLDEST]). */
    public val writePosition: Long
        get() = writePos.load()

    /** Total samples ever consumed or discarded from the head of the buffer. */
    public val readPosition: Long
        get() = readPos.load()

    /** Samples currently readable. */
    public val available: Int
        get() {
            while (true) {
                val r = readPos.load()
                val w = writePos.load()
                val n = w - r
                if (n in 0..capacity) return n.toInt()
                // Raced with a DROP_OLDEST claim between the two loads: retry for a consistent pair.
            }
        }

    /** Free space for [write] before [overflowPolicy] applies. */
    public val remainingCapacity: Int
        get() = capacity - available

    /** Total samples discarded by the [overflowPolicy] (overwritten unread, or rejected on write). */
    public val droppedSamples: Long
        get() = dropped.load()

    /**
     * Producer side: appends `src[offset until offset + length]`.
     *
     * @return the number of samples from `src` that were written. With [OverflowPolicy.DROP_OLDEST] this is
     *   always [length] (only the last [capacity] samples of an oversized write are actually retained, and the
     *   rest are counted in [droppedSamples]); with [OverflowPolicy.DROP_NEWEST] it can be smaller.
     */
    public fun write(src: FloatArray, offset: Int = 0, length: Int = src.size - offset): Int {
        checkRange(src.size, offset, length)
        if (length == 0) return 0
        val w = writePos.load() // only the producer stores writePos
        var from = offset
        var count = length
        when (overflowPolicy) {
            OverflowPolicy.DROP_NEWEST -> {
                val free = capacity - (w - readPos.load()).toInt()
                if (count > free) {
                    dropped.addAndFetch((count - free).toLong())
                    count = free
                }
                if (count == 0) return 0
            }
            OverflowPolicy.DROP_OLDEST -> {
                if (count > capacity) {
                    // Only the newest `capacity` samples of this call can survive.
                    val skipped = count - capacity
                    dropped.addAndFetch(skipped.toLong())
                    from += skipped
                    count = capacity
                }
                val minRead = w + count - capacity
                while (true) {
                    val r = readPos.load()
                    if (r >= minRead) break
                    if (readPos.compareAndSet(r, minRead)) {
                        dropped.addAndFetch(minRead - r)
                        break
                    }
                }
            }
        }
        val start = (w % capacity).toInt()
        val first = minOf(count, capacity - start)
        src.copyInto(samples, start, from, from + first)
        if (count > first) src.copyInto(samples, 0, from + first, from + count)
        writePos.store(w + count)
        return if (overflowPolicy == OverflowPolicy.DROP_OLDEST) length else count
    }

    /**
     * Consumer side: removes up to [length] of the oldest samples into `dst[offset until offset + length]`.
     *
     * @return the number of samples copied (0 when empty).
     */
    public fun read(dst: FloatArray, offset: Int = 0, length: Int = dst.size - offset): Int {
        checkRange(dst.size, offset, length)
        if (length == 0) return 0
        while (true) {
            val r = readPos.load()
            val w = writePos.load()
            val avail = w - r
            if (avail !in 0..capacity) continue
            val n = minOf(length.toLong(), avail).toInt()
            if (n == 0) return 0
            copyOut(r, dst, offset, n)
            if (readPos.compareAndSet(r, r + n)) return n
        }
    }

    /**
     * Consumer side: copies up to [length] samples starting [skip] samples after the oldest unread sample into
     * `dst[offset until offset + length]` without consuming anything (sliding-window extraction: [peek] a
     * window, then [skip] the hop).
     *
     * @return the number of samples copied.
     */
    public fun peek(dst: FloatArray, offset: Int = 0, length: Int = dst.size - offset, skip: Int = 0): Int {
        checkRange(dst.size, offset, length)
        require(skip >= 0) { "skip must be >= 0, was $skip" }
        if (length == 0) return 0
        while (true) {
            val r = readPos.load()
            val w = writePos.load()
            val avail = w - r
            if (avail !in 0..capacity) continue
            val n = minOf(length.toLong(), avail - skip).toInt()
            if (n <= 0) return 0
            copyOut(r + skip, dst, offset, n)
            // CAS to the same value: validates (with write semantics, i.e. ordered before any later producer
            // claim) that no slot we copied was claimed for overwriting while we copied it.
            if (readPos.compareAndSet(r, r)) return n
        }
    }

    /**
     * Consumer side: copies the newest `min(length, available)` unread samples into `dst[offset ...]`, oldest
     * first, without consuming anything (e.g. the trailing window for a partial transcription).
     *
     * @return the number of samples copied.
     */
    public fun peekLatest(dst: FloatArray, offset: Int = 0, length: Int = dst.size - offset): Int {
        checkRange(dst.size, offset, length)
        if (length == 0) return 0
        while (true) {
            val r = readPos.load()
            val w = writePos.load()
            val avail = w - r
            if (avail !in 0..capacity) continue
            val n = minOf(length.toLong(), avail).toInt()
            if (n == 0) return 0
            copyOut(w - n, dst, offset, n)
            if (readPos.compareAndSet(r, r)) return n
        }
    }

    /** Consumer side: discards up to [count] of the oldest unread samples; returns how many were discarded. */
    public fun skip(count: Int): Int {
        require(count >= 0) { "count must be >= 0, was $count" }
        if (count == 0) return 0
        while (true) {
            val r = readPos.load()
            val w = writePos.load()
            val avail = w - r
            if (avail !in 0..capacity) continue
            val n = minOf(count.toLong(), avail).toInt()
            if (n == 0) return 0
            if (readPos.compareAndSet(r, r + n)) return n
        }
    }

    /** Consumer side: discards every unread sample; returns how many were discarded. */
    public fun clear(): Int {
        while (true) {
            val r = readPos.load()
            val w = writePos.load()
            if (w < r) continue
            if (readPos.compareAndSet(r, w)) return (w - r).toInt()
        }
    }

    private fun copyOut(position: Long, dst: FloatArray, offset: Int, n: Int) {
        val start = (position % capacity).toInt()
        val first = minOf(n, capacity - start)
        samples.copyInto(dst, offset, start, start + first)
        if (n > first) samples.copyInto(dst, offset + first, 0, n - first)
    }

    private fun checkRange(size: Int, offset: Int, length: Int) {
        if (offset < 0 || length < 0 || offset > size - length) {
            throw IndexOutOfBoundsException("offset=$offset, length=$length out of bounds for array of size $size")
        }
    }
}
