package com.donfuldauer.kggml.models

/**
 * Random-access, read-only bytes of a model file, used by the header parsers.
 *
 * Parsers only read the header and skip tensor data by position, so model weights are never loaded into memory.
 */
public interface ModelByteSource : AutoCloseable {
    /** Total size in bytes, or `-1` if unknown (a stream whose end is only found by reading). */
    public val size: Long

    /**
     * Reads up to [length] bytes starting at file offset [position] into [buffer] at [offset].
     *
     * @return the number of bytes read (`> 0` when [length] > 0), or `-1` if [position] is at or past the end.
     */
    public fun read(position: Long, buffer: ByteArray, offset: Int, length: Int): Int
}

/** A [ModelByteSource] over an in-memory byte array (for example a header prefix or a test fixture). */
public class ByteArrayModelSource(private val bytes: ByteArray) : ModelByteSource {
    override val size: Long get() = bytes.size.toLong()

    override fun read(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
        require(position >= 0) { "position must be >= 0" }
        require(offset >= 0 && length >= 0 && offset + length <= buffer.size) { "invalid buffer range" }
        if (position >= bytes.size) return -1
        val count = minOf(length.toLong(), bytes.size - position).toInt()
        bytes.copyInto(buffer, offset, position.toInt(), position.toInt() + count)
        return count
    }

    override fun close(): Unit = Unit
}

/** Sequential little-endian reader over a [ModelByteSource] with a small read-ahead buffer. */
internal class LittleEndianReader(private val source: ModelByteSource, start: Long = 0, bufferSize: Int = 64 * 1024) {
    private val buffer = ByteArray(bufferSize)
    private var bufferStart = 0L
    private var bufferLength = 0

    /** Absolute offset of the next byte to read. */
    var position: Long = start
        private set

    /** Bytes left until the end of the source, or `null` if its size is unknown. */
    val remaining: Long? get() = if (source.size < 0) null else source.size - position

    fun readU8(): Int = readByteUnchecked("u8").toInt() and 0xFF

    fun readI8(): Int = readByteUnchecked("i8").toInt()

    fun readU16(): Int = readU8() or (readU8() shl 8)

    fun readI16(): Int = readU16().toShort().toInt()

    fun readI32(): Int = readU8() or (readU8() shl 8) or (readU8() shl 16) or (readU8() shl 24)

    fun readU32(): Long = readI32().toLong() and 0xFFFF_FFFFL

    fun readI64(): Long = (readU32()) or (readU32() shl 32)

    /** Reads an unsigned 64-bit value that must fit in a non-negative [Long] (sizes, counts, offsets). */
    fun readU64AsLong(what: String): Long {
        val value = readI64()
        if (value < 0) throw ModelFormatException("$what is too large (${value.toULong()})")
        return value
    }

    fun readF32(): Float = Float.fromBits(readI32())

    fun readF64(): Double = Double.fromBits(readI64())

    fun readBytes(count: Int): ByteArray {
        require(count >= 0)
        ensureAvailable(count.toLong(), "$count-byte field")
        val out = ByteArray(count)
        var written = 0
        while (written < count) {
            fill("$count-byte field")
            val index = (position - bufferStart).toInt()
            val n = minOf(count - written, bufferLength - index)
            buffer.copyInto(out, written, index, index + n)
            written += n
            position += n
        }
        return out
    }

    /** Skips [count] bytes. Fails when the source size is known and fewer than [count] bytes remain. */
    fun skip(count: Long, what: String) {
        if (count < 0) throw ModelFormatException("negative size for $what")
        ensureAvailable(count, what)
        if (source.size < 0) {
            // Unknown size: make sure the skipped range really exists by reading its last byte.
            if (count > 0) {
                position += count - 1
                readByteUnchecked(what)
            }
        } else {
            position += count
        }
    }

    /** Fails early, before any allocation or skipping, if fewer than [count] bytes remain. */
    fun ensureAvailable(count: Long, what: String) {
        val left = remaining ?: return
        if (count > left) throw ModelFormatException("truncated file: $what needs $count bytes at offset $position, $left left")
    }

    private fun readByteUnchecked(what: String): Byte {
        fill(what)
        val b = buffer[(position - bufferStart).toInt()]
        position++
        return b
    }

    private fun fill(what: String) {
        if (position >= bufferStart && position < bufferStart + bufferLength) return
        val n = source.read(position, buffer, 0, buffer.size)
        if (n <= 0) throw ModelFormatException("truncated file: unexpected end of data reading $what at offset $position")
        bufferStart = position
        bufferLength = n
    }
}
