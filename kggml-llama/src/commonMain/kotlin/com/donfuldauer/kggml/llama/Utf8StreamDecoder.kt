package com.donfuldauer.kggml.llama

/**
 * Incremental UTF-8 decoder for token pieces.
 *
 * A token piece may end in the middle of a multi-byte character (byte-fallback tokens, BPE merges). [decode] returns
 * the text of every complete character seen so far and holds back an incomplete trailing sequence (at most 3 bytes)
 * until the bytes that complete it arrive. Malformed input is decoded with U+FFFD replacement characters, never
 * held back indefinitely.
 */
internal class Utf8StreamDecoder {
    private var pending = ByteArray(0)

    /** Number of bytes currently held back. */
    val pendingBytes: Int get() = pending.size

    /** Appends [bytes] and returns the newly completed text (possibly empty). */
    fun decode(bytes: ByteArray): String {
        val data = if (pending.isEmpty()) bytes else pending + bytes
        val cut = completePrefixLength(data)
        pending = data.copyOfRange(cut, data.size)
        return if (cut == 0) "" else data.decodeToString(0, cut)
    }

    /** Returns any held-back bytes as text (with replacement characters) and resets the decoder. */
    fun flush(): String {
        val rest = pending
        pending = ByteArray(0)
        return rest.decodeToString()
    }

    private companion object {
        /** Length of the longest prefix of [data] that does not end inside an incomplete UTF-8 sequence. */
        fun completePrefixLength(data: ByteArray): Int {
            val n = data.size
            // Look back at most 3 bytes for the lead byte of the final sequence.
            var i = n - 1
            val stop = maxOf(0, n - 4)
            while (i >= stop) {
                val b = data[i].toInt() and 0xFF
                if (b and 0xC0 != 0x80) { // not a continuation byte: ASCII or a lead byte
                    val needed = when {
                        b and 0x80 == 0 -> 1
                        b and 0xE0 == 0xC0 -> 2
                        b and 0xF0 == 0xE0 -> 3
                        b and 0xF8 == 0xF0 -> 4
                        else -> 1 // invalid lead byte: decoded as U+FFFD right away
                    }
                    return if (i + needed > n) i else n
                }
                i--
            }
            return n // only continuation bytes in the window: malformed, decode as is
        }
    }
}
