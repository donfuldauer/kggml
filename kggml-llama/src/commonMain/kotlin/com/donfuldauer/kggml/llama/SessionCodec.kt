package com.donfuldauer.kggml.llama

import com.donfuldauer.kggml.core.InvalidInputException

/**
 * Binary container of a [LlamaEngine] session: the token history held in the KV cache plus llama.cpp's serialized
 * sequence state (`llama_state_seq_get_data`).
 *
 * Layout (little-endian):
 * ```
 * magic "KGLS" | u32 version (1) | i32 n_vocab | i32 n_embd | i32 n_layer | i32 n_tokens | i32 tokens[n_tokens]
 *   | u32 native_size | u32 crc32(native) | u8 native[native_size]
 * ```
 * The model fingerprint (`n_vocab`, `n_embd`, `n_layer`) rejects sessions saved with a different model, and the
 * CRC-32 rejects damaged payloads, before llama.cpp parses any of it (upstream trusts the sizes inside the payload,
 * so corrupt data could otherwise request huge allocations). It is an integrity check, not authentication: only
 * restore sessions from trusted storage.
 */
internal object SessionCodec {
    private val MAGIC = byteArrayOf('K'.code.toByte(), 'G'.code.toByte(), 'L'.code.toByte(), 'S'.code.toByte())
    const val VERSION: Int = 1
    private const val HEADER_SIZE = 4 + 4 * 5

    /** Identifies the model a session belongs to. */
    data class Fingerprint(val vocabularySize: Int, val embeddingSize: Int, val layerCount: Int)

    /** CRC-32 (IEEE 802.3, as zlib's `crc32`). */
    fun crc32(data: ByteArray): Int {
        var crc = -1
        for (b in data) crc = CRC_TABLE[(crc xor b.toInt()) and 0xFF] xor (crc ushr 8)
        return crc.inv()
    }

    private val CRC_TABLE = IntArray(256) { n ->
        var c = n
        repeat(8) { c = if (c and 1 != 0) (c ushr 1) xor 0xEDB88320.toInt() else c ushr 1 }
        c
    }

    class Session(val fingerprint: Fingerprint, val tokens: IntArray, val nativeState: ByteArray)

    fun encode(fingerprint: Fingerprint, tokens: IntArray, nativeState: ByteArray): ByteArray {
        val size = HEADER_SIZE.toLong() + 4L * tokens.size + 8L + nativeState.size
        if (size > Int.MAX_VALUE - 8) throw IllegalStateException("session state is too large to serialize ($size bytes)")
        val out = ByteArray(size.toInt())
        var p = 0
        fun i32(v: Int) {
            out[p] = v.toByte()
            out[p + 1] = (v ushr 8).toByte()
            out[p + 2] = (v ushr 16).toByte()
            out[p + 3] = (v ushr 24).toByte()
            p += 4
        }
        MAGIC.copyInto(out, 0)
        p = 4
        i32(VERSION)
        i32(fingerprint.vocabularySize)
        i32(fingerprint.embeddingSize)
        i32(fingerprint.layerCount)
        i32(tokens.size)
        tokens.forEach(::i32)
        i32(nativeState.size)
        i32(crc32(nativeState))
        nativeState.copyInto(out, p)
        return out
    }

    /** @throws InvalidInputException if [data] is not a well-formed session container. */
    fun decode(data: ByteArray): Session {
        fun fail(reason: String): Nothing = throw InvalidInputException("Invalid LlamaEngine session data: $reason")
        if (data.size < HEADER_SIZE + 8) fail("too short (${data.size} bytes)")
        for (i in MAGIC.indices) if (data[i] != MAGIC[i]) fail("bad magic")
        var p = 4
        fun i32(): Int {
            val v = (data[p].toInt() and 0xFF) or ((data[p + 1].toInt() and 0xFF) shl 8) or
                ((data[p + 2].toInt() and 0xFF) shl 16) or ((data[p + 3].toInt() and 0xFF) shl 24)
            p += 4
            return v
        }
        val version = i32()
        if (version != VERSION) fail("unsupported version $version")
        val fingerprint = Fingerprint(i32(), i32(), i32())
        val count = i32()
        if (count < 0 || count.toLong() * 4 > data.size - p - 8) fail("token count $count exceeds the data")
        val tokens = IntArray(count) { i32() }
        val nativeSize = i32()
        val crc = i32()
        if (nativeSize < 0 || nativeSize != data.size - p) fail("native state size $nativeSize does not match the data")
        val native = data.copyOfRange(p, data.size)
        if (crc32(native) != crc) fail("KV-cache payload checksum mismatch (damaged data)")
        return Session(fingerprint, tokens, native)
    }
}
