package com.donfuldauer.kggml.llama

/**
 * Synthetic llama-architecture GGUF for integration tests — a byte-for-byte Kotlin port of
 * `kggml-llama/native/tests/tiny_model.cpp` (MIT, Kggml; documented in `NOTICE`).
 *
 * A real llama.cpp "llama" model (RMSNorm, RoPE, SwiGLU) with a 32-wide state, one block, deterministic
 * pseudo-random F32 weights and a 288-token SentencePiece vocabulary: `<unk>`, `<s>`, `</s>`, the 256 byte-fallback
 * tokens, `▁`, `a`..`z`, `▁ab` and `▁z`. ~100 KB, generated at test time, never committed. Its output is
 * meaningless; tests use it to drive the real llama.cpp load / decode / sampling / state paths.
 */
internal object TinyLlamaModel {
    const val VOCAB = 288
    const val EMBD = 32
    const val CTX_TRAIN = 512
    const val BOS = 1
    const val EOS = 2
    const val BYTE_BASE = 3
    const val SPACE = 259
    const val LETTER_BASE = 260
    const val SPACE_AB = 286
    const val SPACE_Z = 287

    private const val HEADS = 2
    private const val HEADS_KV = 2
    private const val FF = 64
    private const val ALIGNMENT = 32

    // gguf_type
    private const val GGUF_UINT32 = 4
    private const val GGUF_INT32 = 5
    private const val GGUF_FLOAT32 = 6
    private const val GGUF_BOOL = 7
    private const val GGUF_STRING = 8
    private const val GGUF_ARRAY = 9

    // llama_token_type
    private const val TYPE_NORMAL = 1
    private const val TYPE_UNKNOWN = 2
    private const val TYPE_CONTROL = 3
    private const val TYPE_BYTE = 6

    /** The model file contents. */
    fun bytes(): ByteArray {
        val kv = GgufWriter()
        var kvCount = 0
        fun key(name: String, type: Int) {
            kv.str(name)
            kv.u32(type)
            kvCount++
        }
        fun str(name: String, value: String) { key(name, GGUF_STRING); kv.str(value) }
        fun u32(name: String, value: Int) { key(name, GGUF_UINT32); kv.u32(value) }
        fun f32(name: String, value: Float) { key(name, GGUF_FLOAT32); kv.f32(value) }
        fun bool(name: String, value: Boolean) { key(name, GGUF_BOOL); kv.u8(if (value) 1 else 0) }

        str("general.architecture", "llama")
        str("general.name", "kggml-tiny-llama")
        u32("general.file_type", 0)
        u32("llama.context_length", CTX_TRAIN)
        u32("llama.embedding_length", EMBD)
        u32("llama.block_count", 1)
        u32("llama.feed_forward_length", FF)
        u32("llama.attention.head_count", HEADS)
        u32("llama.attention.head_count_kv", HEADS_KV)
        f32("llama.attention.layer_norm_rms_epsilon", 1e-5f)
        u32("llama.rope.dimension_count", EMBD / HEADS)
        f32("llama.rope.freq_base", 10000.0f)
        u32("llama.vocab_size", VOCAB)

        val tokens = ArrayList<String>()
        val scores = ArrayList<Float>()
        val types = ArrayList<Int>()
        fun add(piece: String, score: Float, type: Int) {
            tokens += piece
            scores += score
            types += type
        }
        add("<unk>", 0f, TYPE_UNKNOWN)
        add("<s>", 0f, TYPE_CONTROL)
        add("</s>", 0f, TYPE_CONTROL)
        for (b in 0 until 256) add("<0x" + b.toString(16).uppercase().padStart(2, '0') + ">", 0f, TYPE_BYTE)
        add("▁", -1f, TYPE_NORMAL)
        for (c in 'a'..'z') add(c.toString(), -2f, TYPE_NORMAL)
        add("▁ab", -0.5f, TYPE_NORMAL)
        add("▁z", -0.5f, TYPE_NORMAL)
        check(tokens.size == VOCAB)

        str("tokenizer.ggml.model", "llama")
        key("tokenizer.ggml.tokens", GGUF_ARRAY)
        kv.u32(GGUF_STRING); kv.u64(tokens.size.toLong()); tokens.forEach(kv::str)
        key("tokenizer.ggml.scores", GGUF_ARRAY)
        kv.u32(GGUF_FLOAT32); kv.u64(scores.size.toLong()); scores.forEach(kv::f32)
        key("tokenizer.ggml.token_type", GGUF_ARRAY)
        kv.u32(GGUF_INT32); kv.u64(types.size.toLong()); types.forEach(kv::u32)
        u32("tokenizer.ggml.unknown_token_id", 0)
        u32("tokenizer.ggml.bos_token_id", BOS)
        u32("tokenizer.ggml.eos_token_id", EOS)
        bool("tokenizer.ggml.add_bos_token", true)
        bool("tokenizer.ggml.add_eos_token", false)

        // Tensors (F32), created in the same order (and drawing from the same xorshift stream) as tiny_model.cpp.
        val nEmbdKv = EMBD / HEADS * HEADS_KV
        val rng = XorShiftRng(0x9E3779B9.toInt())
        class Tensor(val name: String, val ne: LongArray, val data: FloatArray)
        val tensors = ArrayList<Tensor>()
        fun tensor(name: String, ne0: Int, ne1: Int, norm: Boolean) {
            val ne = if (ne1 > 0) longArrayOf(ne0.toLong(), ne1.toLong()) else longArrayOf(ne0.toLong())
            val n = if (ne1 > 0) ne0 * ne1 else ne0
            tensors += Tensor(name, ne, FloatArray(n) { if (norm) 1f else rng.next(0.5f) })
        }
        tensor("token_embd.weight", EMBD, VOCAB, false)
        tensor("output_norm.weight", EMBD, 0, true)
        tensor("output.weight", EMBD, VOCAB, false)
        tensor("blk.0.attn_norm.weight", EMBD, 0, true)
        tensor("blk.0.attn_q.weight", EMBD, EMBD, false)
        tensor("blk.0.attn_k.weight", EMBD, nEmbdKv, false)
        tensor("blk.0.attn_v.weight", EMBD, nEmbdKv, false)
        tensor("blk.0.attn_output.weight", EMBD, EMBD, false)
        tensor("blk.0.ffn_norm.weight", EMBD, 0, true)
        tensor("blk.0.ffn_gate.weight", EMBD, FF, false)
        tensor("blk.0.ffn_down.weight", FF, EMBD, false)
        tensor("blk.0.ffn_up.weight", EMBD, FF, false)

        val out = GgufWriter()
        out.raw(byteArrayOf('G'.code.toByte(), 'G'.code.toByte(), 'U'.code.toByte(), 'F'.code.toByte()))
        out.u32(3)
        out.u64(tensors.size.toLong())
        out.u64(kvCount.toLong())
        out.raw(kv.toByteArray())
        var offset = 0L
        for (t in tensors) {
            out.str(t.name)
            out.u32(t.ne.size)
            t.ne.forEach(out::u64)
            out.u32(0) // GGML_TYPE_F32
            out.u64(offset)
            offset += pad(t.data.size * 4L)
        }
        out.padTo(ALIGNMENT)
        for (t in tensors) {
            t.data.forEach(out::f32)
            out.padTo(ALIGNMENT)
        }
        return out.toByteArray()
    }

    private fun pad(n: Long): Long = (n + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT
}

/** Deterministic xorshift32 weights in [-scale, scale] (same stream as tiny_model.cpp). */
internal class XorShiftRng(private var s: Int) {
    fun next(scale: Float): Float {
        s = s xor (s shl 13)
        s = s xor (s ushr 17)
        s = s xor (s shl 5)
        return ((s and 0xFFFFFF).toFloat() / 0xFFFFFF.toFloat() * 2.0f - 1.0f) * scale
    }
}

/** Little-endian GGUF byte writer shared by the synthetic fixtures. */
internal class GgufWriter {
    private var buf = ByteArray(1 shl 16)
    var size = 0
        private set

    private fun ensure(n: Int) {
        if (size + n > buf.size) buf = buf.copyOf(maxOf(buf.size * 2, size + n))
    }

    fun u8(v: Int) {
        ensure(1)
        buf[size++] = v.toByte()
    }

    fun u32(v: Int) {
        ensure(4)
        for (i in 0 until 4) buf[size++] = (v ushr (8 * i)).toByte()
    }

    fun u64(v: Long) {
        ensure(8)
        for (i in 0 until 8) buf[size++] = (v ushr (8 * i)).toByte()
    }

    fun f32(v: Float) = u32(v.toRawBits())

    fun raw(bytes: ByteArray) {
        ensure(bytes.size)
        bytes.copyInto(buf, size)
        size += bytes.size
    }

    fun str(s: String) {
        val bytes = s.encodeToByteArray()
        u64(bytes.size.toLong())
        raw(bytes)
    }

    fun padTo(alignment: Int) {
        while (size % alignment != 0) u8(0)
    }

    fun toByteArray(): ByteArray = buf.copyOf(size)
}
