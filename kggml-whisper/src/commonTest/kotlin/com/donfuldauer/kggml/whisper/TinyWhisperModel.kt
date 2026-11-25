package com.donfuldauer.kggml.whisper

/**
 * Synthetic whisper ggml model for integration tests — a byte-for-byte Kotlin port of
 * `kggml-whisper/native/tests/tiny_model.cpp` (MIT, Kggml; documented in `NOTICE`).
 *
 * Real whisper.cpp hyper-parameter layout (English vocabulary, 80 mel bins, 1500 audio positions) with a 16-wide
 * state and one encoder / decoder layer and deterministic pseudo-random weights: ~1.7 MB, generated at test time,
 * never committed. Its output text is meaningless; tests use it to drive the real whisper.cpp encode / decode path.
 */
internal object TinyWhisperModel {
    private const val VOCAB = 51864
    private const val AUDIO_CTX = 1500
    private const val STATE = 16
    private const val HEADS = 1
    private const val LAYERS = 1
    private const val TEXT_CTX = 64
    private const val MELS = 80
    private const val FFT = 201
    private const val GGML_MAGIC = 0x67676d6c

    /** The model file contents. */
    fun bytes(): ByteArray {
        val w = Writer()
        w.i32(GGML_MAGIC)
        for (v in intArrayOf(VOCAB, AUDIO_CTX, STATE, HEADS, LAYERS, TEXT_CTX, STATE, HEADS, LAYERS, MELS, 1)) w.i32(v)
        w.i32(MELS)
        w.i32(FFT)
        val filters = FloatArray(MELS * FFT)
        for (m in 0 until MELS) filters[m * FFT + m * 2 + 1] = 1f
        filters.forEach(w::f32)
        w.vocab(256)

        val s = STATE
        w.tensor("encoder.positional_embedding", intArrayOf(s, AUDIO_CTX))
        w.matrix("encoder.conv1.weight", intArrayOf(3, MELS, s))
        w.tensor("encoder.conv1.bias", intArrayOf(1, s))
        w.matrix("encoder.conv2.weight", intArrayOf(3, s, s))
        w.tensor("encoder.conv2.bias", intArrayOf(1, s))
        w.tensor("encoder.ln_post.weight", intArrayOf(s), ones = true)
        w.tensor("encoder.ln_post.bias", intArrayOf(s))
        for (i in 0 until LAYERS) block(w, "encoder.blocks.$i.", cross = false)
        w.tensor("decoder.positional_embedding", intArrayOf(s, TEXT_CTX))
        w.matrix("decoder.token_embedding.weight", intArrayOf(s, VOCAB))
        w.tensor("decoder.ln.weight", intArrayOf(s), ones = true)
        w.tensor("decoder.ln.bias", intArrayOf(s))
        for (i in 0 until LAYERS) block(w, "decoder.blocks.$i.", cross = true)
        return w.toByteArray()
    }

    private fun block(w: Writer, p: String, cross: Boolean) {
        val s = STATE
        w.tensor("${p}mlp_ln.weight", intArrayOf(s), ones = true)
        w.tensor("${p}mlp_ln.bias", intArrayOf(s))
        w.matrix("${p}mlp.0.weight", intArrayOf(s, 4 * s))
        w.tensor("${p}mlp.0.bias", intArrayOf(4 * s))
        w.matrix("${p}mlp.2.weight", intArrayOf(4 * s, s))
        w.tensor("${p}mlp.2.bias", intArrayOf(s))
        w.tensor("${p}attn_ln.weight", intArrayOf(s), ones = true)
        w.tensor("${p}attn_ln.bias", intArrayOf(s))
        w.matrix("${p}attn.query.weight", intArrayOf(s, s))
        w.tensor("${p}attn.query.bias", intArrayOf(s))
        w.matrix("${p}attn.key.weight", intArrayOf(s, s))
        w.matrix("${p}attn.value.weight", intArrayOf(s, s))
        w.tensor("${p}attn.value.bias", intArrayOf(s))
        w.matrix("${p}attn.out.weight", intArrayOf(s, s))
        w.tensor("${p}attn.out.bias", intArrayOf(s))
        if (cross) {
            w.tensor("${p}cross_attn_ln.weight", intArrayOf(s), ones = true)
            w.tensor("${p}cross_attn_ln.bias", intArrayOf(s))
            w.matrix("${p}cross_attn.query.weight", intArrayOf(s, s))
            w.tensor("${p}cross_attn.query.bias", intArrayOf(s))
            w.matrix("${p}cross_attn.key.weight", intArrayOf(s, s))
            w.matrix("${p}cross_attn.value.weight", intArrayOf(s, s))
            w.tensor("${p}cross_attn.value.bias", intArrayOf(s))
            w.matrix("${p}cross_attn.out.weight", intArrayOf(s, s))
            w.tensor("${p}cross_attn.out.bias", intArrayOf(s))
        }
    }

    /** IEEE-754 binary32 -> binary16, round-to-nearest-even, flushing tiny values to zero (as tiny_model.cpp). */
    internal fun toF16(f: Float): Int {
        val x = f.toRawBits()
        val sign = (x ushr 16) and 0x8000
        val exp = ((x ushr 23) and 0xff) - 127 + 15
        val mant = x and 0x7fffff
        if (exp <= 0) return sign
        if (exp >= 31) return sign or 0x7c00
        var h = sign or (exp shl 10) or (mant ushr 13)
        val rest = mant and 0x1fff
        if (rest > 0x1000 || (rest == 0x1000 && (h and 1) != 0)) h++
        return h and 0xffff
    }

    private class Writer {
        private var buf = ByteArray(2 shl 20)
        private var size = 0
        private var state: UInt = 12345u

        private fun ensure(n: Int) {
            if (size + n > buf.size) buf = buf.copyOf(maxOf(buf.size * 2, size + n))
        }

        fun i32(v: Int) {
            ensure(4)
            buf[size++] = v.toByte()
            buf[size++] = (v ushr 8).toByte()
            buf[size++] = (v ushr 16).toByte()
            buf[size++] = (v ushr 24).toByte()
        }

        fun f32(v: Float) = i32(v.toRawBits())

        fun u16(v: Int) {
            ensure(2)
            buf[size++] = v.toByte()
            buf[size++] = (v ushr 8).toByte()
        }

        fun bytes(b: ByteArray) {
            ensure(b.size)
            b.copyInto(buf, size)
            size += b.size
        }

        fun tensor(name: String, ne: IntArray, ones: Boolean = false, f16: Boolean = false) {
            val scale = 0.05f
            val nameBytes = name.encodeToByteArray()
            i32(ne.size)
            i32(nameBytes.size)
            i32(if (f16) 1 else 0)
            var n = 1
            for (d in ne) {
                i32(d)
                n *= d
            }
            bytes(nameBytes)
            repeat(n) {
                state = state * 1664525u + 1013904223u
                val x = if (ones) 1f else scale * ((state shr 8).toFloat() / 8388608f - 1f)
                if (f16) u16(toF16(x)) else f32(x)
            }
        }

        fun matrix(name: String, ne: IntArray) = tensor(name, ne, ones = false, f16 = true)

        fun vocab(n: Int) {
            i32(n)
            for (id in 0 until n) {
                val word = (if (id == 220) " " else "tok$id").encodeToByteArray()
                i32(word.size)
                bytes(word)
            }
        }

        fun toByteArray(): ByteArray = buf.copyOf(size)
    }
}
