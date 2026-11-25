package com.donfuldauer.kggml.llama

/**
 * Synthetic LLaVA-style ("mlp") multimodal projector GGUF for [TinyLlamaModel] — a byte-for-byte Kotlin port of
 * `write_tiny_mmproj()` in `kggml-llama/native/tests/tiny_model.cpp` (MIT, Kggml; documented in `NOTICE`).
 *
 * A real libmtmd CLIP vision tower: 14x14 patch embedding of 28x28 images (4 patches), learned position
 * embeddings, two pre-norm ViT blocks with a 16-wide state and a 2-layer GELU MLP projector into the text model's
 * 32-wide embedding space, deterministic pseudo-random F32 weights, ~64 KB. Every image therefore becomes
 * [IMAGE_TOKENS] embeddings. Its output is meaningless; tests use it to drive the real libmtmd preprocess / encode
 * / eval paths without shipping projector weights.
 */
internal object TinyMmproj {
    const val IMAGE_SIZE = 28
    const val IMAGE_TOKENS = 4

    private const val V_EMBD = 16
    private const val V_FF = 32
    private const val V_LAYERS = 2
    private const val V_HEADS = 2
    private const val PATCH = 14
    private const val PATCHES = (IMAGE_SIZE / PATCH) * (IMAGE_SIZE / PATCH)
    private const val ALIGNMENT = 32

    // gguf_type
    private const val GGUF_UINT32 = 4
    private const val GGUF_FLOAT32 = 6
    private const val GGUF_BOOL = 7
    private const val GGUF_STRING = 8
    private const val GGUF_ARRAY = 9

    /** The projector file contents. */
    fun bytes(): ByteArray {
        check(PATCHES == IMAGE_TOKENS)
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
        fun f32s(name: String, values: FloatArray) {
            key(name, GGUF_ARRAY)
            kv.u32(GGUF_FLOAT32)
            kv.u64(values.size.toLong())
            values.forEach(kv::f32)
        }

        str("general.architecture", "clip")
        str("general.name", "kggml-tiny-mmproj")
        u32("general.file_type", 0)
        bool("clip.has_vision_encoder", true)
        str("clip.projector_type", "mlp")
        u32("clip.vision.image_size", IMAGE_SIZE)
        u32("clip.vision.patch_size", PATCH)
        u32("clip.vision.embedding_length", V_EMBD)
        u32("clip.vision.feed_forward_length", V_FF)
        u32("clip.vision.block_count", V_LAYERS)
        u32("clip.vision.projection_dim", TinyLlamaModel.EMBD)
        u32("clip.vision.attention.head_count", V_HEADS)
        f32("clip.vision.attention.layer_norm_epsilon", 1e-6f)
        f32s("clip.vision.image_mean", floatArrayOf(0.5f, 0.5f, 0.5f))
        f32s("clip.vision.image_std", floatArrayOf(0.5f, 0.5f, 0.5f))

        // Tensors (F32) in the same order, and drawing from the same xorshift stream, as tiny_model.cpp. A fill of 0
        // means "random"; ggml stores n_dims without trailing 1s.
        val rng = XorShiftRng(0x2545F491)
        class Tensor(val name: String, val ne: LongArray, val data: FloatArray)
        val tensors = ArrayList<Tensor>()
        fun tensor(name: String, ne0: Int, ne1: Int, ne2: Int, ne3: Int, fill: Float) {
            val dims = longArrayOf(ne0.toLong(), ne1.toLong(), ne2.toLong(), ne3.toLong())
            var nDims = 4
            while (nDims > 1 && dims[nDims - 1] == 1L) nDims--
            val n = ne0 * ne1 * ne2 * ne3
            tensors += Tensor(name, dims.copyOf(nDims), FloatArray(n) { if (fill != 0f) fill else rng.next(0.25f) })
        }
        tensor("v.patch_embd.weight", PATCH, PATCH, 3, V_EMBD, 0f)
        tensor("v.position_embd.weight", V_EMBD, PATCHES, 1, 1, 0f)
        for (il in 0 until V_LAYERS) {
            val b = "v.blk.$il."
            tensor(b + "ln1.weight", V_EMBD, 1, 1, 1, 1f)
            tensor(b + "ln1.bias", V_EMBD, 1, 1, 1, 0f)
            for (m in listOf("attn_q", "attn_k", "attn_v", "attn_out")) {
                tensor("$b$m.weight", V_EMBD, V_EMBD, 1, 1, 0f)
                tensor("$b$m.bias", V_EMBD, 1, 1, 1, 0f)
            }
            tensor(b + "ln2.weight", V_EMBD, 1, 1, 1, 1f)
            tensor(b + "ln2.bias", V_EMBD, 1, 1, 1, 0f)
            tensor(b + "ffn_up.weight", V_EMBD, V_FF, 1, 1, 0f)
            tensor(b + "ffn_up.bias", V_FF, 1, 1, 1, 0f)
            tensor(b + "ffn_down.weight", V_FF, V_EMBD, 1, 1, 0f)
            tensor(b + "ffn_down.bias", V_EMBD, 1, 1, 1, 0f)
        }
        tensor("mm.0.weight", V_EMBD, TinyLlamaModel.EMBD, 1, 1, 0f)
        tensor("mm.0.bias", TinyLlamaModel.EMBD, 1, 1, 1, 0f)
        tensor("mm.2.weight", TinyLlamaModel.EMBD, TinyLlamaModel.EMBD, 1, 1, 0f)
        tensor("mm.2.bias", TinyLlamaModel.EMBD, 1, 1, 1, 0f)

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

    /** Deterministic packed RGB888 test pattern (same as `rgb_pattern()` in test_mtmd.cpp). */
    fun rgbPattern(width: Int, height: Int, seed: Int): ByteArray =
        ByteArray(width * height * 3) { i -> ((i * 37 + seed * 101 + (i / 3) * 7) and 0xFF).toByte() }

    /** Encodes packed RGB888 as an uncompressed 24-bit BMP, in memory (same as `bmp_from_rgb()` in test_mtmd.cpp). */
    fun bmp(rgb: ByteArray, width: Int, height: Int): ByteArray {
        val row = (width * 3 + 3) and 3.inv()
        val dataSize = row * height
        val out = ByteArray(54 + dataSize)
        fun put32(at: Int, v: Int) { for (i in 0 until 4) out[at + i] = (v ushr (8 * i)).toByte() }
        fun put16(at: Int, v: Int) { out[at] = v.toByte(); out[at + 1] = (v ushr 8).toByte() }
        out[0] = 'B'.code.toByte()
        out[1] = 'M'.code.toByte()
        put32(2, out.size)
        put32(10, 54)
        put32(14, 40)
        put32(18, width)
        put32(22, height)
        put16(26, 1)
        put16(28, 24)
        put32(34, dataSize)
        for (y in 0 until height) {
            val dst = 54 + (height - 1 - y) * row
            for (x in 0 until width) {
                val src = (y * width + x) * 3
                out[dst + x * 3] = rgb[src + 2]
                out[dst + x * 3 + 1] = rgb[src + 1]
                out[dst + x * 3 + 2] = rgb[src]
            }
        }
        return out
    }

    private fun pad(n: Long): Long = (n + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT
}
