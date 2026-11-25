package com.donfuldauer.kggml.models

/** Little-endian byte writer for synthetic model headers. */
class LeWriter {
    private val out = ArrayList<Byte>()
    val size: Int get() = out.size

    fun u8(v: Int) = apply { out += v.toByte() }
    fun i16(v: Int) = apply { u8(v); u8(v shr 8) }
    fun i32(v: Int) = apply { for (i in 0 until 4) u8(v shr (8 * i)) }
    fun i64(v: Long) = apply { for (i in 0 until 8) u8((v shr (8 * i)).toInt()) }
    fun f32(v: Float) = i32(v.toBits())
    fun f64(v: Double) = i64(v.toBits())
    fun bytes(b: ByteArray) = apply { b.forEach { out += it } }
    fun zeros(n: Int) = apply { repeat(n) { out += 0 } }
    fun padTo(alignment: Int) = apply { while (out.size % alignment != 0) out += 0 }
    fun toByteArray(): ByteArray = out.toByteArray()
}

/** Builds GGUF files: metadata of common types and tensor infos (with optional zero-filled data). */
class GgufBuilder(private val version: Int = 3) {
    private val kvs = ArrayList<(LeWriter) -> Unit>()
    private val tensors = ArrayList<Triple<String, List<Long>, Int>>()
    var alignment = 32

    private fun LeWriter.len(n: Long) = if (version == 1) i32(n.toInt()) else i64(n)
    private fun LeWriter.str(s: String) = apply { val b = s.encodeToByteArray(); len(b.size.toLong()); bytes(b) }

    fun string(key: String, value: String) = apply { kvs += { it.str(key).i32(8).str(value) } }
    fun u32(key: String, value: Int) = apply { kvs += { it.str(key).i32(4).i32(value) } }
    fun i32(key: String, value: Int) = apply { kvs += { it.str(key).i32(5).i32(value) } }
    fun u64(key: String, value: Long) = apply { kvs += { it.str(key).i32(10).i64(value) } }
    fun f32(key: String, value: Float) = apply { kvs += { it.str(key).i32(6).f32(value) } }
    fun bool(key: String, value: Boolean) = apply { kvs += { it.str(key).i32(7).u8(if (value) 1 else 0) } }
    fun misc(key: String) = apply {
        kvs += { it.str("$key.u8").i32(0).u8(200) }
        kvs += { it.str("$key.i8").i32(1).u8(-3) }
        kvs += { it.str("$key.u16").i32(2).i16(60000) }
        kvs += { it.str("$key.i16").i32(3).i16(-2) }
        kvs += { it.str("$key.i64").i32(11).i64(-5) }
        kvs += { it.str("$key.f64").i32(12).f64(2.5) }
    }
    fun stringArray(key: String, values: List<String>) = apply {
        kvs += { w -> w.str(key).i32(9).i32(8).len(values.size.toLong()); values.forEach { w.str(it) } }
    }
    fun intArray(key: String, values: List<Int>) = apply {
        kvs += { w -> w.str(key).i32(9).i32(5).len(values.size.toLong()); values.forEach { w.i32(it) } }
    }
    fun rawKv(key: String, typeId: Int) = apply { kvs += { it.str(key).i32(typeId) } }
    fun tensor(name: String, dims: List<Long>, type: GgmlType) = tensor(name, dims, type.id)
    fun tensor(name: String, dims: List<Long>, typeId: Int) = apply { tensors += Triple(name, dims, typeId) }

    fun build(withData: Boolean = false): ByteArray {
        val w = LeWriter().bytes("GGUF".encodeToByteArray()).i32(version)
        w.len(tensors.size.toLong()).len(kvs.size.toLong())
        kvs.forEach { it(w) }
        var offset = 0L
        val sizes = ArrayList<Long>()
        for ((name, dims, typeId) in tensors) {
            w.str(name).i32(dims.size)
            dims.forEach { w.len(it) }
            w.i32(typeId).i64(offset)
            val size = GgmlType.fromId(typeId)?.byteSize(dims.fold(1L) { a, b -> a * b }) ?: 0L
            sizes += size
            offset += (size + alignment - 1) / alignment * alignment
        }
        if (withData) {
            w.padTo(alignment)
            sizes.forEach { w.zeros(it.toInt()); w.padTo(alignment) }
        }
        return w.toByteArray()
    }
}

/** A small llama-architecture GGUF (2 layers, GQA, 64-token context). */
fun tinyLlamaGguf(version: Int = 3, withData: Boolean = false, vocab: Int = 8): ByteArray = GgufBuilder(version)
    .string("general.architecture", "llama")
    .string("general.name", "tiny")
    .u32("general.file_type", 15)
    .u32("llama.context_length", 64)
    .u32("llama.embedding_length", 64)
    .u32("llama.block_count", 2)
    .u32("llama.attention.head_count", 4)
    .u32("llama.attention.head_count_kv", 2)
    .stringArray("tokenizer.ggml.tokens", List(vocab) { "t$it" })
    .tensor("token_embd.weight", listOf(64, vocab.toLong()), GgmlType.F16)
    .tensor("blk.0.attn_q.weight", listOf(256, 64), GgmlType.Q4_K)
    .tensor("blk.1.attn_q.weight", listOf(256, 64), GgmlType.Q4_K)
    .tensor("output_norm.weight", listOf(64), GgmlType.F32)
    .build(withData)

/** A synthetic whisper.cpp ggml file with tiny-like hyper-parameters but small tensors. */
fun tinyWhisperGgml(
    ftype: Int = 1,
    audioLayers: Int = 4,
    nVocabHparam: Int = 51865,
    vocab: List<String> = listOf("a", "bc", "def"),
    tensors: List<Triple<String, List<Int>, GgmlType>> = listOf(
        Triple("encoder.conv1.weight", listOf(3, 80, 8), GgmlType.F16),
        Triple("decoder.token_embedding.weight", listOf(32, 4), GgmlType.Q8_0),
        Triple("decoder.ln.bias", listOf(16), GgmlType.F32),
    ),
): ByteArray {
    val w = LeWriter().i32(0x67676d6c)
    listOf(nVocabHparam, 1500, 384, 6, audioLayers, 448, 384, 6, 4, 80, ftype).forEach { w.i32(it) }
    w.i32(80).i32(201).zeros(80 * 201 * 4)
    w.i32(vocab.size)
    vocab.forEach { val b = it.encodeToByteArray(); w.i32(b.size).bytes(b) }
    for ((name, dims, type) in tensors) {
        val nameBytes = name.encodeToByteArray()
        w.i32(dims.size).i32(nameBytes.size).i32(type.id)
        dims.forEach { w.i32(it) }
        w.bytes(nameBytes)
        w.zeros(type.byteSize(dims.fold(1L) { a, b -> a * b }).toInt())
    }
    return w.toByteArray()
}

/** A source that reports an unknown size (like a network stream prefix). */
class UnknownSizeSource(private val bytes: ByteArray) : ModelByteSource {
    override val size: Long get() = -1
    override fun read(position: Long, buffer: ByteArray, offset: Int, length: Int): Int =
        ByteArrayModelSource(bytes).read(position, buffer, offset, length)
    override fun close() = Unit
}
