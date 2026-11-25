package com.donfuldauer.kggml.models

/** GGUF metadata value types (`gguf_type`). */
public enum class GgufValueType(public val id: Int) {
    UINT8(0), INT8(1), UINT16(2), INT16(3), UINT32(4), INT32(5), FLOAT32(6), BOOL(7), STRING(8), ARRAY(9),
    UINT64(10), INT64(11), FLOAT64(12),
    ;

    public companion object {
        private val byId = entries.associateBy { it.id }

        /** The type with [id], or `null` if unknown. */
        public fun fromId(id: Int): GgufValueType? = byId[id]
    }
}

/** A GGUF metadata value. */
public sealed interface GgufValue {
    /** The stored value type. */
    public val type: GgufValueType

    /** Any integer type. `UINT64` values above [Long.MAX_VALUE] are stored with their raw bits. */
    public data class Integer(override val type: GgufValueType, val value: Long) : GgufValue

    /** `FLOAT32` or `FLOAT64`. */
    public data class Real(override val type: GgufValueType, val value: Double) : GgufValue

    /** `BOOL`. */
    public data class Bool(val value: Boolean) : GgufValue {
        override val type: GgufValueType get() = GgufValueType.BOOL
    }

    /** `STRING` (UTF-8). */
    public data class Text(val value: String) : GgufValue {
        override val type: GgufValueType get() = GgufValueType.STRING
    }

    /**
     * `ARRAY`. Large arrays (tokenizer vocabularies, merges) are skipped rather than kept in memory: [values] is then
     * `null` and only [elementType] and [size] are known.
     */
    public data class Array(val elementType: GgufValueType, val size: Long, val values: List<GgufValue>?) : GgufValue {
        override val type: GgufValueType get() = GgufValueType.ARRAY
    }
}

/**
 * One tensor descriptor from the GGUF header.
 *
 * @property dimensions `ne[0..n_dims)`.
 * @property typeId the raw `ggml_type`; [type] is `null` if it is unknown to this parser.
 * @property offset offset of the tensor data relative to [GgufModelHeader.dataOffset].
 */
public data class GgufTensorInfo(
    val name: String,
    val dimensions: List<Long>,
    val typeId: Int,
    val offset: Long,
) {
    /** The element type, or `null` if [typeId] is unknown. */
    val type: GgmlType? get() = GgmlType.fromId(typeId)

    /** Number of elements (product of [dimensions]). */
    val elementCount: Long get() = dimensions.fold(1L) { acc, d -> checkedMultiply(acc, d) }

    /** Data size in bytes, or `null` if [type] is unknown. */
    val byteSize: Long? get() = type?.byteSize(elementCount)
}

/**
 * Parsed GGUF header: metadata key/values and tensor descriptors. Tensor data is never read.
 *
 * @property version GGUF version (1, 2 or 3).
 * @property alignment tensor data alignment (`general.alignment`, default 32).
 * @property dataOffset absolute file offset at which tensor data starts.
 */
public class GgufModelHeader internal constructor(
    public val version: Int,
    public val metadata: Map<String, GgufValue>,
    public val tensors: List<GgufTensorInfo>,
    public val alignment: Int,
    public val dataOffset: Long,
) {
    /** `general.architecture` (for example `llama`, `qwen2`, `gemma3`). */
    public val architecture: String? get() = string("general.architecture")

    /** `general.name`. */
    public val name: String? get() = string("general.name")

    /** `general.file_type` (`llama_ftype`), or `null` if absent. */
    public val fileTypeId: Int? get() = long("general.file_type")?.toInt()

    /** Total number of parameters (sum of tensor element counts). */
    public val parameterCount: Long get() = tensors.fold(0L) { acc, t -> checkedAdd(acc, t.elementCount) }

    /** Bytes of tensor data, from the tensor types and shapes. Tensors of unknown type count as 0. */
    public val tensorDataBytes: Long get() = tensors.fold(0L) { acc, t -> checkedAdd(acc, t.byteSize ?: 0L) }

    /** The tensor type that holds the most bytes (for example `Q4_K` in a `Q4_K_M` file). */
    public val dominantTensorType: GgmlType?
        get() = tensors.filter { it.type != null }
            .groupBy { it.type!! }
            .maxByOrNull { (_, list) -> list.sumOf { it.byteSize ?: 0L } }?.key

    /**
     * Human-readable quantization label: the `llama_ftype` name (for example `Q4_K_M`) from `general.file_type`, or
     * the [dominantTensorType] name if that key is absent or unknown.
     */
    public val quantization: String? get() = fileTypeId?.let { llamaFileTypeName(it) } ?: dominantTensorType?.name

    /** Training context window (`<arch>.context_length`). */
    public val contextLength: Long? get() = archLong("context_length")

    /** Transformer block count (`<arch>.block_count`). */
    public val blockCount: Long? get() = archLong("block_count")

    /** Embedding width (`<arch>.embedding_length`). */
    public val embeddingLength: Long? get() = archLong("embedding_length")

    /** Attention head count (`<arch>.attention.head_count`; the maximum if it is a per-layer array). */
    public val headCount: Long? get() = archLong("attention.head_count")

    /** KV head count (`<arch>.attention.head_count_kv`), defaulting to [headCount] (no GQA). */
    public val headCountKv: Long? get() = archLong("attention.head_count_kv") ?: headCount

    /** Key head size (`<arch>.attention.key_length`), defaulting to `embeddingLength / headCount`. */
    public val keyLength: Long? get() = archLong("attention.key_length") ?: defaultHeadSize()

    /** Value head size (`<arch>.attention.value_length`), defaulting to `embeddingLength / headCount`. */
    public val valueLength: Long? get() = archLong("attention.value_length") ?: defaultHeadSize()

    /** Vocabulary size (`<arch>.vocab_size`, else the length of `tokenizer.ggml.tokens`). */
    public val vocabularySize: Long?
        get() = archLong("vocab_size") ?: (metadata["tokenizer.ggml.tokens"] as? GgufValue.Array)?.size

    /** String value of [key], or `null`. */
    public fun string(key: String): String? = (metadata[key] as? GgufValue.Text)?.value

    /**
     * Integer value of [key], or `null`. For an array of integers (per-layer hyper-parameters) the maximum is
     * returned.
     */
    public fun long(key: String): Long? = when (val v = metadata[key]) {
        is GgufValue.Integer -> v.value
        is GgufValue.Array -> v.values?.filterIsInstance<GgufValue.Integer>()?.maxOfOrNull { it.value }
        else -> null
    }

    private fun archLong(suffix: String): Long? = architecture?.let { long("$it.$suffix") }

    private fun defaultHeadSize(): Long? {
        val embd = embeddingLength ?: return null
        val heads = headCount ?: return null
        return if (heads > 0) embd / heads else null
    }

    public companion object {
        /** `"GGUF"` read as a little-endian `uint32`. */
        public const val MAGIC: Long = 0x46554747L

        /** Arrays with more elements than this are skipped instead of kept in [GgufValue.Array.values]. */
        public const val MAX_RETAINED_ARRAY_ELEMENTS: Int = 1024

        private const val MAX_STRING_BYTES = 64L * 1024 * 1024
        private const val MAX_COUNT = 1L shl 31
        private const val MAX_DIMS = 4
        private const val DEFAULT_ALIGNMENT = 32

        /**
         * Parses the GGUF header from [source]. Only the header is read; tensor data is not touched.
         *
         * @throws ModelFormatException on a wrong magic, unsupported version, unknown value type, or truncated or
         *   inconsistent header.
         */
        public fun parse(source: ModelByteSource): GgufModelHeader {
            val r = LittleEndianReader(source)
            val magic = r.readU32()
            if (magic != MAGIC) throw ModelFormatException("not a GGUF file (magic 0x${magic.toString(16)})")
            val version = r.readI32()
            if (version !in 1..3) throw ModelFormatException("unsupported GGUF version $version")
            val v1 = version == 1
            fun count(what: String): Long {
                val n = if (v1) r.readU32() else r.readU64AsLong(what)
                if (n > MAX_COUNT) throw ModelFormatException("$what $n is implausibly large")
                return n
            }
            fun string(what: String): String {
                val len = if (v1) r.readU32() else r.readU64AsLong("$what length")
                if (len > MAX_STRING_BYTES) throw ModelFormatException("$what length $len is implausibly large")
                return r.readBytes(len.toInt()).decodeToString()
            }
            fun skipString(what: String) {
                val len = if (v1) r.readU32() else r.readU64AsLong("$what length")
                if (len > MAX_STRING_BYTES) throw ModelFormatException("$what length $len is implausibly large")
                r.skip(len, what)
            }
            fun valueType(what: String): GgufValueType {
                val id = r.readI32()
                return GgufValueType.fromId(id) ?: throw ModelFormatException("unknown GGUF value type $id for $what")
            }
            fun scalar(type: GgufValueType, key: String): GgufValue = when (type) {
                GgufValueType.UINT8 -> GgufValue.Integer(type, r.readU8().toLong())
                GgufValueType.INT8 -> GgufValue.Integer(type, r.readI8().toLong())
                GgufValueType.UINT16 -> GgufValue.Integer(type, r.readU16().toLong())
                GgufValueType.INT16 -> GgufValue.Integer(type, r.readI16().toLong())
                GgufValueType.UINT32 -> GgufValue.Integer(type, r.readU32())
                GgufValueType.INT32 -> GgufValue.Integer(type, r.readI32().toLong())
                GgufValueType.UINT64, GgufValueType.INT64 -> GgufValue.Integer(type, r.readI64())
                GgufValueType.FLOAT32 -> GgufValue.Real(type, r.readF32().toDouble())
                GgufValueType.FLOAT64 -> GgufValue.Real(type, r.readF64())
                GgufValueType.BOOL -> GgufValue.Bool(r.readU8() != 0)
                GgufValueType.STRING -> GgufValue.Text(string("value of $key"))
                GgufValueType.ARRAY -> throw ModelFormatException("nested GGUF arrays are not supported ($key)")
            }
            fun scalarSize(type: GgufValueType): Long = when (type) {
                GgufValueType.UINT8, GgufValueType.INT8, GgufValueType.BOOL -> 1
                GgufValueType.UINT16, GgufValueType.INT16 -> 2
                GgufValueType.UINT32, GgufValueType.INT32, GgufValueType.FLOAT32 -> 4
                GgufValueType.UINT64, GgufValueType.INT64, GgufValueType.FLOAT64 -> 8
                GgufValueType.STRING, GgufValueType.ARRAY -> -1
            }

            val tensorCount = count("tensor count")
            val kvCount = count("metadata count")
            val metadata = LinkedHashMap<String, GgufValue>()
            for (i in 0 until kvCount) {
                val key = string("metadata key #$i")
                val type = valueType(key)
                val value = if (type == GgufValueType.ARRAY) {
                    val elementType = valueType("$key[]")
                    if (elementType == GgufValueType.ARRAY) {
                        throw ModelFormatException("nested GGUF arrays are not supported ($key)")
                    }
                    val n = count("$key array length")
                    if (n <= MAX_RETAINED_ARRAY_ELEMENTS) {
                        GgufValue.Array(elementType, n, List(n.toInt()) { scalar(elementType, key) })
                    } else {
                        val width = scalarSize(elementType)
                        if (width > 0) {
                            r.skip(checkedMultiply(n, width), "$key array")
                        } else {
                            for (j in 0 until n) skipString("$key[$j]")
                        }
                        GgufValue.Array(elementType, n, null)
                    }
                } else {
                    scalar(type, key)
                }
                if (metadata.put(key, value) != null) throw ModelFormatException("duplicate GGUF metadata key $key")
            }

            val alignment = when (val a = metadata["general.alignment"]) {
                null -> DEFAULT_ALIGNMENT
                is GgufValue.Integer -> a.value.toInt()
                else -> throw ModelFormatException("general.alignment must be an integer")
            }
            if (alignment <= 0 || alignment and (alignment - 1) != 0) {
                throw ModelFormatException("general.alignment $alignment is not a power of two")
            }

            val tensors = ArrayList<GgufTensorInfo>(minOf(tensorCount, 65_536L).toInt())
            for (i in 0 until tensorCount) {
                val name = string("tensor name #$i")
                val nDims = r.readU32()
                if (nDims < 1 || nDims > MAX_DIMS) throw ModelFormatException("tensor $name has $nDims dimensions")
                val dims = List(nDims.toInt()) {
                    val d = if (v1) r.readU32() else r.readU64AsLong("tensor $name dimension")
                    d
                }
                val typeId = r.readI32()
                val offset = r.readU64AsLong("tensor $name offset")
                if (offset % alignment != 0L) throw ModelFormatException("tensor $name offset $offset is not aligned")
                val info = GgufTensorInfo(name, dims, typeId, offset)
                info.elementCount // validates overflow
                info.byteSize // validates block multiple
                tensors += info
            }
            val dataOffset = (r.position + alignment - 1) / alignment * alignment
            return GgufModelHeader(version, metadata, tensors, alignment, dataOffset)
        }
    }
}

/** `llama_ftype` names from the pinned `llama.h`, or `null` for unknown values. */
internal fun llamaFileTypeName(id: Int): String? = when (id) {
    0 -> "F32"
    1 -> "F16"
    2 -> "Q4_0"
    3 -> "Q4_1"
    7 -> "Q8_0"
    8 -> "Q5_0"
    9 -> "Q5_1"
    10 -> "Q2_K"
    11 -> "Q3_K_S"
    12 -> "Q3_K_M"
    13 -> "Q3_K_L"
    14 -> "Q4_K_S"
    15 -> "Q4_K_M"
    16 -> "Q5_K_S"
    17 -> "Q5_K_M"
    18 -> "Q6_K"
    19 -> "IQ2_XXS"
    20 -> "IQ2_XS"
    21 -> "Q2_K_S"
    22 -> "IQ3_XS"
    23 -> "IQ3_XXS"
    24 -> "IQ1_S"
    25 -> "IQ4_NL"
    26 -> "IQ3_S"
    27 -> "IQ3_M"
    28 -> "IQ2_S"
    29 -> "IQ2_M"
    30 -> "IQ4_XS"
    31 -> "IQ1_M"
    32 -> "BF16"
    36 -> "TQ1_0"
    37 -> "TQ2_0"
    38 -> "MXFP4_MOE"
    39 -> "NVFP4"
    else -> null
}
