package com.donfuldauer.kggml.models

/**
 * `ggml_type` tensor element types with their block layout, as defined by the pinned `ggml`
 * (`ggml/include/ggml.h`, block structs in `ggml/src/ggml-common.h`).
 *
 * A tensor of `n` elements occupies `n / blockSize * typeSize` bytes; `n` must be a multiple of [blockSize].
 *
 * @property id the numeric `ggml_type` value stored in GGUF tensor infos and Whisper tensor headers.
 * @property blockSize elements per quantization block (1 for plain scalar types).
 * @property typeSize bytes per block.
 */
public enum class GgmlType(public val id: Int, public val blockSize: Int, public val typeSize: Int) {
    F32(0, 1, 4),
    F16(1, 1, 2),
    Q4_0(2, 32, 18),
    Q4_1(3, 32, 20),
    Q5_0(6, 32, 22),
    Q5_1(7, 32, 24),
    Q8_0(8, 32, 34),
    Q8_1(9, 32, 36),
    Q2_K(10, 256, 84),
    Q3_K(11, 256, 110),
    Q4_K(12, 256, 144),
    Q5_K(13, 256, 176),
    Q6_K(14, 256, 210),
    Q8_K(15, 256, 292),
    IQ2_XXS(16, 256, 66),
    IQ2_XS(17, 256, 74),
    IQ3_XXS(18, 256, 98),
    IQ1_S(19, 256, 50),
    IQ4_NL(20, 32, 18),
    IQ3_S(21, 256, 110),
    IQ2_S(22, 256, 82),
    IQ4_XS(23, 256, 136),
    I8(24, 1, 1),
    I16(25, 1, 2),
    I32(26, 1, 4),
    I64(27, 1, 8),
    F64(28, 1, 8),
    IQ1_M(29, 256, 56),
    BF16(30, 1, 2),
    TQ1_0(34, 256, 54),
    TQ2_0(35, 256, 66),
    MXFP4(39, 32, 17),
    NVFP4(40, 64, 36),
    ;

    /** Whether this is a block-quantized type (as opposed to a plain float/int type). */
    public val isQuantized: Boolean get() = blockSize > 1

    /**
     * Bytes needed for [elementCount] elements of this type.
     *
     * @throws ModelFormatException if [elementCount] is negative or not a multiple of [blockSize].
     */
    public fun byteSize(elementCount: Long): Long {
        if (elementCount < 0) throw ModelFormatException("negative element count $elementCount for $name")
        if (elementCount % blockSize != 0L) {
            throw ModelFormatException("element count $elementCount is not a multiple of the $name block size $blockSize")
        }
        return checkedMultiply(elementCount / blockSize, typeSize.toLong())
    }

    public companion object {
        private val byId: Map<Int, GgmlType> = entries.associateBy { it.id }

        /** The type with `ggml_type` value [id], or `null` if it is unknown or was removed upstream. */
        public fun fromId(id: Int): GgmlType? = byId[id]
    }
}

internal fun checkedMultiply(a: Long, b: Long): Long {
    if (a == 0L || b == 0L) return 0L
    val result = a * b
    if (a < 0 || b < 0 || result / b != a) throw ModelFormatException("size overflow ($a * $b)")
    return result
}

internal fun checkedAdd(a: Long, b: Long): Long {
    val result = a + b
    if (((a xor result) and (b xor result)) < 0) throw ModelFormatException("size overflow ($a + $b)")
    return result
}
