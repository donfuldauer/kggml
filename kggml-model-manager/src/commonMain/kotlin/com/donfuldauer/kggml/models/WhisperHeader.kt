package com.donfuldauer.kggml.models

/**
 * Hyper-parameters of a `whisper.cpp` `ggml` model file (`whisper_hparams`, in file order).
 *
 * @property vocabularySize `n_vocab`: tokens in the vocabulary.
 * @property audioContext `n_audio_ctx`: encoder positions (1500 = 30 s of audio).
 * @property audioState `n_audio_state`: encoder width.
 * @property audioHeads `n_audio_head`: encoder attention heads.
 * @property audioLayers `n_audio_layer`: encoder layers (identifies the model size, see
 *   [WhisperModelHeader.modelType]).
 * @property textContext `n_text_ctx`: decoder positions.
 * @property textState `n_text_state`: decoder width.
 * @property textHeads `n_text_head`: decoder attention heads.
 * @property textLayers `n_text_layer`: decoder layers.
 * @property melBands `n_mels`: mel frequency bands of the input spectrogram (80, or 128 for large-v3).
 * @property fileTypeId `ggml_ftype` with the quantization version factor removed.
 * @property quantizationVersion `GGML_QNT_VERSION` the file was quantized with (0 for unquantized files).
 */
public data class WhisperHyperParameters(
    val vocabularySize: Int,
    val audioContext: Int,
    val audioState: Int,
    val audioHeads: Int,
    val audioLayers: Int,
    val textContext: Int,
    val textState: Int,
    val textHeads: Int,
    val textLayers: Int,
    val melBands: Int,
    val fileTypeId: Int,
    val quantizationVersion: Int,
)

/**
 * One tensor descriptor of a Whisper `ggml` file.
 *
 * @property name tensor name (for example `encoder.conv1.weight`).
 * @property dimensions `ne[0..n_dims)`.
 * @property typeId the raw `ggml_type`; [type] is `null` if it is unknown to this parser.
 * @property dataOffset absolute file offset of the tensor data.
 */
public data class WhisperTensorInfo(val name: String, val dimensions: List<Long>, val typeId: Int, val dataOffset: Long) {
    /** The element type, or `null` if [typeId] is unknown. */
    val type: GgmlType? get() = GgmlType.fromId(typeId)

    /** Number of elements. */
    val elementCount: Long get() = dimensions.fold(1L) { acc, d -> checkedMultiply(acc, d) }

    /** Data size in bytes, or `null` if [type] is unknown. */
    val byteSize: Long? get() = type?.byteSize(elementCount)
}

/**
 * Parsed `whisper.cpp` `ggml` model header (legacy, non-GGUF format): hyper-parameters, mel filter shape, vocabulary
 * size and tensor descriptors. Tensor data is skipped, never read.
 *
 * @property hyperParameters the model hyper-parameters.
 * @property melFilterShape `(n_mel, n_fft)` of the embedded mel filter bank.
 * @property vocabularyEntries number of vocabulary entries stored in the file.
 * @property tensors every tensor descriptor, in file order.
 */
public class WhisperModelHeader internal constructor(
    public val hyperParameters: WhisperHyperParameters,
    public val melFilterShape: Pair<Int, Int>,
    public val vocabularyEntries: Int,
    public val tensors: List<WhisperTensorInfo>,
) {
    /** Model size name from the encoder depth (`tiny`, `base`, `small`, `medium`, `large`), or `null`. */
    public val modelType: String?
        get() = when (hyperParameters.audioLayers) {
            4 -> "tiny"
            6 -> "base"
            12 -> "small"
            24 -> "medium"
            32 -> "large"
            else -> null
        }

    /** Whether the model is multilingual (`n_vocab >= 51865`) rather than English-only. */
    public val isMultilingual: Boolean get() = hyperParameters.vocabularySize >= 51865

    /** Quantization label (`ggml_ftype` name, for example `F16` or `Q5_0`). */
    public val quantization: String? get() = ggmlFileTypeName(hyperParameters.fileTypeId)

    /** Total parameters (sum of tensor element counts). */
    public val parameterCount: Long get() = tensors.fold(0L) { acc, t -> checkedAdd(acc, t.elementCount) }

    /** Bytes of tensor data. Tensors of unknown type count as 0. */
    public val tensorDataBytes: Long get() = tensors.fold(0L) { acc, t -> checkedAdd(acc, t.byteSize ?: 0L) }

    public companion object {
        /** `"ggml"` magic as a little-endian `uint32` (bytes `6c 6d 67 67`). */
        public const val MAGIC: Long = 0x67676d6cL

        private const val QNT_VERSION_FACTOR = 1000
        private const val MAX_DIMS = 4
        private const val MAX_NAME_BYTES = 1024
        private const val MAX_TOKEN_BYTES = 1024 * 1024
        private const val MAX_HPARAM = 1 shl 24

        /**
         * Parses [source]. With [scanTensors] (default) the tensor table is walked to the end of the file by skipping
         * tensor data, which needs a source of known size or a complete stream; pass `false` to read only the
         * hyper-parameters, mel filters and vocabulary (for example from a downloaded header prefix).
         *
         * @throws ModelFormatException on a wrong magic, invalid hyper-parameters or a truncated or corrupt file.
         */
        public fun parse(source: ModelByteSource, scanTensors: Boolean = true): WhisperModelHeader {
            val r = LittleEndianReader(source)
            val magic = r.readU32()
            if (magic != MAGIC) throw ModelFormatException("not a whisper.cpp ggml file (magic 0x${magic.toString(16)})")
            val raw = IntArray(11) { r.readI32() }
            val names = listOf(
                "n_vocab", "n_audio_ctx", "n_audio_state", "n_audio_head", "n_audio_layer",
                "n_text_ctx", "n_text_state", "n_text_head", "n_text_layer", "n_mels",
            )
            for (i in names.indices) {
                if (raw[i] <= 0 || raw[i] > MAX_HPARAM) throw ModelFormatException("invalid ${names[i]} ${raw[i]}")
            }
            if (raw[10] < 0) throw ModelFormatException("invalid ftype ${raw[10]}")
            val hp = WhisperHyperParameters(
                vocabularySize = raw[0], audioContext = raw[1], audioState = raw[2], audioHeads = raw[3],
                audioLayers = raw[4], textContext = raw[5], textState = raw[6], textHeads = raw[7],
                textLayers = raw[8], melBands = raw[9],
                fileTypeId = raw[10] % QNT_VERSION_FACTOR, quantizationVersion = raw[10] / QNT_VERSION_FACTOR,
            )

            val nMel = r.readI32()
            val nFft = r.readI32()
            if (nMel <= 0 || nFft <= 0 || nMel > MAX_HPARAM || nFft > MAX_HPARAM) {
                throw ModelFormatException("invalid mel filter shape $nMel x $nFft")
            }
            r.skip(checkedMultiply(nMel.toLong() * nFft, 4), "mel filters")

            val nVocab = r.readI32()
            if (nVocab < 0 || nVocab > MAX_HPARAM) throw ModelFormatException("invalid vocabulary count $nVocab")
            for (i in 0 until nVocab) {
                val len = r.readU32()
                if (len > MAX_TOKEN_BYTES) throw ModelFormatException("token #$i length $len is implausibly large")
                r.skip(len, "token #$i")
            }

            val tensors = ArrayList<WhisperTensorInfo>()
            if (scanTensors) {
                while (true) {
                    // whisper.cpp stops at end of file; a clean EOF can only happen at a tensor boundary.
                    val left = r.remaining
                    if (left == 0L) break
                    if (left == null && !hasMore(source, r.position)) break
                    val nDims = r.readI32()
                    val nameLen = r.readI32()
                    val typeId = r.readI32()
                    if (nDims !in 1..MAX_DIMS) throw ModelFormatException("tensor #${tensors.size} has $nDims dimensions")
                    if (nameLen !in 1..MAX_NAME_BYTES) throw ModelFormatException("tensor #${tensors.size} name length $nameLen")
                    val dims = List(nDims) {
                        val d = r.readI32()
                        if (d <= 0) throw ModelFormatException("tensor #${tensors.size} has dimension $d")
                        d.toLong()
                    }
                    val name = r.readBytes(nameLen).decodeToString()
                    val info = WhisperTensorInfo(name, dims, typeId, r.position)
                    val bytes = info.byteSize ?: throw ModelFormatException("tensor $name has unknown ggml type $typeId")
                    r.skip(bytes, "tensor $name data")
                    tensors += info
                }
            }
            return WhisperModelHeader(hp, nMel to nFft, nVocab, tensors)
        }

        private fun hasMore(source: ModelByteSource, position: Long): Boolean =
            source.read(position, ByteArray(1), 0, 1) > 0
    }
}

/** `ggml_ftype` names from the pinned `ggml.h`, or `null` for unknown values. */
internal fun ggmlFileTypeName(id: Int): String? = when (id) {
    0 -> "F32"
    1 -> "F16"
    2 -> "Q4_0"
    3 -> "Q4_1"
    4 -> "Q4_1_SOME_F16"
    7 -> "Q8_0"
    8 -> "Q5_0"
    9 -> "Q5_1"
    10 -> "Q2_K"
    11 -> "Q3_K"
    12 -> "Q4_K"
    13 -> "Q5_K"
    14 -> "Q6_K"
    15 -> "IQ2_XXS"
    16 -> "IQ2_XS"
    17 -> "IQ3_XXS"
    18 -> "IQ1_S"
    19 -> "IQ4_NL"
    20 -> "IQ3_S"
    21 -> "IQ2_S"
    22 -> "IQ4_XS"
    23 -> "IQ1_M"
    24 -> "BF16"
    25 -> "MXFP4"
    26 -> "NVFP4"
    else -> null
}
