package com.donfuldauer.kggml.models

/** Container format of a model file. */
public enum class ModelFormat {
    /** GGUF (llama.cpp, libmtmd projectors, stable-diffusion.cpp). */
    GGUF,

    /** Legacy whisper.cpp `ggml` binary. */
    WHISPER_GGML,
}

/**
 * Summary of a model file from its header, without loading weights.
 *
 * @property parameterCount total tensor elements.
 * @property quantization quantization label (for example `Q4_K_M`, `F16`), or `null` if unknown.
 * @property contextLength training context (LLMs) or text context (Whisper), or `null`.
 * @property estimatedPeakRam pre-load peak RAM estimate with default options.
 */
public sealed class ModelHeaderInfo {
    public abstract val format: ModelFormat
    public abstract val parameterCount: Long
    public abstract val quantization: String?
    public abstract val contextLength: Long?
    public abstract val estimatedPeakRam: PeakRamEstimate

    /** A GGUF file. */
    public class Gguf(public val header: GgufModelHeader) : ModelHeaderInfo() {
        override val format: ModelFormat get() = ModelFormat.GGUF
        override val parameterCount: Long get() = header.parameterCount
        override val quantization: String? get() = header.quantization
        override val contextLength: Long? get() = header.contextLength
        override val estimatedPeakRam: PeakRamEstimate by lazy { estimateLlmPeakRam(header) }
    }

    /** A whisper.cpp `ggml` file. */
    public class Whisper(public val header: WhisperModelHeader) : ModelHeaderInfo() {
        override val format: ModelFormat get() = ModelFormat.WHISPER_GGML
        override val parameterCount: Long get() = header.parameterCount
        override val quantization: String? get() = header.quantization
        override val contextLength: Long? get() = header.hyperParameters.textContext.toLong()
        override val estimatedPeakRam: PeakRamEstimate by lazy { estimateWhisperPeakRam(header) }
    }
}

/** Detects the model format from the magic number and parses the header. */
public object ModelInspector {
    /**
     * Inspects [source] (GGUF or whisper.cpp `ggml`).
     *
     * @throws ModelFormatException if the format is unknown or the header is corrupt or truncated.
     */
    public fun inspect(source: ModelByteSource): ModelHeaderInfo {
        val magicBytes = ByteArray(4)
        var read = 0
        while (read < 4) {
            val n = source.read(read.toLong(), magicBytes, read, 4 - read)
            if (n <= 0) throw ModelFormatException("file is too short to be a model ($read bytes)")
            read += n
        }
        val magic = LittleEndianReader(ByteArrayModelSource(magicBytes)).readU32()
        return when (magic) {
            GgufModelHeader.MAGIC -> ModelHeaderInfo.Gguf(GgufModelHeader.parse(source))
            WhisperModelHeader.MAGIC -> ModelHeaderInfo.Whisper(WhisperModelHeader.parse(source))
            else -> throw ModelFormatException("unknown model format (magic 0x${magic.toString(16)})")
        }
    }

    /** Opens [path] with [fileSystem] and inspects it. */
    public fun inspectFile(path: String, fileSystem: ModelFileSystem = defaultModelFileSystem()): ModelHeaderInfo =
        fileSystem.openRead(path).use { inspect(it) }
}
