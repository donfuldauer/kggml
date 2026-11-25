package com.donfuldauer.kggml.samples.showcase.whisper

/**
 * Audio decoded from a PCM file, kept in its original sample format so it can be handed to
 * `WhisperEngine.transcribe` without an extra conversion pass (the engine downmixes / resamples natively).
 */
sealed class PcmAudio {
    abstract val sampleRate: Int
    abstract val channels: Int

    /** Number of interleaved samples (all channels). */
    abstract val sampleCount: Int

    /** Duration in milliseconds. */
    val durationMs: Long
        get() = if (sampleRate == 0 || channels == 0) 0 else sampleCount.toLong() / channels * 1000 / sampleRate

    /** Signed 16-bit little-endian PCM (WAV format 1, or a raw `.pcm` file). */
    class S16(val samples: ShortArray, override val sampleRate: Int, override val channels: Int) : PcmAudio() {
        override val sampleCount: Int get() = samples.size
    }

    /** IEEE float 32-bit PCM (WAV format 3). */
    class F32(val samples: FloatArray, override val sampleRate: Int, override val channels: Int) : PcmAudio() {
        override val sampleCount: Int get() = samples.size
    }
}

/** Thrown by [parsePcmFile] for files it cannot decode. */
class UnsupportedAudioException(message: String) : IllegalArgumentException(message)

private const val WAVE_FORMAT_PCM = 1
private const val WAVE_FORMAT_IEEE_FLOAT = 3
private const val WAVE_FORMAT_EXTENSIBLE = 0xFFFE

/**
 * Decodes [bytes] as either a RIFF/WAVE file (16-bit integer or 32-bit float PCM, any rate / channel count,
 * including `WAVE_FORMAT_EXTENSIBLE`) or, when there is no RIFF header, as raw signed 16-bit little-endian PCM at
 * [rawSampleRate] Hz with [rawChannels] channels.
 *
 * @throws UnsupportedAudioException for malformed or unsupported files.
 */
fun parsePcmFile(bytes: ByteArray, rawSampleRate: Int = 16_000, rawChannels: Int = 1): PcmAudio {
    if (bytes.size >= 12 && bytes.ascii(0, 4) == "RIFF" && bytes.ascii(8, 4) == "WAVE") return parseWav(bytes)
    if (bytes.size % (2 * rawChannels) != 0) {
        throw UnsupportedAudioException("Raw PCM must be 16-bit with $rawChannels channel(s); ${bytes.size} bytes is not a whole frame count")
    }
    return PcmAudio.S16(decodeS16(bytes, 0, bytes.size), rawSampleRate, rawChannels)
}

private fun parseWav(bytes: ByteArray): PcmAudio {
    var offset = 12
    var format = -1
    var channels = 0
    var sampleRate = 0
    var bitsPerSample = 0
    while (offset + 8 <= bytes.size) {
        val id = bytes.ascii(offset, 4)
        val size = bytes.le32(offset + 4)
        val body = offset + 8
        // Streaming writers often leave a placeholder size on the data chunk, so only that one is clamped below.
        val truncated = size < 0 || body + size > bytes.size
        if (truncated && id != "data") throw UnsupportedAudioException("Truncated '$id' chunk")
        when (id) {
            "fmt " -> {
                if (size < 16) throw UnsupportedAudioException("fmt chunk too short ($size bytes)")
                format = bytes.le16(body)
                channels = bytes.le16(body + 2)
                sampleRate = bytes.le32(body + 4)
                bitsPerSample = bytes.le16(body + 14)
                if (format == WAVE_FORMAT_EXTENSIBLE) {
                    if (size < 40) throw UnsupportedAudioException("WAVE_FORMAT_EXTENSIBLE fmt chunk too short")
                    format = bytes.le16(body + 24) // first two bytes of the SubFormat GUID
                }
            }
            "data" -> {
                if (format == -1) throw UnsupportedAudioException("data chunk before fmt chunk")
                if (channels !in 1..8) throw UnsupportedAudioException("Unsupported channel count $channels")
                if (sampleRate <= 0) throw UnsupportedAudioException("Invalid sample rate $sampleRate")
                val length = if (truncated) bytes.size - body else size
                return when {
                    format == WAVE_FORMAT_PCM && bitsPerSample == 16 -> {
                        val usable = length - length % (2 * channels)
                        PcmAudio.S16(decodeS16(bytes, body, usable), sampleRate, channels)
                    }
                    format == WAVE_FORMAT_IEEE_FLOAT && bitsPerSample == 32 -> {
                        val usable = length - length % (4 * channels)
                        PcmAudio.F32(FloatArray(usable / 4) { Float.fromBits(bytes.le32(body + it * 4)) }, sampleRate, channels)
                    }
                    else -> throw UnsupportedAudioException(
                        "Unsupported WAV encoding (format $format, $bitsPerSample bits); use 16-bit PCM or 32-bit float",
                    )
                }
            }
        }
        // Chunks are word-aligned.
        offset = body + size + (size and 1)
    }
    throw UnsupportedAudioException(if (format == -1) "No fmt chunk" else "No data chunk")
}

private fun decodeS16(bytes: ByteArray, from: Int, length: Int): ShortArray =
    ShortArray(length / 2) { bytes.le16(from + it * 2).toShort() }

private fun ByteArray.ascii(from: Int, length: Int): String =
    CharArray(length) { (this[from + it].toInt() and 0xFF).toChar() }.concatToString()

private fun ByteArray.le16(at: Int): Int = (this[at].toInt() and 0xFF) or ((this[at + 1].toInt() and 0xFF) shl 8)

private fun ByteArray.le32(at: Int): Int = le16(at) or (le16(at + 2) shl 16)
