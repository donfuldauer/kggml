package com.donfuldauer.kggml.whisper.streaming

import com.donfuldauer.kggml.whisper.WhisperEngine

/**
 * The mandatory >= 1,000 ms window floor applied to every slice before it reaches `whisper_full_with_state`
 * (whisper.cpp rejects / mis-decodes inputs shorter than one second), and the 30 s window ceiling of the model.
 */
public object WindowFloor {
    /** Minimum slice duration handed to inference. */
    public const val MIN_WINDOW_MS: Int = 1_000

    /** Minimum slice length handed to inference: 1,000 ms at 16 kHz = 16,000 samples. */
    public const val MIN_WINDOW_SAMPLES: Int = WhisperEngine.SAMPLE_RATE / 1_000 * MIN_WINDOW_MS

    /** Whisper's fixed encoder window: 30,000 ms at 16 kHz = 480,000 samples. */
    public const val MAX_WINDOW_SAMPLES: Int = WhisperEngine.SAMPLE_RATE / 1_000 * 30_000

    /** Number of 16 kHz samples in [ms] milliseconds. */
    public fun samplesForMs(ms: Int): Int {
        require(ms >= 0) { "ms must be >= 0, was $ms" }
        return (ms.toLong() * WhisperEngine.SAMPLE_RATE / 1_000).toInt()
    }

    /**
     * Returns `src[0 until length]` as an array of at least [MIN_WINDOW_SAMPLES] samples: shorter input is
     * copied to the front of a [MIN_WINDOW_SAMPLES]-long array and zero-padded at the end; longer input is
     * copied unchanged. Always returns a fresh array the caller owns.
     */
    public fun padToMinimum(src: FloatArray, length: Int = src.size): FloatArray {
        require(length in 0..src.size) { "length must be in 0..${src.size}, was $length" }
        val out = FloatArray(maxOf(length, MIN_WINDOW_SAMPLES))
        src.copyInto(out, 0, 0, length)
        return out
    }
}
