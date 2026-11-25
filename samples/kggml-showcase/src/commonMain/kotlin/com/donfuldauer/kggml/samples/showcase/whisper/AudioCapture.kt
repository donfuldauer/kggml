package com.donfuldauer.kggml.samples.showcase.whisper

import kotlin.math.sqrt
import kotlinx.coroutines.flow.Flow

/** Format of the interleaved float PCM chunks emitted by an [AudioCapture]. */
data class CaptureFormat(val sampleRate: Int, val channels: Int) {
    init {
        require(sampleRate > 0) { "sampleRate must be > 0, was $sampleRate" }
        require(channels in 1..8) { "channels must be in 1..8, was $channels" }
    }

    /** Duration of [sampleCount] interleaved samples, in milliseconds. */
    fun durationMs(sampleCount: Int): Long = sampleCount.toLong() / channels * 1000 / sampleRate
}

/**
 * Live microphone PCM source (44.1 / 48 kHz, mono or stereo, as delivered by the platform).
 *
 * Audio never touches the file system: chunks are handed straight to `WhisperEngine.transcribeStream`, which
 * downmixes and resamples them natively.
 */
interface AudioCapture {
    /**
     * Prepares the platform audio session (permission / session category where applicable) and returns the format
     * [frames] will emit.
     *
     * @throws Exception if the microphone is unavailable (e.g. permission denied).
     */
    suspend fun prepare(): CaptureFormat

    /**
     * Cold flow of interleaved float PCM chunks in `[-1, 1]`. Collecting it opens the microphone; cancelling the
     * collection stops and releases it. Each emitted array is a fresh copy owned by the collector.
     */
    fun frames(): Flow<FloatArray>
}

/** The platform microphone, or `null` when this platform has no supported capture path. */
expect fun platformAudioCapture(): AudioCapture?

/** Reads a local audio file fully into memory (no temporary copies are written). */
expect fun readAudioFile(path: String): ByteArray

/** Default model path shown in the Whisper Studio (e.g. from `KGGML_WHISPER_MODEL`), or empty. */
expect fun defaultWhisperModelPath(): String

/** Root-mean-square level of [chunk] in `[0, 1]` (input is clamped), used by the waveform visualizer. */
fun rmsLevel(chunk: FloatArray): Float {
    if (chunk.isEmpty()) return 0f
    var sum = 0.0
    for (sample in chunk) {
        val s = sample.coerceIn(-1f, 1f).toDouble()
        sum += s * s
    }
    return sqrt(sum / chunk.size).toFloat().coerceIn(0f, 1f)
}
