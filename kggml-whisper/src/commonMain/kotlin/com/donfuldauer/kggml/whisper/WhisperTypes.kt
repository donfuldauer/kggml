package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.TranscriptionEvent

/**
 * Construction-time options of a [WhisperEngine] that are specific to Whisper (the compute configuration
 * itself is a [com.donfuldauer.kggml.core.BackendConfig]).
 *
 * @property gpuDevice ggml GPU device index used by accelerated backends.
 * @property allowBackendFallback when `true` (default), an unavailable requested backend falls back along
 *   [ComputeBackend.fallback] (CoreML → Metal → CPU, Vulkan → CPU); when `false`, it fails with
 *   [com.donfuldauer.kggml.core.BackendUnavailableException].
 */
public data class WhisperEngineOptions(
    val gpuDevice: Int = 0,
    val allowBackendFallback: Boolean = true,
) {
    init {
        require(gpuDevice >= 0) { "gpuDevice must be >= 0, was $gpuDevice" }
    }
}

/**
 * Per-call decoding options of [WhisperEngine.transcribe].
 *
 * @property language ISO 639-1 code (e.g. `"en"`, `"de"`), `"auto"` to detect, or `null` for the model default
 *   (`"en"` for English-only models, otherwise auto-detection).
 * @property translate translate the speech to English.
 * @property initialPrompt optional decoder prompt (vocabulary hints, previous text).
 * @property noContext do not carry decoder text context over from the previous call on this engine.
 * @property singleSegment force a single output segment (useful for short streaming slices).
 * @property noTimestamps do not decode timestamp tokens (segment times are then coarse).
 * @property maxTokens maximum tokens per segment, `0` for no limit.
 * @property temperature initial sampling temperature, `0` for greedy decoding.
 * @property temperatureIncrement temperature fallback increment; `0` disables temperature fallback.
 */
public data class TranscriptionOptions(
    val language: String? = null,
    val translate: Boolean = false,
    val initialPrompt: String? = null,
    val noContext: Boolean = true,
    val singleSegment: Boolean = false,
    val noTimestamps: Boolean = false,
    val maxTokens: Int = 0,
    val temperature: Float = 0f,
    val temperatureIncrement: Float = 0.2f,
) {
    init {
        require(maxTokens >= 0) { "maxTokens must be >= 0, was $maxTokens" }
        require(temperature >= 0f) { "temperature must be >= 0, was $temperature" }
        require(temperatureIncrement >= 0f) { "temperatureIncrement must be >= 0, was $temperatureIncrement" }
        require(language == null || language.isNotBlank()) { "language must be null or non-blank" }
    }

    public companion object {
        public val DEFAULT: TranscriptionOptions = TranscriptionOptions()
    }
}

/**
 * One decoded segment.
 *
 * @property index zero-based segment index within its [TranscriptionResult].
 * @property text segment text (UTF-8 decoded; invalid byte sequences become U+FFFD).
 * @property startMs segment start, in milliseconds from the start of the transcribed buffer.
 * @property endMs segment end (`>= startMs`).
 * @property noSpeechProbability whisper's no-speech probability for the segment.
 */
public data class WhisperSegment(
    val index: Int,
    val text: String,
    val startMs: Long,
    val endMs: Long,
    val noSpeechProbability: Float,
) {
    /** This segment as a committed [TranscriptionEvent]. */
    public fun toEvent(): TranscriptionEvent.CommittedSegment =
        TranscriptionEvent.CommittedSegment(index = index, text = text, startMs = startMs, endMs = endMs)
}

/**
 * Result of one [WhisperEngine.transcribe] call.
 *
 * @property segments decoded segments, in order.
 * @property audioDurationMs duration of the input audio (before the >= 1000 ms zero-padding floor).
 * @property backend the backend the transcription ran on.
 */
public data class TranscriptionResult(
    val segments: List<WhisperSegment>,
    val audioDurationMs: Long,
    val backend: ComputeBackend,
) {
    /** All segment texts concatenated and trimmed. */
    public val text: String
        get() = segments.joinToString(separator = "") { it.text }.trim()

    /** The result as transcription events: every segment as committed, followed by [TranscriptionEvent.Completed]. */
    public fun toEvents(): List<TranscriptionEvent> =
        segments.map { it.toEvent() } + TranscriptionEvent.Completed(segments.size, audioDurationMs)
}
