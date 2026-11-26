package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.TranscriptionEvent
import com.donfuldauer.kggml.whisper.streaming.VadConfig
import com.donfuldauer.kggml.whisper.streaming.WindowFloor

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
        /** Model-default language, no translation, greedy decoding with temperature fallback. */
        public val DEFAULT: TranscriptionOptions = TranscriptionOptions()
    }
}

/**
 * Options of [WhisperEngine.transcribeStream].
 *
 * The stream is segmented into speech turns by an energy [VadConfig] gate (pure silence never reaches the model;
 * turns end after [VadConfig.trailingSilenceMs] of silence or are split at [VadConfig.maxTurnMs] <= 30 s, Whisper's
 * encoder window). Each finished turn is decoded once into locked
 * [com.donfuldauer.kggml.core.TranscriptionEvent.CommittedSegment]s; while a turn is still in progress, its audio so
 * far is re-decoded every [partialStepMs] of new audio into an unstable
 * [com.donfuldauer.kggml.core.TranscriptionEvent.PartialHypothesis].
 *
 * @property transcription decoding options for every decode of the stream. [TranscriptionOptions.noContext] is
 *   ignored: a stream always decodes with `no_context` and supplies its history explicitly as prompt tokens (see
 *   [carryPromptTokens]). [TranscriptionOptions.initialPrompt] conditions decodes until the first committed text
 *   exists; from then on the carried-over tokens replace it (upstream whisper.cpp semantics).
 * @property vad voice-activity gate and turn segmentation.
 * @property partialStepMs audio (in ms of new turn audio) between two partial decodes of the turn in progress;
 *   `0` disables partial hypotheses. Partial decodes are conflated: if the engine is still busy when the next step
 *   is due, the backlog is skipped and only the newest audio is decoded. The default of 300 ms keeps partial
 *   latency below ~400 ms on hardware that decodes a turn within ~100 ms.
 * @property carryPromptTokens condition each decode on the text tokens of the previously committed segments
 *   (`whisper_full_params.prompt_tokens`), across pauses and across 30 s turn splits.
 * @property maxPromptTokens upper bound of carried-over tokens (the newest are kept); further capped by the model's
 *   own budget (`n_text_ctx / 2 - 1`, 223 for released Whisper models).
 * @property bufferMs capacity of the ring buffer between the audio producer and the decoder, in ms of 16 kHz audio.
 *   When it is full, collection of the audio flow is suspended (back-pressure) until the decoder catches up, so
 *   audio is never dropped and timestamps stay exact.
 */
public data class StreamingOptions(
    val transcription: TranscriptionOptions = TranscriptionOptions.DEFAULT,
    val vad: VadConfig = VadConfig(),
    val partialStepMs: Int = 300,
    val carryPromptTokens: Boolean = true,
    val maxPromptTokens: Int = DEFAULT_MAX_PROMPT_TOKENS,
    val bufferMs: Int = 30_000,
) {
    init {
        require(partialStepMs == 0 || partialStepMs in MIN_PARTIAL_STEP_MS..30_000) {
            "partialStepMs must be 0 or in $MIN_PARTIAL_STEP_MS..30000, was $partialStepMs"
        }
        require(maxPromptTokens >= 0) { "maxPromptTokens must be >= 0, was $maxPromptTokens" }
        require(bufferMs in WindowFloor.MIN_WINDOW_MS..MAX_BUFFER_MS) {
            "bufferMs must be in ${WindowFloor.MIN_WINDOW_MS}..$MAX_BUFFER_MS, was $bufferMs"
        }
    }

    public companion object {
        /** Defaults: 300 ms partial step, prompt carry-over on, 30 s buffer, default [VadConfig]. */
        public val DEFAULT: StreamingOptions = StreamingOptions()

        /** `n_text_ctx / 2` of every released Whisper model (448 / 2). */
        public const val DEFAULT_MAX_PROMPT_TOKENS: Int = 224

        /** Smallest non-zero [partialStepMs]. */
        public const val MIN_PARTIAL_STEP_MS: Int = 100

        /** Largest [bufferMs] (5 minutes of 16 kHz audio, ~19 MB). */
        public const val MAX_BUFFER_MS: Int = 300_000
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
