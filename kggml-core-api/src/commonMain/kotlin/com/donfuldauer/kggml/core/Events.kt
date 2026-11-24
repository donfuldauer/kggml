package com.donfuldauer.kggml.core

/** Events emitted by a streaming speech-to-text engine (e.g. `Flow<TranscriptionEvent>`). */
public sealed interface TranscriptionEvent {

    /**
     * An unstable hypothesis for audio that has not yet been committed. Later events may revise it.
     *
     * @property text hypothesized text.
     * @property startMs start of the hypothesized span, in milliseconds from the start of the stream.
     * @property endMs end of the hypothesized span (`>= startMs`).
     */
    public data class PartialHypothesis(val text: String, val startMs: Long, val endMs: Long) : TranscriptionEvent {
        init {
            requireValidSpan(startMs, endMs)
        }
    }

    /**
     * A finalized segment. Committed segments are never revised.
     *
     * @property index zero-based, monotonically increasing segment index.
     * @property text segment text.
     * @property startMs segment start, in milliseconds from the start of the stream.
     * @property endMs segment end (`>= startMs`).
     */
    public data class CommittedSegment(
        val index: Int,
        val text: String,
        val startMs: Long,
        val endMs: Long,
    ) : TranscriptionEvent {
        init {
            require(index >= 0) { "index must be >= 0, was $index" }
            requireValidSpan(startMs, endMs)
        }
    }

    /**
     * The stream finished successfully. This is the last event emitted.
     *
     * @property segmentCount number of [CommittedSegment]s emitted.
     * @property audioDurationMs total duration of the audio processed.
     */
    public data class Completed(val segmentCount: Int, val audioDurationMs: Long) : TranscriptionEvent {
        init {
            require(segmentCount >= 0) { "segmentCount must be >= 0, was $segmentCount" }
            require(audioDurationMs >= 0) { "audioDurationMs must be >= 0, was $audioDurationMs" }
        }
    }

    /**
     * A non-fatal or terminal error reported in-band. If terminal, no further events follow.
     *
     * @property exception the typed failure.
     */
    public data class Error(val exception: InferenceException) : TranscriptionEvent
}

/** Events emitted by a streaming text-generation engine (e.g. `Flow<TokenEvent>`). */
public sealed interface TokenEvent {

    /**
     * A generated token.
     *
     * @property index zero-based position of this token within the generation.
     * @property tokenId vocabulary id of the token.
     * @property text detokenized text piece (may be empty for partial UTF-8 sequences).
     */
    public data class Token(val index: Int, val tokenId: Int, val text: String) : TokenEvent {
        init {
            require(index >= 0) { "index must be >= 0, was $index" }
        }
    }

    /**
     * Generation finished. This is the last event emitted.
     *
     * @property tokenCount number of [Token]s emitted.
     * @property stopReason why generation stopped.
     */
    public data class Completed(val tokenCount: Int, val stopReason: StopReason) : TokenEvent {
        init {
            require(tokenCount >= 0) { "tokenCount must be >= 0, was $tokenCount" }
        }
    }

    /**
     * An error reported in-band. If terminal, no further events follow.
     *
     * @property exception the typed failure.
     */
    public data class Error(val exception: InferenceException) : TokenEvent
}

/** Why text generation stopped. Cancellation is reported through coroutine cancellation, not here. */
public enum class StopReason {
    /** The model emitted an end-of-generation token. */
    END_OF_GENERATION,

    /** The configured maximum token count was reached. */
    MAX_TOKENS,

    /** A configured stop sequence was produced. */
    STOP_SEQUENCE,

    /** The context window is full. */
    CONTEXT_FULL,
}

private fun requireValidSpan(startMs: Long, endMs: Long) {
    require(startMs >= 0) { "startMs must be >= 0, was $startMs" }
    require(endMs >= startMs) { "endMs ($endMs) must be >= startMs ($startMs)" }
}
