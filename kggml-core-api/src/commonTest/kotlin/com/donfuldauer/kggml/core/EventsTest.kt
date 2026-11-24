package com.donfuldauer.kggml.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class EventsTest {

    @Test
    fun transcriptionEventsAcceptValidSpans() {
        TranscriptionEvent.PartialHypothesis("he", 0, 0)
        TranscriptionEvent.PartialHypothesis("hello", 100, 900)
        TranscriptionEvent.CommittedSegment(0, "hello", 0, 1_000)
        TranscriptionEvent.Completed(segmentCount = 0, audioDurationMs = 0)
    }

    @Test
    fun transcriptionEventsRejectInvalidSpans() {
        assertFailsWith<IllegalArgumentException> { TranscriptionEvent.PartialHypothesis("x", -1, 10) }
        assertFailsWith<IllegalArgumentException> { TranscriptionEvent.PartialHypothesis("x", 10, 9) }
        assertFailsWith<IllegalArgumentException> { TranscriptionEvent.CommittedSegment(0, "x", -5, 0) }
        assertFailsWith<IllegalArgumentException> { TranscriptionEvent.CommittedSegment(0, "x", 500, 100) }
        assertFailsWith<IllegalArgumentException> { TranscriptionEvent.CommittedSegment(-1, "x", 0, 100) }
        assertFailsWith<IllegalArgumentException> { TranscriptionEvent.Completed(-1, 0) }
        assertFailsWith<IllegalArgumentException> { TranscriptionEvent.Completed(0, -1) }
    }

    @Test
    fun transcriptionEventsAreExhaustive() {
        fun kind(event: TranscriptionEvent): String = when (event) {
            is TranscriptionEvent.PartialHypothesis -> "partial"
            is TranscriptionEvent.CommittedSegment -> "segment"
            is TranscriptionEvent.Completed -> "completed"
            is TranscriptionEvent.Error -> "error"
        }
        assertEquals(
            listOf("partial", "segment", "completed", "error"),
            listOf(
                TranscriptionEvent.PartialHypothesis("a", 0, 1),
                TranscriptionEvent.CommittedSegment(0, "a", 0, 1),
                TranscriptionEvent.Completed(1, 1),
                TranscriptionEvent.Error(InvalidInputException("bad pcm")),
            ).map(::kind),
        )
    }

    @Test
    fun errorEventsWrapTypedException() {
        val e = OutOfMemoryException("oom")
        assertEquals(e, TranscriptionEvent.Error(e).exception)
        assertEquals(e, TokenEvent.Error(e).exception)
    }

    @Test
    fun tokenEventsValidate() {
        TokenEvent.Token(index = 0, tokenId = 0, text = "")
        TokenEvent.Completed(tokenCount = 0, stopReason = StopReason.MAX_TOKENS)
        assertFailsWith<IllegalArgumentException> { TokenEvent.Token(index = -1, tokenId = 1, text = "a") }
        assertFailsWith<IllegalArgumentException> { TokenEvent.Completed(tokenCount = -1, StopReason.END_OF_GENERATION) }
    }

    @Test
    fun tokenEventsAreExhaustiveAndStructurallyEqual() {
        fun kind(event: TokenEvent): String = when (event) {
            is TokenEvent.Token -> "token"
            is TokenEvent.Completed -> "completed:${event.stopReason}"
            is TokenEvent.Error -> "error"
        }
        assertEquals("token", kind(TokenEvent.Token(0, 7, "hi")))
        assertEquals("completed:STOP_SEQUENCE", kind(TokenEvent.Completed(3, StopReason.STOP_SEQUENCE)))
        assertEquals("error", kind(TokenEvent.Error(InferenceCancelledException("x"))))
        assertEquals(TokenEvent.Token(1, 2, "x"), TokenEvent.Token(1, 2, "x"))
    }

    @Test
    fun stopReasons() {
        assertEquals(
            listOf("END_OF_GENERATION", "MAX_TOKENS", "STOP_SEQUENCE", "CONTEXT_FULL"),
            StopReason.entries.map { it.name },
        )
    }
}
