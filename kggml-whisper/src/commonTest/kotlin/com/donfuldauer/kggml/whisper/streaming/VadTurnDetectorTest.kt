package com.donfuldauer.kggml.whisper.streaming

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class VadTurnDetectorTest {
    // 10 ms frames (160 samples), 3-frame onset, 5-frame hangover, 2-frame (320-sample) pre-roll.
    private val small = VadConfig(
        frameMs = 10,
        speechThreshold = 0.05f,
        minSpeechMs = 30,
        trailingSilenceMs = 50,
        preRollMs = 20,
        maxTurnMs = 1_000,
    )

    private fun speech(ms: Int) = FloatArray(WindowFloor.samplesForMs(ms)) { if (it % 2 == 0) 0.1f else -0.1f }
    private fun silence(ms: Int) = FloatArray(WindowFloor.samplesForMs(ms))

    private class Harness(config: VadConfig) {
        val turns = mutableListOf<SpeechTurn>()
        val vad = VadTurnDetector(config) { turns += it }

        /** Feeds frame by frame and records the state after each frame. */
        fun feed(audio: FloatArray): List<VadState> {
            val frame = vad.config.frameSamples
            val states = mutableListOf<VadState>()
            var pos = 0
            while (pos < audio.size) {
                val n = minOf(frame, audio.size - pos)
                vad.accept(audio, pos, n)
                pos += n
                if (n == frame) states += vad.state
            }
            return states
        }
    }

    @Test
    fun pureSilenceNeverReachesInference() {
        val h = Harness(small)
        val states = h.feed(silence(10_000))
        assertTrue(states.all { it == VadState.SILENCE })
        assertFalse(h.vad.flush())
        assertTrue(h.turns.isEmpty())
        assertEquals(160_000L, h.vad.streamPosition)
    }

    @Test
    fun blipShorterThanMinSpeechIsIgnored() {
        val h = Harness(small)
        h.feed(silence(100))
        h.feed(speech(20)) // 2 voiced frames < 3-frame onset
        h.feed(silence(500))
        h.feed(speech(20))
        assertEquals(VadState.SILENCE, h.vad.state)
        assertFalse(h.vad.flush())
        assertTrue(h.turns.isEmpty())
    }

    @Test
    fun fullTurnLifecycleWithPreRollAndPadding() {
        val h = Harness(small)
        val states = h.feed(silence(100)) + h.feed(speech(300)) + h.feed(silence(100))
        // frames 0..9 silence; 10,11 onset (still SILENCE); 12 -> SPEECH; ... 39 last speech.
        assertEquals(List(12) { VadState.SILENCE }, states.subList(0, 12))
        assertEquals(List(28) { VadState.SPEECH }, states.subList(12, 40))
        assertEquals(List(4) { VadState.TRAILING_PAUSE }, states.subList(40, 44))
        assertEquals(VadState.TURN_COMMITTED, states[44])
        assertEquals(VadState.SILENCE, states[45])

        val turn = h.turns.single()
        assertEquals(TurnEndReason.PAUSE, turn.reason)
        assertEquals(320 + 4_800 + 800, turn.audioSamples) // pre-roll + speech + hangover
        assertEquals(1_600L - 320L, turn.startSample)
        assertEquals(16_000, turn.samples.size)
        assertTrue(turn.isPadded)
        assertEquals(370L, turn.audioDurationMs)
        assertTrue(turn.samples.copyOfRange(0, 320).all { it == 0f })
        assertContentEquals(speech(300), turn.samples.copyOfRange(320, 5_120))
        assertTrue(turn.samples.copyOfRange(5_120, 16_000).all { it == 0f })
        assertEquals(1L, h.vad.turnsEmitted)
    }

    @Test
    fun pauseShorterThanHangoverDoesNotSplitTheTurn() {
        val h = Harness(small)
        val states = h.feed(speech(100)) + h.feed(silence(30)) + h.feed(speech(100)) + h.feed(silence(100))
        assertTrue(VadState.TRAILING_PAUSE in states.subList(10, 13))
        assertEquals(VadState.SPEECH, states[13])
        val turn = h.turns.single()
        assertEquals(0L, turn.startSample)
        assertEquals(TurnEndReason.PAUSE, turn.reason)
        assertEquals(WindowFloor.samplesForMs(100 + 30 + 100 + 50), turn.audioSamples)
    }

    @Test
    fun longSpeechIsSplitAtMaxTurnAndContinuesWithoutGaps() {
        val h = Harness(small)
        h.feed(silence(100))
        h.feed(speech(2_500))
        h.feed(silence(100))
        assertEquals(
            listOf(TurnEndReason.MAX_LENGTH, TurnEndReason.MAX_LENGTH, TurnEndReason.PAUSE),
            h.turns.map { it.reason },
        )
        assertEquals(listOf(16_000, 16_000, 320 + 40_000 + 800 - 32_000), h.turns.map { it.audioSamples })
        assertEquals(listOf(1_280L, 17_280L, 33_280L), h.turns.map { it.startSample })
        assertFalse(h.turns[0].isPadded)
        assertFalse(h.turns[1].isPadded)
        assertTrue(h.turns[2].isPadded)
        // The split loses no audio: the turns concatenate to pre-roll + speech + hangover.
        val joined = h.turns.flatMap { it.samples.copyOf(it.audioSamples).asList() }
        assertContentEquals(silence(20).asList() + speech(2_500).asList() + silence(50).asList(), joined)
    }

    @Test
    fun splitFollowedBySilenceReturnsToIdle() {
        val h = Harness(small)
        // 800-sample history at onset + 95 frames fills the 16,000-sample turn exactly.
        val states = h.feed(silence(100)) + h.feed(speech(980)) + h.feed(silence(300))
        assertEquals(VadState.TURN_COMMITTED, states[10 + 97])
        assertEquals(VadState.SILENCE, states[10 + 98])
        val turn = h.turns.single()
        assertEquals(TurnEndReason.MAX_LENGTH, turn.reason)
        assertEquals(16_000, turn.audioSamples)
        assertFalse(h.vad.flush())
    }

    @Test
    fun speechAfterACommittedTurnNeedsAFreshOnset() {
        val h = Harness(small)
        h.feed(speech(100))
        h.feed(silence(50)) // commits
        assertEquals(VadState.TURN_COMMITTED, h.vad.state)
        val states = h.feed(speech(30))
        assertEquals(listOf(VadState.SILENCE, VadState.SILENCE, VadState.SPEECH), states)
        h.feed(silence(50))
        assertEquals(2, h.turns.size)
        // No pre-roll for the second turn: history was cleared at the commit, so it holds the 3 onset frames only.
        assertEquals(480 + 800, h.turns[1].audioSamples)
        assertEquals(h.turns[0].startSample + h.turns[0].audioSamples, h.turns[1].startSample)
    }

    @Test
    fun finalFlushBelowOneSecondIsZeroPaddedIncludingPartialFrame() {
        val h = Harness(small)
        h.feed(silence(50))
        val tail = speech(205) // 20 full frames + 80 samples
        h.vad.accept(tail)
        assertEquals(VadState.SPEECH, h.vad.state)
        assertTrue(h.vad.flush())
        val turn = h.turns.single()
        assertEquals(TurnEndReason.FLUSH, turn.reason)
        assertEquals(320 + 3_280, turn.audioSamples)
        assertEquals(16_000, turn.samples.size)
        assertContentEquals(tail, turn.samples.copyOfRange(320, 320 + 3_280))
        assertEquals(VadState.SILENCE, h.vad.state)
        assertEquals(800L + 3_280L, h.vad.streamPosition)
        assertFalse(h.vad.flush())
    }

    @Test
    fun flushDuringTrailingPauseCommitsTheTurn() {
        val h = Harness(small)
        h.feed(speech(100))
        h.feed(silence(20))
        assertEquals(VadState.TRAILING_PAUSE, h.vad.state)
        assertTrue(h.vad.flush())
        assertEquals(TurnEndReason.FLUSH, h.turns.single().reason)
        assertEquals(1_600 + 320, h.turns.single().audioSamples)
    }

    @Test
    fun flushWithPendingOnsetDiscardsIt() {
        val h = Harness(small)
        h.feed(speech(20))
        h.vad.accept(speech(5))
        assertFalse(h.vad.flush())
        assertTrue(h.turns.isEmpty())
    }

    @Test
    fun zeroMinSpeechAndHangoverUseSingleFrames() {
        val h = Harness(small.copy(minSpeechMs = 0, trailingSilenceMs = 0, preRollMs = 0))
        val states = h.feed(speech(10)) + h.feed(silence(10)) + h.feed(silence(10))
        assertEquals(listOf(VadState.SPEECH, VadState.TURN_COMMITTED, VadState.SILENCE), states)
        assertEquals(320, h.turns.single().audioSamples)
    }

    @Test
    fun thirtySecondTurnIsEmittedUnpaddedAtTheWhisperWindow() {
        val turns = mutableListOf<SpeechTurn>()
        val vad = VadTurnDetector(VadConfig(preRollMs = 0)) { turns += it }
        vad.accept(speech(30_500))
        vad.flush()
        assertEquals(TurnEndReason.MAX_LENGTH, turns[0].reason)
        assertEquals(WindowFloor.MAX_WINDOW_SAMPLES, turns[0].samples.size)
        assertEquals(WindowFloor.MAX_WINDOW_SAMPLES, turns[0].audioSamples)
        assertEquals(TurnEndReason.FLUSH, turns[1].reason)
        assertEquals(8_000, turns[1].audioSamples)
        assertEquals(16_000, turns[1].samples.size)
    }

    @Test
    fun noEmittedSliceIsEverBelowTheFloorOnRandomAudio() {
        val rnd = Random(42)
        val h = Harness(small)
        repeat(400) {
            val ms = rnd.nextInt(0, 400)
            h.vad.accept(if (rnd.nextBoolean()) speech(ms) else silence(ms), 0, WindowFloor.samplesForMs(ms))
            h.vad.accept(FloatArray(rnd.nextInt(0, 200)))
            if (rnd.nextInt(20) == 0) h.vad.flush()
        }
        h.vad.flush()
        assertTrue(h.turns.size > 10)
        for (t in h.turns) {
            assertTrue(t.samples.size >= WindowFloor.MIN_WINDOW_SAMPLES, "$t")
            assertTrue(t.audioSamples <= small.maxTurnSamples, "$t")
            assertTrue(t.audioSamples >= 1)
        }
    }

    @Test
    fun drainingARingBufferMatchesDirectFeeding() {
        val rnd = Random(7)
        val audio = FloatArray(WindowFloor.samplesForMs(6_000))
        var pos = 0
        while (pos < audio.size) {
            val n = minOf(rnd.nextInt(50, 4_000), audio.size - pos)
            if (rnd.nextBoolean()) speech(n / 16 + 1).copyInto(audio, pos, 0, n)
            pos += n
        }
        val direct = mutableListOf<SpeechTurn>()
        VadTurnDetector(small) { direct += it }.apply { accept(audio); flush() }

        val viaRing = mutableListOf<SpeechTurn>()
        val vad = VadTurnDetector(small) { viaRing += it }
        val ring = PcmRingBuffer(4_096)
        pos = 0
        while (pos < audio.size) {
            val n = minOf(rnd.nextInt(1, 3_000), audio.size - pos)
            ring.write(audio, pos, n)
            pos += n
            assertEquals(n.toLong(), vad.drain(ring))
        }
        vad.flush()

        assertTrue(direct.isNotEmpty())
        assertEquals(direct.map { it.startSample to it.audioSamples }, viaRing.map { it.startSample to it.audioSamples })
        direct.zip(viaRing).forEach { (a, b) -> assertContentEquals(a.samples, b.samples) }
    }

    @Test
    fun resetDropsInProgressTurnSilently() {
        val h = Harness(small)
        h.feed(speech(200))
        h.vad.accept(speech(3))
        h.vad.reset()
        assertEquals(VadState.SILENCE, h.vad.state)
        assertEquals(0L, h.vad.streamPosition)
        assertFalse(h.vad.flush())
        assertTrue(h.turns.isEmpty())
        h.feed(speech(100))
        h.feed(silence(50))
        assertEquals(0L, h.turns.single().startSample)
    }

    @Test
    fun acceptValidatesRange() {
        val vad = VadTurnDetector(small) {}
        assertFailsWith<IndexOutOfBoundsException> { vad.accept(FloatArray(4), 2, 3) }
        assertFailsWith<IndexOutOfBoundsException> { vad.accept(FloatArray(4), -1, 1) }
    }

    @Test
    fun configValidation() {
        assertFailsWith<IllegalArgumentException> { VadConfig(frameMs = 5) }
        assertFailsWith<IllegalArgumentException> { VadConfig(frameMs = 101) }
        assertFailsWith<IllegalArgumentException> { VadConfig(speechThreshold = 0f) }
        assertFailsWith<IllegalArgumentException> { VadConfig(speechThreshold = Float.NaN) }
        assertFailsWith<IllegalArgumentException> { VadConfig(minSpeechMs = -1) }
        assertFailsWith<IllegalArgumentException> { VadConfig(trailingSilenceMs = -1) }
        assertFailsWith<IllegalArgumentException> { VadConfig(preRollMs = -1) }
        assertFailsWith<IllegalArgumentException> { VadConfig(maxTurnMs = 999) }
        assertFailsWith<IllegalArgumentException> { VadConfig(maxTurnMs = 30_001) }
        assertFailsWith<IllegalArgumentException> { VadConfig(preRollMs = 950, maxTurnMs = 1_000) }
        val cfg = VadConfig()
        assertEquals(480, cfg.frameSamples)
        assertEquals(3, cfg.onsetFrames)
        assertEquals(20, cfg.pauseFrames)
        assertEquals(480_000, cfg.maxTurnSamples)
    }
}
