package com.donfuldauer.kggml.whisper.streaming

import com.donfuldauer.kggml.whisper.WhisperEngine
import kotlin.math.sqrt

/** State of a [VadTurnDetector]. */
public enum class VadState {
    /** Idle: no turn in progress. Frames are only kept as pre-roll; nothing is sent to inference. */
    SILENCE,

    /** Voiced frames are being accumulated into the current turn. */
    SPEECH,

    /** The current turn's speech stopped; waiting [VadConfig.trailingSilenceMs] before committing it. */
    TRAILING_PAUSE,

    /** A turn was just emitted. The next frame starts from [SILENCE] (or a new turn, after a forced split). */
    TURN_COMMITTED,
}

/** Why a [SpeechTurn] was emitted. */
public enum class TurnEndReason {
    /** The speaker paused for [VadConfig.trailingSilenceMs]. */
    PAUSE,

    /** The turn reached [VadConfig.maxTurnMs] and was split; speech may continue in the next turn. */
    MAX_LENGTH,

    /** [VadTurnDetector.flush] ended the stream while a turn was in progress. */
    FLUSH,
}

/**
 * Deterministic energy-gate parameters of a [VadTurnDetector]. All durations are at 16 kHz.
 *
 * @property frameMs analysis frame length; each frame is classified voiced / unvoiced by its RMS.
 * @property speechThreshold RMS (linear, full scale = 1.0) at or above which a frame is voiced.
 * @property minSpeechMs consecutive voiced audio needed to start a turn (shorter blips are ignored).
 * @property trailingSilenceMs consecutive unvoiced audio after speech that commits the turn.
 * @property preRollMs audio preceding the speech onset that is prepended to a turn (soft word starts).
 * @property maxTurnMs longest turn before it is force-committed; at most Whisper's 30,000 ms window.
 */
public data class VadConfig(
    val frameMs: Int = 30,
    val speechThreshold: Float = 0.015f,
    val minSpeechMs: Int = 90,
    val trailingSilenceMs: Int = 600,
    val preRollMs: Int = 300,
    val maxTurnMs: Int = 30_000,
) {
    init {
        require(frameMs in 10..100) { "frameMs must be in 10..100, was $frameMs" }
        require(speechThreshold > 0f && speechThreshold.isFinite()) {
            "speechThreshold must be finite and > 0, was $speechThreshold"
        }
        require(minSpeechMs >= 0) { "minSpeechMs must be >= 0, was $minSpeechMs" }
        require(trailingSilenceMs >= 0) { "trailingSilenceMs must be >= 0, was $trailingSilenceMs" }
        require(preRollMs >= 0) { "preRollMs must be >= 0, was $preRollMs" }
        require(maxTurnMs in WindowFloor.MIN_WINDOW_MS..30_000) {
            "maxTurnMs must be in ${WindowFloor.MIN_WINDOW_MS}..30000, was $maxTurnMs"
        }
        require(preRollMs + maxOf(minSpeechMs, frameMs) + frameMs <= maxTurnMs) {
            "preRollMs + minSpeechMs + frameMs must fit in maxTurnMs"
        }
    }

    internal val frameSamples: Int get() = WindowFloor.samplesForMs(frameMs)
    internal val onsetFrames: Int get() = maxOf(1, ceilDiv(minSpeechMs, frameMs))
    internal val pauseFrames: Int get() = maxOf(1, ceilDiv(trailingSilenceMs, frameMs))
    internal val maxTurnSamples: Int get() = WindowFloor.samplesForMs(maxTurnMs)

    private fun ceilDiv(a: Int, b: Int) = (a + b - 1) / b
}

/**
 * A committed speech turn, ready for `whisper_full_with_state`.
 *
 * [samples] is guaranteed to hold at least [WindowFloor.MIN_WINDOW_SAMPLES] (1,000 ms): turns with less audio
 * are zero-padded at the end. The detector allocates a fresh array per turn, which the receiver owns.
 *
 * @property samples 16 kHz mono PCM, `size >= WindowFloor.MIN_WINDOW_SAMPLES`.
 * @property audioSamples real (unpadded) audio at the front of [samples].
 * @property startSample stream position (in samples since the detector started or was [VadTurnDetector.reset])
 *   of `samples[0]`, including pre-roll.
 * @property reason why the turn ended.
 */
public class SpeechTurn internal constructor(
    public val samples: FloatArray,
    public val audioSamples: Int,
    public val startSample: Long,
    public val reason: TurnEndReason,
) {
    init {
        check(samples.size >= WindowFloor.MIN_WINDOW_SAMPLES) { "turn below the 1,000 ms floor: ${samples.size}" }
        check(audioSamples in 1..samples.size) { "audioSamples out of range: $audioSamples" }
    }

    /** `true` when [samples] carries zero padding after [audioSamples]. */
    public val isPadded: Boolean get() = audioSamples < samples.size

    /** Duration of the real audio in milliseconds. */
    public val audioDurationMs: Long get() = audioSamples * 1_000L / WhisperEngine.SAMPLE_RATE

    override fun toString(): String =
        "SpeechTurn(start=$startSample, audioSamples=$audioSamples, size=${samples.size}, reason=$reason)"
}

/**
 * Deterministic energy-based Voice Activity Detection turn segmenter for 16 kHz mono PCM.
 *
 * Audio is cut into fixed [VadConfig.frameMs] frames; each frame is voiced when its RMS is at least
 * [VadConfig.speechThreshold]. Per frame:
 * - [VadState.SILENCE]: the frame is kept as pre-roll only. [VadConfig.minSpeechMs] of consecutive voiced frames
 *   starts a turn (pre-roll + onset frames) → [VadState.SPEECH]. Pure silence never reaches [onTurn].
 * - [VadState.SPEECH]: the frame is appended; an unvoiced frame → [VadState.TRAILING_PAUSE].
 * - [VadState.TRAILING_PAUSE]: the frame is appended; a voiced frame → [VadState.SPEECH];
 *   [VadConfig.trailingSilenceMs] of unvoiced frames commits the turn ([TurnEndReason.PAUSE]).
 * - A turn that cannot take another frame without exceeding [VadConfig.maxTurnMs] is committed
 *   ([TurnEndReason.MAX_LENGTH]); if the next frame is voiced it immediately starts the next turn.
 * - [VadState.TURN_COMMITTED]: reported after each commit until the next frame is processed.
 *
 * Every emitted [SpeechTurn] is zero-padded to the 1,000 ms [WindowFloor]; no shorter slice is ever passed to
 * [onTurn]. Not thread-safe: drive it from the single consumer of the [PcmRingBuffer].
 * Internal buffers are allocated once; the only per-turn allocation is the emitted [SpeechTurn.samples].
 */
public class VadTurnDetector(
    public val config: VadConfig = VadConfig(),
    private val onTurn: (SpeechTurn) -> Unit,
) {
    private val frameSamples = config.frameSamples
    private val frame = FloatArray(frameSamples)
    private var frameFill = 0

    // Pre-roll + onset history while in SILENCE (circular).
    private val historyCapacity = WindowFloor.samplesForMs(config.preRollMs) + config.onsetFrames * frameSamples
    private val history = FloatArray(historyCapacity)
    private var historyEnd = 0L // total samples ever appended to history since it was last cleared

    private val turn = FloatArray(config.maxTurnSamples)
    private var turnFill = 0
    private var turnStart = 0L

    private var onsetCount = 0
    private var pauseCount = 0
    private var continueAfterSplit = false

    /** Samples processed into complete frames since construction or [reset]. */
    public var streamPosition: Long = 0L
        private set

    /** Current state (see [VadState]). */
    public var state: VadState = VadState.SILENCE
        private set

    /** Number of turns emitted since construction or [reset]. */
    public var turnsEmitted: Long = 0L
        private set

    /**
     * Audio samples (pre-roll included) accumulated so far in the turn in progress, or `0` when no turn is in
     * progress ([VadState.SILENCE] / [VadState.TURN_COMMITTED]). Used for partial (unstable) transcriptions.
     */
    public val activeTurnSamples: Int
        get() = if (state == VadState.SPEECH || state == VadState.TRAILING_PAUSE) turnFill else 0

    /** Stream position (see [SpeechTurn.startSample]) of the first sample of the turn in progress; `-1` if none. */
    public val activeTurnStartSample: Long
        get() = if (activeTurnSamples > 0) turnStart else -1L

    /**
     * Copies the [activeTurnSamples] samples of the turn in progress into `dst[offset ...]` without changing any
     * state, and returns how many were copied (`0` when no turn is in progress).
     */
    public fun copyActiveTurn(dst: FloatArray, offset: Int = 0): Int {
        val n = activeTurnSamples
        if (offset < 0 || offset > dst.size - n) {
            throw IndexOutOfBoundsException("offset=$offset, length=$n out of bounds for array of size ${dst.size}")
        }
        turn.copyInto(dst, offset, 0, n)
        return n
    }

    /** Feeds `src[offset until offset + length]`; may invoke [onTurn] zero or more times before returning. */
    public fun accept(src: FloatArray, offset: Int = 0, length: Int = src.size - offset) {
        if (offset < 0 || length < 0 || offset > src.size - length) {
            throw IndexOutOfBoundsException("offset=$offset, length=$length out of bounds for array of size ${src.size}")
        }
        var pos = offset
        val end = offset + length
        while (pos < end) {
            val n = minOf(frameSamples - frameFill, end - pos)
            src.copyInto(frame, frameFill, pos, pos + n)
            frameFill += n
            pos += n
            if (frameFill == frameSamples) processFrame()
        }
    }

    /**
     * Drains every currently readable sample of [buffer] (as its consumer) through the detector, reading
     * straight into the frame buffer without intermediate copies. Returns the number of samples consumed.
     */
    public fun drain(buffer: PcmRingBuffer): Long {
        var total = 0L
        while (true) {
            val n = buffer.read(frame, frameFill, frameSamples - frameFill)
            if (n == 0) return total
            total += n
            frameFill += n
            if (frameFill == frameSamples) processFrame()
        }
    }

    /**
     * Ends the stream: a turn in progress (plus any partial trailing frame) is committed with
     * [TurnEndReason.FLUSH], zero-padded to the 1,000 ms floor if shorter. Pending onset audio that never
     * reached [VadConfig.minSpeechMs] is discarded. The detector is then back in [VadState.SILENCE] and can be
     * reused; [streamPosition] keeps counting.
     *
     * @return `true` if a turn was emitted.
     */
    public fun flush(): Boolean {
        val inTurn = state == VadState.SPEECH || state == VadState.TRAILING_PAUSE
        if (inTurn && frameFill > 0) {
            // A turn always has room for one more full frame (otherwise it was already split).
            frame.copyInto(turn, turnFill, 0, frameFill)
            turnFill += frameFill
        }
        streamPosition += frameFill
        frameFill = 0
        val emitted = inTurn && turnFill > 0
        if (emitted) commit(TurnEndReason.FLUSH)
        toSilence()
        return emitted
    }

    /** Discards all buffered audio and state without emitting anything, and restarts [streamPosition] at 0. */
    public fun reset() {
        frameFill = 0
        toSilence()
        streamPosition = 0L
        turnsEmitted = 0L
    }

    private fun processFrame() {
        frameFill = 0
        streamPosition += frameSamples
        val voiced = rms(frame) >= config.speechThreshold
        if (state == VadState.TURN_COMMITTED) {
            if (continueAfterSplit && voiced) {
                continueAfterSplit = false
                startTurn(includeHistory = false)
                appendToTurn()
                state = VadState.SPEECH
                splitIfFull()
                return
            }
            toSilence()
        }
        when (state) {
            VadState.SILENCE -> {
                appendToHistory()
                if (voiced) {
                    if (++onsetCount >= config.onsetFrames) {
                        startTurn(includeHistory = true)
                        state = VadState.SPEECH
                        splitIfFull()
                    }
                } else {
                    onsetCount = 0
                }
            }
            VadState.SPEECH -> {
                appendToTurn()
                if (!voiced) {
                    pauseCount = 1
                    state = VadState.TRAILING_PAUSE
                    if (pauseCount >= config.pauseFrames) return commit(TurnEndReason.PAUSE)
                }
                splitIfFull()
            }
            VadState.TRAILING_PAUSE -> {
                appendToTurn()
                if (voiced) {
                    pauseCount = 0
                    state = VadState.SPEECH
                } else if (++pauseCount >= config.pauseFrames) {
                    return commit(TurnEndReason.PAUSE)
                }
                splitIfFull()
            }
            VadState.TURN_COMMITTED -> error("unreachable")
        }
    }

    private fun splitIfFull() {
        if (turnFill + frameSamples > turn.size) {
            val speaking = state == VadState.SPEECH
            commit(TurnEndReason.MAX_LENGTH)
            continueAfterSplit = speaking
        }
    }

    private fun startTurn(includeHistory: Boolean) {
        turnFill = 0
        if (includeHistory) {
            val n = minOf(historyEnd, historyCapacity.toLong()).toInt()
            val start = ((historyEnd - n) % historyCapacity).toInt()
            val first = minOf(n, historyCapacity - start)
            history.copyInto(turn, 0, start, start + first)
            if (n > first) history.copyInto(turn, first, 0, n - first)
            turnFill = n
        }
        turnStart = streamPosition - turnFill - if (includeHistory) 0 else frameSamples
        historyEnd = 0L
        onsetCount = 0
        pauseCount = 0
    }

    private fun appendToTurn() {
        frame.copyInto(turn, turnFill)
        turnFill += frameSamples
    }

    private fun appendToHistory() {
        // frameSamples <= historyCapacity (onsetFrames >= 1), so a frame wraps at most once.
        val start = (historyEnd % historyCapacity).toInt()
        val first = minOf(frameSamples, historyCapacity - start)
        frame.copyInto(history, start, 0, first)
        if (frameSamples > first) frame.copyInto(history, 0, first, frameSamples)
        historyEnd += frameSamples
    }

    private fun commit(reason: TurnEndReason) {
        val emitted = SpeechTurn(WindowFloor.padToMinimum(turn, turnFill), turnFill, turnStart, reason)
        turnFill = 0
        pauseCount = 0
        onsetCount = 0
        continueAfterSplit = false
        state = VadState.TURN_COMMITTED
        turnsEmitted++
        onTurn(emitted)
    }

    private fun toSilence() {
        state = VadState.SILENCE
        turnFill = 0
        historyEnd = 0L
        onsetCount = 0
        pauseCount = 0
        continueAfterSplit = false
    }

    private fun rms(x: FloatArray): Float {
        var sum = 0.0
        for (v in x) sum += v.toDouble() * v
        return sqrt(sum / x.size).toFloat()
    }
}
