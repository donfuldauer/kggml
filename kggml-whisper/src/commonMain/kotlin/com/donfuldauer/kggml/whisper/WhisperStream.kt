package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.InferenceException
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.TranscriptionEvent
import com.donfuldauer.kggml.whisper.streaming.OverflowPolicy
import com.donfuldauer.kggml.whisper.streaming.PcmRingBuffer
import com.donfuldauer.kggml.whisper.streaming.SpeechTurn
import com.donfuldauer.kggml.whisper.streaming.VadTurnDetector
import com.donfuldauer.kggml.whisper.streaming.WindowFloor
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.AtomicLong
import kotlin.concurrent.atomics.AtomicReference
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ProducerScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** The audio source of a [WhisperStream]. */
internal sealed interface StreamSource {
    class F32(val flow: Flow<FloatArray>) : StreamSource

    class S16(val flow: Flow<ShortArray>) : StreamSource
}

/**
 * One [WhisperEngine.transcribeStream] pipeline (a fresh instance per call; every collection of [flow] runs its own
 * independent pipeline with its own native state):
 *
 * ```
 * audio Flow ─(producer: native streaming resampler)─▶ PcmRingBuffer ─(consumer)─▶ VadTurnDetector
 *     ├─ turn in progress, every partialStep of new audio ─▶ decode (single segment, no timestamps) ─▶ Partial
 *     └─ finished turn ─▶ decode with carried prompt tokens ─▶ CommittedSegment… ─▶ prompt history
 * end of audio ─▶ VAD flush ─▶ last turn ─▶ Completed
 * ```
 *
 * The producer and the consumer are separate coroutines joined only by the lock-free single-producer /
 * single-consumer ring buffer and two conflated wake-up channels, so audio keeps being ingested (and resampled)
 * while the consumer waits for a decode on the engine thread. A full ring buffer suspends the producer.
 */
internal class WhisperStream(
    private val engine: WhisperEngine,
    private val bridges: WhisperBridgeHolder,
    private val sampleRate: Int,
    private val channels: Int,
    private val options: StreamingOptions,
    private val source: StreamSource,
) {
    fun flow(): Flow<TranscriptionEvent> = channelFlow { Collection(this).run() }.flowOn(Dispatchers.Default)

    /** State of one collection. */
    private inner class Collection(private val out: ProducerScope<TranscriptionEvent>) {
        private val ring = PcmRingBuffer(WindowFloor.samplesForMs(options.bufferMs), OverflowPolicy.DROP_NEWEST)
        private val dataSignal = Channel<Unit>(Channel.CONFLATED)
        private val spaceSignal = Channel<Unit>(Channel.CONFLATED)
        private val inputDone = AtomicBoolean(false)
        private val inputFrames = AtomicLong(0L)
        private val producerError = AtomicReference<InferenceException?>(null)

        private val pendingTurns = ArrayDeque<SpeechTurn>()
        private val vad = VadTurnDetector(options.vad) { pendingTurns.addLast(it) }
        private val partialStepSamples = WindowFloor.samplesForMs(options.partialStepMs)
        private val partialOptions = options.transcription.copy(
            singleSegment = true,
            noTimestamps = true,
            temperatureIncrement = 0f, // one greedy pass: partials are about latency, not robustness
        )
        private var partialBuffer = FloatArray(0)
        private var partialTurnStart = -1L
        private var partialTurnSamples = 0
        private var committed = 0

        suspend fun run() {
            val slot = try {
                engine.openStream { dataSignal.trySend(Unit) }
            } catch (e: CancellationException) {
                throw e // a CancellationException is an IllegalStateException: never treat it as "closed"
            } catch (e: IllegalStateException) {
                if (!engine.isClosed) throw e
                out.send(TranscriptionEvent.Error(closedError()))
                return
            } catch (e: InferenceException) {
                out.send(TranscriptionEvent.Error(e))
                return
            }
            val prompt = PromptHistory(if (options.carryPromptTokens) minOf(options.maxPromptTokens, slot.maxPromptTokens) else 0)
            val producer = out.launch { produce() }
            try {
                consume(slot, prompt)
            } catch (e: CancellationException) {
                throw e
            } catch (e: InferenceException) {
                out.send(TranscriptionEvent.Error(e))
            } catch (e: IllegalStateException) {
                if (!engine.isClosed) throw e
                out.send(TranscriptionEvent.Error(closedError()))
            } finally {
                producer.cancel()
                withContext(NonCancellable) { engine.closeStream(slot) }
            }
        }

        // ---- consumer: VAD, decodes, events ------------------------------------------------------------

        private suspend fun consume(slot: StreamSlot, prompt: PromptHistory) {
            while (true) {
                if (engine.isClosed) throw closedError()
                val done = inputDone.load() // read before draining: everything written before `done` is drained
                vad.drain(ring)
                spaceSignal.trySend(Unit)
                producerError.load()?.let { throw it }
                commitPending(slot, prompt)
                if (done && ring.available == 0) {
                    vad.flush()
                    commitPending(slot, prompt)
                    out.send(TranscriptionEvent.Completed(committed, inputFrames.load() * 1_000L / sampleRate))
                    return
                }
                maybeDecodePartial(slot, prompt)
                if (ring.available == 0 && !inputDone.load()) dataSignal.receive()
            }
        }

        private suspend fun commitPending(slot: StreamSlot, prompt: PromptHistory) {
            while (pendingTurns.isNotEmpty()) {
                val turn = pendingTurns.removeFirst()
                val segments = engine.decodeStream(
                    slot, turn.samples, turn.samples.size, options.transcription, prompt.tokens, prompt.size,
                )
                val turnStartMs = samplesToMs(turn.startSample)
                val turnEndMs = samplesToMs(turn.startSample + turn.audioSamples)
                for (segment in segments) {
                    if (segment.text.isBlank()) continue
                    val start = (turnStartMs + segment.startMs).coerceIn(turnStartMs, turnEndMs)
                    val end = (turnStartMs + segment.endMs).coerceIn(start, turnEndMs)
                    out.send(TranscriptionEvent.CommittedSegment(committed++, segment.text, start, end))
                    prompt.append(segment.tokens)
                }
            }
        }

        private suspend fun maybeDecodePartial(slot: StreamSlot, prompt: PromptHistory) {
            if (partialStepSamples == 0) return
            val active = vad.activeTurnSamples
            if (active == 0) return
            val start = vad.activeTurnStartSample
            if (start != partialTurnStart) {
                partialTurnStart = start
                partialTurnSamples = 0
            }
            // Only the newest audio is decoded: a backlog that built up during the previous decode is skipped.
            if (active - partialTurnSamples < partialStepSamples) return
            partialTurnSamples = active
            if (partialBuffer.size < active) partialBuffer = FloatArray(options.vad.maxTurnSamples)
            val n = vad.copyActiveTurn(partialBuffer)
            val segments = engine.decodeStream(slot, partialBuffer, n, partialOptions, prompt.tokens, prompt.size)
            val text = segments.joinToString(separator = "") { it.text }
            if (text.isBlank()) return
            val startMs = samplesToMs(start)
            out.send(TranscriptionEvent.PartialHypothesis(text, startMs, samplesToMs(start + n)))
        }

        // ---- producer: audio flow -> 16 kHz mono -> ring buffer ---------------------------------------

        private suspend fun produce() {
            val converter = Converter()
            try {
                when (source) {
                    is StreamSource.F32 -> source.flow.collect { chunk ->
                        val frames = frames(chunk.size)
                        if (sampleRate == WhisperEngine.SAMPLE_RATE && channels == 1) {
                            write(chunk, chunk.size) // already 16 kHz mono float
                        } else {
                            val (pcm, n) = converter.f32(chunk, frames)
                            write(pcm, n)
                        }
                    }
                    is StreamSource.S16 -> source.flow.collect { chunk ->
                        val (pcm, n) = converter.s16(chunk, frames(chunk.size))
                        write(pcm, n)
                    }
                }
                val (tail, n) = converter.flush()
                write(tail, n)
            } catch (e: InferenceException) {
                producerError.store(e)
            } finally {
                converter.close()
                inputDone.store(true)
                dataSignal.trySend(Unit)
            }
        }

        private fun frames(size: Int): Int {
            if (size % channels != 0) {
                throw InvalidInputException("audio chunk of $size samples is not a multiple of channels ($channels)")
            }
            val frames = size / channels
            inputFrames.addAndFetch(frames.toLong())
            return frames
        }

        /** Copies `src[0 until count]` into the ring buffer, suspending while it is full. */
        private suspend fun write(src: FloatArray, count: Int) {
            var offset = 0
            while (offset < count) {
                val free = ring.remainingCapacity
                if (free == 0) {
                    spaceSignal.receive()
                    continue
                }
                val n = minOf(free, count - offset)
                ring.write(src, offset, n)
                offset += n
                dataSignal.trySend(Unit)
            }
        }

        private fun samplesToMs(samples: Long): Long = samples * 1_000L / WhisperEngine.SAMPLE_RATE

        private fun closedError() = InferenceCancelledException("WhisperEngine was closed during transcribeStream")

        /** Lazily created native streaming resampler (chunk boundaries are seamless). Producer coroutine only. */
        private inner class Converter {
            private var handle = 0L
            private var buffer = FloatArray(0)

            fun f32(chunk: FloatArray, frames: Int): Pair<FloatArray, Int> {
                val bridge = bridges.get()
                val r = resampler(bridge, PcmFormat.F32)
                ensureCapacity(bridge, r, frames)
                val result = bridge.resamplerProcessF32(r, chunk, frames, channels, buffer)
                bridge.check(result.status, "kggml_whisper_resampler_process")
                return buffer to result.value
            }

            fun s16(chunk: ShortArray, frames: Int): Pair<FloatArray, Int> {
                val bridge = bridges.get()
                val r = resampler(bridge, PcmFormat.S16)
                ensureCapacity(bridge, r, frames)
                val result = bridge.resamplerProcessS16(r, chunk, frames, channels, buffer)
                bridge.check(result.status, "kggml_whisper_resampler_process")
                return buffer to result.value
            }

            fun flush(): Pair<FloatArray, Int> {
                if (handle == 0L) return buffer to 0
                val bridge = bridges.get()
                ensureCapacity(bridge, handle, 0)
                val result = bridge.resamplerFlush(handle, buffer)
                bridge.check(result.status, "kggml_whisper_resampler_flush")
                return buffer to result.value
            }

            fun close() {
                val h = handle
                handle = 0L
                if (h != 0L) bridges.get().resamplerFree(h)
            }

            private fun resampler(bridge: NativeWhisperBridge, format: Int): Long {
                if (handle == 0L) {
                    val created = bridge.resamplerCreate(sampleRate, channels, format)
                    bridge.check(created.status, "kggml_whisper_resampler_create")
                    handle = created.handle
                }
                return handle
            }

            private fun ensureCapacity(bridge: NativeWhisperBridge, r: Long, frames: Int) {
                val bound = bridge.resamplerMaxOutput(r, frames)
                if (bound < 0 || bound > Int.MAX_VALUE) {
                    throw InvalidInputException("cannot resample $frames frames at $sampleRate Hz")
                }
                if (buffer.size < bound) buffer = FloatArray(bound.toInt())
            }
        }
    }
}

/**
 * Bounded history of committed text tokens carried into `whisper_full_params.prompt_tokens`; keeps the newest
 * [capacity] tokens. [tokens]`[0 until size]` is the current prompt, oldest first.
 */
internal class PromptHistory(val capacity: Int) {
    init {
        require(capacity >= 0) { "capacity must be >= 0, was $capacity" }
    }

    val tokens: IntArray = IntArray(capacity)

    var size: Int = 0
        private set

    /** Appends [newTokens], dropping the oldest tokens beyond [capacity]. */
    fun append(newTokens: IntArray) {
        if (capacity == 0 || newTokens.isEmpty()) return
        if (newTokens.size >= capacity) {
            newTokens.copyInto(tokens, 0, newTokens.size - capacity, newTokens.size)
            size = capacity
            return
        }
        val overflow = size + newTokens.size - capacity
        if (overflow > 0) {
            tokens.copyInto(tokens, 0, overflow, size)
            size -= overflow
        }
        newTokens.copyInto(tokens, size)
        size += newTokens.size
    }

    fun toList(): List<Int> = tokens.copyOf(size).toList()
}
