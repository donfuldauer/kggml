package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.NativeFailureException
import com.donfuldauer.kggml.core.OutOfMemoryException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import com.donfuldauer.kggml.core.TranscriptionEvent
import com.donfuldauer.kggml.core.TranscriptionEvent.Completed
import com.donfuldauer.kggml.core.TranscriptionEvent.CommittedSegment
import com.donfuldauer.kggml.core.TranscriptionEvent.PartialHypothesis
import com.donfuldauer.kggml.whisper.streaming.VadConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.consumeAsFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

/**
 * [WhisperEngine.transcribeStream] against [FakeWhisperBridge]: event ordering, partial cadence, prompt-token
 * carry-over, cancellation / close cleanup and error semantics. Engines run on real dedicated threads, so waits use
 * real time with a timeout.
 */
class WhisperStreamTest {
    private suspend fun open(bridge: FakeWhisperBridge, config: BackendConfig = BackendConfig.CPU): WhisperEngine =
        WhisperEngine.open(
            modelPath = "/models/ggml-tiny.bin",
            backendConfig = config,
            options = WhisperEngineOptions(),
            bridges = WhisperBridgeHolder { bridge },
            availableBackends = { setOf(ComputeBackend.CPU_ONLY) },
            dispatcherFactory = SerialDispatcherFactory.Default,
        )

    private suspend fun <T> real(block: suspend CoroutineScope.() -> T): T =
        withContext(Dispatchers.Default) { withTimeout(10.seconds, block) }

    private suspend fun awaitCondition(condition: () -> Boolean) = real { while (!condition()) delay(1) }

    // 30 ms VAD frames = 480 samples; use frame-aligned durations so turn boundaries are exact.
    private fun speech(ms: Int) = FloatArray(ms * 16) { if (it % 2 == 0) 0.1f else -0.1f }
    private fun silence(ms: Int) = FloatArray(ms * 16)
    private fun concat(vararg parts: FloatArray): FloatArray {
        val out = FloatArray(parts.sumOf { it.size })
        var pos = 0
        for (p in parts) {
            p.copyInto(out, pos)
            pos += p.size
        }
        return out
    }

    /** Splits [audio] into 20 ms chunks. */
    private fun chunked(audio: FloatArray, chunk: Int = 320): Flow<FloatArray> =
        flowOf(*(audio.indices step chunk).map { audio.copyOfRange(it, minOf(audio.size, it + chunk)) }.toTypedArray())

    private val noPartials = StreamingOptions(partialStepMs = 0)

    // Default VadConfig: 30 ms frames, 90 ms onset (3 frames), 600 ms pause, 300 ms pre-roll.
    // A turn whose speech starts at frame f (frame-aligned) starts at sample (f + 3) * 480 - (4800 + 3 * 480).
    private fun turnStartMs(speechStartMs: Int) = ((speechStartMs / 30 + 3) * 480 - 6_240).coerceAtLeast(0) / 16L

    // ---- ordering & timestamps --------------------------------------------------------------------------

    @Test
    fun committedSegmentsOfEachTurnThenCompleted() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val audio = concat(silence(480), speech(960), silence(990), speech(810), silence(990))
        val events = real { engine.transcribeStream(chunked(audio), options = noPartials).toList() }

        assertEquals(3, events.size, "$events")
        val first = assertIs<CommittedSegment>(events[0])
        val second = assertIs<CommittedSegment>(events[1])
        assertEquals(Completed(segmentCount = 2, audioDurationMs = audio.size / 16L), events[2])
        assertEquals(listOf(0, 1), listOf(first.index, second.index))
        assertEquals(" seg0", first.text)
        assertEquals(turnStartMs(480), first.startMs)
        assertEquals(turnStartMs(480 + 960 + 990), second.startMs)
        // The fake segment spans the whole (1 s padded) buffer: it is clamped to the turn's real audio.
        assertTrue(first.endMs in first.startMs..second.startMs, "$first / $second")
        assertTrue(second.endMs <= audio.size / 16L)

        // Two turns -> two committed decodes on the stream's own state, never the context's transcribe().
        assertEquals(2, bridge.stateTranscribes.items.size)
        assertTrue(bridge.transcribes.items.isEmpty())
        assertTrue(bridge.stateTranscribes.items.all { it.hasListener && it.params.noContext })
        awaitCondition { bridge.liveStateCount == 0 }
        engine.close()
    }

    @Test
    fun pureSilenceCompletesWithoutAnyDecode() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val events = real { engine.transcribeStream(chunked(silence(3_000))).toList() }
        assertEquals(listOf<TranscriptionEvent>(Completed(0, 3_000)), events)
        assertTrue(bridge.stateTranscribes.items.isEmpty())
        engine.close()
    }

    @Test
    fun finalFlushCommitsSpeechThatIsStillInProgress() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        // Audio ends mid-speech: the VAD flush commits the turn (zero-padded to the 1 s floor).
        val events = real { engine.transcribeStream(chunked(concat(silence(300), speech(450))), options = noPartials).toList() }
        assertEquals(2, events.size, "$events")
        assertIs<CommittedSegment>(events[0])
        assertEquals(Completed(1, 750), events[1])
        val decoded = bridge.stateTranscribes.items.single()
        assertTrue(decoded.count >= 16_000, "flushed turn must be padded to the 1,000 ms floor")
        engine.close()
    }

    @Test
    fun blankSegmentsAreNotCommitted() = runTest {
        val bridge = FakeWhisperBridge().apply { streamSegments = { listOf(FakeSegment(0, 500, "   ", listOf(7))) } }
        val engine = open(bridge)
        val events = real { engine.transcribeStream(chunked(concat(speech(900), silence(900))), options = noPartials).toList() }
        assertEquals(listOf<TranscriptionEvent>(Completed(0, 1_800)), events)
        engine.close()
    }

    @Test
    fun partialsPrecedeTheCommitOfTheirTurnAndUseFastDecodeOptions() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val audio = Channel<FloatArray>(Channel.UNLIMITED)
        val events = ConcurrentLog<TranscriptionEvent>()
        val options = StreamingOptions(partialStepMs = 300, transcription = TranscriptionOptions(temperatureIncrement = 0.4f))
        val job = launch(Dispatchers.Default) { engine.transcribeStream(audio.consumeAsFlow(), options = options).collect { events.add(it) } }

        // Feed speech in 300 ms steps and wait for each partial decode: deterministic regardless of scheduling.
        audio.send(speech(450))
        awaitCondition { events.items.count { it is PartialHypothesis } == 1 }
        audio.send(speech(300))
        awaitCondition { events.items.count { it is PartialHypothesis } == 2 }
        audio.send(speech(120)) // less than one step of new audio: no new partial
        audio.send(silence(900))
        audio.close()
        real { job.join() }

        val list = events.items
        val partials = list.filterIsInstance<PartialHypothesis>()
        val committed = list.filterIsInstance<CommittedSegment>().single()
        assertEquals(2, partials.size, "$list")
        assertTrue(list.indexOf(partials.last()) < list.indexOf(committed), "partials of a turn come before its commit")
        assertIs<Completed>(list.last())
        assertTrue(partials.all { it.startMs == committed.startMs }, "partials span the turn in progress")
        assertTrue(partials[0].endMs < partials[1].endMs)
        assertTrue(partials[1].endMs <= committed.endMs)

        val (partialCalls, commitCalls) = bridge.stateTranscribes.items.partition { it.isPartial }
        assertEquals(2, partialCalls.size)
        assertTrue(partialCalls.all { it.params.temperatureIncrement == 0f && it.params.singleSegment && it.params.noTimestamps })
        val commit = commitCalls.single()
        assertEquals(0.4f, commit.params.temperatureIncrement)
        assertTrue(!commit.params.singleSegment && !commit.params.noTimestamps)
        engine.close()
    }

    @Test
    fun partialsAreConflatedWhenTheEngineIsBehind() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        // All 6 s of speech is available at once: instead of 20 partial decodes (one per 300 ms step), the consumer
        // drains everything and decodes the newest audio, so at most a handful of partials run.
        val audio = concat(speech(6_000), silence(900))
        val events = real {
            engine.transcribeStream(flowOf(audio), options = StreamingOptions(partialStepMs = 300)).toList()
        }
        val partialDecodes = bridge.stateTranscribes.items.count { it.isPartial }
        assertTrue(partialDecodes <= 2, "expected conflated partials, got $partialDecodes")
        assertEquals(1, events.filterIsInstance<CommittedSegment>().size)
        engine.close()
    }

    // ---- prompt-token carry-over -----------------------------------------------------------------------

    /** Speech turns separated by pauses plus one 5 s turn split at maxTurnMs = 2 s (sliding windows). */
    private fun multiTurnAudio() = concat(
        speech(900), silence(900),
        speech(5_010), silence(900),
        speech(600), silence(900),
    )

    private val splitVad = VadConfig(maxTurnMs = 2_000)

    @Test
    fun committedTokensAreCarriedIntoLaterDecodesAcrossPausesAndWindowSplits() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val events = real {
            engine.transcribeStream(chunked(multiTurnAudio()), options = StreamingOptions(vad = splitVad, partialStepMs = 0)).toList()
        }
        val calls = bridge.stateTranscribes.items
        assertTrue(calls.size >= 5, "pauses + 2 s splits must produce >= 5 turns, got ${calls.size}")
        val committed = events.filterIsInstance<CommittedSegment>()
        assertEquals(calls.size, committed.size)
        assertEquals(committed.indices.toList(), committed.map { it.index })
        for ((i, call) in calls.withIndex()) {
            assertEquals((0 until i).map { 1000 + it }, call.prompt, "decode $i must see every earlier committed token")
        }
        // Stream-absolute timestamps are monotonic across turns and windows.
        committed.zipWithNext().forEach { (a, b) -> assertTrue(a.endMs <= b.startMs, "$a then $b") }
        engine.close()
    }

    @Test
    fun promptHistoryIsCappedByOptionsAndByTheModelBudget() = runTest {
        for ((optionCap, modelCap, expected) in listOf(Triple(2, 223, 2), Triple(224, 3, 3))) {
            val bridge = FakeWhisperBridge().apply { maxPrompt = modelCap }
            val engine = open(bridge)
            real {
                engine.transcribeStream(
                    chunked(multiTurnAudio()),
                    options = StreamingOptions(vad = splitVad, partialStepMs = 0, maxPromptTokens = optionCap),
                ).toList()
            }
            val calls = bridge.stateTranscribes.items
            for ((i, call) in calls.withIndex()) {
                val all = (0 until i).map { 1000 + it }
                assertEquals(all.takeLast(expected), call.prompt, "cap $optionCap/$modelCap, decode $i")
            }
            engine.close()
        }
    }

    @Test
    fun promptCarryOverCanBeDisabled() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        real {
            engine.transcribeStream(chunked(multiTurnAudio()), options = StreamingOptions(partialStepMs = 0, carryPromptTokens = false)).toList()
        }
        assertTrue(bridge.stateTranscribes.items.size >= 3)
        assertTrue(bridge.stateTranscribes.items.all { it.prompt.isEmpty() })
        engine.close()
    }

    @Test
    fun partialsAreConditionedOnThePromptButDoNotExtendIt() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val audio = Channel<FloatArray>(Channel.UNLIMITED)
        val job = launch(Dispatchers.Default) { engine.transcribeStream(audio.consumeAsFlow()).collect {} }
        audio.send(concat(speech(900), silence(900))) // turn 1 (+ maybe partials)
        awaitCondition { bridge.stateTranscribes.items.any { !it.isPartial } }
        val firstCommit = bridge.stateTranscribes.items.first { !it.isPartial }
        audio.send(speech(450))
        awaitCondition { bridge.stateTranscribes.items.last().let { it.isPartial && it.ordinal > firstCommit.ordinal } }
        audio.send(silence(900))
        audio.close()
        real { job.join() }
        val calls = bridge.stateTranscribes.items
        val laterPartial = calls.last { it.isPartial }
        val secondCommit = calls.last { !it.isPartial }
        assertEquals(listOf(1000 + firstCommit.ordinal), laterPartial.prompt)
        assertEquals(listOf(1000 + firstCommit.ordinal), secondCommit.prompt, "partial tokens are never carried")
        engine.close()
    }

    @Test
    fun transcriptionOptionsReachEveryDecodeAndNoContextIsForced() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge, BackendConfig.CPU.copy(cpuThreads = 3))
        val options = StreamingOptions(
            partialStepMs = 0,
            transcription = TranscriptionOptions(language = "de", translate = true, noContext = false, maxTokens = 12, initialPrompt = "Kggml"),
        )
        real { engine.transcribeStream(chunked(concat(speech(900), silence(900))), options = options).toList() }
        val params = bridge.stateTranscribes.items.single().params
        assertEquals("de", params.language)
        assertTrue(params.translate)
        assertTrue(params.noContext, "streams always supply their history explicitly")
        assertEquals(12, params.maxTokens)
        assertEquals(3, params.threads)
        assertEquals("Kggml", params.initialPrompt)
        engine.close()
    }

    // ---- input formats -----------------------------------------------------------------------------------

    @Test
    fun sixteenKhzMonoFloatBypassesTheResampler() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        real { engine.transcribeStream(chunked(concat(speech(900), silence(900)))).toList() }
        assertTrue(bridge.createdResamplers.items.isEmpty())
        assertTrue(bridge.conversions.items.isEmpty())
        engine.close()
    }

    @Test
    fun otherRatesChannelsAndPcm16GoThroughOneStreamingResamplerThatIsFreed() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        // 48 kHz stereo float: 1.8 s.
        val mono = concat(speech(900), silence(900))
        val stereo48k = FloatArray(mono.size * 3 * 2) { mono[it / 6] }
        val events = real {
            engine.transcribeStream(flowOf(stereo48k.copyOfRange(0, 9_600), stereo48k.copyOfRange(9_600, stereo48k.size)), 48_000, 2).toList()
        }
        assertEquals(Completed(1, 1_800), events.last())
        assertEquals(listOf("stream-f32:4800:2:48000", "stream-f32:${stereo48k.size / 2 - 4_800}:2:48000", "stream-flush"), bridge.conversions.items)
        assertEquals(1, bridge.createdResamplers.items.size)
        awaitCondition { bridge.freedResamplers.items == bridge.createdResamplers.items }

        // 16 kHz mono int16 still needs normalization: S16 resampler at 16 kHz.
        val pcm16 = ShortArray(mono.size) { (mono[it] * 32767).toInt().toShort() }
        val events16 = real { engine.transcribeStream(flowOf(pcm16), options = noPartials).toList() }
        assertEquals(Completed(1, 1_800), events16.last())
        assertTrue(bridge.conversions.items.contains("stream-s16:${pcm16.size}:1:16000"))
        awaitCondition { bridge.freedResamplers.items.size == 2 }
        engine.close()
    }

    @Test
    fun invalidFormatOrClosedEngineFailsImmediately() = runTest {
        val engine = open(FakeWhisperBridge())
        assertFailsWith<InvalidInputException> { engine.transcribeStream(flowOf(FloatArray(10)), sampleRate = 500) }
        assertFailsWith<InvalidInputException> { engine.transcribeStream(flowOf(ShortArray(10)), channels = 9) }
        engine.close()
        assertFailsWith<IllegalStateException> { engine.transcribeStream(flowOf(FloatArray(10))) }
    }

    @Test
    fun malformedChunkIsATerminalInvalidInputError() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val events = real { engine.transcribeStream(flowOf(FloatArray(4), FloatArray(3)), channels = 2).toList() }
        val error = assertIs<TranscriptionEvent.Error>(events.single())
        assertIs<InvalidInputException>(error.exception)
        awaitCondition { bridge.liveStateCount == 0 && bridge.freedResamplers.items.size == 1 }
        engine.close()
    }

    // ---- errors --------------------------------------------------------------------------------------------

    @Test
    fun nativeDecodeFailureIsATerminalErrorEventAndTheFlowCompletesNormally() = runTest {
        val bridge = FakeWhisperBridge().apply { stateTranscribeStatus = { WhisperStatus.NATIVE_FAILURE } }
        val engine = open(bridge)
        val events = real { engine.transcribeStream(chunked(concat(speech(900), silence(900), speech(900), silence(900))), options = noPartials).toList() }
        val error = assertIs<TranscriptionEvent.Error>(events.single(), "$events")
        assertIs<NativeFailureException>(error.exception)
        assertTrue(error.exception.message!!.contains("kggml_whisper_transcribe_with_state"), error.exception.message)
        assertEquals(1, bridge.stateTranscribes.items.size, "nothing is decoded after a terminal error")
        awaitCondition { bridge.liveStateCount == 0 }
        engine.transcribe(FloatArray(16_000)) // engine still usable
        engine.close()
    }

    @Test
    fun stateAllocationFailureIsAnErrorEvent() = runTest {
        val bridge = FakeWhisperBridge().apply { stateCreateStatus = WhisperStatus.OUT_OF_MEMORY }
        val engine = open(bridge)
        val events = real { engine.transcribeStream(chunked(speech(900))).toList() }
        assertIs<OutOfMemoryException>(assertIs<TranscriptionEvent.Error>(events.single()).exception)
        engine.close()
    }

    @Test
    fun listenerStopStatusSurfacesAsTypedError() = runTest {
        // A failing read inside new_segment_callback stops the decode; the Kotlin failure is reported, typed.
        val bridge = object : NativeWhisperBridge by FakeWhisperBridge() {
            override fun stateSegmentTokens(state: Long, index: Int) = RawTokensResult(WhisperStatus.OUT_OF_MEMORY)
        }
        val engine = WhisperEngine.open(
            "/m.bin", BackendConfig.CPU, WhisperEngineOptions(), WhisperBridgeHolder { bridge },
            { setOf(ComputeBackend.CPU_ONLY) }, SerialDispatcherFactory.Default,
        )
        val events = real { engine.transcribeStream(chunked(concat(speech(900), silence(900))), options = noPartials).toList() }
        val error = assertIs<TranscriptionEvent.Error>(events.single())
        assertIs<OutOfMemoryException>(error.exception)
        assertTrue(error.exception.message!!.contains("kggml_whisper_state_get_segment_tokens"))
        engine.close()
    }

    @Test
    fun upstreamAudioFailurePropagatesAndStillFreesEverything() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val failing = flow {
            emit(speech(300).let { FloatArray(it.size * 2) { i -> it[i / 2] } })
            throw IllegalArgumentException("microphone unplugged")
        }
        val e = assertFailsWith<IllegalArgumentException> { real { engine.transcribeStream(failing, channels = 2).toList() } }
        assertEquals("microphone unplugged", e.message)
        awaitCondition { bridge.liveStateCount == 0 && bridge.freedResamplers.items.size == 1 }
        engine.close()
    }

    // ---- cancellation & lifecycle --------------------------------------------------------------------------

    @Test
    fun cancellingTheCollectorAbortsNativeDecodingAndFreesTheStateBeforeJoinReturns() = runTest {
        val bridge = FakeWhisperBridge().apply { blockStateTranscribe = { true } }
        val engine = open(bridge)
        val audio = Channel<FloatArray>(Channel.UNLIMITED)
        val job: Job = launch(Dispatchers.Default) {
            engine.transcribeStream(audio.consumeAsFlow(), sampleRate = 48_000).collect {}
        }
        audio.send(FloatArray(48 * 900) { if (it % 2 == 0) 0.1f else -0.1f })
        audio.send(FloatArray(48 * 900))
        awaitCondition { bridge.transcriptionStarted.load() == 1 }
        assertTrue(engine.isTranscribing)
        real { job.cancelAndJoin() }

        val handle = bridge.createdHandles.single()
        assertTrue(handle in bridge.abortRequests.items, "collector cancellation must reach the native abort flag")
        assertEquals(0, bridge.liveStateCount, "the stream's whisper_state is freed before the collection finishes")
        assertEquals(bridge.createdResamplers.items, bridge.freedResamplers.items)
        assertTrue(bridge.freedHandles.isEmpty(), "the engine's own context stays alive")

        bridge.blockStateTranscribe = { false }
        assertEquals("hello world", engine.transcribe(FloatArray(16_000)).text)
        engine.close()
    }

    @Test
    fun closingTheEngineEndsARunningStreamWithInferenceCancelled() = runTest {
        val bridge = FakeWhisperBridge().apply { blockStateTranscribe = { true } }
        val engine = open(bridge)
        val audio = Channel<FloatArray>(Channel.UNLIMITED)
        val events = ConcurrentLog<TranscriptionEvent>()
        val job = launch(Dispatchers.Default) { engine.transcribeStream(audio.consumeAsFlow(), options = noPartials).collect { events.add(it) } }
        audio.send(concat(speech(900), silence(900)))
        awaitCondition { bridge.transcriptionStarted.load() == 1 }
        engine.close()
        real { job.join() }
        assertIs<InferenceCancelledException>(assertIs<TranscriptionEvent.Error>(events.items.single()).exception)
        awaitCondition { bridge.freedHandles.size == 1 }
        assertEquals(0, bridge.liveStateCount, "close frees the stream's state (before the context)")
    }

    @Test
    fun closingTheEngineEndsAnIdleStreamPromptly() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val audio = Channel<FloatArray>(Channel.UNLIMITED) // never closed, no speech: the consumer is idle
        val events = ConcurrentLog<TranscriptionEvent>()
        val job = launch(Dispatchers.Default) { engine.transcribeStream(audio.consumeAsFlow()).collect { events.add(it) } }
        audio.send(silence(300))
        awaitCondition { bridge.createdStates.size == 1 }
        engine.close()
        real { job.join() }
        assertIs<InferenceCancelledException>(assertIs<TranscriptionEvent.Error>(events.items.single()).exception)
        awaitCondition { bridge.freedHandles.size == 1 }
        assertEquals(0, bridge.liveStateCount)
    }

    @Test
    fun collectingAfterCloseEmitsInferenceCancelled() = runTest {
        val engine = open(FakeWhisperBridge())
        val stream = engine.transcribeStream(flowOf(speech(300)))
        engine.close()
        val events = real { stream.toList() }
        assertIs<InferenceCancelledException>(assertIs<TranscriptionEvent.Error>(events.single()).exception)
    }

    @Test
    fun abortActiveTranscriptionEndsTheStreamWithAnError() = runTest {
        val bridge = FakeWhisperBridge().apply { blockStateTranscribe = { true } }
        val engine = open(bridge)
        val events = ConcurrentLog<TranscriptionEvent>()
        val job = launch(Dispatchers.Default) {
            engine.transcribeStream(chunked(concat(speech(900), silence(900))), options = noPartials).collect { events.add(it) }
        }
        awaitCondition { engine.isTranscribing }
        assertTrue(engine.abortActiveTranscription())
        real { job.join() }
        assertIs<InferenceCancelledException>(assertIs<TranscriptionEvent.Error>(events.items.single()).exception)
        awaitCondition { bridge.liveStateCount == 0 }
        engine.close()
    }

    @Test
    fun concurrentCollectionsOwnIndependentStates() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val audio = chunked(concat(speech(900), silence(900)))
        val stream = engine.transcribeStream(audio, options = noPartials)
        val a = launch(Dispatchers.Default) { stream.toList() }
        val b = launch(Dispatchers.Default) { stream.toList() }
        real {
            a.join()
            b.join()
        }
        val states = bridge.stateTranscribes.items.map { it.state }.toSet()
        assertEquals(2, states.size, "each collection decodes on its own whisper_state")
        assertEquals(2, bridge.createdStates.size)
        assertEquals(0, bridge.liveStateCount)
        engine.close()
    }

    @Test
    fun reinitializeDuringAStreamRecreatesItsStateOnTheNewContext() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        val audio = Channel<FloatArray>(Channel.UNLIMITED)
        val events = ConcurrentLog<TranscriptionEvent>()
        val job = launch(Dispatchers.Default) {
            engine.transcribeStream(audio.consumeAsFlow(), options = noPartials).collect { events.add(it) }
        }
        audio.send(concat(speech(900), silence(900)))
        awaitCondition { events.items.size == 1 }
        engine.reinitialize(BackendConfig.CPU)
        audio.send(concat(speech(900), silence(900)))
        audio.close()
        real { job.join() }

        val calls = bridge.stateTranscribes.items
        assertEquals(2, calls.size)
        assertNotEquals(calls[0].handle, calls[1].handle, "second turn runs on the reloaded context")
        assertNotEquals(calls[0].state, calls[1].state)
        assertEquals(listOf(1000), calls[1].prompt, "prompt history survives the reload")
        assertEquals(Completed(2, 3_600), events.items.last())
        assertEquals(0, bridge.liveStateCount)
        engine.close()
    }

    // ---- options & helpers ---------------------------------------------------------------------------------

    @Test
    fun streamingOptionsAreValidated() {
        assertFailsWith<IllegalArgumentException> { StreamingOptions(partialStepMs = 50) }
        assertFailsWith<IllegalArgumentException> { StreamingOptions(partialStepMs = 30_001) }
        assertFailsWith<IllegalArgumentException> { StreamingOptions(maxPromptTokens = -1) }
        assertFailsWith<IllegalArgumentException> { StreamingOptions(bufferMs = 999) }
        assertFailsWith<IllegalArgumentException> { StreamingOptions(bufferMs = 300_001) }
        StreamingOptions(partialStepMs = 0, maxPromptTokens = 0, bufferMs = 1_000)
        assertEquals(300, StreamingOptions.DEFAULT.partialStepMs)
        assertEquals(224, StreamingOptions.DEFAULT.maxPromptTokens)
    }

    @Test
    fun promptHistoryKeepsTheNewestTokens() {
        val h = PromptHistory(4)
        h.append(intArrayOf(1, 2))
        assertEquals(listOf(1, 2), h.toList())
        h.append(intArrayOf())
        h.append(intArrayOf(3, 4, 5))
        assertEquals(listOf(2, 3, 4, 5), h.toList())
        h.append(intArrayOf(6, 7, 8, 9, 10))
        assertEquals(listOf(7, 8, 9, 10), h.toList())
        h.append(intArrayOf(11))
        assertEquals(listOf(8, 9, 10, 11), h.toList())
        val none = PromptHistory(0)
        none.append(intArrayOf(1))
        assertEquals(0, none.size)
        assertFailsWith<IllegalArgumentException> { PromptHistory(-1) }
    }

    @Test
    fun smallBufferAppliesBackPressureWithoutDroppingAudio() = runTest {
        val bridge = FakeWhisperBridge()
        val engine = open(bridge)
        // 1 s ring buffer, 8 s of audio in 2 s chunks: the producer must wait for the consumer, never drop.
        val audio = concat(speech(3_000), silence(900), speech(3_000), silence(1_100))
        val chunks = (audio.indices step 32_000).map { audio.copyOfRange(it, minOf(audio.size, it + 32_000)) }
        val events = real { engine.transcribeStream(flowOf(*chunks.toTypedArray()), options = StreamingOptions(bufferMs = 1_000, partialStepMs = 0)).toList() }
        assertEquals(Completed(2, 8_000), events.last())
        val decoded = bridge.stateTranscribes.items.sumOf { it.count }
        assertTrue(decoded >= 6_000 * 16, "all speech reached the decoder ($decoded samples)")
        engine.close()
    }
}
