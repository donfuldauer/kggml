package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import com.donfuldauer.kggml.core.TranscriptionEvent
import com.donfuldauer.kggml.core.TranscriptionEvent.Completed
import com.donfuldauer.kggml.core.TranscriptionEvent.CommittedSegment
import com.donfuldauer.kggml.core.TranscriptionEvent.PartialHypothesis
import com.donfuldauer.kggml.whisper.streaming.VadConfig
import java.io.File
import java.util.Collections
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * [WhisperEngine.transcribeStream] against the real native library (JNI -> kggml_whisper -> whisper.cpp
 * `whisper_full_with_state` with `new_segment_callback` and `prompt_tokens`) and the synthetic [TinyWhisperModel].
 *
 * A recording [NativeWhisperBridge] wraps the JNI bridge to observe what crosses the boundary (prompt tokens, the
 * tokens read back, state lifecycle) and reads the prompt whisper.cpp actually received through a test hook.
 */
class JvmNativeWhisperStreamTest {
    private val requireNative = System.getProperty("kggml.test.requireNative").toBoolean()
    private lateinit var modelFile: File

    @BeforeTest
    fun setUp() {
        val present = System.getProperty("kggml.whisper.native.path") != null
        if (requireNative) assertTrue(present, "-Pkggml.requireNative=true but the whisper host build was not found")
        assumeTrue(present, "native whisper host build not found; run kggml-whisper/native/scripts/build-host.sh")
        modelFile = File.createTempFile("kggml-whisper-tiny-stream-", ".bin").apply {
            deleteOnExit()
            writeBytes(TinyWhisperModel.bytes())
        }
    }

    @AfterTest
    fun tearDown() {
        if (::modelFile.isInitialized) modelFile.delete()
    }

    /** One `transcribeWithState` call as observed at the JNI boundary. */
    private class Call(val partial: Boolean, val prompt: List<Int>) {
        @Volatile var nativePrompt: List<Int>? = null
        val segments: MutableList<Pair<String, List<Int>>> = Collections.synchronizedList(mutableListOf())
    }

    private class RecordingBridge(private val delegate: NativeWhisperBridge) : NativeWhisperBridge by delegate {
        val calls: MutableList<Call> = Collections.synchronizedList(mutableListOf())
        val states: MutableList<Long> = Collections.synchronizedList(mutableListOf())

        @Volatile private var lastText = ""

        override fun stateCreate(handle: Long): RawHandleResult = delegate.stateCreate(handle).also { states += it.handle }

        override fun transcribeWithState(
            handle: Long,
            state: Long,
            samples: FloatArray,
            count: Int,
            params: RawTranscribeParams,
            promptTokens: IntArray,
            promptCount: Int,
            listener: WhisperSegmentListener?,
        ): Int {
            val call = Call(params.singleSegment && params.noTimestamps, promptTokens.copyOf(promptCount).toList())
            calls += call
            val status = delegate.transcribeWithState(handle, state, samples, count, params, promptTokens, promptCount, listener)
            call.nativePrompt = KggmlWhisperJni.nativeTestStateLastPrompt(state)?.toList()
            return status
        }

        override fun stateSegment(state: Long, index: Int): RawSegmentResult =
            delegate.stateSegment(state, index).also { lastText = it.text.decodeToString() }

        override fun stateSegmentTokens(state: Long, index: Int): RawTokensResult =
            delegate.stateSegmentTokens(state, index).also { calls.last().segments += lastText to it.tokens.toList() }
    }

    private suspend fun engine(bridge: NativeWhisperBridge): WhisperEngine = WhisperEngine.open(
        modelPath = modelFile.absolutePath,
        backendConfig = BackendConfig.CPU.copy(cpuThreads = 2),
        options = WhisperEngineOptions(),
        bridges = WhisperBridgeHolder { bridge },
        availableBackends = { setOf(ComputeBackend.CPU_ONLY) },
        dispatcherFactory = SerialDispatcherFactory.Default,
    )

    /** Deterministic "speech": a 0.1-amplitude sawtooth (the tiny model decodes text segments from it). */
    private fun speech(ms: Int) = FloatArray(ms * 16) { 0.1f * ((it % 100) / 50f - 1f) }
    private fun silence(ms: Int) = FloatArray(ms * 16)

    /** [parts] streamed in 40 ms chunks. */
    private fun chunked(vararg parts: FloatArray, chunk: Int = 640): Flow<FloatArray> = flow {
        for (p in parts) for (i in p.indices step chunk) emit(p.copyOfRange(i, minOf(p.size, i + chunk)))
    }

    private fun liveStates(): Long = KggmlWhisperJni.nativeTestLiveStateCount()

    @Test
    fun multiMinuteStreamCommitsOrderedSegmentsAndCarriesPromptTokensThroughWhisper() = runBlocking {
        val recording = RecordingBridge(createPlatformWhisperBridge())
        val baseline = liveStates()
        val engine = engine(recording)
        try {
            // ~2.5 minutes: 30 turns of 3 s speech + 0.9 s pause, plus one 40 s monologue split into 30 s windows.
            val parts = buildList {
                repeat(30) {
                    add(speech(3_000))
                    add(silence(900))
                }
                add(speech(40_020))
                add(silence(990))
            }.toTypedArray()
            val totalMs = parts.sumOf { it.size } / 16L
            val events = withTimeout(120.seconds) {
                engine.transcribeStream(chunked(*parts), options = StreamingOptions(partialStepMs = 1_000)).toList()
            }

            assertTrue(events.none { it is TranscriptionEvent.Error }, "$events")
            val completed = assertIs<Completed>(events.last())
            assertEquals(totalMs, completed.audioDurationMs)
            val committed = events.filterIsInstance<CommittedSegment>()
            assertEquals(completed.segmentCount, committed.size)
            assertEquals(committed.indices.toList(), committed.map { it.index })
            committed.zipWithNext().forEach { (a, b) -> assertTrue(a.endMs <= b.startMs, "$a then $b") }
            assertTrue(committed.all { it.endMs <= totalMs })
            assertTrue(committed.isNotEmpty(), "the tiny model should commit text from the sawtooth bursts")

            // Prompt carry-over, verified at whisper_full_params level: each commit is conditioned on the text tokens
            // of every earlier non-blank committed segment (newest kept, model budget n_text_ctx/2 - 1 = 31).
            val commits = recording.calls.filter { !it.partial }
            assertTrue(commits.size >= 32, "30 pauses + a 30 s split, got ${commits.size}")
            val history = mutableListOf<Int>()
            for ((i, call) in commits.withIndex()) {
                assertEquals(history.takeLast(31), call.prompt, "commit $i prompt")
                assertEquals(call.prompt, call.nativePrompt, "commit $i: prompt_tokens reached whisper.cpp unchanged")
                call.segments.filter { it.first.isNotBlank() }.forEach { history += it.second }
            }
            assertTrue(commits.drop(1).any { it.prompt.isNotEmpty() }, "some commits must carry tokens forward")
            assertTrue(recording.calls.any { it.partial }, "partials run while a turn is in progress")

            assertEquals(1, recording.states.size, "one whisper_state per collection")
            assertEquals(baseline, liveStates(), "the stream's whisper_state is freed when the collection completes")
        } finally {
            engine.close()
        }
    }

    @Test
    fun cancellingTheCollectorFreesTheNativeStateDeterministically() = runBlocking {
        val baseline = liveStates()
        WhisperEngine(modelFile.absolutePath, BackendConfig.CPU.copy(cpuThreads = 2)).use { engine ->
            // Endless 48 kHz stereo audio (through the native streaming resampler) with short pauses.
            val endless = flow {
                val burst = FloatArray(48 * 2 * 1_500) { 0.1f * ((it / 2 % 300) / 150f - 1f) }
                val pause = FloatArray(48 * 2 * 900)
                while (true) {
                    emit(burst)
                    emit(pause)
                }
            }
            val events: MutableList<TranscriptionEvent> = Collections.synchronizedList(mutableListOf())
            val job = launch(Dispatchers.Default) {
                engine.transcribeStream(endless, sampleRate = 48_000, channels = 2, options = StreamingOptions(partialStepMs = 300))
                    .collect { events += it }
            }
            withTimeout(60.seconds) { while (events.size < 3) delay(1) }
            assertEquals(baseline + 1, liveStates())
            withTimeout(30.seconds) { job.cancelAndJoin() }
            assertEquals(baseline, liveStates(), "state freed before cancelAndJoin returned")
            assertTrue(events.toList().none { it is TranscriptionEvent.Error || it is Completed })

            // The engine is still fully usable, including for a new stream.
            val again = engine.transcribeStream(chunked(speech(1_200), silence(900))).toList()
            assertIs<Completed>(again.last())
            assertEquals(baseline, liveStates())
        }
    }

    @Test
    fun closingTheEngineMidStreamEndsWithInferenceCancelledAndFreesEverything() = runBlocking {
        val baseline = liveStates()
        val engine = WhisperEngine(modelFile.absolutePath, BackendConfig.CPU)
        val endless = flow { while (true) emit(speech(100)) } // continuous speech: 30 s window splits forever
        val events: MutableList<TranscriptionEvent> = Collections.synchronizedList(mutableListOf())
        val job = launch(Dispatchers.Default) {
            engine.transcribeStream(endless, options = StreamingOptions(vad = VadConfig(maxTurnMs = 2_000)))
                .collect { events += it }
        }
        withTimeout(60.seconds) { while (events.toList().none { it is CommittedSegment || it is PartialHypothesis }) delay(1) }
        engine.close()
        withTimeout(30.seconds) { job.join() }
        assertIs<InferenceCancelledException>(assertIs<TranscriptionEvent.Error>(events.last()).exception)
        withTimeout(10.seconds) { while (liveStates() != baseline) delay(5) }
    }

    @Test
    fun pcm16StreamAndNoTemporaryFiles() = runBlocking {
        val tmp = File(System.getProperty("java.io.tmpdir"))
        val cwd = File(".").absoluteFile
        val before = tmp.list().orEmpty().toSet() to cwd.list().orEmpty().toSet()
        WhisperEngine(modelFile.absolutePath, BackendConfig.CPU).use { engine ->
            val pcm = (speech(2_000) + silence(900) + speech(1_500)).let { f -> ShortArray(f.size) { (f[it] * 32767).toInt().toShort() } }
            val chunks = flow { for (i in pcm.indices step 800) emit(pcm.copyOfRange(i, minOf(pcm.size, i + 800))) }
            val events = engine.transcribeStream(chunks).toList()
            assertEquals(4_400L, assertIs<Completed>(events.last()).audioDurationMs)

            // Taking only the first event cancels the rest of the stream: still no files, no leaks.
            val first = engine.transcribeStream(chunked(speech(1_000), silence(900))).first()
            assertTrue(first !is TranscriptionEvent.Error, "$first")
        }
        val after = tmp.list().orEmpty().toSet() to cwd.list().orEmpty().toSet()
        assertEquals(emptySet(), after.first - before.first, "no temporary files in ${tmp.path}")
        assertEquals(emptySet(), after.second - before.second, "no files in the working directory")
    }
}
