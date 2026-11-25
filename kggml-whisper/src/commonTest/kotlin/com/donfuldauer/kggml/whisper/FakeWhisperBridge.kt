package com.donfuldauer.kggml.whisper

import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.AtomicLong
import kotlin.concurrent.atomics.AtomicReference
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource

/** Thread-safe append-only list (fake bridge methods are called from several engine threads at once). */
internal class ConcurrentLog<T> {
    private val ref = AtomicReference<List<T>>(emptyList())

    val items: List<T> get() = ref.load()

    fun add(item: T) {
        while (true) {
            val current = ref.load()
            if (ref.compareAndSet(current, current + item)) return
        }
    }
}

/** One recorded `kggml_whisper_init_from_file()` call. */
internal data class InitCall(val modelPath: String, val backendKind: Int, val gpuDevice: Int, val flashAttention: Boolean)

/** One recorded `kggml_whisper_transcribe()` call; [samples] is the exact array instance that crossed the bridge. */
internal class TranscribeCall(val handle: Long, val samples: FloatArray, val count: Int, val params: RawTranscribeParams)

/** One recorded `kggml_whisper_transcribe_with_state()` call (inputs copied at call time). */
internal class StateTranscribeCall(
    val ordinal: Int,
    val handle: Long,
    val state: Long,
    val samples: FloatArray,
    val params: RawTranscribeParams,
    val prompt: List<Int>,
    val hasListener: Boolean,
) {
    val count: Int get() = samples.size

    /** Partial decodes of the stream run single-segment without timestamps. */
    val isPartial: Boolean get() = params.singleSegment && params.noTimestamps
}

/** A scripted streaming segment (times relative to the decoded buffer). */
internal data class FakeSegment(val startMs: Long, val endMs: Long, val text: String, val tokens: List<Int>)

/**
 * Deterministic in-memory stand-in for the native whisper library.
 *
 * Contexts are numbered handles. [initStatus] decides the status per requested backend kind; GPU kinds report
 * `use_gpu = 1`. [blockTranscription] makes `transcribe` spin (on the engine's own thread) until an abort is
 * requested for that context, exactly like whisper.cpp polling its abort callback.
 */
internal class FakeWhisperBridge(
    var coreMl: Boolean = false,
    var initStatus: (backendKind: Int) -> Int = { WhisperStatus.OK },
    var transcribeStatus: Int = WhisperStatus.OK,
    var segments: List<Triple<Long, Long, String>> = listOf(Triple(0L, 1000L, " hello"), Triple(1000L, 2000L, " world")),
) : NativeWhisperBridge {
    private val nextHandle = AtomicLong(0x1000)
    private val live = ConcurrentLog<Long>()
    private val freed = ConcurrentLog<Long>()
    private val aborts = AtomicReference<Set<Long>>(emptySet())
    private val runningCount = AtomicInt(0)
    private val maxRunning = AtomicInt(0)

    val inits = ConcurrentLog<InitCall>()
    val transcribes = ConcurrentLog<TranscribeCall>()
    val conversions = ConcurrentLog<String>()
    val abortRequests = ConcurrentLog<Long>()
    val blockTranscription = AtomicBoolean(false)
    val transcriptionStarted = AtomicInt(0)

    /** Highest number of transcriptions observed executing at the same time (across engines). */
    val maxConcurrentTranscriptions: Int get() = maxRunning.load()

    val freedHandles: List<Long> get() = freed.items
    val createdHandles: List<Long> get() = live.items

    var lastErrorMessage: String = ""

    // ---- streaming states ----
    private val nextState = AtomicLong(0x5000)
    private val liveStates = ConcurrentLog<Pair<Long, Long>>() // state -> owning context
    private val freedStateLog = ConcurrentLog<Long>()
    private val stateResults = AtomicReference<Map<Long, List<FakeSegment>>>(emptyMap())
    private val stateCallOrdinal = AtomicInt(0)

    val stateTranscribes = ConcurrentLog<StateTranscribeCall>()
    var stateCreateStatus: Int = WhisperStatus.OK
    var maxPrompt: Int = 223

    /** Status returned by every `transcribeWithState` (after the listener ran); OK by default. */
    var stateTranscribeStatus: (StateTranscribeCall) -> Int = { WhisperStatus.OK }

    /** Blocks streaming decodes selected by this predicate until aborted (like [blockTranscription]). */
    var blockStateTranscribe: (StateTranscribeCall) -> Boolean = { false }

    /** Segments produced by a streaming decode; by default one segment per call with a unique token. */
    var streamSegments: (StateTranscribeCall) -> List<FakeSegment> = { call ->
        val durationMs = call.count * 1000L / 16_000
        listOf(FakeSegment(0, durationMs, if (call.isPartial) " partial${call.ordinal}" else " seg${call.ordinal}", listOf(1000 + call.ordinal)))
    }

    val createdStates: List<Long> get() = liveStates.items.map { it.first }
    val freedStates: List<Long> get() = freedStateLog.items
    val liveStateCount: Int get() = createdStates.size - freedStates.size

    // ---- streaming resamplers ----
    private class FakeResampler(val inRate: Int, val channels: Int, val format: Int) {
        var inTotal = 0L
        var outTotal = 0L
    }

    private val nextResampler = AtomicLong(0x9000)
    private val resamplers = AtomicReference<Map<Long, FakeResampler>>(emptyMap())
    val createdResamplers = ConcurrentLog<Long>()
    val freedResamplers = ConcurrentLog<Long>()

    override fun hasCoreMl(): Boolean = coreMl

    override fun moduleVersion(): String = "1.0.0"

    override fun upstreamVersion(): String = "1.8.4"

    override fun upstreamCommit(): String = "9386f239401074690479731c1e41683fbbeac557"

    override fun systemInfo(): String = "FAKE"

    override fun initFromFile(modelPath: String, backendKind: Int, gpuDevice: Int, flashAttention: Boolean): RawHandleResult {
        inits.add(InitCall(modelPath, backendKind, gpuDevice, flashAttention))
        val status = initStatus(backendKind)
        if (status != WhisperStatus.OK) {
            lastErrorMessage = "fake init failure for kind $backendKind"
            return RawHandleResult(status, 0L)
        }
        val handle = nextHandle.fetchAndAdd(1L)
        live.add(handle)
        backendOf[handle] = backendKind
        return RawHandleResult(status, handle)
    }

    private val backendOfRef = AtomicReference<Map<Long, Int>>(emptyMap())
    private val backendOf = object {
        operator fun set(handle: Long, kind: Int) {
            while (true) {
                val cur = backendOfRef.load()
                if (backendOfRef.compareAndSet(cur, cur + (handle to kind))) return
            }
        }

        operator fun get(handle: Long): Int = backendOfRef.load()[handle] ?: -1
    }

    override fun free(handle: Long) {
        check(handle in live.items) { "free of unknown handle $handle" }
        check(handle !in freed.items) { "double free of $handle" }
        freed.add(handle)
    }

    private fun checkAlive(handle: Long) {
        check(handle in live.items && handle !in freed.items) { "use of dead handle $handle" }
    }

    override fun effectiveBackend(handle: Long): RawIntResult {
        checkAlive(handle)
        // CoreML without an .mlmodelc encoder falls back to Metal, like the real library.
        val kind = backendOf[handle]
        return RawIntResult(WhisperStatus.OK, if (kind == WhisperBackendKind.COREML && !coreMlEncoderPresent) WhisperBackendKind.METAL else kind)
    }

    var coreMlEncoderPresent: Boolean = true

    override fun usesGpu(handle: Long): RawIntResult {
        checkAlive(handle)
        return RawIntResult(WhisperStatus.OK, if (backendOf[handle] == WhisperBackendKind.CPU) 0 else 1)
    }

    override fun requestAbort(handle: Long) {
        abortRequests.add(handle)
        while (true) {
            val cur = aborts.load()
            if (aborts.compareAndSet(cur, cur + handle)) return
        }
    }

    override fun clearAbort(handle: Long) {
        checkAlive(handle)
        while (true) {
            val cur = aborts.load()
            if (aborts.compareAndSet(cur, cur - handle)) return
        }
    }

    override fun abortRequested(handle: Long): Boolean = handle in aborts.load()

    override fun transcribe(handle: Long, samples: FloatArray, count: Int, params: RawTranscribeParams): Int {
        checkAlive(handle)
        transcribes.add(TranscribeCall(handle, samples, count, params))
        if (abortRequested(handle)) {
            lastErrorMessage = "transcription cancelled before it started"
            return WhisperStatus.CANCELLED
        }
        val running = runningCount.addAndFetch(1)
        while (true) {
            val max = maxRunning.load()
            if (running <= max || maxRunning.compareAndSet(max, running)) break
        }
        transcriptionStarted.addAndFetch(1)
        try {
            if (blockTranscription.load()) {
                val deadline = TimeSource.Monotonic.markNow() + 10.seconds
                while (!abortRequested(handle)) {
                    check(deadline.hasNotPassedNow()) { "fake transcription was never aborted" }
                }
            }
            if (abortRequested(handle)) {
                lastErrorMessage = "transcription cancelled"
                return WhisperStatus.CANCELLED
            }
            if (transcribeStatus != WhisperStatus.OK) lastErrorMessage = "fake transcribe failure"
            return transcribeStatus
        } finally {
            runningCount.addAndFetch(-1)
        }
    }

    override fun segmentCount(handle: Long): RawIntResult {
        checkAlive(handle)
        return RawIntResult(WhisperStatus.OK, segments.size)
    }

    override fun segment(handle: Long, index: Int): RawSegmentResult {
        checkAlive(handle)
        val (t0, t1, text) = segments.getOrNull(index) ?: return RawSegmentResult(WhisperStatus.INVALID_ARGUMENT)
        return RawSegmentResult(WhisperStatus.OK, t0, t1, 0.25f, text.encodeToByteArray())
    }

    override fun stateCreate(handle: Long): RawHandleResult {
        checkAlive(handle)
        if (stateCreateStatus != WhisperStatus.OK) {
            lastErrorMessage = "fake whisper_init_state failure"
            return RawHandleResult(stateCreateStatus, 0L)
        }
        val state = nextState.fetchAndAdd(1L)
        liveStates.add(state to handle)
        return RawHandleResult(WhisperStatus.OK, state)
    }

    override fun stateFree(state: Long) {
        check(state in createdStates) { "free of unknown state $state" }
        check(state !in freedStates) { "double free of state $state" }
        val owner = liveStates.items.first { it.first == state }.second
        check(owner !in freed.items) { "state $state freed after its context $owner" }
        freedStateLog.add(state)
    }

    private fun checkStateAlive(state: Long) {
        check(state in createdStates && state !in freedStates) { "use of dead state $state" }
    }

    override fun maxPromptTokens(handle: Long): RawIntResult {
        checkAlive(handle)
        return RawIntResult(WhisperStatus.OK, maxPrompt)
    }

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
        checkAlive(handle)
        checkStateAlive(state)
        check(liveStates.items.first { it.first == state }.second == handle) { "state $state used with foreign context" }
        check(promptCount in 0..promptTokens.size) { "promptCount $promptCount out of range" }
        val call = StateTranscribeCall(
            ordinal = stateCallOrdinal.fetchAndAdd(1),
            handle = handle,
            state = state,
            samples = samples.copyOf(count),
            params = params,
            prompt = promptTokens.copyOf(promptCount).toList(),
            hasListener = listener != null,
        )
        stateTranscribes.add(call)
        setResults(state, emptyList())
        if (abortRequested(handle)) {
            lastErrorMessage = "transcription cancelled before it started"
            return WhisperStatus.CANCELLED
        }
        transcriptionStarted.addAndFetch(1)
        if (blockStateTranscribe(call)) {
            val deadline = TimeSource.Monotonic.markNow() + 10.seconds
            while (!abortRequested(handle)) {
                check(deadline.hasNotPassedNow()) { "fake streaming decode was never aborted" }
            }
        }
        if (abortRequested(handle)) {
            lastErrorMessage = "transcription cancelled"
            return WhisperStatus.CANCELLED
        }
        val produced = streamSegments(call)
        val visible = ArrayList<FakeSegment>()
        for ((i, segment) in produced.withIndex()) {
            visible += segment
            setResults(state, visible.toList())
            val status = listener?.onNewSegments(i, 1) ?: WhisperStatus.OK
            if (status != WhisperStatus.OK) {
                setResults(state, emptyList())
                lastErrorMessage = "new_segment_callback stopped the transcription with status $status"
                return status
            }
        }
        val status = stateTranscribeStatus(call)
        if (status != WhisperStatus.OK) {
            setResults(state, emptyList())
            lastErrorMessage = "fake streaming decode failure"
        }
        return status
    }

    private fun setResults(state: Long, segments: List<FakeSegment>) {
        while (true) {
            val cur = stateResults.load()
            if (stateResults.compareAndSet(cur, cur + (state to segments))) return
        }
    }

    override fun stateSegmentCount(state: Long): RawIntResult {
        checkStateAlive(state)
        return RawIntResult(WhisperStatus.OK, stateResults.load()[state].orEmpty().size)
    }

    override fun stateSegment(state: Long, index: Int): RawSegmentResult {
        checkStateAlive(state)
        val segment = stateResults.load()[state].orEmpty().getOrNull(index) ?: return RawSegmentResult(WhisperStatus.INVALID_ARGUMENT)
        return RawSegmentResult(WhisperStatus.OK, segment.startMs, segment.endMs, 0.1f, segment.text.encodeToByteArray())
    }

    override fun stateSegmentTokens(state: Long, index: Int): RawTokensResult {
        checkStateAlive(state)
        val segment = stateResults.load()[state].orEmpty().getOrNull(index) ?: return RawTokensResult(WhisperStatus.INVALID_ARGUMENT)
        return RawTokensResult(WhisperStatus.OK, segment.tokens.toIntArray())
    }

    override fun resamplerCreate(inputRate: Int, channels: Int, format: Int): RawHandleResult {
        val h = nextResampler.fetchAndAdd(1L)
        while (true) {
            val cur = resamplers.load()
            if (resamplers.compareAndSet(cur, cur + (h to FakeResampler(inputRate, channels, format)))) break
        }
        createdResamplers.add(h)
        return RawHandleResult(WhisperStatus.OK, h)
    }

    override fun resamplerFree(resampler: Long) {
        check(resampler in createdResamplers.items) { "free of unknown resampler $resampler" }
        check(resampler !in freedResamplers.items) { "double free of resampler $resampler" }
        freedResamplers.add(resampler)
    }

    private fun liveResampler(h: Long): FakeResampler {
        check(h !in freedResamplers.items) { "use of freed resampler $h" }
        return checkNotNull(resamplers.load()[h]) { "unknown resampler $h" }
    }

    override fun resamplerMaxOutput(resampler: Long, frames: Int): Long {
        val r = liveResampler(resampler)
        return (frames.toLong() * 16_000 + r.inRate - 1) / r.inRate + 1
    }

    override fun resamplerProcessF32(resampler: Long, input: FloatArray, frames: Int, channels: Int, output: FloatArray): RawIntResult {
        val r = liveResampler(resampler)
        check(r.format == PcmFormat.F32 && r.channels == channels)
        conversions.add("stream-f32:$frames:$channels:${r.inRate}")
        return streamConvert(r, frames, output) { f -> (0 until channels).sumOf { input[f * channels + it].toDouble() }.toFloat() / channels }
    }

    override fun resamplerProcessS16(resampler: Long, input: ShortArray, frames: Int, channels: Int, output: FloatArray): RawIntResult {
        val r = liveResampler(resampler)
        check(r.format == PcmFormat.S16 && r.channels == channels)
        conversions.add("stream-s16:$frames:$channels:${r.inRate}")
        return streamConvert(r, frames, output) { f -> (0 until channels).sumOf { input[f * channels + it] / 32768.0 }.toFloat() / channels }
    }

    override fun resamplerFlush(resampler: Long, output: FloatArray): RawIntResult {
        liveResampler(resampler)
        conversions.add("stream-flush")
        return RawIntResult(WhisperStatus.OK, 0)
    }

    /** Stateful nearest-neighbour decimation: output k takes input frame floor(k * inRate / 16000). */
    private inline fun streamConvert(r: FakeResampler, frames: Int, output: FloatArray, mono: (Int) -> Float): RawIntResult {
        val base = r.inTotal
        r.inTotal += frames
        var n = 0
        while (true) {
            val src = r.outTotal * r.inRate / 16_000
            if (src >= r.inTotal) break
            if (n >= output.size) return RawIntResult(WhisperStatus.BUFFER_TOO_SMALL, n)
            output[n++] = mono((src - base).toInt())
            r.outTotal++
        }
        return RawIntResult(WhisperStatus.OK, n)
    }

    /** ceil(frames * 16000 / rate), like the native resampler. */
    override fun resampledLength(frames: Int, sampleRate: Int): Long {
        if (frames < 0 || sampleRate <= 0) return -1
        return (frames.toLong() * 16_000 + sampleRate - 1) / sampleRate
    }

    override fun convertF32(input: FloatArray, frames: Int, channels: Int, sampleRate: Int, output: FloatArray): RawIntResult {
        conversions.add("f32:$frames:$channels:$sampleRate")
        return fakeConvert(frames, sampleRate, output) { f -> (0 until channels).sumOf { input[f * channels + it].toDouble() }.toFloat() / channels }
    }

    override fun convertS16(input: ShortArray, frames: Int, channels: Int, sampleRate: Int, output: FloatArray): RawIntResult {
        conversions.add("s16:$frames:$channels:$sampleRate")
        return fakeConvert(frames, sampleRate, output) { f -> (0 until channels).sumOf { input[f * channels + it] / 32768.0 }.toFloat() / channels }
    }

    /** Nearest-neighbour "resampling" of the downmixed signal: good enough to check data flow. */
    private inline fun fakeConvert(frames: Int, sampleRate: Int, output: FloatArray, mono: (Int) -> Float): RawIntResult {
        val needed = resampledLength(frames, sampleRate).toInt()
        if (output.size < needed) return RawIntResult(WhisperStatus.BUFFER_TOO_SMALL, needed)
        for (i in 0 until needed) {
            val src = (i.toLong() * sampleRate / 16_000).toInt().coerceAtMost(frames - 1)
            output[i] = if (src >= 0) mono(src) else 0f
        }
        return RawIntResult(WhisperStatus.OK, needed)
    }

    override fun statusName(status: Int): String = "KGGML_STATUS_$status"

    override fun lastError(): String = lastErrorMessage
}
