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
