package com.donfuldauer.kggml.samples.showcase.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceException
import com.donfuldauer.kggml.core.TranscriptionEvent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.receiveAsFlow

/** Microphone fake: the test pushes chunks with [send]; [finish] ends the capture. */
class FakeAudioCapture(val format: CaptureFormat = CaptureFormat(48_000, 1)) : AudioCapture {
    private val chunks = Channel<FloatArray>(Channel.UNLIMITED)
    var prepareCount = 0
        private set
    var prepareFailure: Throwable? = null

    fun send(chunk: FloatArray) {
        chunks.trySend(chunk)
    }

    /** A chunk of [ms] milliseconds at a constant [amplitude]. */
    fun sendMs(ms: Int, amplitude: Float = 0.25f) = send(FloatArray(format.sampleRate * format.channels * ms / 1000) { amplitude })

    fun finish() {
        chunks.close()
    }

    override suspend fun prepare(): CaptureFormat {
        prepareCount++
        prepareFailure?.let { throw it }
        return format
    }

    override fun frames(): Flow<FloatArray> = chunks.receiveAsFlow()
}

/**
 * Deterministic transcriber: after every audio chunk it emits a [TranscriptionEvent.PartialHypothesis] covering the
 * audio so far ("word1 word2 …", one word per chunk); when the audio ends it commits the turn as one segment and
 * completes. [streamError] instead ends the stream with an in-band error after the first chunk.
 */
class FakeTranscriber(
    override val modelPath: String,
    var config: BackendConfig,
    private val available: Set<ComputeBackend> = setOf(ComputeBackend.CPU_ONLY, ComputeBackend.GPU_METAL),
    var streamError: InferenceException? = null,
    var fileSegments: List<TranscriptionEvent.CommittedSegment> = emptyList(),
) : Transcriber {
    val reinitialized = mutableListOf<BackendConfig>()
    val transcribedFiles = mutableListOf<PcmAudio>()
    var closed = false
        private set
    var fallbackInstalls = 0
        private set
    var fallbackCloses = 0
        private set

    override val activeBackend: ComputeBackend
        get() = ComputeBackend.resolve(config.backend, available)

    override val usesGpu: Boolean get() = activeBackend != ComputeBackend.CPU_ONLY

    override fun stream(audio: Flow<FloatArray>, format: CaptureFormat): Flow<TranscriptionEvent> = flow {
        var samples = 0L
        val words = mutableListOf<String>()
        var failed = false
        audio.collect { chunk ->
            if (failed) return@collect
            samples += chunk.size
            words += "word${words.size + 1}"
            val endMs = format.durationMs(samples.toInt())
            val error = streamError
            if (error != null) {
                emit(TranscriptionEvent.Error(error))
                failed = true
            } else {
                emit(TranscriptionEvent.PartialHypothesis(" ${words.joinToString(" ")}", 0, endMs))
            }
        }
        if (failed) return@flow
        val endMs = format.durationMs(samples.toInt())
        if (words.isNotEmpty()) emit(TranscriptionEvent.CommittedSegment(0, " ${words.joinToString(" ")}.", 0, endMs))
        emit(TranscriptionEvent.Completed(if (words.isEmpty()) 0 else 1, endMs))
    }

    override suspend fun transcribe(audio: PcmAudio): List<TranscriptionEvent.CommittedSegment> {
        transcribedFiles += audio
        return fileSegments
    }

    override suspend fun reinitialize(config: BackendConfig): ComputeBackend {
        reinitialized += config
        this.config = config
        return activeBackend
    }

    override fun installBackgroundFallback(scope: CoroutineScope, onError: (Throwable) -> Unit): AutoCloseable {
        fallbackInstalls++
        return AutoCloseable { fallbackCloses++ }
    }

    override fun close() {
        closed = true
    }
}

/** Records every [open] and hands out [FakeTranscriber]s (or throws [failure]). */
class FakeTranscriberFactory(
    var failure: Throwable? = null,
    private val configure: (FakeTranscriber) -> Unit = {},
) : TranscriberFactory {
    val opened = mutableListOf<FakeTranscriber>()

    override suspend fun open(modelPath: String, config: BackendConfig): Transcriber {
        failure?.let { throw it }
        return FakeTranscriber(modelPath, config).also(configure).also { opened += it }
    }
}

/** A mono 16-bit RIFF/WAVE file of [samples]. */
fun wavBytes(samples: ShortArray, sampleRate: Int = 16_000): ByteArray {
    val out = ArrayList<Byte>()
    fun ascii(s: String) = s.forEach { out += it.code.toByte() }
    fun le16(v: Int) {
        out += (v and 0xFF).toByte(); out += ((v shr 8) and 0xFF).toByte()
    }
    fun le32(v: Int) {
        le16(v and 0xFFFF); le16((v ushr 16) and 0xFFFF)
    }
    val dataBytes = samples.size * 2
    ascii("RIFF"); le32(36 + dataBytes); ascii("WAVE")
    ascii("fmt "); le32(16); le16(1); le16(1); le32(sampleRate); le32(sampleRate * 2); le16(2); le16(16)
    ascii("data"); le32(dataBytes)
    samples.forEach { le16(it.toInt()) }
    return out.toByteArray()
}
