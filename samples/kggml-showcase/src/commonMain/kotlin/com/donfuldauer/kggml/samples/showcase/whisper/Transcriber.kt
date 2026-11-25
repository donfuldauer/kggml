package com.donfuldauer.kggml.samples.showcase.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.TranscriptionEvent
import com.donfuldauer.kggml.whisper.WhisperEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.Flow

/** A loaded speech-to-text model used by the Whisper Studio ([WhisperTranscriber] in the app, fakes in tests). */
interface Transcriber : AutoCloseable {
    val modelPath: String

    /** The backend the model actually runs on (never [ComputeBackend.AUTO]). */
    val activeBackend: ComputeBackend

    /** Whether the loaded model submits GPU work (always `false` for [ComputeBackend.CPU_ONLY]). */
    val usesGpu: Boolean

    /** Streams [audio] (interleaved float PCM in [format]) into transcription events. */
    fun stream(audio: Flow<FloatArray>, format: CaptureFormat): Flow<TranscriptionEvent>

    /** Transcribes a decoded file in one pass and returns its segments in order. */
    suspend fun transcribe(audio: PcmAudio): List<TranscriptionEvent.CommittedSegment>

    /** Reloads the model with [config]; returns the new [activeBackend]. */
    suspend fun reinitialize(config: BackendConfig): ComputeBackend

    /**
     * Installs the platform's background-safety fallback (iOS: switch to `CPU_ONLY` while the app is in the
     * background), or returns `null` when the platform needs none.
     */
    fun installBackgroundFallback(scope: CoroutineScope, onError: (Throwable) -> Unit): AutoCloseable?
}

/** Opens [Transcriber]s; the suspend call loads the model and may block on native code. */
fun interface TranscriberFactory {
    suspend fun open(modelPath: String, config: BackendConfig): Transcriber
}

/** Production [TranscriberFactory] backed by `:kggml-whisper`. */
object WhisperTranscriberFactory : TranscriberFactory {
    override suspend fun open(modelPath: String, config: BackendConfig): Transcriber =
        WhisperTranscriber(WhisperEngine(modelPath, config))
}

/** [Transcriber] over a [WhisperEngine]. */
class WhisperTranscriber(private val engine: WhisperEngine) : Transcriber {
    override val modelPath: String get() = engine.modelPath
    override val activeBackend: ComputeBackend get() = engine.activeBackend
    override val usesGpu: Boolean get() = engine.usesGpu

    override fun stream(audio: Flow<FloatArray>, format: CaptureFormat): Flow<TranscriptionEvent> =
        engine.transcribeStream(audio, sampleRate = format.sampleRate, channels = format.channels)

    override suspend fun transcribe(audio: PcmAudio): List<TranscriptionEvent.CommittedSegment> {
        val result = when (audio) {
            is PcmAudio.F32 -> engine.transcribe(audio.samples, audio.sampleRate, audio.channels)
            is PcmAudio.S16 -> engine.transcribe(audio.samples, audio.sampleRate, audio.channels)
        }
        return result.segments.map { TranscriptionEvent.CommittedSegment(it.index, it.text, it.startMs, it.endMs) }
    }

    override suspend fun reinitialize(config: BackendConfig): ComputeBackend = engine.reinitialize(config)

    override fun installBackgroundFallback(scope: CoroutineScope, onError: (Throwable) -> Unit): AutoCloseable? =
        platformBackgroundFallback(engine, scope, onError)

    override fun close() = engine.close()
}

/**
 * `true` on platforms where GPU work is revoked in the background (iOS), so the studio offers the background
 * stress toggle comparing `CPU_ONLY` against Metal.
 */
expect val supportsBackgroundStress: Boolean

/** iOS: `WhisperEngine.observeAppLifecycle`; elsewhere `null`. */
expect fun platformBackgroundFallback(
    engine: WhisperEngine,
    scope: CoroutineScope,
    onError: (Throwable) -> Unit,
): AutoCloseable?
