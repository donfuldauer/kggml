package com.donfuldauer.kggml.samples.showcase.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.TranscriptionEvent
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** What the Whisper Studio is currently doing. */
enum class StudioActivity(val label: String) {
    IDLE("Idle"),
    LOADING_MODEL("Loading model…"),
    LISTENING("Listening"),
    STOPPING("Finishing the last turn…"),
    TRANSCRIBING_FILE("Transcribing file…"),
}

/** Where the transcript on screen came from. */
enum class TranscriptSource { LIVE, FILE }

/** A locked, timestamped transcript entry (a `CommittedSegment`). */
data class TranscriptLine(val index: Int, val text: String, val startMs: Long, val endMs: Long) {
    val timeLabel: String get() = "${formatTimestamp(startMs)} → ${formatTimestamp(endMs)}"
    val testTag: String get() = "committed-$index"
}

/**
 * The unstable hypothesis of the turn in progress.
 *
 * @property lagMs captured audio duration when the hypothesis arrived minus its [endMs]: how far the displayed
 *   text trails the microphone (target < 400 ms).
 */
data class PartialLine(val text: String, val startMs: Long, val endMs: Long, val lagMs: Long)

/** View state of the Live Whisper Dictation & Audio Studio. */
data class WhisperStudioState(
    val modelPath: String = "",
    val filePath: String = "",
    val selectedBackend: ComputeBackend = ComputeBackend.AUTO,
    /** Backend of the loaded model, or `null` before the first load. */
    val activeBackend: ComputeBackend? = null,
    val usesGpu: Boolean = false,
    val micAvailable: Boolean = false,
    val backgroundStressSupported: Boolean = false,
    /**
     * iOS background stress mode: when `true` the automatic `CPU_ONLY` background fallback is *not* installed, so a
     * `GPU_METAL` stream runs into iOS revoking GPU access while a `CPU_ONLY` stream keeps transcribing.
     */
    val backgroundStress: Boolean = false,
    val activity: StudioActivity = StudioActivity.IDLE,
    val captureFormat: CaptureFormat? = null,
    val capturedMs: Long = 0,
    /** Recent microphone RMS levels in `[0, 1]`, oldest first. */
    val levels: List<Float> = emptyList(),
    val partial: PartialLine? = null,
    val committed: List<TranscriptLine> = emptyList(),
    val source: TranscriptSource? = null,
    val status: String? = null,
    val error: String? = null,
) {
    val isBusy: Boolean get() = activity != StudioActivity.IDLE
    val canStartLive: Boolean get() = !isBusy && micAvailable && modelPath.isNotBlank()
    val canTranscribeFile: Boolean get() = !isBusy && modelPath.isNotBlank() && filePath.isNotBlank()
    val canStop: Boolean get() = activity == StudioActivity.LISTENING
    val canCancel: Boolean get() = isBusy
}

/** Backends offered by the runtime backend switcher, in display order. */
val STUDIO_BACKENDS: List<ComputeBackend> = listOf(
    ComputeBackend.AUTO,
    ComputeBackend.CPU_ONLY,
    ComputeBackend.GPU_METAL,
    ComputeBackend.GPU_VULKAN,
    ComputeBackend.NPU_COREML,
)

/** The engine configuration requested for [backend]. */
fun studioBackendConfig(backend: ComputeBackend): BackendConfig =
    if (backend == ComputeBackend.CPU_ONLY) BackendConfig.CPU else BackendConfig(backend = backend)

/**
 * State holder of the Whisper Studio. All public methods must be called from the thread of [scope] (the UI
 * thread in the app); work runs in coroutines on [scope], native model loading on [ioDispatcher].
 */
class WhisperStudioController(
    private val scope: CoroutineScope,
    private val factory: TranscriberFactory,
    private val capture: AudioCapture?,
    private val readFile: (String) -> ByteArray,
    private val ioDispatcher: CoroutineDispatcher,
    initialModelPath: String = "",
    backgroundStressSupported: Boolean = false,
    private val maxLevels: Int = DEFAULT_MAX_LEVELS,
) : AutoCloseable {
    private val mutableState = MutableStateFlow(
        WhisperStudioState(
            modelPath = initialModelPath,
            micAvailable = capture != null,
            backgroundStressSupported = backgroundStressSupported,
        ),
    )

    val state: StateFlow<WhisperStudioState> = mutableState.asStateFlow()

    private var transcriber: Transcriber? = null
    private var loadedConfig: BackendConfig? = null
    private var backgroundFallback: AutoCloseable? = null
    private var job: Job? = null
    private var stopSignal: CompletableDeferred<Unit>? = null
    private var closed = false

    fun setModelPath(path: String) = mutableState.update { it.copy(modelPath = path) }

    fun setFilePath(path: String) = mutableState.update { it.copy(filePath = path) }

    /** Selects the compute backend; an already loaded idle model is reloaded on it right away. */
    fun selectBackend(backend: ComputeBackend) {
        val current = mutableState.value
        if (current.isBusy || current.selectedBackend == backend) return
        mutableState.update { it.copy(selectedBackend = backend, error = null) }
        val loaded = transcriber ?: return
        if (loaded.modelPath != current.modelPath) return
        launchWork(StudioActivity.LOADING_MODEL) { ensureTranscriber() }
    }

    fun setBackgroundStress(enabled: Boolean) {
        if (!mutableState.value.backgroundStressSupported) return
        mutableState.update { it.copy(backgroundStress = enabled) }
        updateBackgroundFallback()
    }

    /** Starts live dictation from the microphone. */
    fun startLive() {
        val audioCapture = capture
        if (!mutableState.value.canStartLive || audioCapture == null) return
        resetTranscript(TranscriptSource.LIVE)
        launchWork(StudioActivity.LOADING_MODEL) {
            val loaded = ensureTranscriber()
            val format = audioCapture.prepare()
            val stop = CompletableDeferred<Unit>()
            stopSignal = stop
            mutableState.update {
                it.copy(activity = StudioActivity.LISTENING, captureFormat = format, capturedMs = 0, levels = emptyList())
            }
            loaded.stream(meteredAudio(audioCapture, format, stop), format).collect(::onEvent)
        }
    }

    /** Stops the microphone; the last turn is still committed before the stream completes. */
    fun stop() {
        val stop = stopSignal ?: return
        if (mutableState.value.activity != StudioActivity.LISTENING) return
        mutableState.update { it.copy(activity = StudioActivity.STOPPING) }
        stop.complete(Unit)
    }

    /** Abandons the running stream or file transcription immediately. */
    fun cancel() {
        job?.cancel()
    }

    /** Transcribes the WAV / raw 16 kHz PCM file at [WhisperStudioState.filePath] in one pass. */
    fun transcribeFile() {
        val path = mutableState.value.filePath
        if (!mutableState.value.canTranscribeFile) return
        resetTranscript(TranscriptSource.FILE)
        launchWork(StudioActivity.LOADING_MODEL) {
            val loaded = ensureTranscriber()
            mutableState.update { it.copy(activity = StudioActivity.TRANSCRIBING_FILE) }
            val audio = withContext(ioDispatcher) { parsePcmFile(readFile(path)) }
            val segments = loaded.transcribe(audio)
            mutableState.update { state ->
                state.copy(
                    committed = segments.map { it.toLine() },
                    status = "Transcribed ${formatDuration(audio.durationMs)} of audio " +
                        "(${audio.sampleRate} Hz, ${audio.channels} ch): ${segments.size} segment(s)",
                )
            }
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        job?.cancel()
        releaseTranscriber()
    }

    // ----------------------------------------------------------------------------------------------------------

    private fun launchWork(initial: StudioActivity, block: suspend () -> Unit) {
        if (closed) return
        mutableState.update { it.copy(activity = initial, error = null) }
        job = scope.launch {
            try {
                block()
            } catch (e: CancellationException) {
                mutableState.update { it.copy(status = "Cancelled", partial = null) }
                throw e
            } catch (t: Throwable) {
                mutableState.update { it.copy(error = t.describe(), partial = null) }
            } finally {
                stopSignal = null
                mutableState.update { it.copy(activity = StudioActivity.IDLE) }
            }
        }
    }

    /**
     * Microphone chunks until [stop] completes (or the capture ends), metering the waveform and captured duration
     * on the way to the engine.
     */
    private fun meteredAudio(capture: AudioCapture, format: CaptureFormat, stop: CompletableDeferred<Unit>): Flow<FloatArray> {
        var samples = 0L
        return channelFlow {
            val pump = launch { capture.frames().collect { send(it) } }
            val stopper = launch {
                stop.await()
                pump.cancel()
            }
            pump.join()
            stopper.cancel()
        }.onEach { chunk ->
            samples += chunk.size
            val level = rmsLevel(chunk)
            val capturedMs = samples / format.channels * 1000 / format.sampleRate
            mutableState.update { it.copy(capturedMs = capturedMs, levels = (it.levels + level).takeLast(maxLevels)) }
        }
    }

    private fun onEvent(event: TranscriptionEvent) {
        when (event) {
            is TranscriptionEvent.PartialHypothesis -> mutableState.update {
                val lag = (it.capturedMs - event.endMs).coerceAtLeast(0)
                it.copy(partial = PartialLine(event.text.trim(), event.startMs, event.endMs, lag))
            }

            is TranscriptionEvent.CommittedSegment -> mutableState.update {
                it.copy(committed = it.committed + event.toLine(), partial = null)
            }

            is TranscriptionEvent.Completed -> mutableState.update {
                it.copy(
                    partial = null,
                    status = "Completed: ${event.segmentCount} segment(s), " +
                        "${formatDuration(event.audioDurationMs)} of audio",
                )
            }

            is TranscriptionEvent.Error -> mutableState.update {
                it.copy(partial = null, error = event.exception.describe())
            }
        }
    }

    /** Loads (or reloads) the model for the current path and selected backend. Runs on [scope]. */
    private suspend fun ensureTranscriber(): Transcriber {
        val state = mutableState.value
        val path = state.modelPath.trim()
        val config = studioBackendConfig(state.selectedBackend)
        val current = transcriber
        val loaded = if (current != null && current.modelPath == path) {
            if (loadedConfig != config) {
                mutableState.update { it.copy(activity = StudioActivity.LOADING_MODEL) }
                current.reinitialize(config)
                loadedConfig = config
            }
            current
        } else {
            releaseTranscriber()
            mutableState.update { it.copy(activity = StudioActivity.LOADING_MODEL, activeBackend = null) }
            withContext(ioDispatcher) { factory.open(path, config) }.also {
                if (closed) {
                    it.close()
                    throw CancellationException("Whisper Studio closed")
                }
                transcriber = it
                loadedConfig = config
            }
        }
        mutableState.update { it.copy(activeBackend = loaded.activeBackend, usesGpu = loaded.usesGpu) }
        updateBackgroundFallback()
        return loaded
    }

    private fun updateBackgroundFallback() {
        backgroundFallback?.close()
        backgroundFallback = null
        val loaded = transcriber ?: return
        if (mutableState.value.backgroundStress) return
        backgroundFallback = loaded.installBackgroundFallback(scope) { failure ->
            mutableState.update { it.copy(error = "Background fallback failed: ${failure.describe()}") }
        }
    }

    private fun releaseTranscriber() {
        backgroundFallback?.close()
        backgroundFallback = null
        transcriber?.close()
        transcriber = null
        loadedConfig = null
    }

    private fun resetTranscript(source: TranscriptSource) = mutableState.update {
        it.copy(source = source, committed = emptyList(), partial = null, status = null, error = null, capturedMs = 0)
    }

    companion object {
        const val DEFAULT_MAX_LEVELS: Int = 96
    }
}

private fun TranscriptionEvent.CommittedSegment.toLine() = TranscriptLine(index, text.trim(), startMs, endMs)

private fun Throwable.describe(): String = "${this::class.simpleName ?: "Error"}: ${message ?: "(no message)"}"

/** `m:ss.s` for a stream position in milliseconds. */
fun formatTimestamp(ms: Long): String {
    val clamped = ms.coerceAtLeast(0)
    val minutes = clamped / 60_000
    val seconds = (clamped % 60_000) / 1000
    val tenths = (clamped % 1000) / 100
    return "$minutes:${seconds.toString().padStart(2, '0')}.$tenths"
}

/** `12.3 s` for a duration in milliseconds. */
fun formatDuration(ms: Long): String = "${ms / 1000}.${(ms % 1000) / 100} s"
