package com.donfuldauer.kggml.samples.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.whisper.TranscriptionResult
import com.donfuldauer.kggml.whisper.WhisperEngine
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlin.time.Duration
import kotlin.time.TimeSource
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/** Backend choices offered by the sample UI: the runtime toggle required by AGENTS.md §2.2. */
enum class BackendChoice {
    /** `ComputeBackend.CPU_ONLY`: `use_gpu = false` (Accelerate / NEON only on iOS; safe in the background). */
    CPU_ONLY,

    /** The platform GPU: `GPU_METAL` on iOS, `GPU_VULKAN` on Android (falls back to the CPU when unavailable). */
    GPU,

    /** `ComputeBackend.AUTO`: CoreML > Metal > Vulkan > CPU, whichever is available first. */
    AUTO,
    ;

    fun toComputeBackend(platformGpu: ComputeBackend = platformGpuBackend): ComputeBackend = when (this) {
        CPU_ONLY -> ComputeBackend.CPU_ONLY
        GPU -> platformGpu
        AUTO -> ComputeBackend.AUTO
    }

    fun label(platformGpu: ComputeBackend = platformGpuBackend): String = when (this) {
        CPU_ONLY -> "CPU only"
        GPU -> if (platformGpu == ComputeBackend.GPU_METAL) "GPU (Metal)" else "GPU (Vulkan)"
        AUTO -> "Auto"
    }
}

/** The GPU backend of this platform: `GPU_METAL` on iOS, `GPU_VULKAN` on Android. */
expect val platformGpuBackend: ComputeBackend

/** The subset of [WhisperEngine] the sample uses, so the session logic can be tested with fakes. */
interface SpeechEngine : AutoCloseable {
    val activeBackend: ComputeBackend
    val usesGpu: Boolean

    /** The wrapped engine, or `null` for test fakes. */
    val whisperEngine: WhisperEngine?

    suspend fun reinitialize(config: BackendConfig): ComputeBackend

    suspend fun transcribe(audio: PcmAudio): TranscriptionResult
}

/** Opens a [SpeechEngine] for a model path and backend configuration. */
fun interface SpeechEngineFactory {
    suspend fun open(modelPath: String, config: BackendConfig): SpeechEngine
}

/** [SpeechEngine] backed by a real [WhisperEngine]. */
class WhisperSpeechEngine(override val whisperEngine: WhisperEngine) : SpeechEngine {
    override val activeBackend: ComputeBackend get() = whisperEngine.activeBackend
    override val usesGpu: Boolean get() = whisperEngine.usesGpu

    override suspend fun reinitialize(config: BackendConfig): ComputeBackend = whisperEngine.reinitialize(config)

    override suspend fun transcribe(audio: PcmAudio): TranscriptionResult = when (audio) {
        is PcmAudio.S16 -> whisperEngine.transcribe(audio.samples, audio.sampleRate, audio.channels)
        is PcmAudio.F32 -> whisperEngine.transcribe(audio.samples, audio.sampleRate, audio.channels)
    }

    override fun close() = whisperEngine.close()

    companion object {
        /** Factory that loads real whisper.cpp models. */
        val Factory: SpeechEngineFactory = SpeechEngineFactory { path, config -> WhisperSpeechEngine(WhisperEngine(path, config)) }
    }
}

/** Outcome of one [WhisperSampleSession.transcribe] call. */
data class TranscriptionReport(
    val requestedBackend: ComputeBackend,
    val activeBackend: ComputeBackend,
    val usesGpu: Boolean,
    val audioDurationMs: Long,
    val elapsed: Duration,
    val text: String,
    val segmentCount: Int,
)

/**
 * Keeps at most one engine open and reuses it between transcriptions. Changing the backend reloads the same
 * engine through `WhisperEngine.reinitialize` (the runtime toggle); changing the model opens a new engine.
 * Calls are serialized; the engine itself additionally serializes all native work on its own thread.
 *
 * @param onEngineOpened called (inside the session lock) after a new engine was opened, e.g. to attach the iOS
 *   app-lifecycle observer.
 * @param onEngineClosing called (inside the session lock) before an engine is closed.
 */
@OptIn(ExperimentalAtomicApi::class)
class WhisperSampleSession(
    private val factory: SpeechEngineFactory = WhisperSpeechEngine.Factory,
    private val timeSource: TimeSource = TimeSource.Monotonic,
    private val onEngineOpened: (SpeechEngine) -> Unit = {},
    private val onEngineClosing: (SpeechEngine) -> Unit = {},
) : AutoCloseable {
    private val mutex = Mutex()
    private val closed = AtomicBoolean(false)
    private var engine: SpeechEngine? = null
    private var modelPath: String? = null
    private var requested: ComputeBackend? = null

    /** The currently open engine, if any. */
    val currentEngine: SpeechEngine? get() = engine

    /** Transcribes [audio] with the model at [modelPath] on [backend], loading / reloading the engine as needed. */
    suspend fun transcribe(modelPath: String, audio: PcmAudio, backend: ComputeBackend): TranscriptionReport =
        mutex.withLock {
            check(!closed.load()) { "session is closed" }
            val active = ensureEngine(modelPath, backend)
            val mark = timeSource.markNow()
            val result = active.transcribe(audio)
            TranscriptionReport(
                requestedBackend = backend,
                activeBackend = result.backend,
                // From the result, not the engine: the iOS lifecycle fallback may have reloaded it on the CPU since.
                usesGpu = result.backend != ComputeBackend.CPU_ONLY,
                audioDurationMs = result.audioDurationMs,
                elapsed = mark.elapsedNow(),
                text = result.text,
                segmentCount = result.segments.size,
            )
        }

    private suspend fun ensureEngine(path: String, backend: ComputeBackend): SpeechEngine {
        val current = engine
        if (current != null && modelPath == path) {
            if (requested != backend) {
                current.reinitialize(BackendConfig(backend = backend))
                requested = backend
            }
            return current
        }
        releaseEngine()
        val opened = factory.open(path, BackendConfig(backend = backend))
        engine = opened
        modelPath = path
        requested = backend
        onEngineOpened(opened)
        return opened
    }

    private fun releaseEngine() {
        val current = engine ?: return
        engine = null
        modelPath = null
        requested = null
        try {
            onEngineClosing(current)
        } finally {
            current.close()
        }
    }

    /** Closes the open engine (aborting a running transcription). Idempotent. */
    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        // A running transcription holds the lock: abort it natively first so the lock is released promptly.
        engine?.whisperEngine?.abortActiveTranscription()
        if (mutex.tryLock()) {
            try {
                releaseEngine()
            } finally {
                mutex.unlock()
            }
        } else {
            // Closing the engine from here is safe (WhisperEngine.close is thread-safe and idempotent).
            engine?.close()
        }
    }
}

/** Renders a [TranscriptionReport] as the sample's result text. */
fun formatReport(report: TranscriptionReport): String = buildString {
    appendLine("Requested ${report.requestedBackend}, ran on ${report.activeBackend} (use_gpu=${report.usesGpu})")
    appendLine("Audio ${report.audioDurationMs} ms, ${report.segmentCount} segment(s) in ${report.elapsed.inWholeMilliseconds} ms")
    append(report.text.ifEmpty { "(no speech)" })
}
