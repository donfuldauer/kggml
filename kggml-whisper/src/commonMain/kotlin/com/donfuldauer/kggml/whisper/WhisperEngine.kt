package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.InferenceEngine
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import com.donfuldauer.kggml.core.SerialExecutor
import com.donfuldauer.kggml.runtime.KggmlRuntime
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.AtomicReference
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.InternalCoroutinesApi
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.job

/**
 * Loads a whisper.cpp ggml model and returns a ready [WhisperEngine].
 *
 * The model is loaded on the new engine's own dedicated thread. [backendConfig] is enforced at load time:
 *
 * - [ComputeBackend.CPU_ONLY] sets `whisper_context_params.use_gpu = false` *before*
 *   `whisper_init_from_file_with_params()`, so ggml never creates a GPU backend. On iOS all work then runs on
 *   Apple Accelerate / ARM NEON, which remains legal while the app is in the background.
 * - [ComputeBackend.GPU_METAL] / [ComputeBackend.GPU_VULKAN] set `use_gpu = true` on that device kind.
 * - [ComputeBackend.NPU_COREML] additionally enables the Core ML encoder (iOS builds with `WHISPER_COREML`); it
 *   uses `<model>-encoder.mlmodelc` next to the model when present and otherwise falls back to Metal/CPU.
 * - [ComputeBackend.AUTO] picks the first available of CoreML, Metal, Vulkan, CPU.
 *
 * `BackendConfig.useFlashAttention` maps to whisper's flash attention and `cpuThreads` to the decoder thread
 * count ([BackendConfig.AUTO_THREADS] = number of performance cores). `gpuLayers` and `useMmap` do not apply to
 * Whisper models and are ignored.
 *
 * @throws com.donfuldauer.kggml.core.ModelLoadException if the file is missing or not a whisper model.
 * @throws com.donfuldauer.kggml.core.BackendUnavailableException if the backend is unavailable and
 *   [WhisperEngineOptions.allowBackendFallback] is `false`.
 * @throws com.donfuldauer.kggml.core.NativeFailureException if the native libraries cannot be loaded.
 */
public suspend fun WhisperEngine(
    modelPath: String,
    backendConfig: BackendConfig = BackendConfig.DEFAULT,
    options: WhisperEngineOptions = WhisperEngineOptions(),
): WhisperEngine = WhisperEngine.open(
    modelPath = modelPath,
    backendConfig = backendConfig,
    options = options,
    bridges = WhisperBridgeHolder.platform,
    availableBackends = ::platformAvailableBackends,
    dispatcherFactory = SerialDispatcherFactory.Default,
)

/** Backends usable by whisper on this device: the runtime's GPU backends plus whisper's own CoreML capability. */
internal fun platformAvailableBackends(bridge: NativeWhisperBridge): Set<ComputeBackend> {
    val backends = KggmlRuntime.availableBackends().toMutableSet()
    backends -= ComputeBackend.NPU_COREML
    if (bridge.hasCoreMl()) backends += ComputeBackend.NPU_COREML
    return backends
}

/**
 * whisper.cpp speech-to-text engine.
 *
 * Create instances with the [WhisperEngine] factory function. Engines are ordinary objects — never singletons —
 * and any number of them (with the same or different models and backends) can coexist and transcribe
 * concurrently: each owns its native whisper context and a dedicated single-threaded dispatcher
 * ([SerialExecutor]) that serializes model loading, [transcribe], [reinitialize] and the native release in
 * [close].
 *
 * Audio crosses into native code zero-copy: 16 kHz mono `FloatArray`s are read in place by whisper.cpp; other
 * rates / channel counts and `ShortArray` input are converted natively (pinned input, one output buffer).
 *
 * Cancellation is cooperative and prompt: cancelling the coroutine that called [transcribe] sets the native
 * abort flag, which whisper.cpp polls inside every ggml graph computation, so decoding stops within one graph
 * step and the call throws `CancellationException`. The engine stays usable afterwards.
 */
public class WhisperEngine internal constructor(
    /** Path of the loaded whisper ggml model (`.bin`). */
    public val modelPath: String,
    backendConfig: BackendConfig,
    /** Whisper-specific construction options. */
    public val options: WhisperEngineOptions,
    private val bridges: WhisperBridgeHolder,
    private val availableBackends: (NativeWhisperBridge) -> Set<ComputeBackend>,
    dispatcherFactory: SerialDispatcherFactory,
) : InferenceEngine {

    /** Configuration and outcome of the most recent successful model load. */
    private class LoadState(val config: BackendConfig, val backend: ComputeBackend, val usesGpu: Boolean)

    private val executor = SerialExecutor("kggml-whisper-${nextInstanceId.addAndFetch(1)}", dispatcherFactory)
    private val state = AtomicReference(LoadState(backendConfig, ComputeBackend.CPU_ONLY, usesGpu = false))
    private val closed = AtomicBoolean(false)
    private val abortGate = AbortGate()

    /** Native context handle. Only read or written on [executor]'s thread; `0` when no model is loaded. */
    private var handle: Long = 0L

    /** The configuration of the currently loaded model (see [reinitialize]). */
    override val backendConfig: BackendConfig
        get() = state.load().config

    /** The backend whisper.cpp actually runs on (never [ComputeBackend.AUTO]). */
    override val activeBackend: ComputeBackend
        get() = state.load().backend

    /** `whisper_context_params.use_gpu` of the loaded model: always `false` for [ComputeBackend.CPU_ONLY]. */
    public val usesGpu: Boolean
        get() = state.load().usesGpu

    override val isClosed: Boolean
        get() = closed.load()

    /** `true` while a [transcribe] call is executing native code. */
    public val isTranscribing: Boolean
        get() = abortGate.isArmed

    /**
     * Transcribes interleaved float PCM in `[-1, 1]`.
     *
     * 16 kHz mono input is passed to whisper.cpp zero-copy; any other [sampleRate] / [channels] is downmixed and
     * resampled natively first. Input shorter than 1000 ms is zero-padded to 1000 ms (empty input yields silence).
     *
     * @param samples interleaved samples; `samples.size` must be a multiple of [channels].
     * @param sampleRate input sample rate in Hz, `1000..384000`.
     * @param channels interleaved channel count, `1..8`.
     * @throws InvalidInputException for malformed audio or options the native layer rejects (e.g. unknown language).
     * @throws IllegalStateException if the engine is closed.
     * @throws InferenceCancelledException if [close] or [abortActiveTranscription] interrupted the call.
     * @throws kotlinx.coroutines.CancellationException if the calling coroutine was cancelled.
     */
    public suspend fun transcribe(
        samples: FloatArray,
        sampleRate: Int = SAMPLE_RATE,
        channels: Int = 1,
        options: TranscriptionOptions = TranscriptionOptions.DEFAULT,
    ): TranscriptionResult {
        val frames = validateShape(samples.size, sampleRate, channels)
        return executor.execute {
            if (sampleRate == SAMPLE_RATE && channels == 1) {
                runTranscription(samples, samples.size, frames, sampleRate, options)
            } else {
                val (pcm, count) = convert(frames, sampleRate) { bridge, out ->
                    bridge.convertF32(samples, frames, channels, sampleRate, out)
                }
                runTranscription(pcm, count, frames, sampleRate, options)
            }
        }
    }

    /**
     * Transcribes interleaved signed 16-bit PCM (normalized as `sample / 32768`). The array is read in place by
     * the native converter; see the `FloatArray` overload for the remaining semantics.
     */
    public suspend fun transcribe(
        samples: ShortArray,
        sampleRate: Int = SAMPLE_RATE,
        channels: Int = 1,
        options: TranscriptionOptions = TranscriptionOptions.DEFAULT,
    ): TranscriptionResult {
        val frames = validateShape(samples.size, sampleRate, channels)
        return executor.execute {
            val (pcm, count) = convert(frames, sampleRate) { bridge, out ->
                bridge.convertS16(samples, frames, channels, sampleRate, out)
            }
            runTranscription(pcm, count, frames, sampleRate, options)
        }
    }

    /**
     * Reloads the model with [backendConfig] on this engine's thread, after any transcription already queued. The
     * old native context is freed *before* the new one is created, so GPU resources are released promptly (e.g.
     * when switching to [ComputeBackend.CPU_ONLY] as an iOS app enters the background).
     *
     * If loading with [backendConfig] fails, the previous configuration is restored and the original error is
     * rethrown.
     *
     * @return the new [activeBackend].
     * @throws IllegalStateException if the engine is closed.
     */
    public suspend fun reinitialize(backendConfig: BackendConfig): ComputeBackend = executor.execute {
        val previous = state.load().config
        freeModel()
        try {
            loadModel(backendConfig)
        } catch (failure: Throwable) {
            try {
                loadModel(previous)
            } catch (restoreFailure: Throwable) {
                failure.addSuppressed(restoreFailure)
            }
            throw failure
        }
        activeBackend
    }

    /**
     * Requests the native abort of the transcription currently executing, if any; it then fails with
     * [InferenceCancelledException]. Safe from any thread. Queued calls are not affected.
     *
     * @return `true` if a running transcription was signalled.
     */
    public fun abortActiveTranscription(): Boolean = abortGate.fire { bridges.get().requestAbort(it) }

    /**
     * Closes the engine: aborts a running transcription, rejects queued and future calls with
     * [IllegalStateException], frees the native context on the engine's thread and stops that thread. Idempotent
     * and safe from any thread.
     */
    override fun close() {
        if (!closed.compareAndSet(expectedValue = false, newValue = true)) return
        abortActiveTranscription()
        executor.close(release = ::freeModel)
    }

    // ------------------------------------------------------------------------------------------------------
    // Executor-thread internals
    // ------------------------------------------------------------------------------------------------------

    internal suspend fun start() {
        try {
            executor.execute { loadModel(state.load().config) }
        } catch (failure: Throwable) {
            close()
            throw failure
        }
    }

    private fun loadModel(config: BackendConfig) {
        check(handle == 0L) { "a model is already loaded" }
        val bridge = bridges.get()
        var backend = ComputeBackend.resolve(config.backend, availableBackends(bridge), options.allowBackendFallback)
        while (true) {
            val result = bridge.initFromFile(
                modelPath = modelPath,
                backendKind = WhisperBackendKind.of(backend),
                gpuDevice = options.gpuDevice,
                flashAttention = config.useFlashAttention,
            )
            if (result.status == WhisperStatus.OK) {
                adopt(bridge, result.handle, config, backend)
                return
            }
            val next = backend.fallback
            if (result.status != WhisperStatus.BACKEND_UNAVAILABLE || !options.allowBackendFallback || next == null) {
                throw bridge.nativeError(result.status, "kggml_whisper_init_from_file", backend)
            }
            backend = next
        }
    }

    private fun adopt(bridge: NativeWhisperBridge, newHandle: Long, config: BackendConfig, requested: ComputeBackend) {
        try {
            val effective = bridge.effectiveBackend(newHandle)
            bridge.check(effective.status, "kggml_whisper_get_backend", requested)
            val gpu = bridge.usesGpu(newHandle)
            bridge.check(gpu.status, "kggml_whisper_uses_gpu", requested)
            val backend = WhisperBackendKind.toComputeBackend(effective.value) ?: requested
            handle = newHandle
            state.store(LoadState(config, backend, gpu.value != 0))
        } catch (failure: Throwable) {
            bridge.free(newHandle)
            throw failure
        }
    }

    private fun freeModel() {
        val current = handle
        if (current == 0L) return
        handle = 0L
        bridges.get().free(current)
    }

    private inline fun convert(
        frames: Int,
        sampleRate: Int,
        call: (NativeWhisperBridge, FloatArray) -> RawIntResult,
    ): Pair<FloatArray, Int> {
        val bridge = bridges.get()
        val length = bridge.resampledLength(frames, sampleRate)
        if (length < 0 || length > Int.MAX_VALUE) {
            throw InvalidInputException("Cannot resample $frames frames at $sampleRate Hz to $SAMPLE_RATE Hz")
        }
        val out = FloatArray(length.toInt())
        val result = call(bridge, out)
        bridge.check(result.status, "kggml_whisper_pcm_to_whisper")
        return out to result.value.coerceIn(0, out.size)
    }

    private suspend fun runTranscription(
        pcm: FloatArray,
        count: Int,
        inputFrames: Int,
        inputRate: Int,
        options: TranscriptionOptions,
    ): TranscriptionResult {
        val current = handle
        check(current != 0L) { "WhisperEngine has no model loaded (a previous reinitialize failed)" }
        val bridge = bridges.get()
        val job = currentCoroutineContext().job
        val config = state.load().config

        bridge.clearAbort(current)
        abortGate.arm(current)
        // onCancelling = true: the handler must run while this thread is still blocked in native code (a plain
        // completion handler would only run after the job completes, i.e. after whisper finished).
        @OptIn(InternalCoroutinesApi::class)
        val registration = job.invokeOnCompletion(onCancelling = true, invokeImmediately = true) { cause ->
            if (cause != null) abortActiveTranscription()
        }
        val status = try {
            bridge.transcribe(current, pcm, count, options.toRaw(config))
        } finally {
            registration.dispose()
            abortGate.disarm()
        }

        if (status != WhisperStatus.OK) {
            val error = bridge.nativeError(status, "kggml_whisper_transcribe", activeBackend)
            if (status == WhisperStatus.CANCELLED) {
                currentCoroutineContext().ensureActive() // caller cancelled -> CancellationException
                if (isClosed) throw InferenceCancelledException("WhisperEngine was closed during transcription", error)
            }
            throw error
        }

        val countResult = bridge.segmentCount(current)
        bridge.check(countResult.status, "kggml_whisper_segment_count")
        val segments = (0 until countResult.value.coerceAtLeast(0)).map { index ->
            val raw = bridge.segment(current, index)
            bridge.check(raw.status, "kggml_whisper_get_segment")
            val start = raw.startMs.coerceAtLeast(0)
            WhisperSegment(
                index = index,
                text = raw.text.decodeToString(),
                startMs = start,
                endMs = raw.endMs.coerceAtLeast(start),
                noSpeechProbability = raw.noSpeechProbability,
            )
        }
        return TranscriptionResult(
            segments = segments,
            audioDurationMs = inputFrames.toLong() * 1000L / inputRate,
            backend = activeBackend,
        )
    }

    public companion object {
        /** The sample rate whisper.cpp consumes (Hz). */
        public const val SAMPLE_RATE: Int = 16_000

        /** Smallest accepted input sample rate (Hz). */
        public const val MIN_SAMPLE_RATE: Int = 1_000

        /** Largest accepted input sample rate (Hz). */
        public const val MAX_SAMPLE_RATE: Int = 384_000

        /** Largest accepted interleaved channel count. */
        public const val MAX_CHANNELS: Int = 8

        private val nextInstanceId = AtomicInt(0)

        /** Version and capability information of the loaded native whisper library. */
        public fun nativeInfo(): WhisperNativeInfo = nativeInfo(WhisperBridgeHolder.platform.get())

        internal fun nativeInfo(bridge: NativeWhisperBridge): WhisperNativeInfo = WhisperNativeInfo(
            moduleVersion = bridge.moduleVersion(),
            whisperCppVersion = bridge.upstreamVersion(),
            whisperCppCommit = bridge.upstreamCommit(),
            systemInfo = bridge.systemInfo(),
            hasCoreMl = bridge.hasCoreMl(),
        )

        internal suspend fun open(
            modelPath: String,
            backendConfig: BackendConfig,
            options: WhisperEngineOptions,
            bridges: WhisperBridgeHolder,
            availableBackends: (NativeWhisperBridge) -> Set<ComputeBackend>,
            dispatcherFactory: SerialDispatcherFactory,
        ): WhisperEngine {
            require(modelPath.isNotEmpty()) { "modelPath must not be empty" }
            val engine = WhisperEngine(modelPath, backendConfig, options, bridges, availableBackends, dispatcherFactory)
            engine.start()
            return engine
        }

        private fun validateShape(size: Int, sampleRate: Int, channels: Int): Int {
            if (channels !in 1..MAX_CHANNELS) {
                throw InvalidInputException("channels must be in 1..$MAX_CHANNELS, was $channels")
            }
            if (sampleRate !in MIN_SAMPLE_RATE..MAX_SAMPLE_RATE) {
                throw InvalidInputException("sampleRate must be in $MIN_SAMPLE_RATE..$MAX_SAMPLE_RATE, was $sampleRate")
            }
            if (size % channels != 0) {
                throw InvalidInputException("sample count $size is not a multiple of channels ($channels)")
            }
            return size / channels
        }

        private fun TranscriptionOptions.toRaw(config: BackendConfig) = RawTranscribeParams(
            language = language,
            initialPrompt = initialPrompt,
            threads = if (config.cpuThreads == BackendConfig.AUTO_THREADS) 0 else config.cpuThreads,
            translate = translate,
            noContext = noContext,
            singleSegment = singleSegment,
            noTimestamps = noTimestamps,
            maxTokens = maxTokens,
            temperature = temperature,
            temperatureIncrement = temperatureIncrement,
        )
    }
}

/**
 * Version and capability information of the native whisper library.
 *
 * @property moduleVersion Kggml whisper module version, e.g. `"1.0.0"`.
 * @property whisperCppVersion pinned upstream whisper.cpp release, e.g. `"1.8.4"`.
 * @property whisperCppCommit full upstream whisper.cpp commit SHA (see `NOTICE`).
 * @property systemInfo `whisper_print_system_info()` (CPU features of this build).
 * @property hasCoreMl `true` if the Core ML encoder is compiled in (iOS builds).
 */
public data class WhisperNativeInfo(
    val moduleVersion: String,
    val whisperCppVersion: String,
    val whisperCppCommit: String,
    val systemInfo: String,
    val hasCoreMl: Boolean,
)
