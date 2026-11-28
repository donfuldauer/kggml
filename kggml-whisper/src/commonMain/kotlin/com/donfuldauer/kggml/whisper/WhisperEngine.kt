package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.InferenceEngine
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import com.donfuldauer.kggml.core.SerialExecutor
import com.donfuldauer.kggml.core.TranscriptionEvent
import com.donfuldauer.kggml.runtime.KggmlRuntime
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.AtomicLong
import kotlin.concurrent.atomics.AtomicReference
import kotlin.jvm.JvmName
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.InternalCoroutinesApi
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
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
 * - [ComputeBackend.AUTO] picks the first available of CoreML, Metal, OpenCL, Vulkan, CPU.
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
 *
 * Live audio is transcribed with [transcribeStream], which turns a `Flow` of PCM chunks into a
 * `Flow<TranscriptionEvent>` without writing any audio to disk. Each collection owns its own native
 * `whisper_state`, so streams, one-shot [transcribe] calls and other streams on the same engine never disturb each
 * other's decoder state; their decodes are serialized on the engine's thread like every other native call.
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

    /** Native streaming states of the [transcribeStream] collections currently running on this engine. */
    private val streamSlots = AtomicReference<Set<StreamSlot>>(emptySet())

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

    /** `true` while a [transcribe] call or a [transcribeStream] decode is executing native code. */
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
     * Transcribes a live stream of interleaved float PCM chunks in `[-1, 1]` into a cold `Flow` of
     * [TranscriptionEvent]s. Nothing is written to disk: audio flows from [audioFlow] through the native streaming
     * resampler (skipped for 16 kHz mono) into an in-memory ring buffer, a voice-activity turn detector, and
     * `whisper_full_with_state()` on a `whisper_state` owned by this collection.
     *
     * Events, in order:
     * - [TranscriptionEvent.PartialHypothesis] — unstable text of the speech turn in progress, re-decoded every
     *   [StreamingOptions.partialStepMs] of new audio (conflated when the engine is behind).
     * - [TranscriptionEvent.CommittedSegment] — locked, timestamped segments of a finished turn (a pause of
     *   [com.donfuldauer.kggml.whisper.streaming.VadConfig.trailingSilenceMs], a 30 s window split, or the end of
     *   the input). Indices increase monotonically from 0 and timestamps are milliseconds from the start of the
     *   stream. The text tokens of committed segments are carried forward as `prompt_tokens` of later decodes
     *   ([StreamingOptions.carryPromptTokens]) across pauses and 30 s windows.
     * - [TranscriptionEvent.Completed] — after [audioFlow] completed and the last turn was committed; the last event.
     * - [TranscriptionEvent.Error] — a typed [com.donfuldauer.kggml.core.InferenceException] (native failure,
     *   malformed chunk, [abortActiveTranscription], or [close] of the engine, reported as
     *   [InferenceCancelledException]). It is terminal: no event follows it and the flow then **completes
     *   normally** rather than throwing.
     *
     * Exceptions thrown by [audioFlow] itself propagate to the collector unchanged. Cancelling the collector aborts
     * a running native decode through whisper.cpp's abort callback and frees the stream's `whisper_state` and
     * resampler on the engine thread before the collection finishes.
     *
     * The audio flow is collected on [kotlinx.coroutines.Dispatchers.Default]. If the decoder falls behind by
     * [StreamingOptions.bufferMs], collection of [audioFlow] is suspended (back-pressure) — audio is never dropped.
     *
     * @param sampleRate input sample rate in Hz, `1000..384000`.
     * @param channels interleaved channel count, `1..8`; every chunk's size must be a multiple of it.
     * @throws InvalidInputException immediately for an unsupported [sampleRate] / [channels].
     * @throws IllegalStateException immediately if the engine is closed.
     */
    public fun transcribeStream(
        audioFlow: Flow<FloatArray>,
        sampleRate: Int = SAMPLE_RATE,
        channels: Int = 1,
        options: StreamingOptions = StreamingOptions.DEFAULT,
    ): Flow<TranscriptionEvent> {
        validateFormat(sampleRate, channels)
        check(!isClosed) { "WhisperEngine is closed" }
        return WhisperStream(this, bridges, sampleRate, channels, options, StreamSource.F32(audioFlow)).flow()
    }

    /**
     * [transcribeStream] for interleaved signed 16-bit PCM chunks (normalized natively as `sample / 32768`). See the
     * `FloatArray` overload for the event contract. (On the JVM this overload is named `transcribeStreamPcm16`,
     * since both overloads erase to `Flow`.)
     */
    @JvmName("transcribeStreamPcm16")
    public fun transcribeStream(
        audioFlow: Flow<ShortArray>,
        sampleRate: Int = SAMPLE_RATE,
        channels: Int = 1,
        options: StreamingOptions = StreamingOptions.DEFAULT,
    ): Flow<TranscriptionEvent> {
        validateFormat(sampleRate, channels)
        check(!isClosed) { "WhisperEngine is closed" }
        return WhisperStream(this, bridges, sampleRate, channels, options, StreamSource.S16(audioFlow)).flow()
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
     * [InferenceCancelledException] (a [transcribeStream] decode ends its stream with a terminal
     * [TranscriptionEvent.Error]). Safe from any thread. Queued calls are not affected.
     *
     * @return `true` if a running transcription was signalled.
     */
    public fun abortActiveTranscription(): Boolean = abortGate.fire { bridges.get().requestAbort(it) }

    /**
     * Closes the engine: aborts a running transcription, rejects queued and future calls with
     * [IllegalStateException], ends running [transcribeStream] collections with a terminal
     * [TranscriptionEvent.Error] ([InferenceCancelledException]), frees their native states and then the native
     * context on the engine's thread, and stops that thread. Idempotent and safe from any thread.
     */
    override fun close() {
        if (!closed.compareAndSet(expectedValue = false, newValue = true)) return
        abortActiveTranscription()
        streamSlots.load().forEach { it.onEngineClosed() } // wake idle streams so they end promptly
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
        val bridge = bridges.get()
        // States first (they belong to the context). A stream whose state is released here re-creates it lazily on
        // the next model (reinitialize) or ends with an error (close).
        streamSlots.load().forEach { it.release(bridge) }
        bridge.free(current)
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
        val current = loadedHandle()
        val bridge = bridges.get()
        val config = state.load().config

        val status = abortable(bridge, current) { bridge.transcribe(current, pcm, count, options.toRaw(config)) }
        throwOnFailure(bridge, status, "kggml_whisper_transcribe")

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

    private fun loadedHandle(): Long {
        val current = handle
        check(current != 0L) { "WhisperEngine has no model loaded (a previous reinitialize failed)" }
        return current
    }

    /**
     * Runs the blocking native [call] on context [current] with the native abort wired to the calling coroutine's
     * job: cancelling it (or [abortActiveTranscription] / [close]) makes whisper.cpp stop at its next abort poll.
     */
    private suspend inline fun <T> abortable(bridge: NativeWhisperBridge, current: Long, call: () -> T): T {
        val job = currentCoroutineContext().job
        bridge.clearAbort(current)
        abortGate.arm(current)
        // onCancelling = true: the handler must run while this thread is still blocked in native code (a plain
        // completion handler would only run after the job completes, i.e. after whisper finished).
        @OptIn(InternalCoroutinesApi::class)
        val registration = job.invokeOnCompletion(onCancelling = true, invokeImmediately = true) { cause ->
            if (cause != null) abortActiveTranscription()
        }
        try {
            return call()
        } finally {
            registration.dispose()
            abortGate.disarm()
        }
    }

    /** Maps a non-OK transcription [status] to the right exception. Must run right after the native call. */
    private suspend fun throwOnFailure(bridge: NativeWhisperBridge, status: Int, operation: String) {
        if (status == WhisperStatus.OK) return
        val error = bridge.nativeError(status, operation, activeBackend)
        if (status == WhisperStatus.CANCELLED) {
            currentCoroutineContext().ensureActive() // caller cancelled -> CancellationException
            if (isClosed) throw InferenceCancelledException("WhisperEngine was closed during transcription", error)
        }
        throw error
    }

    // ------------------------------------------------------------------------------------------------------
    // Streaming support (used by WhisperStream; every native call runs on the executor)
    // ------------------------------------------------------------------------------------------------------

    /**
     * Registers a new stream and allocates its native `whisper_state`.
     *
     * @throws IllegalStateException if the engine is closed.
     */
    internal suspend fun openStream(onEngineClosed: () -> Unit): StreamSlot = executor.execute {
        val current = loadedHandle()
        val bridge = bridges.get()
        val max = bridge.maxPromptTokens(current)
        bridge.check(max.status, "kggml_whisper_max_prompt_tokens")
        val slot = StreamSlot(max.value.coerceAtLeast(0), onEngineClosed)
        register(slot)
        try {
            slot.ensureState(bridge, current)
        } catch (failure: Throwable) {
            unregister(slot)
            slot.release(bridge)
            throw failure
        }
        slot
    }

    /**
     * Decodes `pcm[0 until count]` on the stream's own `whisper_state`, conditioned on `prompt[0 until promptCount]`.
     * Segments are read through `new_segment_callback` as whisper.cpp finalizes them. Timestamps are relative to
     * `pcm[0]`.
     */
    internal suspend fun decodeStream(
        slot: StreamSlot,
        pcm: FloatArray,
        count: Int,
        options: TranscriptionOptions,
        prompt: IntArray,
        promptCount: Int,
    ): List<StreamedSegment> = executor.execute {
        val current = loadedHandle()
        val bridge = bridges.get()
        val stateHandle = slot.ensureState(bridge, current)
        val params = options.toRaw(state.load().config).copy(noContext = true)
        val segments = ArrayList<StreamedSegment>()
        var listenerFailure: Throwable? = null
        val listener = WhisperSegmentListener { first, n ->
            try {
                for (i in maxOf(first, segments.size) until first + n) segments += readStreamSegment(bridge, stateHandle, i)
                WhisperStatus.OK
            } catch (t: Throwable) {
                listenerFailure = t // rethrown below; never unwinds through native frames
                WhisperStatus.NATIVE_FAILURE
            }
        }
        val status = abortable(bridge, current) {
            bridge.transcribeWithState(current, stateHandle, pcm, count, params, prompt, promptCount, listener)
        }
        if (status != WhisperStatus.CANCELLED) listenerFailure?.let { throw it }
        throwOnFailure(bridge, status, "kggml_whisper_transcribe_with_state")
        // Defensive: pick up any finalized segment whisper.cpp did not announce through the callback.
        val total = bridge.stateSegmentCount(stateHandle)
        bridge.check(total.status, "kggml_whisper_state_segment_count")
        for (i in segments.size until total.value) segments += readStreamSegment(bridge, stateHandle, i)
        segments
    }

    /**
     * Unregisters [slot] and frees its native state on the engine thread. If the engine is already closed, its
     * pending release step frees every still-registered state instead (so this never races a running decode).
     */
    internal suspend fun closeStream(slot: StreamSlot) {
        try {
            executor.execute {
                unregister(slot)
                slot.release(bridges.get())
            }
        } catch (_: IllegalStateException) {
            // Executor closed: freeModel() (scheduled by close()) releases the state.
        }
    }

    private fun readStreamSegment(bridge: NativeWhisperBridge, stateHandle: Long, index: Int): StreamedSegment {
        val raw = bridge.stateSegment(stateHandle, index)
        bridge.check(raw.status, "kggml_whisper_state_get_segment")
        val tokens = bridge.stateSegmentTokens(stateHandle, index)
        bridge.check(tokens.status, "kggml_whisper_state_get_segment_tokens")
        val start = raw.startMs.coerceAtLeast(0)
        return StreamedSegment(raw.text.decodeToString(), start, raw.endMs.coerceAtLeast(start), tokens.tokens)
    }

    private fun register(slot: StreamSlot) {
        while (true) {
            val current = streamSlots.load()
            if (streamSlots.compareAndSet(current, current + slot)) return
        }
    }

    private fun unregister(slot: StreamSlot) {
        while (true) {
            val current = streamSlots.load()
            if (streamSlots.compareAndSet(current, current - slot)) return
        }
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

        internal fun validateFormat(sampleRate: Int, channels: Int) {
            if (channels !in 1..MAX_CHANNELS) {
                throw InvalidInputException("channels must be in 1..$MAX_CHANNELS, was $channels")
            }
            if (sampleRate !in MIN_SAMPLE_RATE..MAX_SAMPLE_RATE) {
                throw InvalidInputException("sampleRate must be in $MIN_SAMPLE_RATE..$MAX_SAMPLE_RATE, was $sampleRate")
            }
        }

        private fun validateShape(size: Int, sampleRate: Int, channels: Int): Int {
            validateFormat(sampleRate, channels)
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
 * @property moduleVersion Kggml whisper module version, e.g. `"0.1.0"`.
 * @property whisperCppVersion pinned upstream whisper.cpp release, e.g. `"1.9.6"`.
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

/**
 * The native `whisper_state` of one [WhisperEngine.transcribeStream] collection.
 *
 * [ensureState] and [release] run on the engine thread only; the handle is swapped atomically so that a state is
 * freed exactly once whichever of the stream's own cleanup or the engine's [WhisperEngine.close] /
 * [WhisperEngine.reinitialize] gets to it first.
 *
 * @property maxPromptTokens the model's prompt-token budget (`kggml_whisper_max_prompt_tokens`).
 */
internal class StreamSlot(val maxPromptTokens: Int, private val engineClosed: () -> Unit) {
    private val stateHandle = AtomicLong(0L)

    /** `true` while a native state is allocated. */
    val hasState: Boolean get() = stateHandle.load() != 0L

    /** Returns the state, creating it on context [context] if there is none (first use or after reinitialize). */
    fun ensureState(bridge: NativeWhisperBridge, context: Long): Long {
        stateHandle.load().let { if (it != 0L) return it }
        val created = bridge.stateCreate(context)
        bridge.check(created.status, "kggml_whisper_state_create")
        stateHandle.store(created.handle)
        return created.handle
    }

    /** Frees the native state, if any. Idempotent. */
    fun release(bridge: NativeWhisperBridge) {
        val h = stateHandle.exchange(0L)
        if (h != 0L) bridge.stateFree(h)
    }

    fun onEngineClosed() = engineClosed()
}

/** One segment of a streaming decode; times in ms relative to the decoded buffer. */
internal class StreamedSegment(val text: String, val startMs: Long, val endMs: Long, val tokens: IntArray)
