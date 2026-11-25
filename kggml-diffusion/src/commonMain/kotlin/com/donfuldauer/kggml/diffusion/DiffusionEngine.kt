package com.donfuldauer.kggml.diffusion

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.InferenceEngine
import com.donfuldauer.kggml.core.InferenceException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.NativeFailureException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import com.donfuldauer.kggml.core.SerialExecutor
import com.donfuldauer.kggml.runtime.KggmlRuntime
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.AtomicReference
import kotlinx.coroutines.InternalCoroutinesApi
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ProducerScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.job

/**
 * Loads a full Stable Diffusion checkpoint (`.gguf`, `.safetensors` or `.ckpt`) with stable-diffusion.cpp and returns
 * a ready [DiffusionEngine]. Same as the [DiffusionModelFiles] overload with `DiffusionModelFiles(modelPath)`.
 */
public suspend fun DiffusionEngine(
    modelPath: String,
    backendConfig: BackendConfig = BackendConfig.DEFAULT,
    options: DiffusionEngineOptions = DiffusionEngineOptions(),
): DiffusionEngine = DiffusionEngine(DiffusionModelFiles(modelPath = modelPath), backendConfig, options)

/**
 * Loads [model] with stable-diffusion.cpp and returns a ready [DiffusionEngine].
 *
 * The weights are loaded on the new engine's own dedicated thread. [backendConfig] is enforced at load time:
 *
 * - [ComputeBackend.CPU_ONLY] pins every stable-diffusion.cpp module to the ggml CPU device, so no GPU backend is
 *   ever created (Accelerate / NEON on Apple, legal in the iOS background).
 * - [ComputeBackend.GPU_METAL] / [ComputeBackend.GPU_VULKAN] run on the [DiffusionEngineOptions.gpuDevice]-th device
 *   of that kind. [BackendConfig.gpuLayers] `= 0` keeps the weights in host memory and streams them to the GPU per
 *   graph (stable-diffusion.cpp `offload_params_to_cpu`); any other value keeps them on the GPU.
 * - [ComputeBackend.NPU_COREML] is not supported by stable-diffusion.cpp; with
 *   [DiffusionEngineOptions.allowBackendFallback] it falls back to Metal / CPU, otherwise loading fails with
 *   [com.donfuldauer.kggml.core.BackendUnavailableException].
 * - [ComputeBackend.AUTO] picks the first available of Metal, Vulkan, CPU.
 *
 * [BackendConfig.cpuThreads] sets the CPU threads ([BackendConfig.AUTO_THREADS] = performance cores),
 * [BackendConfig.useFlashAttention] flash attention in every module, and [BackendConfig.useMmap] whether the weight
 * files are memory-mapped.
 *
 * @throws com.donfuldauer.kggml.core.ModelLoadException if a file is missing or not a loadable diffusion model.
 * @throws com.donfuldauer.kggml.core.BackendUnavailableException if the backend is unavailable and fallback is off.
 * @throws com.donfuldauer.kggml.core.OutOfMemoryException if the weights cannot be allocated.
 * @throws com.donfuldauer.kggml.core.NativeFailureException if the native libraries cannot be loaded.
 */
public suspend fun DiffusionEngine(
    model: DiffusionModelFiles,
    backendConfig: BackendConfig = BackendConfig.DEFAULT,
    options: DiffusionEngineOptions = DiffusionEngineOptions(),
): DiffusionEngine = DiffusionEngine.open(
    model = model,
    backendConfig = backendConfig,
    options = options,
    bridges = DiffusionBridgeHolder.platform,
    availableBackends = { platformAvailableBackends() },
    dispatcherFactory = SerialDispatcherFactory.Default,
)

/** Backends stable-diffusion.cpp can use on this device: the runtime's GPU backends (no Core ML path). */
internal fun platformAvailableBackends(): Set<ComputeBackend> = KggmlRuntime.availableBackends() - ComputeBackend.NPU_COREML

/**
 * stable-diffusion.cpp text-to-image engine with streamed progress and cooperative cancellation.
 *
 * Create instances with the [DiffusionEngine] factory functions. Engines are ordinary objects — never singletons —
 * and any number of them can coexist and generate concurrently: each owns its native context (weights + backends)
 * and a dedicated single-threaded dispatcher ([SerialExecutor]) that serializes loading, every generation and the
 * native release in [close].
 *
 * **Cancellation** is cooperative at step granularity: cancelling the collector of [generate] (or
 * [abortActiveGeneration] / [close]) latches the native abort flag, and stable-diffusion.cpp stops at its next
 * progress point — after the denoising step (or VAE tile) that is computing finishes. No native image buffer
 * outlives a call, and the engine stays usable afterwards.
 */
public class DiffusionEngine internal constructor(
    /** The model files this engine was loaded from. */
    public val model: DiffusionModelFiles,
    override val backendConfig: BackendConfig,
    /** Diffusion-specific construction options. */
    public val options: DiffusionEngineOptions,
    private val bridges: DiffusionBridgeHolder,
    private val availableBackends: () -> Set<ComputeBackend>,
    dispatcherFactory: SerialDispatcherFactory,
) : InferenceEngine {

    private class Loaded(val backend: ComputeBackend, val usesGpu: Boolean, val info: DiffusionModelInfo)

    private val executor = SerialExecutor("kggml-diffusion-${nextInstanceId.addAndFetch(1)}", dispatcherFactory)
    private val loaded = AtomicReference<Loaded?>(null)
    private val closed = AtomicBoolean(false)
    private val abortGate = AbortGate()

    /** [generate] collections between entering and leaving the engine thread (see [close]). */
    private val activeGenerations = AtomicInt(0)

    /** Native context handle: only touched on the executor thread. */
    private var context: Long = 0L

    private fun loadedState(): Loaded = loaded.load() ?: error("DiffusionEngine is not loaded")

    /** The backend stable-diffusion.cpp actually runs on (never [ComputeBackend.AUTO]). */
    override val activeBackend: ComputeBackend
        get() = loadedState().backend

    /** `true` if a GPU device was handed to stable-diffusion.cpp; always `false` for [ComputeBackend.CPU_ONLY]. */
    public val usesGpu: Boolean
        get() = loadedState().usesGpu

    /** Static properties of the loaded model. */
    public val modelInfo: DiffusionModelInfo
        get() = loadedState().info

    override val isClosed: Boolean
        get() = closed.load()

    /** `true` while a generation is executing native code on this engine. */
    public val isGenerating: Boolean
        get() = abortGate.isArmed

    /**
     * Generates one image from [params] as a cold `Flow` of [DiffusionEvent]s. Each collection runs one generation on
     * the engine thread (collections on the same engine are serialized).
     *
     * Events, in order:
     * - [DiffusionEvent.Progress] — `step = 0` when sampling starts, then one per completed denoising step
     *   ([DiffusionPhase.DENOISING]); with [DiffusionGenerationParams.vaeTiling] the VAE tiles follow
     *   ([DiffusionPhase.VAE_TILES]). Progress is buffered, so a slow collector never stalls the sampler.
     * - [DiffusionEvent.Completed] — the last event, with the RGB8 [DiffusionImage].
     * - [DiffusionEvent.Error] — a typed [InferenceException] (native failure, out of memory,
     *   [abortActiveGeneration], or [close] of the engine, reported as [InferenceCancelledException]). It is terminal:
     *   no event follows it and the flow then **completes normally**.
     *
     * Cancelling the collector aborts the running generation at its next step boundary and throws
     * `CancellationException` as usual.
     *
     * @throws IllegalStateException immediately if the engine is closed.
     */
    public fun generate(params: DiffusionGenerationParams): Flow<DiffusionEvent> {
        check(!isClosed) { "DiffusionEngine is closed" }
        return channelFlow {
            // The final send() may suspend on the engine thread while the collector is slow. close() must not shut
            // that thread down underneath it (on Kotlin/Native the terminated worker would never resume the producer),
            // so the last generation to leave shuts the executor down instead (see close()).
            activeGenerations.addAndFetch(1)
            try {
                serial { runGeneration(params, this@channelFlow) }
            } finally {
                if (activeGenerations.addAndFetch(-1) == 0 && isClosed) shutDownExecutor()
            }
        }.buffer(Channel.UNLIMITED)
    }

    /**
     * Runs [generate] to completion and returns the image. [onProgress] (optional) receives every
     * [DiffusionEvent.Progress] on the calling coroutine.
     *
     * @throws InferenceException for the failure a [DiffusionEvent.Error] would report.
     * @throws IllegalStateException if the engine is closed.
     */
    public suspend fun generateImage(
        params: DiffusionGenerationParams,
        onProgress: ((DiffusionEvent.Progress) -> Unit)? = null,
    ): DiffusionImage {
        var image: DiffusionImage? = null
        generate(params).collect { event ->
            when (event) {
                is DiffusionEvent.Progress -> onProgress?.invoke(event)
                is DiffusionEvent.Completed -> image = event.image
                is DiffusionEvent.Error -> throw event.exception
            }
        }
        return image ?: error("generation ended without an image")
    }

    /**
     * Requests the native abort of the generation currently executing, if any; its [generate] flow then ends with a
     * terminal [DiffusionEvent.Error] ([InferenceCancelledException]) at the next step boundary. Safe from any
     * thread. Queued generations are not affected.
     *
     * @return `true` if a running generation was signalled.
     */
    public fun abortActiveGeneration(): Boolean = abortGate.fire { bridges.get().requestAbort(it) }

    /**
     * Closes the engine: aborts a running generation, rejects queued and future calls with [IllegalStateException],
     * frees the native context on the engine thread, and stops that thread. Idempotent and safe from any thread.
     */
    override fun close() {
        if (!closed.compareAndSet(expectedValue = false, newValue = true)) return
        abortActiveGeneration()
        // With a generation in flight, its collector shuts the executor down when the generation leaves the engine
        // thread.
        if (activeGenerations.load() == 0) shutDownExecutor()
    }

    /** Frees the native context on the engine thread after queued work, then stops it. Idempotent. */
    private fun shutDownExecutor() = executor.close(release = ::freeAll)

    /**
     * Runs [block] on the engine thread, serialized with every other call. Calls still queued when the engine closes
     * fail with [IllegalStateException] instead of touching native state that is about to be released.
     */
    private suspend fun <T> serial(block: suspend () -> T): T = executor.execute {
        check(!isClosed) { "DiffusionEngine is closed" }
        block()
    }

    // ------------------------------------------------------------------------------------------------------
    // Executor-thread internals
    // ------------------------------------------------------------------------------------------------------

    internal suspend fun start() {
        try {
            executor.execute { load() }
        } catch (failure: Throwable) {
            close()
            throw failure
        }
    }

    private fun load() {
        val bridge = bridges.get()
        var backend = ComputeBackend.resolve(backendConfig.backend, availableBackends(), options.allowBackendFallback)
        while (true) {
            val result = bridge.contextCreate(contextParams(backend))
            if (result.status == DiffusionStatus.OK) {
                adopt(bridge, result.handle, backend)
                return
            }
            val next = backend.fallback
            if (result.status != DiffusionStatus.BACKEND_UNAVAILABLE || !options.allowBackendFallback || next == null) {
                throw bridge.nativeError(result.status, "kggml_diffusion_context_create", backend)
            }
            backend = next
        }
    }

    internal fun contextParams(backend: ComputeBackend): RawContextParams = RawContextParams(
        paths = model.toRaw(),
        backend = DiffusionBackendKind.of(backend),
        gpuDevice = options.gpuDevice,
        threads = if (backendConfig.cpuThreads == BackendConfig.AUTO_THREADS) 0 else backendConfig.cpuThreads,
        useMmap = backendConfig.useMmap,
        flashAttention = backendConfig.useFlashAttention,
        offloadParamsToCpu = backend != ComputeBackend.CPU_ONLY && backendConfig.effectiveGpuLayers(backend) == 0,
        keepClipOnCpu = options.keepTextEncoderOnCpu,
        keepVaeOnCpu = options.keepVaeOnCpu,
        vaeDecodeOnly = true,
        weightType = options.weightType,
    )

    private fun adopt(bridge: NativeDiffusionBridge, handle: Long, requested: ComputeBackend) {
        context = handle
        try {
            val raw = bridge.contextInfo(handle)
            bridge.check(raw.status, "kggml_diffusion_context_get_info", requested)
            if (!raw.supportsImageGeneration) throw ModelLoadException("model cannot generate images (txt2img)")
            val info = DiffusionModelInfo(
                supportsImageGeneration = true,
                defaultSampler = DiffusionSampler.fromNative(raw.defaultSampleMethod) ?: DiffusionSampler.MODEL_DEFAULT,
                threads = raw.threads,
            )
            val backend = DiffusionBackendKind.toComputeBackend(raw.backend) ?: requested
            loaded.store(Loaded(backend, raw.usesGpu, info))
        } catch (failure: Throwable) {
            freeAll()
            throw failure
        }
    }

    private fun freeAll() {
        // No abort may reach a context once it is freed.
        abortGate.disarm()
        val bridge = runCatching { bridges.get() }.getOrNull() ?: return
        if (context != 0L) bridge.contextFree(context)
        context = 0L
    }

    private suspend fun runGeneration(params: DiffusionGenerationParams, out: ProducerScope<DiffusionEvent>) {
        val bridge = bridges.get()
        try {
            val tracker = DiffusionProgressTracker()
            // Runs inside the blocking native call: never suspends and never throws into native code. The flow's
            // channel is unbounded, so trySend only fails once the collector is gone.
            val listener = DiffusionProgressListener { step, total, secondsPerStep ->
                try {
                    tracker.onProgress(step, total, secondsPerStep)?.let { out.trySend(it) }
                } catch (_: Throwable) {
                    // A progress event is never worth failing a generation for.
                }
            }
            val ctx = context
            val result = abortable(bridge, ctx) { bridge.generate(ctx, params.toRaw(), listener) }
            throwOnFailure(bridge, result.status, "kggml_diffusion_generate")
            val image = try {
                DiffusionImage(result.width, result.height, result.channels, result.pixels)
            } catch (e: IllegalArgumentException) {
                throw NativeFailureException(message = "kggml_diffusion_generate returned an invalid image: ${e.message}", cause = e)
            }
            if (image.width != params.width || image.height != params.height || image.channels != RGB_CHANNELS) {
                throw NativeFailureException(
                    message = "kggml_diffusion_generate returned ${image.width}x${image.height}x${image.channels}, " +
                        "expected ${params.width}x${params.height}x$RGB_CHANNELS",
                )
            }
            out.send(DiffusionEvent.Completed(image))
        } catch (e: InferenceException) {
            out.send(DiffusionEvent.Error(e))
        }
    }

    /**
     * Runs [call] on context [ctx] with the native abort wired to the calling coroutine's job: cancelling it (or
     * [abortActiveGeneration] / [close]) latches the context's abort flag, which stable-diffusion.cpp observes at its
     * next step boundary. The latch is cleared once on entry, so an abort aimed at an earlier generation is not
     * inherited.
     */
    private suspend inline fun <T> abortable(bridge: NativeDiffusionBridge, ctx: Long, call: () -> T): T {
        val job = currentCoroutineContext().job
        bridge.clearAbort(ctx)
        abortGate.arm(ctx)
        // onCancelling = true: the handler must run while this thread is still blocked in native code.
        @OptIn(InternalCoroutinesApi::class)
        val registration = job.invokeOnCompletion(onCancelling = true, invokeImmediately = true) { cause ->
            if (cause != null) abortActiveGeneration()
        }
        try {
            return call()
        } finally {
            registration.dispose()
            abortGate.disarm()
        }
    }

    /** Throws the right exception for a non-OK [status]. Must run right after the native call. */
    private suspend fun throwOnFailure(bridge: NativeDiffusionBridge, status: Int, operation: String) {
        if (status == DiffusionStatus.OK) return
        val error = bridge.nativeError(status, operation, activeBackend)
        if (status == DiffusionStatus.CANCELLED) {
            currentCoroutineContext().ensureActive() // caller cancelled -> CancellationException
            if (isClosed) throw InferenceCancelledException("DiffusionEngine was closed during generation", error)
        }
        throw error
    }

    public companion object {
        private const val RGB_CHANNELS = 3
        private val nextInstanceId = AtomicInt(0)

        /** Version information of the loaded native diffusion library. */
        public fun nativeInfo(): DiffusionNativeInfo = nativeInfo(DiffusionBridgeHolder.platform.get())

        internal fun nativeInfo(bridge: NativeDiffusionBridge): DiffusionNativeInfo = DiffusionNativeInfo(
            moduleVersion = bridge.moduleVersion(),
            stableDiffusionCppVersion = bridge.upstreamVersion(),
            stableDiffusionCppCommit = bridge.upstreamCommit(),
            systemInfo = bridge.systemInfo(),
        )

        internal suspend fun open(
            model: DiffusionModelFiles,
            backendConfig: BackendConfig,
            options: DiffusionEngineOptions,
            bridges: DiffusionBridgeHolder,
            availableBackends: () -> Set<ComputeBackend>,
            dispatcherFactory: SerialDispatcherFactory,
        ): DiffusionEngine {
            val engine = DiffusionEngine(model, backendConfig, options, bridges, availableBackends, dispatcherFactory)
            engine.start()
            return engine
        }
    }
}

internal fun DiffusionGenerationParams.toRaw(): RawGenerateParams = RawGenerateParams(
    prompt = prompt,
    negativePrompt = negativePrompt,
    width = width,
    height = height,
    sampleSteps = sampleSteps,
    cfgScale = cfgScale,
    distilledGuidance = distilledGuidance,
    sampleMethod = sampler.nativeValue,
    scheduler = scheduler.nativeValue,
    seed = if (seed < 0) DiffusionGenerationParams.RANDOM_SEED else seed,
    clipSkip = clipSkip,
    vaeTiling = vaeTiling,
)
