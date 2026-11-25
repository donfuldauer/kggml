package com.donfuldauer.kggml.diffusion

import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.AtomicReference

/** Lock-free append-only log readable from any thread. */
internal class ConcurrentLog<T> {
    private val ref = AtomicReference<List<T>>(emptyList())

    fun add(item: T) {
        while (true) {
            val current = ref.load()
            if (ref.compareAndSet(current, current + item)) return
        }
    }

    val items: List<T> get() = ref.load()
}

internal data class GenerateCall(val context: Long, val params: RawGenerateParams, val hadListener: Boolean)

/**
 * Deterministic in-memory stand-in for the `kggml_diffusion.h` ABI.
 *
 * - [contextCreate] returns [createStatus] for the requested backend kind (default OK) and tracks the handle.
 * - [generate] behaves like the native wrapper: a latched abort cancels before any work; then it reports `0..steps`
 *   to the listener, calls [onStep] after each point (tests may block there, e.g. until an abort is requested) and
 *   stops with CANCELLED at the first point after an abort was latched. On success it "allocates" a native image,
 *   copies a deterministic gradient out of it and "frees" it, so tests can assert that no image buffer leaks.
 * - Every handle is tracked so tests can assert that each context is freed exactly once.
 */
internal class FakeDiffusionBridge(
    /** Status returned by contextCreate per requested backend kind. */
    val createStatus: (backendKind: Int) -> Int = { DiffusionStatus.OK },
    /** Effective backend reported by contextInfo (`null` = the requested one). */
    var reportedBackend: Int? = null,
    var supportsImageGeneration: Boolean = true,
    /** Invoked inside generate after each progress point (may block or request an abort). */
    var onStep: (FakeDiffusionBridge, Long, Int) -> Unit = { _, _, _ -> },
    /** Forced status of generate after the steps ran (e.g. NATIVE_FAILURE / OUT_OF_MEMORY). */
    var generateStatus: Int = DiffusionStatus.OK,
    /** Overrides the produced image dimensions (to simulate a misbehaving native layer). */
    var imageDims: ((RawGenerateParams) -> Triple<Int, Int, Int>)? = null,
    /** Extra progress points reported after the denoising steps (simulated VAE tiles), `0..vaeTiles`. */
    var vaeTiles: Int = 0,
) : NativeDiffusionBridge {
    private val nextHandle = AtomicInt(1000)
    private val liveRef = AtomicReference<Set<Long>>(emptySet())
    private val abortRef = AtomicReference<Set<Long>>(emptySet())
    private val backendOf = AtomicReference<Map<Long, Int>>(emptyMap())

    val creates = ConcurrentLog<RawContextParams>()
    val generates = ConcurrentLog<GenerateCall>()
    val frees = ConcurrentLog<Long>()
    val clears = AtomicInt(0)
    val imagesAllocated = AtomicInt(0)
    val imagesFreed = AtomicInt(0)
    val cancelledAtStep = ConcurrentLog<Int>()
    var lastErrorText = ""

    val live: Set<Long> get() = liveRef.load()

    private fun update(ref: AtomicReference<Set<Long>>, f: (Set<Long>) -> Set<Long>) {
        while (true) {
            val current = ref.load()
            if (ref.compareAndSet(current, f(current))) return
        }
    }

    override fun moduleVersion(): String = "1.0.0"

    override fun upstreamVersion(): String = "fake-sd"

    override fun upstreamCommit(): String = "0".repeat(40)

    override fun systemInfo(): String = "fake"

    override fun statusName(status: Int): String = "KGGML_STATUS_$status"

    override fun lastError(): String = lastErrorText

    override fun contextCreate(params: RawContextParams): RawHandleResult {
        creates.add(params)
        val status = createStatus(params.backend)
        if (status != DiffusionStatus.OK) {
            lastErrorText = "create failed for backend ${params.backend}"
            return RawHandleResult(status, 0L)
        }
        val handle = nextHandle.addAndFetch(1).toLong()
        while (true) {
            val current = backendOf.load()
            if (backendOf.compareAndSet(current, current + (handle to params.backend))) break
        }
        update(liveRef) { it + handle }
        return RawHandleResult(DiffusionStatus.OK, handle)
    }

    override fun contextFree(context: Long) {
        frees.add(context)
        check(context in live) { "double free / unknown context $context" }
        update(liveRef) { it - context }
        update(abortRef) { it - context }
    }

    override fun contextInfo(context: Long): RawContextInfo {
        check(context in live) { "contextInfo on dead context $context" }
        val backend = reportedBackend ?: backendOf.load().getValue(context)
        return RawContextInfo(
            status = DiffusionStatus.OK,
            backend = backend,
            usesGpu = backend != DiffusionBackendKind.CPU,
            threads = 4,
            supportsImageGeneration = supportsImageGeneration,
            defaultSampleMethod = DiffusionSampler.EULER_A.nativeValue,
        )
    }

    override fun requestAbort(context: Long) = update(abortRef) { it + context }

    override fun clearAbort(context: Long) {
        clears.addAndFetch(1)
        update(abortRef) { it - context }
    }

    override fun abortRequested(context: Long): Boolean = context in abortRef.load()

    override fun generate(context: Long, params: RawGenerateParams, listener: DiffusionProgressListener?): RawImageResult {
        check(context in live) { "generate on dead context $context" }
        generates.add(GenerateCall(context, params, listener != null))
        if (abortRequested(context)) {
            lastErrorText = "an abort is latched on this context"
            cancelledAtStep.add(-1)
            return RawImageResult(DiffusionStatus.CANCELLED)
        }
        fun phase(total: Int): Boolean {
            for (step in 0..total) {
                listener?.onProgress(step, total, 0.25f)
                onStep(this, context, step)
                if (abortRequested(context)) {
                    lastErrorText = "generation aborted at progress step $step/$total"
                    cancelledAtStep.add(step)
                    return false
                }
            }
            return true
        }
        if (!phase(params.sampleSteps)) return RawImageResult(DiffusionStatus.CANCELLED)
        if (vaeTiles > 0 && !phase(vaeTiles)) return RawImageResult(DiffusionStatus.CANCELLED)
        if (generateStatus != DiffusionStatus.OK) {
            lastErrorText = "forced failure"
            return RawImageResult(generateStatus)
        }
        val (w, h, c) = imageDims?.invoke(params) ?: Triple(params.width, params.height, 3)
        imagesAllocated.addAndFetch(1)
        try {
            return RawImageResult(DiffusionStatus.OK, w, h, c, gradient(w, h, c, params.seed))
        } finally {
            imagesFreed.addAndFetch(1) // kggml_diffusion_image_free()
        }
    }

    companion object {
        /** Pixel (x, y) channel k = (x + 3y + 7k + seed) & 255. */
        fun gradient(width: Int, height: Int, channels: Int, seed: Long): ByteArray {
            val out = ByteArray(maxOf(0, width * height * channels))
            var i = 0
            for (y in 0 until height) for (x in 0 until width) for (k in 0 until channels) {
                out[i++] = ((x + 3 * y + 7 * k + seed.toInt()) and 255).toByte()
            }
            return out
        }
    }
}
