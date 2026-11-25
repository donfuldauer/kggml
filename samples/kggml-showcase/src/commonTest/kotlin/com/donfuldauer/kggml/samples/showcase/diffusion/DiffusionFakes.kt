package com.donfuldauer.kggml.samples.showcase.diffusion

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceException
import com.donfuldauer.kggml.diffusion.DiffusionEvent
import com.donfuldauer.kggml.diffusion.DiffusionGenerationParams
import com.donfuldauer.kggml.diffusion.DiffusionImage
import com.donfuldauer.kggml.diffusion.DiffusionPhase
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Deterministic generator: emits `Progress(0..steps)` then a `width x height` RGB image whose pixels derive from the
 * seed. With [stepGate] set, every step after the first waits for an element from the channel.
 */
class FakeImageGenerator(
    override val modelPath: String,
    override val config: BackendConfig,
    var stepGate: Channel<Unit>? = null,
    var failAtStep: Int? = null,
    var failure: InferenceException? = null,
) : ImageGenerator {
    val requests = mutableListOf<DiffusionGenerationParams>()
    var closed = false
        private set

    override val activeBackend: ComputeBackend
        get() = if (config.backend == ComputeBackend.CPU_ONLY) ComputeBackend.CPU_ONLY else ComputeBackend.GPU_METAL
    override val usesGpu: Boolean get() = activeBackend != ComputeBackend.CPU_ONLY
    override val estimatedBytes: Long = 2000

    override fun generate(params: DiffusionGenerationParams): Flow<DiffusionEvent> = flow {
        requests += params
        emit(DiffusionEvent.Progress(DiffusionPhase.DENOISING, 0, params.sampleSteps, 0f))
        for (step in 1..params.sampleSteps) {
            stepGate?.receive()
            if (step == failAtStep) {
                emit(DiffusionEvent.Error(failure!!))
                return@flow
            }
            emit(DiffusionEvent.Progress(DiffusionPhase.DENOISING, step, params.sampleSteps, 0.5f))
        }
        emit(DiffusionEvent.Completed(fakeImage(params.width, params.height, params.seed)))
    }

    override fun close() {
        closed = true
    }
}

fun fakeImage(width: Int, height: Int, seed: Long): DiffusionImage =
    DiffusionImage(width, height, 3, ByteArray(width * height * 3) { i -> ((i + seed) % 251).toByte() })

class FakeImageGeneratorFactory(
    var failure: Throwable? = null,
    private val configure: (FakeImageGenerator) -> Unit = {},
) : ImageGeneratorFactory {
    val opened = mutableListOf<FakeImageGenerator>()

    override suspend fun open(modelPath: String, config: BackendConfig): ImageGenerator {
        failure?.let { throw it }
        return FakeImageGenerator(modelPath, config).also(configure).also { opened += it }
    }
}
