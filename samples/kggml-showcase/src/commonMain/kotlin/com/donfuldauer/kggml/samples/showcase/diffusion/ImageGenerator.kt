package com.donfuldauer.kggml.samples.showcase.diffusion

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.diffusion.DiffusionEngine
import com.donfuldauer.kggml.diffusion.DiffusionEvent
import com.donfuldauer.kggml.diffusion.DiffusionGenerationParams
import com.donfuldauer.kggml.models.ModelInspector
import com.donfuldauer.kggml.models.defaultModelFileSystem
import kotlinx.coroutines.flow.Flow

/** A loaded text-to-image model used by the Diffusion Image Studio ([DiffusionImageGenerator] in the app). */
interface ImageGenerator : AutoCloseable {
    val modelPath: String
    val config: BackendConfig

    /** The backend the model actually runs on (never [ComputeBackend.AUTO]). */
    val activeBackend: ComputeBackend
    val usesGpu: Boolean

    /** Estimated resident bytes (header-based estimate for GGUF, otherwise the file size). */
    val estimatedBytes: Long

    /** One generation as progress events followed by the image (see `DiffusionEngine.generate`). */
    fun generate(params: DiffusionGenerationParams): Flow<DiffusionEvent>
}

/** Opens [ImageGenerator]s; may block on native model loading (called on the I/O dispatcher). */
fun interface ImageGeneratorFactory {
    suspend fun open(modelPath: String, config: BackendConfig): ImageGenerator
}

/** Production [ImageGeneratorFactory] backed by `:kggml-diffusion`. */
object DiffusionImageGeneratorFactory : ImageGeneratorFactory {
    override suspend fun open(modelPath: String, config: BackendConfig): ImageGenerator {
        val engine = DiffusionEngine(modelPath, config)
        val estimate = runCatching { ModelInspector.inspectFile(modelPath).estimatedPeakRam.totalBytes }
            .getOrElse { defaultModelFileSystem().size(modelPath).coerceAtLeast(0) }
        return DiffusionImageGenerator(engine, modelPath, estimate)
    }
}

/** [ImageGenerator] over a [DiffusionEngine]. */
class DiffusionImageGenerator(
    private val engine: DiffusionEngine,
    override val modelPath: String,
    override val estimatedBytes: Long,
) : ImageGenerator {
    override val config: BackendConfig get() = engine.backendConfig
    override val activeBackend: ComputeBackend get() = engine.activeBackend
    override val usesGpu: Boolean get() = engine.usesGpu

    override fun generate(params: DiffusionGenerationParams): Flow<DiffusionEvent> = engine.generate(params)

    override fun close() = engine.close()
}
