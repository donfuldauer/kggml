package com.donfuldauer.kggml.samples.showcase.diffusion

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.diffusion.DiffusionEvent
import com.donfuldauer.kggml.diffusion.DiffusionGenerationParams
import com.donfuldauer.kggml.diffusion.DiffusionImage
import com.donfuldauer.kggml.diffusion.DiffusionPhase
import com.donfuldauer.kggml.samples.showcase.describe
import com.donfuldauer.kggml.samples.showcase.formatBytes
import com.donfuldauer.kggml.samples.showcase.models.ResidentModels
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.random.Random

enum class DiffusionActivity(val label: String) {
    IDLE("Idle"),
    LOADING_MODEL("Loading model…"),
    GENERATING("Generating…"),
    EXPORTING("Exporting PNG…"),
}

/** Progress of the running generation. */
data class GenerationProgress(val phase: DiffusionPhase, val step: Int, val totalSteps: Int, val secondsPerStep: Float) {
    val fraction: Float get() = if (totalSteps <= 0) 0f else step.toFloat() / totalSteps
    val label: String
        get() = when (phase) {
            DiffusionPhase.DENOISING -> "Denoising step $step / $totalSteps"
            DiffusionPhase.VAE_TILES -> "Decoding VAE tile $step / $totalSteps"
        }
}

/** A finished image with the seed and settings that reproduce it. */
data class GeneratedImage(val image: DiffusionImage, val seed: Long, val prompt: String, val steps: Int, val cfgScale: Float)

data class DiffusionStudioState(
    val modelPath: String = "",
    val backend: ComputeBackend = ComputeBackend.AUTO,
    val prompt: String = "a watercolor painting of a lighthouse at sunset",
    val negativePrompt: String = "blurry, low quality",
    val steps: Int = DiffusionGenerationParams.DEFAULT_SAMPLE_STEPS,
    val cfgScale: Float = DiffusionGenerationParams.DEFAULT_CFG_SCALE,
    val size: Int = 512,
    val seedText: String = "42",
    /** Pick a new random seed for every generation (the chosen seed is shown so the image can be reproduced). */
    val randomSeed: Boolean = false,
    val loadedBackend: ComputeBackend? = null,
    val usesGpu: Boolean = false,
    val activity: DiffusionActivity = DiffusionActivity.IDLE,
    val progress: GenerationProgress? = null,
    val result: GeneratedImage? = null,
    val exportPath: String = "",
    val status: String? = null,
    val error: String? = null,
) {
    val isBusy: Boolean get() = activity != DiffusionActivity.IDLE
    val seed: Long? get() = seedText.trim().toLongOrNull()?.takeIf { it >= 0 }
    val canGenerate: Boolean get() = !isBusy && modelPath.isNotBlank() && prompt.isNotBlank() && (randomSeed || seed != null)
    val canCancel: Boolean get() = activity == DiffusionActivity.GENERATING || activity == DiffusionActivity.LOADING_MODEL
    val canExport: Boolean get() = !isBusy && result != null && exportPath.isNotBlank()
}

/**
 * State holder of the Diffusion Image Studio. All public methods must be called on [scope]'s thread (the UI
 * thread); model loading and PNG export run on [ioDispatcher]. The engine is leased from [residents] for each
 * generation, so it can be evicted on low memory while idle.
 *
 * @param writeFile writes the exported PNG (in the app: the local file system).
 * @param exportDirectory default directory of exported images.
 */
class DiffusionStudioController(
    private val scope: CoroutineScope,
    private val factory: ImageGeneratorFactory,
    private val residents: ResidentModels,
    private val ioDispatcher: CoroutineDispatcher,
    private val writeFile: (String, ByteArray) -> Unit,
    private val exportDirectory: String,
    initialModelPath: String = "",
    private val random: Random = Random.Default,
) : AutoCloseable {
    private val mutableState = MutableStateFlow(DiffusionStudioState(modelPath = initialModelPath))
    val state: StateFlow<DiffusionStudioState> = mutableState.asStateFlow()

    private var job: Job? = null
    private var loadedKey: String? = null
    private var closed = false

    fun setModelPath(path: String) = mutableState.update { it.copy(modelPath = path) }
    fun setBackend(backend: ComputeBackend) = mutableState.update { it.copy(backend = backend) }
    fun setPrompt(text: String) = mutableState.update { it.copy(prompt = text) }
    fun setNegativePrompt(text: String) = mutableState.update { it.copy(negativePrompt = text) }
    fun setSteps(steps: Int) = mutableState.update { it.copy(steps = steps.coerceIn(1, MAX_STEPS)) }

    /** CFG scale, rounded to 0.5 and clamped to `1..20`. */
    fun setCfgScale(scale: Float) = mutableState.update { it.copy(cfgScale = (kotlin.math.round(scale * 2) / 2f).coerceIn(1f, MAX_CFG)) }
    fun setSize(size: Int) = mutableState.update { it.copy(size = size) }
    fun setSeedText(text: String) = mutableState.update { it.copy(seedText = text.filter { c -> c.isDigit() }.take(10)) }
    fun setRandomSeed(enabled: Boolean) = mutableState.update { it.copy(randomSeed = enabled) }
    fun setExportPath(path: String) = mutableState.update { it.copy(exportPath = path) }

    /** Rolls a new seed into the seed field. */
    fun randomizeSeed() = mutableState.update { it.copy(seedText = nextSeed().toString()) }

    /** Generates an image with the current settings, streaming step progress. */
    fun generate() {
        val current = mutableState.value
        if (!current.canGenerate || closed) return
        val seed = if (current.randomSeed) nextSeed() else current.seed ?: return
        val params = try {
            DiffusionGenerationParams(
                prompt = current.prompt.trim(),
                negativePrompt = current.negativePrompt.trim(),
                width = current.size,
                height = current.size,
                sampleSteps = current.steps,
                cfgScale = current.cfgScale,
                seed = seed,
            )
        } catch (e: IllegalArgumentException) {
            mutableState.update { it.copy(error = "Invalid parameters: ${e.message}") }
            return
        }
        mutableState.update { it.copy(seedText = seed.toString(), activity = DiffusionActivity.LOADING_MODEL, progress = null, error = null, status = null) }
        job = scope.launch {
            try {
                val config = if (current.backend == ComputeBackend.CPU_ONLY) BackendConfig.CPU else BackendConfig(backend = current.backend)
                val path = current.modelPath.trim()
                val key = "sd:$path|${config.backend}"
                loadedKey?.takeIf { it != key }?.let { residents.remove(it) }
                val lease = residents.lease(key) {
                    var opened: ImageGenerator? = null
                    val generator = try {
                        withContext(ioDispatcher) { factory.open(path, config).also { opened = it } }
                    } catch (e: CancellationException) {
                        opened?.close()
                        throw e
                    }
                    if (closed) {
                        generator.close()
                        throw CancellationException("Diffusion Studio closed")
                    }
                    generator to generator.estimatedBytes
                }
                loadedKey = key
                try {
                    val generator = lease.resource
                    mutableState.update {
                        it.copy(activity = DiffusionActivity.GENERATING, loadedBackend = generator.activeBackend, usesGpu = generator.usesGpu)
                    }
                    generator.generate(params).collect { event ->
                        when (event) {
                            is DiffusionEvent.Progress -> mutableState.update {
                                it.copy(progress = GenerationProgress(event.phase, event.step, event.totalSteps, event.secondsPerStep))
                            }

                            is DiffusionEvent.Completed -> mutableState.update {
                                it.copy(
                                    result = GeneratedImage(event.image, seed, params.prompt, params.sampleSteps, params.cfgScale),
                                    exportPath = joinExportPath(exportDirectory, "kggml-diffusion-$seed.png"),
                                    status = "Generated ${event.image.width}x${event.image.height} image (seed $seed)",
                                )
                            }

                            is DiffusionEvent.Error -> throw event.exception
                        }
                    }
                } finally {
                    lease.close()
                    residents.refresh()
                }
            } catch (e: CancellationException) {
                val at = mutableState.value.progress?.let { " at ${it.label.lowercase()}" }.orEmpty()
                mutableState.update { it.copy(status = "Cancelled$at") }
                throw e
            } catch (t: Throwable) {
                mutableState.update { it.copy(error = t.describe()) }
            } finally {
                mutableState.update { it.copy(activity = DiffusionActivity.IDLE) }
            }
        }
    }

    /** Cancels the running generation (stable-diffusion.cpp stops after the current step). */
    fun cancel() {
        job?.cancel()
    }

    /** Encodes the current image as PNG and writes it to [DiffusionStudioState.exportPath]. */
    fun export() {
        val current = mutableState.value
        val result = current.result ?: return
        if (!current.canExport) return
        val path = current.exportPath.trim()
        mutableState.update { it.copy(activity = DiffusionActivity.EXPORTING, error = null) }
        scope.launch {
            try {
                val size = withContext(ioDispatcher) {
                    val png = result.image.toPng()
                    writeFile(path, png)
                    png.size
                }
                mutableState.update { it.copy(status = "Exported PNG (${formatBytes(size.toLong())}) to $path") }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                mutableState.update { it.copy(error = "Export failed: ${t.describe()}") }
            } finally {
                mutableState.update { it.copy(activity = DiffusionActivity.IDLE) }
            }
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        job?.cancel()
        loadedKey?.let { residents.remove(it) }
    }

    private fun nextSeed(): Long = random.nextLong(0, MAX_SEED + 1)

    companion object {
        const val MAX_STEPS: Int = 50
        const val MAX_CFG: Float = 20f
        const val MAX_SEED: Long = 4_294_967_295L
        val SIZES: List<Int> = listOf(256, 384, 512)
    }
}

/** PNG bytes of a generated image (RGB, RGBA or gray). */
fun DiffusionImage.toPng(): ByteArray = PngEncoder.encode(width, height, channels, pixels)

private fun joinExportPath(directory: String, name: String): String =
    if (directory.isEmpty()) name else if (directory.endsWith("/") || directory.endsWith("\\")) directory + name else "$directory/$name"
