package com.donfuldauer.kggml.diffusion

import com.donfuldauer.kggml.core.InferenceException

/**
 * The model files of a [DiffusionEngine] (UTF-8 paths to `.gguf`, `.safetensors` or `.ckpt` files).
 *
 * Exactly one of [modelPath] (a full checkpoint, e.g. Stable Diffusion 1.x / 2.x / XL) and [diffusionModelPath] (a
 * standalone diffusion model such as Flux, SD 3.5 or Z-Image, which needs its component files below) is required.
 *
 * @property modelPath full checkpoint containing the diffusion model, VAE and text encoders.
 * @property diffusionModelPath standalone diffusion model; pair it with the component files below.
 * @property vaePath optional VAE override.
 * @property taesdPath optional tiny autoencoder (faster, lower-quality decode).
 * @property clipLPath optional CLIP-L text encoder (SD 3 / Flux).
 * @property clipGPath optional CLIP-G text encoder (SD 3).
 * @property t5xxlPath optional T5-XXL text encoder (SD 3 / Flux).
 * @property llmPath optional LLM text encoder (Qwen-Image, Z-Image, ...).
 */
public data class DiffusionModelFiles(
    val modelPath: String? = null,
    val diffusionModelPath: String? = null,
    val vaePath: String? = null,
    val taesdPath: String? = null,
    val clipLPath: String? = null,
    val clipGPath: String? = null,
    val t5xxlPath: String? = null,
    val llmPath: String? = null,
) {
    init {
        require(!modelPath.isNullOrEmpty() || !diffusionModelPath.isNullOrEmpty()) {
            "modelPath or diffusionModelPath is required"
        }
        require(modelPath.isNullOrEmpty() || diffusionModelPath.isNullOrEmpty()) {
            "set either modelPath (full checkpoint) or diffusionModelPath (standalone diffusion model), not both"
        }
    }

    /** The paths in `kggml_diffusion_context_params` order (`null` = unused). */
    internal fun toRaw(): List<String?> =
        listOf(modelPath, diffusionModelPath, vaePath, taesdPath, clipLPath, clipGPath, t5xxlPath, llmPath)
            .map { it?.takeIf(String::isNotEmpty) }
}

/**
 * Diffusion-specific construction options of a [DiffusionEngine].
 *
 * @property gpuDevice index among the GPU devices of the selected kind (Metal / Vulkan).
 * @property allowBackendFallback when the requested backend is unavailable, fall back along
 *   [com.donfuldauer.kggml.core.ComputeBackend.fallback] (eventually CPU) instead of failing.
 * @property keepTextEncoderOnCpu run the text encoders on the CPU even with a GPU backend (saves GPU memory).
 * @property keepVaeOnCpu run the VAE on the CPU even with a GPU backend (saves GPU memory, slower decode).
 * @property weightType stable-diffusion.cpp `sd_type_t` (= `ggml_type`) to convert the weights to at load time;
 *   [WEIGHT_TYPE_AS_STORED] keeps them as stored.
 */
public data class DiffusionEngineOptions(
    val gpuDevice: Int = 0,
    val allowBackendFallback: Boolean = true,
    val keepTextEncoderOnCpu: Boolean = false,
    val keepVaeOnCpu: Boolean = false,
    val weightType: Int = WEIGHT_TYPE_AS_STORED,
) {
    init {
        require(gpuDevice >= 0) { "gpuDevice must be >= 0, was $gpuDevice" }
        require(weightType >= WEIGHT_TYPE_AS_STORED) { "weightType must be >= $WEIGHT_TYPE_AS_STORED, was $weightType" }
    }

    public companion object {
        /** [weightType] sentinel: keep the weight types stored in the model files. */
        public const val WEIGHT_TYPE_AS_STORED: Int = -1
    }
}

/** stable-diffusion.cpp sampling method (`sample_method_t`). */
public enum class DiffusionSampler(internal val nativeValue: Int) {
    /** The model's own default (stable-diffusion.cpp picks per model, e.g. Euler A for SD 1.x). */
    MODEL_DEFAULT(-1),
    EULER(0),
    EULER_A(1),
    HEUN(2),
    DPM2(3),
    DPMPP_2S_A(4),
    DPMPP_2M(5),
    DPMPP_2M_V2(6),
    IPNDM(7),
    IPNDM_V(8),
    LCM(9),
    DDIM_TRAILING(10),
    TCD(11),
    RES_MULTISTEP(12),
    RES_2S(13),
    ER_SDE(14),
    EULER_CFG_PP(15),
    EULER_A_CFG_PP(16),
    ;

    internal companion object {
        fun fromNative(value: Int): DiffusionSampler? = entries.firstOrNull { it.nativeValue == value }
    }
}

/** stable-diffusion.cpp noise schedule (`scheduler_t`). */
public enum class DiffusionScheduler(internal val nativeValue: Int) {
    /** The model's own default. */
    MODEL_DEFAULT(-1),
    DISCRETE(0),
    KARRAS(1),
    EXPONENTIAL(2),
    AYS(3),
    GITS(4),
    SGM_UNIFORM(5),
    SIMPLE(6),
    SMOOTHSTEP(7),
    KL_OPTIMAL(8),
    LCM(9),
    BONG_TANGENT(10),
}

/**
 * One txt2img request. Validated on construction (including via [copy]); invalid values throw
 * [IllegalArgumentException] before any native call.
 *
 * @property prompt what to draw; must not be blank.
 * @property negativePrompt what to steer away from (only used when [cfgScale] != 1).
 * @property width image width in pixels: a positive multiple of 8, at most [MAX_DIMENSION].
 * @property height image height in pixels: a positive multiple of 8, at most [MAX_DIMENSION].
 * @property sampleSteps denoising steps, `1..`[MAX_SAMPLE_STEPS]. Cancellation takes effect between steps.
 * @property cfgScale classifier-free guidance scale (finite, `>= 0`; `1` disables CFG and the negative prompt).
 * @property distilledGuidance guidance of distilled-guidance models such as Flux (finite, `>= 0`).
 * @property seed RNG seed; any negative value ([RANDOM_SEED]) picks a random one. The same seed, parameters,
 *   model and backend reproduce the same image.
 * @property sampler sampling method.
 * @property scheduler noise schedule.
 * @property clipSkip CLIP layers to skip, [CLIP_SKIP_MODEL_DEFAULT] (`-1`) = the model's default.
 * @property vaeTiling decode the latent in tiles (lower peak memory for large images; reported as
 *   [DiffusionPhase.VAE_TILES] progress).
 */
public data class DiffusionGenerationParams(
    val prompt: String,
    val negativePrompt: String = "",
    val width: Int = DEFAULT_SIZE,
    val height: Int = DEFAULT_SIZE,
    val sampleSteps: Int = DEFAULT_SAMPLE_STEPS,
    val cfgScale: Float = DEFAULT_CFG_SCALE,
    val distilledGuidance: Float = DEFAULT_DISTILLED_GUIDANCE,
    val seed: Long = RANDOM_SEED,
    val sampler: DiffusionSampler = DiffusionSampler.MODEL_DEFAULT,
    val scheduler: DiffusionScheduler = DiffusionScheduler.MODEL_DEFAULT,
    val clipSkip: Int = CLIP_SKIP_MODEL_DEFAULT,
    val vaeTiling: Boolean = false,
) {
    init {
        require(prompt.isNotBlank()) { "prompt must not be blank" }
        requireDimension("width", width)
        requireDimension("height", height)
        require(sampleSteps in 1..MAX_SAMPLE_STEPS) { "sampleSteps must be in 1..$MAX_SAMPLE_STEPS, was $sampleSteps" }
        require(cfgScale.isFinite() && cfgScale >= 0f) { "cfgScale must be finite and >= 0, was $cfgScale" }
        require(distilledGuidance.isFinite() && distilledGuidance >= 0f) {
            "distilledGuidance must be finite and >= 0, was $distilledGuidance"
        }
        require(clipSkip == CLIP_SKIP_MODEL_DEFAULT || clipSkip >= 1) {
            "clipSkip must be $CLIP_SKIP_MODEL_DEFAULT (model default) or >= 1, was $clipSkip"
        }
    }

    public companion object {
        /** Default [width] and [height] in pixels. */
        public const val DEFAULT_SIZE: Int = 512

        /** Default [sampleSteps]. */
        public const val DEFAULT_SAMPLE_STEPS: Int = 20

        /** Default [cfgScale]. */
        public const val DEFAULT_CFG_SCALE: Float = 7.0f

        /** Default [distilledGuidance] (used by guidance-distilled models such as Flux). */
        public const val DEFAULT_DISTILLED_GUIDANCE: Float = 3.5f

        /** [seed] sentinel: random seed. */
        public const val RANDOM_SEED: Long = -1L

        /** [clipSkip] sentinel: the model's default. */
        public const val CLIP_SKIP_MODEL_DEFAULT: Int = -1

        /** Largest accepted [width] / [height] (the native wrapper's limit). */
        public const val MAX_DIMENSION: Int = 16384

        /** Largest accepted [sampleSteps] (the native wrapper's limit). */
        public const val MAX_SAMPLE_STEPS: Int = 10000

        /** [width] and [height] must be multiples of this. */
        public const val DIMENSION_MULTIPLE: Int = 8

        private fun requireDimension(name: String, value: Int) {
            require(value in DIMENSION_MULTIPLE..MAX_DIMENSION && value % DIMENSION_MULTIPLE == 0) {
                "$name must be a positive multiple of $DIMENSION_MULTIPLE, at most $MAX_DIMENSION, was $value"
            }
        }
    }
}

/**
 * A generated image: packed 8-bit RGB ([channels] = 3), row-major, no row padding, `width * height * channels` bytes.
 * The pixels were copied out of the native buffer, which is already freed; this object owns plain heap memory.
 *
 * @property width image width in pixels.
 * @property height image height in pixels.
 * @property channels bytes per pixel (3 for the RGB images [DiffusionEngine] produces).
 * @property pixels `width * height * channels` bytes, row-major, top row first.
 */
public class DiffusionImage(
    public val width: Int,
    public val height: Int,
    public val channels: Int,
    public val pixels: ByteArray,
) {
    init {
        require(width > 0 && height > 0) { "width / height must be > 0, was ${width}x$height" }
        require(channels in 1..4) { "channels must be in 1..4, was $channels" }
        require(pixels.size.toLong() == width.toLong() * height * channels) {
            "pixels has ${pixels.size} bytes, expected ${width.toLong() * height * channels} (${width}x${height}x$channels)"
        }
    }

    /** Byte offset of pixel ([x], [y]) in [pixels]. */
    public fun offsetOf(x: Int, y: Int): Int {
        require(x in 0 until width && y in 0 until height) { "($x, $y) is outside ${width}x$height" }
        return (y * width + x) * channels
    }

    override fun equals(other: Any?): Boolean = other is DiffusionImage && width == other.width &&
        height == other.height && channels == other.channels && pixels.contentEquals(other.pixels)

    override fun hashCode(): Int = ((width * 31 + height) * 31 + channels) * 31 + pixels.contentHashCode()

    override fun toString(): String = "DiffusionImage(${width}x${height}x$channels)"
}

/** Which part of a generation a [DiffusionEvent.Progress] belongs to. */
public enum class DiffusionPhase {
    /** Denoising steps of the sampler (`totalSteps` = [DiffusionGenerationParams.sampleSteps] for most samplers). */
    DENOISING,

    /** Tiles of a tiled VAE decode ([DiffusionGenerationParams.vaeTiling]), reported after the denoising steps. */
    VAE_TILES,
}

/** Events of one [DiffusionEngine.generate] collection. */
public sealed interface DiffusionEvent {
    /**
     * Progress of the running generation. [step] is `0` when a phase starts, then `1..`[totalSteps] after each
     * completed step. Cancellation takes effect at these step boundaries.
     *
     * @property phase the part of the generation being reported.
     * @property step completed steps of [phase], `0..`[totalSteps].
     * @property totalSteps number of steps of [phase].
     * @property secondsPerStep stable-diffusion.cpp's timing estimate (`0` when unknown).
     */
    public data class Progress(
        val phase: DiffusionPhase,
        val step: Int,
        val totalSteps: Int,
        val secondsPerStep: Float,
    ) : DiffusionEvent {
        init {
            require(totalSteps >= 0) { "totalSteps must be >= 0, was $totalSteps" }
            require(step in 0..maxOf(totalSteps, 0)) { "step must be in 0..$totalSteps, was $step" }
        }
    }

    /** The finished image. This is the last event. */
    public data class Completed(val image: DiffusionImage) : DiffusionEvent

    /** A typed failure. It is terminal: no event follows it and the flow then completes normally. */
    public data class Error(val exception: InferenceException) : DiffusionEvent
}

/** Static properties of a loaded diffusion model. */
public data class DiffusionModelInfo(
    /** `true` if the model can generate images (txt2img). */
    val supportsImageGeneration: Boolean,
    /** stable-diffusion.cpp's default sampler for this model ([DiffusionSampler.MODEL_DEFAULT] if unknown). */
    val defaultSampler: DiffusionSampler,
    /** CPU threads stable-diffusion.cpp uses. */
    val threads: Int,
)

/**
 * Version information of the native diffusion library.
 *
 * @property moduleVersion version of the Kggml diffusion wrapper (`kggml_diffusion`).
 * @property stableDiffusionCppVersion version string reported by stable-diffusion.cpp.
 * @property stableDiffusionCppCommit upstream stable-diffusion.cpp commit the library is pinned to (see `NOTICE`).
 * @property systemInfo stable-diffusion.cpp / ggml CPU feature summary (`sd_get_system_info()`).
 */
public data class DiffusionNativeInfo(
    val moduleVersion: String,
    val stableDiffusionCppVersion: String,
    val stableDiffusionCppCommit: String,
    val systemInfo: String,
)
