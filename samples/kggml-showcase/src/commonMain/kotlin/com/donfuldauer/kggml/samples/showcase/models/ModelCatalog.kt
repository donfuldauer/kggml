package com.donfuldauer.kggml.samples.showcase.models

/** What a catalog model is used for (decides which screen can use it). */
enum class ModelKind(val label: String) {
    WHISPER("Speech (whisper.cpp)"),
    LLM("LLM (llama.cpp)"),
    VISION_LLM("Vision LLM (llama.cpp + libmtmd)"),
    VISION_PROJECTOR("Vision projector (libmtmd mmproj)"),
    DIFFUSION("Image generation (stable-diffusion.cpp)"),
}

/**
 * Generation settings a diffusion model is made for, applied when it is chosen for the Diffusion Image Studio (e.g. a
 * distilled turbo model needs few steps and no classifier-free guidance).
 */
data class DiffusionPreset(val steps: Int, val cfgScale: Float, val size: Int) {
    init {
        require(steps >= 1 && cfgScale >= 1f && size > 0) { "invalid preset $this" }
    }

    val label: String get() = "$steps steps, CFG ${if (cfgScale % 1f == 0f) cfgScale.toInt().toString() else cfgScale.toString()}, ${size}x$size"
}

/**
 * One downloadable model of the curated catalog.
 *
 * @property approxSizeBytes approximate size, for display only (never passed as `expectedSizeBytes`, so an upstream
 *   re-upload does not break the download).
 * @property sha256 expected SHA-256 (64 hex digits) when known; `null` downloads unverified and the UI says so.
 * @property projectorId catalog id of the `mmproj` file a vision LLM needs.
 * @property diffusionPreset settings a diffusion model is made for, applied when it is chosen.
 */
data class CatalogModel(
    val id: String,
    val displayName: String,
    val kind: ModelKind,
    val url: String,
    val fileName: String,
    val approxSizeBytes: Long,
    val license: String,
    val sha256: String? = null,
    val projectorId: String? = null,
    val diffusionPreset: DiffusionPreset? = null,
) {
    init {
        require(sha256 == null || SHA256.matches(sha256)) { "sha256 must be 64 hex digits" }
    }

    private companion object {
        val SHA256 = Regex("^[0-9a-fA-F]{64}$")
    }
}

/**
 * The curated catalog: small public models that run on phones. SHA-256 digests are left `null` unless they were
 * checked against the upstream file, so those downloads are reported as "unverified" (size and integrity of the
 * transfer are still checked; the digest is computed and shown).
 */
val DEFAULT_MODEL_CATALOG: List<CatalogModel> = listOf(
    CatalogModel(
        id = "whisper-tiny-en",
        displayName = "Whisper tiny.en",
        kind = ModelKind.WHISPER,
        url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
        fileName = "ggml-tiny.en.bin",
        approxSizeBytes = 77_700_000L,
        license = "MIT",
    ),
    CatalogModel(
        id = "qwen2.5-0.5b-instruct-q4km",
        displayName = "Qwen2.5 0.5B Instruct Q4_K_M",
        kind = ModelKind.LLM,
        url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        fileName = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        approxSizeBytes = 491_000_000L,
        license = "Apache-2.0",
    ),
    CatalogModel(
        id = "smolvlm-256m-instruct-q8",
        displayName = "SmolVLM 256M Instruct Q8_0",
        kind = ModelKind.VISION_LLM,
        url = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf",
        fileName = "SmolVLM-256M-Instruct-Q8_0.gguf",
        approxSizeBytes = 175_000_000L,
        license = "Apache-2.0",
        projectorId = "smolvlm-256m-mmproj-q8",
    ),
    CatalogModel(
        id = "smolvlm-256m-mmproj-q8",
        displayName = "SmolVLM 256M projector Q8_0",
        kind = ModelKind.VISION_PROJECTOR,
        url = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
        fileName = "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
        approxSizeBytes = 104_000_000L,
        license = "Apache-2.0",
    ),
    // Distilled SD 2.1 (SD-Turbo): a usable 512x512 image in 1-4 steps without classifier-free guidance, i.e. 1-4
    // UNet passes instead of 40 for SD 1.5 at 20 steps with CFG. f16 weights with q8_0 matrices for stable-diffusion.cpp
    // (pinned revision; its upstream SHA-256 is verified on download).
    CatalogModel(
        id = "sd-turbo-q8",
        displayName = "SD-Turbo Q8_0 (fast: 4 steps, CFG 1)",
        kind = ModelKind.DIFFUSION,
        url = "https://huggingface.co/Green-Sky/SD-Turbo-GGUF/resolve/19a31586d02d64a73b4419bc193b3ecfaf38e1f0/sd_turbo-f16-q8_0.gguf",
        fileName = "sd_turbo-f16-q8_0.gguf",
        approxSizeBytes = 2_023_700_000L,
        license = "Stability AI Community License (Powered by Stability AI)",
        sha256 = "d50be7655f0a554cf8041c145d88b210bd5f3c545423119dee62ae08cae51580",
        diffusionPreset = DiffusionPreset(steps = 4, cfgScale = 1f, size = 512),
    ),
    CatalogModel(
        id = "sd-v1-5-pruned-emaonly",
        displayName = "Stable Diffusion v1.5 (pruned, EMA only; slow on phones)",
        kind = ModelKind.DIFFUSION,
        url = "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
        fileName = "v1-5-pruned-emaonly.safetensors",
        approxSizeBytes = 4_270_000_000L,
        license = "CreativeML OpenRAIL-M",
    ),
)
