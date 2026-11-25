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
 * One downloadable model of the curated catalog.
 *
 * @property approxSizeBytes approximate size, for display only (never passed as `expectedSizeBytes`, so an upstream
 *   re-upload does not break the download).
 * @property sha256 expected SHA-256 (64 hex digits) when known; `null` downloads unverified and the UI says so.
 * @property projectorId catalog id of the `mmproj` file a vision LLM needs.
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
    CatalogModel(
        id = "sd-v1-5-pruned-emaonly",
        displayName = "Stable Diffusion v1.5 (pruned, EMA only)",
        kind = ModelKind.DIFFUSION,
        url = "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
        fileName = "v1-5-pruned-emaonly.safetensors",
        approxSizeBytes = 4_270_000_000L,
        license = "CreativeML OpenRAIL-M",
    ),
)
