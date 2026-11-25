package com.donfuldauer.kggml.samples.showcase

/**
 * Top-level screens of the showcase navigation shell.
 *
 * [BACKEND_INSPECTOR] (Phase 1) and [WHISPER_STUDIO] (Phase 3) are functional; the remaining screens are
 * placeholders filled in by the phase that delivers the corresponding engine.
 */
enum class ShowcaseDestination(
    val title: String,
    val shortLabel: String,
    val summary: String,
    /** Phase that implements this screen, or `null` if it is already functional. */
    val plannedPhase: Int?,
) {
    BACKEND_INSPECTOR(
        title = "Hardware Backend Inspector",
        shortLabel = "Backends",
        summary = "Compute backends, thread counts and version metadata reported by the shared Kggml runtime.",
        plannedPhase = null,
    ),
    MODEL_MANAGER(
        title = "Model Manager",
        shortLabel = "Models",
        summary = "Resumable model downloads, SHA-256 verification, RAM estimates and cache eviction.",
        plannedPhase = 4,
    ),
    WHISPER_STUDIO(
        title = "Live Whisper Dictation & Audio Studio",
        shortLabel = "Whisper",
        summary = "Live microphone dictation with partial and committed segments, plus PCM file transcription, " +
            "with whisper.cpp.",
        plannedPhase = null,
    ),
    LLM_CHAT(
        title = "LLM Chat / VLM",
        shortLabel = "Chat",
        summary = "Streaming chat, structured output and image understanding with llama.cpp + libmtmd.",
        plannedPhase = 4,
    ),
    DIFFUSION_STUDIO(
        title = "Diffusion Studio",
        shortLabel = "Diffusion",
        summary = "On-device image generation with stable-diffusion.cpp.",
        plannedPhase = 4,
    ),
    ;

    val isPlaceholder: Boolean get() = plannedPhase != null

    /** Stable test tag of this destination's navigation item. */
    val navTag: String get() = "nav-${name.lowercase()}"

    /** Stable test tag of this destination's screen root. */
    val screenTag: String get() = "screen-${name.lowercase()}"
}
