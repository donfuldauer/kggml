package com.donfuldauer.kggml.samples.showcase

/**
 * Top-level screens of the showcase navigation shell.
 *
 * Every screen is functional: [BACKEND_INSPECTOR] (Phase 1), [WHISPER_STUDIO] (Phase 3), and [MODEL_MANAGER],
 * [LLM_CHAT] and [DIFFUSION_STUDIO] (Phase 4).
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
        summary = "Curated model catalog with resumable downloads, SHA-256 verification, header metadata with peak-RAM " +
            "estimates, and manual / OS low-memory eviction of loaded engines.",
        plannedPhase = null,
    ),
    WHISPER_STUDIO(
        title = "Live Whisper Dictation & Audio Studio",
        shortLabel = "Whisper",
        summary = "Live microphone dictation with partial and committed segments, plus PCM file transcription, " +
            "with whisper.cpp.",
        plannedPhase = null,
    ),
    LLM_CHAT(
        title = "LLM Chat, Structured Output & Vision",
        shortLabel = "Chat",
        summary = "Multi-turn streaming chat, KV-cache snapshots, JSON-schema constrained extraction, embeddings and " +
            "in-memory image understanding with llama.cpp + libmtmd.",
        plannedPhase = null,
    ),
    DIFFUSION_STUDIO(
        title = "Diffusion Image Studio",
        shortLabel = "Diffusion",
        summary = "On-device text-to-image generation with live step progress, cancellation and PNG export, with " +
            "stable-diffusion.cpp.",
        plannedPhase = null,
    ),
    ;

    val isPlaceholder: Boolean get() = plannedPhase != null

    /** Stable test tag of this destination's navigation item. */
    val navTag: String get() = "nav-${name.lowercase()}"

    /** Stable test tag of this destination's screen root. */
    val screenTag: String get() = "screen-${name.lowercase()}"
}
