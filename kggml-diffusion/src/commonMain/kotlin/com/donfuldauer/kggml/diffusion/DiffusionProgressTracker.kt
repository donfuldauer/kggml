package com.donfuldauer.kggml.diffusion

/**
 * Turns raw `kggml_diffusion_progress_callback` points into [DiffusionEvent.Progress] events.
 *
 * stable-diffusion.cpp reports `0` when the sampler starts, then `1..total` after each denoising step. With a tiled
 * VAE decode a second sequence (`0..tiles`) follows. The tracker:
 *
 * - starts in [DiffusionPhase.DENOISING] and switches to [DiffusionPhase.VAE_TILES] once the denoising sequence has
 *   completed (`step == total`) and a new sequence starts (the step goes backwards). It never switches back;
 * - clamps out-of-range values (`total < 0`, `step` outside `0..total`) instead of failing inside a native callback;
 * - drops exact repeats of the previous point.
 *
 * Not thread-safe: used by one generation on the engine thread.
 */
internal class DiffusionProgressTracker {
    var phase: DiffusionPhase = DiffusionPhase.DENOISING
        private set

    private var lastStep = -1
    private var lastTotal = -1
    private var denoisingCompleted = false

    /** Maps one raw progress point; returns `null` for a repeat of the previous one. */
    fun onProgress(step: Int, totalSteps: Int, secondsPerStep: Float): DiffusionEvent.Progress? {
        val total = totalSteps.coerceAtLeast(0)
        val current = step.coerceIn(0, total)
        if (phase == DiffusionPhase.DENOISING && denoisingCompleted && (current < lastStep || total != lastTotal)) {
            phase = DiffusionPhase.VAE_TILES
        } else if (current == lastStep && total == lastTotal) {
            return null
        }
        if (phase == DiffusionPhase.DENOISING && total > 0 && current == total) denoisingCompleted = true
        lastStep = current
        lastTotal = total
        val seconds = if (secondsPerStep.isFinite() && secondsPerStep >= 0f) secondsPerStep else 0f
        return DiffusionEvent.Progress(phase, current, total, seconds)
    }
}
