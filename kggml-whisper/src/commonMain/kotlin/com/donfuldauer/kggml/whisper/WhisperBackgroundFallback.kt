package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import kotlin.concurrent.atomics.AtomicReference
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch

/**
 * Keeps a [WhisperEngine] off the GPU while the app is in the background.
 *
 * iOS revokes GPU access from background apps: a Metal command buffer submitted after the app has entered the
 * background fails with `MTLCommandBufferErrorDomain` / `accessRevoked` and aborts the process. This controller
 * reloads the engine with [ComputeBackend.CPU_ONLY] (`use_gpu = false`, Accelerate / NEON only) when the app
 * enters the background and restores the previous configuration when it returns to the foreground.
 *
 * On iOS, wire it to `UIApplication` notifications with `WhisperEngine.observeAppLifecycle(...)` (iosMain); on
 * other platforms call [onEnterBackground] / [onEnterForeground] from your own lifecycle callbacks.
 *
 * Transitions are applied in order on [scope] through [WhisperEngine.reinitialize] (i.e. on the engine's own
 * thread, after transcriptions already queued). Rapid background/foreground toggles are conflated to the latest
 * state. When [abortInFlightOnBackground] is `true` (default), a GPU transcription that is executing when the
 * app enters the background is aborted immediately (it fails with
 * [com.donfuldauer.kggml.core.InferenceCancelledException]); transcriptions queued behind it still run before
 * the switch, so cancel those too if they must not touch the GPU.
 *
 * @param onError receives failures of the reloads (the engine keeps its previous configuration on failure).
 */
public class WhisperBackgroundFallback(
    private val engine: WhisperEngine,
    scope: CoroutineScope,
    private val abortInFlightOnBackground: Boolean = true,
    private val onError: (Throwable) -> Unit = {},
) : AutoCloseable {
    private val transitions = Channel<Boolean>(Channel.CONFLATED)

    /** Foreground configuration to restore; written only by the worker coroutine. */
    private val foregroundConfig = AtomicReference<BackendConfig?>(null)

    private val worker = scope.launch {
        for (background in transitions) apply(background)
    }

    /** `true` while the engine runs on the CPU fallback installed by this controller. */
    public val isFallbackActive: Boolean
        get() = foregroundConfig.load() != null

    /** Call when the app enters the background (e.g. `UIApplicationDidEnterBackgroundNotification`). */
    public fun onEnterBackground() {
        if (abortInFlightOnBackground && engine.usesGpu) engine.abortActiveTranscription()
        transitions.trySend(true)
    }

    /** Call when the app returns to the foreground (e.g. `UIApplicationWillEnterForegroundNotification`). */
    public fun onEnterForeground() {
        transitions.trySend(false)
    }

    /** Stops observing transitions. Does not change the engine's current configuration. */
    override fun close() {
        transitions.close()
        worker.cancel()
    }

    private suspend fun apply(background: Boolean) {
        if (engine.isClosed) return
        try {
            if (background) {
                if (foregroundConfig.load() != null || (!engine.usesGpu && engine.activeBackend == ComputeBackend.CPU_ONLY)) {
                    return // already on the CPU
                }
                val current = engine.backendConfig
                engine.reinitialize(cpuFallbackOf(current))
                foregroundConfig.store(current)
            } else {
                val restore = foregroundConfig.exchange(null) ?: return
                engine.reinitialize(restore)
            }
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Throwable) {
            onError(failure)
        }
    }

    internal companion object {
        /** [config] with the backend forced to [ComputeBackend.CPU_ONLY]; threads / flash attention are kept. */
        fun cpuFallbackOf(config: BackendConfig): BackendConfig =
            config.copy(backend = ComputeBackend.CPU_ONLY, gpuLayers = 0)
    }
}
