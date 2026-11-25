package com.donfuldauer.kggml.whisper

import kotlinx.coroutines.CoroutineScope
import platform.Foundation.NSNotificationCenter
import platform.Foundation.NSOperationQueue
import platform.UIKit.UIApplicationDidEnterBackgroundNotification
import platform.UIKit.UIApplicationWillEnterForegroundNotification
import platform.darwin.NSObjectProtocol

/**
 * Observes `UIApplication` lifecycle notifications and keeps this engine on `ComputeBackend.CPU_ONLY` while the
 * app is in the background, preventing Metal `MTLCommandBufferErrorDomain` / `accessRevoked` aborts:
 *
 * - `UIApplicationDidEnterBackgroundNotification`: aborts a running GPU transcription (when
 *   [abortInFlightOnBackground]) and reloads the model with `use_gpu = false` (Accelerate / NEON only).
 * - `UIApplicationWillEnterForegroundNotification`: restores the previous backend configuration.
 *
 * Reloads run on [scope] through [WhisperEngine.reinitialize]; see [WhisperBackgroundFallback] for details.
 * Close the returned handle (before or together with the engine) to stop observing.
 *
 * ```kotlin
 * val engine = WhisperEngine(modelPath, BackendConfig(backend = ComputeBackend.GPU_METAL))
 * val lifecycle = engine.observeAppLifecycle(appScope)
 * // ...
 * lifecycle.close(); engine.close()
 * ```
 */
public fun WhisperEngine.observeAppLifecycle(
    scope: CoroutineScope,
    abortInFlightOnBackground: Boolean = true,
    onError: (Throwable) -> Unit = {},
): AutoCloseable {
    val fallback = WhisperBackgroundFallback(this, scope, abortInFlightOnBackground, onError)
    val center = NSNotificationCenter.defaultCenter
    // queue = null: the block runs synchronously on the posting (main) thread, so the in-flight GPU work is
    // aborted before the app finishes transitioning to the background.
    val observers: List<NSObjectProtocol> = listOf(
        center.addObserverForName(UIApplicationDidEnterBackgroundNotification, null, null as NSOperationQueue?) {
            fallback.onEnterBackground()
        },
        center.addObserverForName(UIApplicationWillEnterForegroundNotification, null, null as NSOperationQueue?) {
            fallback.onEnterForeground()
        },
    )
    return AutoCloseable {
        observers.forEach { center.removeObserver(it) }
        fallback.close()
    }
}
