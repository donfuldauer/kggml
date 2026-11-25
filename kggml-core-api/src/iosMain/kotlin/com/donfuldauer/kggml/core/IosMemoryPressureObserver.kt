package com.donfuldauer.kggml.core

import platform.Foundation.NSNotificationCenter
import platform.Foundation.NSOperationQueue
import platform.UIKit.UIApplicationDidReceiveMemoryWarningNotification
import platform.darwin.NSObjectProtocol
import kotlin.concurrent.atomics.AtomicBoolean

/**
 * [MemoryPressureObserver] driven by iOS `UIApplicationDidReceiveMemoryWarningNotification`. iOS has a single
 * warning level that precedes jetsam termination, so every warning is reported as [MemoryPressureLevel.CRITICAL].
 *
 * Registers with the default notification center at construction; [close] unregisters it (idempotent).
 *
 * @param queue the operation queue listeners are notified on (the main queue by default, where UIKit posts it).
 */
public class IosMemoryPressureObserver(
    queue: NSOperationQueue = NSOperationQueue.mainQueue,
) : ManualMemoryPressureObserver(), AutoCloseable {
    private val closed = AtomicBoolean(false)
    private val token: NSObjectProtocol = NSNotificationCenter.defaultCenter.addObserverForName(
        name = UIApplicationDidReceiveMemoryWarningNotification,
        `object` = null,
        queue = queue,
    ) { _ -> emit(MemoryPressureLevel.CRITICAL) }

    /** Unregisters from the notification center. Idempotent. */
    override fun close() {
        if (closed.exchange(true)) return
        NSNotificationCenter.defaultCenter.removeObserver(token)
    }
}
