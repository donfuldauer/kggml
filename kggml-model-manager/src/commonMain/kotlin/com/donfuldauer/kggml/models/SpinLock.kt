package com.donfuldauer.kggml.models

import kotlin.concurrent.atomics.AtomicBoolean

/** Tiny lock for short, non-suspending critical sections (common code has no `synchronized`). */
internal class SpinLock {
    val locked: AtomicBoolean = AtomicBoolean(false)

    inline fun <T> withLock(block: () -> T): T {
        while (!locked.compareAndSet(false, true)) {
            // Busy-wait: every critical section in this module is a few map operations.
        }
        try {
            return block()
        } finally {
            locked.store(false)
        }
    }
}
