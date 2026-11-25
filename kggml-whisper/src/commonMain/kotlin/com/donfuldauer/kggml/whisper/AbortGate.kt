package com.donfuldauer.kggml.whisper

import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.AtomicLong

/**
 * Hands the native context of the running transcription to cancellation handlers on other threads without
 * ever letting them touch a context that has been freed.
 *
 * The engine [arm]s the gate with the context handle right before the blocking native call and [disarm]s it right
 * after. A coroutine cancellation handler (running on the cancelling thread) calls [fire], which invokes the
 * abort action only while armed. [disarm] waits until every [fire] that may still observe the handle has left,
 * so once it returns the handle can be reused or freed safely. The abort action itself
 * (`kggml_whisper_request_abort`) is a lock-free atomic store, so that wait is a few instructions at most.
 */
internal class AbortGate {
    private val armedHandle = AtomicLong(NONE)
    private val firing = AtomicInt(0)

    val isArmed: Boolean
        get() = armedHandle.load() != NONE

    fun arm(handle: Long) {
        require(handle != NONE) { "cannot arm with a null handle" }
        armedHandle.store(handle)
    }

    /** Runs [abort] with the armed handle; returns `false` (doing nothing) when no transcription is running. */
    fun fire(abort: (Long) -> Unit): Boolean {
        firing.addAndFetch(1)
        try {
            val handle = armedHandle.load()
            if (handle == NONE) return false
            abort(handle)
            return true
        } finally {
            firing.addAndFetch(-1)
        }
    }

    fun disarm() {
        armedHandle.store(NONE)
        // A fire() that incremented before the store may still hold the old handle: wait for it to leave.
        while (firing.load() != 0) {
            // Busy-wait: the critical section is a single atomic store in native code.
        }
    }

    private companion object {
        const val NONE: Long = 0L
    }
}
