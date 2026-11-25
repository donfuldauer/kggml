package com.donfuldauer.kggml.whisper

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class AbortGateTest {
    @Test
    fun firesOnlyWhileArmed() {
        val gate = AbortGate()
        val fired = mutableListOf<Long>()
        assertFalse(gate.fire { fired += it })
        gate.arm(42L)
        assertTrue(gate.isArmed)
        assertTrue(gate.fire { fired += it })
        gate.disarm()
        assertFalse(gate.isArmed)
        assertFalse(gate.fire { fired += it })
        assertEquals(listOf(42L), fired)
    }

    @Test
    fun rearmingTargetsTheNewHandle() {
        val gate = AbortGate()
        val fired = mutableListOf<Long>()
        gate.arm(1L)
        gate.disarm()
        gate.arm(2L)
        gate.fire { fired += it }
        assertEquals(listOf(2L), fired)
    }

    @Test
    fun nullHandleCannotBeArmed() {
        assertFailsWith<IllegalArgumentException> { AbortGate().arm(0L) }
    }

    @Test
    fun abortActionFailureLeavesTheGateConsistent() {
        val gate = AbortGate()
        gate.arm(7L)
        assertFailsWith<IllegalStateException> { gate.fire { error("boom") } }
        gate.disarm() // must not spin forever: the failed fire() has left
        assertFalse(gate.isArmed)
    }
}
