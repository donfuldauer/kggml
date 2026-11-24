package com.donfuldauer.kggml.core

import kotlin.test.Test
import kotlin.test.assertEquals

class AndroidMemoryPressureObserverTest {

    @Test
    fun mapsTrimLevels() {
        // Raw values of ComponentCallbacks2.TRIM_MEMORY_* (stable since API 14/16).
        val expected = mapOf(
            0 to MemoryPressureLevel.NORMAL,
            5 to MemoryPressureLevel.NORMAL, // RUNNING_MODERATE
            10 to MemoryPressureLevel.MODERATE, // RUNNING_LOW
            15 to MemoryPressureLevel.CRITICAL, // RUNNING_CRITICAL
            20 to MemoryPressureLevel.NORMAL, // UI_HIDDEN
            40 to MemoryPressureLevel.MODERATE, // BACKGROUND
            60 to MemoryPressureLevel.MODERATE, // MODERATE
            80 to MemoryPressureLevel.CRITICAL, // COMPLETE
            100 to MemoryPressureLevel.CRITICAL,
        )
        for ((level, pressure) in expected) {
            assertEquals(pressure, memoryPressureLevelForTrim(level), "trim level $level")
        }
    }
}
