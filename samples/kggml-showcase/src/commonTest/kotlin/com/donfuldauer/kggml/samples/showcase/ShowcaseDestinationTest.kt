package com.donfuldauer.kggml.samples.showcase

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull

class ShowcaseDestinationTest {
    @Test
    fun shellContainsInspectorAndAllEngineScreens() {
        assertEquals(
            listOf("Backends", "Models", "Whisper", "Chat", "Diffusion"),
            ShowcaseDestination.entries.map { it.shortLabel },
        )
    }

    @Test
    fun everyScreenIsFunctional() {
        ShowcaseDestination.entries.forEach {
            assertFalse(it.isPlaceholder, "$it should be functional")
            assertNull(it.plannedPhase, "$it should not be planned for a later phase")
        }
    }

    @Test
    fun testTagsAreUnique() {
        val tags = ShowcaseDestination.entries.flatMap { listOf(it.navTag, it.screenTag) }
        assertEquals(tags.size, tags.toSet().size)
    }
}
