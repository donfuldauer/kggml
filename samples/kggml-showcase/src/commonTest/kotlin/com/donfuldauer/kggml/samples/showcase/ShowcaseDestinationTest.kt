package com.donfuldauer.kggml.samples.showcase

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ShowcaseDestinationTest {
    @Test
    fun shellContainsInspectorAndAllPlannedEngineScreens() {
        assertEquals(
            listOf("Backends", "Models", "Whisper", "Chat", "Diffusion"),
            ShowcaseDestination.entries.map { it.shortLabel },
        )
    }

    @Test
    fun onlyBackendInspectorIsFunctionalInPhase1() {
        assertFalse(ShowcaseDestination.BACKEND_INSPECTOR.isPlaceholder)
        ShowcaseDestination.entries.filter { it != ShowcaseDestination.BACKEND_INSPECTOR }.forEach {
            assertTrue(it.isPlaceholder, "$it should be a placeholder")
        }
    }

    @Test
    fun testTagsAreUnique() {
        val tags = ShowcaseDestination.entries.flatMap { listOf(it.navTag, it.screenTag) }
        assertEquals(tags.size, tags.toSet().size)
    }
}
