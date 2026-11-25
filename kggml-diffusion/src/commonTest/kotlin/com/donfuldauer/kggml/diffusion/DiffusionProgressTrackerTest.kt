package com.donfuldauer.kggml.diffusion

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class DiffusionProgressTrackerTest {
    private fun DiffusionProgressTracker.feed(vararg points: Pair<Int, Int>): List<Pair<DiffusionPhase, Int>> =
        points.mapNotNull { (step, total) -> onProgress(step, total, 0.5f)?.let { it.phase to it.step } }

    @Test
    fun denoisingStepsAreReportedInOrder() {
        val tracker = DiffusionProgressTracker()
        val events = (0..3).map { tracker.onProgress(it, 3, 1.5f) }
        assertEquals(listOf(0, 1, 2, 3), events.map { it!!.step })
        assertEquals(setOf(DiffusionPhase.DENOISING), events.map { it!!.phase }.toSet())
        assertEquals(1.5f, events.last()!!.secondsPerStep)
        assertEquals(DiffusionPhase.DENOISING, tracker.phase)
    }

    @Test
    fun aNewSequenceAfterTheCompletedDenoisingIsTheTiledVae() {
        val tracker = DiffusionProgressTracker()
        val events = tracker.feed(0 to 2, 1 to 2, 2 to 2, 0 to 6, 3 to 6, 6 to 6, 0 to 6)
        assertEquals(
            listOf(
                DiffusionPhase.DENOISING to 0, DiffusionPhase.DENOISING to 1, DiffusionPhase.DENOISING to 2,
                DiffusionPhase.VAE_TILES to 0, DiffusionPhase.VAE_TILES to 3, DiffusionPhase.VAE_TILES to 6,
                DiffusionPhase.VAE_TILES to 0,
            ),
            events,
        )
        assertEquals(DiffusionPhase.VAE_TILES, tracker.phase, "never switches back")
    }

    @Test
    fun aTileSequenceWithTheSameTotalStillSwitchesPhase() {
        val tracker = DiffusionProgressTracker()
        assertEquals(DiffusionPhase.VAE_TILES, tracker.feed(0 to 2, 1 to 2, 2 to 2, 0 to 2).last().first)
    }

    @Test
    fun anIncompleteSequenceThatRestartsStaysDenoising() {
        // e.g. a sampler reporting a sub-step twice: only a *completed* denoising sequence starts the VAE phase.
        val tracker = DiffusionProgressTracker()
        assertEquals(setOf(DiffusionPhase.DENOISING), tracker.feed(0 to 4, 1 to 4, 0 to 4, 2 to 4).map { it.first }.toSet())
    }

    @Test
    fun exactRepeatsAreDropped() {
        val tracker = DiffusionProgressTracker()
        assertEquals(listOf(0, 1, 2), tracker.feed(0 to 2, 0 to 2, 1 to 2, 1 to 2, 2 to 2, 2 to 2).map { it.second })
    }

    @Test
    fun outOfRangeValuesAreClampedInsteadOfFailingInsideTheNativeCallback() {
        val tracker = DiffusionProgressTracker()
        val over = tracker.onProgress(9, 4, Float.NaN)!!
        assertEquals(4, over.step)
        assertEquals(4, over.totalSteps)
        assertEquals(0f, over.secondsPerStep)
        val negative = DiffusionProgressTracker().onProgress(-3, -1, -2f)!!
        assertEquals(0, negative.step)
        assertEquals(0, negative.totalSteps)
        assertEquals(0f, negative.secondsPerStep)
        assertNull(DiffusionProgressTracker().apply { onProgress(1, 2, 0f) }.onProgress(1, 2, 0f))
    }
}
