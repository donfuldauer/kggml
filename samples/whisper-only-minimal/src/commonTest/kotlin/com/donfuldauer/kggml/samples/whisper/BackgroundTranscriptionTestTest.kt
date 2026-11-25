package com.donfuldauer.kggml.samples.whisper

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TestTimeSource
import kotlinx.coroutines.test.runTest

class BackgroundTranscriptionTestTest {
    private val audio = PcmAudio.S16(ShortArray(16_000), 16_000, 1)

    @Test
    fun cpuOnlyRunsKeepCompletingInTheBackground() = runTest {
        val time = TestTimeSource()
        val session = WhisperSampleSession(factory = { p, c -> FakeSpeechEngine(p, c, timeSource = time) }, timeSource = time)
        var calls = 0
        val seen = mutableListOf<BackgroundRun>()
        val summary = runBackgroundTranscriptionTest(
            session, "a.bin", audio, ComputeBackend.CPU_ONLY,
            isInBackground = { ++calls > 2 },
            duration = 1.seconds,
            timeSource = time,
        ) { seen += it }

        // 120 ms per run: runs until the 1 s deadline has passed.
        assertEquals(9, summary.runs.size)
        assertEquals(summary.runs, seen)
        assertEquals(7, summary.backgroundRuns.size)
        assertTrue(summary.passed)
        assertEquals(
            "PASS: 9 run(s) on CPU_ONLY, 7 in background, 7 background run(s) completed",
            summary.describe(),
        )
        assertTrue(seen.last().describe().startsWith("#9 [background] CPU_ONLY use_gpu=false 120 ms: And so my fellow"))
        assertTrue(seen.first().describe().startsWith("#1 [foreground]"))
    }

    @Test
    fun failsWhenNeverBackgrounded() = runTest {
        val session = WhisperSampleSession(factory = { p, c -> FakeSpeechEngine(p, c) })
        val summary = runBackgroundTranscriptionTest(
            session, "a.bin", audio, ComputeBackend.CPU_ONLY, isInBackground = { false }, maxRuns = 3,
        )
        assertEquals(3, summary.runs.size)
        assertFalse(summary.passed)
        assertTrue(summary.describe().endsWith("(send the app to the background while the test runs)"))
    }

    @Test
    fun failsWhenABackgroundRunUsesTheGpu() = runTest {
        val session = WhisperSampleSession(factory = { p, c -> FakeSpeechEngine(p, c) })
        val summary = runBackgroundTranscriptionTest(
            session, "a.bin", audio, ComputeBackend.GPU_METAL, isInBackground = { true }, maxRuns = 2,
        )
        assertFalse(summary.passed)
        assertTrue(summary.describe().endsWith("(a background run used the GPU)"))
    }

    @Test
    fun recordsFailedRunsWithoutStopping() = runTest {
        var engine: FakeSpeechEngine? = null
        val session = WhisperSampleSession(factory = { p, c -> FakeSpeechEngine(p, c).also { engine = it } })
        var runs = 0
        val summary = runBackgroundTranscriptionTest(
            session, "a.bin", audio, ComputeBackend.CPU_ONLY,
            isInBackground = { true },
            maxRuns = 3,
        ) {
            if (++runs == 1) engine!!.failNext = InferenceCancelledException("accessRevoked")
        }
        assertEquals(listOf(true, false, true), summary.runs.map { it.succeeded })
        assertEquals("#2 [background] FAILED: InferenceCancelledException: accessRevoked", summary.runs[1].describe())
        assertFalse(summary.passed)
        assertTrue(summary.describe().endsWith("(a background run failed)"))
    }

    @Test
    fun alwaysRunsAtLeastOnce() = runTest {
        val session = WhisperSampleSession(factory = { p, c -> FakeSpeechEngine(p, c) })
        val summary = runBackgroundTranscriptionTest(
            session, "a.bin", audio, ComputeBackend.CPU_ONLY, isInBackground = { true }, duration = 0.milliseconds,
        )
        assertEquals(1, summary.runs.size)
        assertTrue(summary.passed)
    }
}
