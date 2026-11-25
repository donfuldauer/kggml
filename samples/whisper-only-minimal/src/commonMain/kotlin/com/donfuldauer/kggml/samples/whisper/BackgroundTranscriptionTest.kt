package com.donfuldauer.kggml.samples.whisper

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive

/** One transcription of a [runBackgroundTranscriptionTest]. */
data class BackgroundRun(
    val index: Int,
    /** Whether the app was in the background when the transcription finished. */
    val inBackground: Boolean,
    val report: TranscriptionReport?,
    val error: String?,
    /**
     * `true` when an accelerated (GPU / CoreML) run was aborted with `InferenceCancelledException`: the expected
     * outcome for the run in flight when the lifecycle fallback switches the engine to the CPU. Ignored by
     * [BackgroundTestSummary.passed]; a cancelled `CPU_ONLY` run still counts as a failure.
     */
    val abortedAtTransition: Boolean = false,
) {
    val succeeded: Boolean get() = report != null

    fun describe(): String = buildString {
        append("#$index ").append(if (inBackground) "[background]" else "[foreground]").append(' ')
        if (report != null) {
            append("${report.activeBackend} use_gpu=${report.usesGpu} ${report.elapsed.inWholeMilliseconds} ms: ")
            append(report.text.ifEmpty { "(no speech)" })
        } else {
            append(if (abortedAtTransition) "ABORTED (GPU run at background transition): " else "FAILED: ").append(error)
        }
    }
}

/** Outcome of a [runBackgroundTranscriptionTest]. */
data class BackgroundTestSummary(val backend: ComputeBackend, val runs: List<BackgroundRun>) {
    /** Background runs, excluding GPU runs aborted at the background transition. */
    val backgroundRuns: List<BackgroundRun> get() = runs.filter { it.inBackground && !it.abortedAtTransition }

    /**
     * `true` when at least one transcription completed while the app was in the background, every background
     * transcription succeeded, and none of them ran with `use_gpu = true` (which iOS would abort with Metal
     * `accessRevoked`).
     */
    val passed: Boolean
        get() = backgroundRuns.isNotEmpty() && backgroundRuns.all { it.succeeded && it.report?.usesGpu == false }

    fun describe(): String = buildString {
        val background = backgroundRuns
        append(if (passed) "PASS" else "FAIL")
        append(": ${runs.size} run(s) on $backend, ${background.size} in background")
        append(", ${background.count { it.succeeded }} background run(s) completed")
        when {
            background.isEmpty() -> append(" (send the app to the background while the test runs)")
            background.any { it.report?.usesGpu == true } -> append(" (a background run used the GPU)")
            background.any { !it.succeeded } -> append(" (a background run failed)")
        }
    }
}

/**
 * iOS background transcription test mode: transcribes [audio] repeatedly on [backend] until [duration] has
 * elapsed (or the coroutine is cancelled, e.g. by the background-task expiration handler), recording for each
 * run whether the app was in the background ([isInBackground]). The user starts the test and then sends the app
 * to the background; a run of [com.donfuldauer.kggml.core.ComputeBackend.CPU_ONLY] must keep completing there.
 *
 * Failures of individual runs are recorded, not thrown, so one error does not hide the remaining runs.
 */
suspend fun runBackgroundTranscriptionTest(
    session: WhisperSampleSession,
    modelPath: String,
    audio: PcmAudio,
    backend: ComputeBackend,
    isInBackground: suspend () -> Boolean,
    duration: Duration = 25.seconds,
    maxRuns: Int = 1_000,
    timeSource: TimeSource = TimeSource.Monotonic,
    onRun: suspend (BackgroundRun) -> Unit = {},
): BackgroundTestSummary {
    val deadline = timeSource.markNow() + duration
    val runs = mutableListOf<BackgroundRun>()
    while (currentCoroutineContext().isActive && runs.size < maxRuns && (runs.isEmpty() || deadline.hasNotPassedNow())) {
        val run = try {
            val report = session.transcribe(modelPath, audio, backend)
            BackgroundRun(runs.size + 1, isInBackground(), report, null)
        } catch (e: CancellationException) {
            throw e
        } catch (t: Throwable) {
            val aborted = t is InferenceCancelledException && backend != ComputeBackend.CPU_ONLY
            BackgroundRun(runs.size + 1, isInBackground(), null, "${t::class.simpleName}: ${t.message}", aborted)
        }
        runs += run
        onRun(run)
    }
    return BackgroundTestSummary(backend, runs)
}
