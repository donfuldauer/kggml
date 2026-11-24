package com.donfuldauer.kggml.samples.whisper

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.runtime.KggmlRuntime
import kotlin.coroutines.cancellation.CancellationException

/** Values shown by the minimal sample, read from the shared Kggml runtime. */
data class RuntimeSummary(
    val runtimeVersion: String,
    val ggmlVersion: String,
    val availableBackends: Set<ComputeBackend>,
    val recommendedThreadCount: Int,
)

/** Reads a [RuntimeSummary] from [KggmlRuntime] (blocking native calls; call off the main thread). */
fun readRuntimeSummary(): RuntimeSummary = KggmlRuntime.acquire().use {
    val info = KggmlRuntime.info()
    RuntimeSummary(
        runtimeVersion = info.runtimeVersion,
        ggmlVersion = info.ggmlVersion,
        availableBackends = KggmlRuntime.availableBackends(),
        recommendedThreadCount = KggmlRuntime.recommendedThreadCount(),
    )
}

/** Renders the outcome of [read] as the sample's status text. Never throws except for cancellation. */
fun describeRuntime(read: () -> RuntimeSummary = ::readRuntimeSummary): String =
    try {
        val summary = read()
        buildString {
            appendLine("Kggml whisper-only minimal sample")
            appendLine("Runtime ${summary.runtimeVersion} (ggml ${summary.ggmlVersion})")
            appendLine("Backends: ${summary.availableBackends.joinToString { it.name }}")
            append("Recommended threads: ${summary.recommendedThreadCount}")
        }
    } catch (e: CancellationException) {
        throw e
    } catch (t: Throwable) {
        "Kggml whisper-only minimal sample\nNative runtime unavailable: ${t.message ?: t::class.simpleName}"
    }
