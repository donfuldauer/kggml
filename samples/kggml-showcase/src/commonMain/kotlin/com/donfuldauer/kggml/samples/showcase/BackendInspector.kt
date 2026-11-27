package com.donfuldauer.kggml.samples.showcase

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.runtime.BackendInfo
import com.donfuldauer.kggml.runtime.KggmlRuntime
import com.donfuldauer.kggml.runtime.RuntimeInfo
import kotlin.coroutines.cancellation.CancellationException

/** Everything the inspector shows, read from the runtime in one go. */
data class RuntimeSnapshot(
    val backends: List<BackendInfo>,
    val availableBackends: Set<ComputeBackend>,
    val recommendedThreadCount: Int,
    val hardwareThreadCount: Int,
    val info: RuntimeInfo,
)

/** Source of [RuntimeSnapshot]s; [KggmlRuntimeInspector] in the app, fakes in tests. */
fun interface RuntimeInspector {
    /** Reads a snapshot. May block on native calls and throws on failure (e.g. the runtime cannot load). */
    fun inspect(): RuntimeSnapshot
}

/** Production [RuntimeInspector] querying the shared native runtime through [KggmlRuntime]. */
object KggmlRuntimeInspector : RuntimeInspector {
    override fun inspect(): RuntimeSnapshot {
        // Hold one reference across all queries so the runtime is initialized once for the whole snapshot.
        KggmlRuntime.acquire().use {
            return RuntimeSnapshot(
                backends = KggmlRuntime.backends(),
                availableBackends = KggmlRuntime.availableBackends(),
                recommendedThreadCount = KggmlRuntime.recommendedThreadCount(),
                hardwareThreadCount = KggmlRuntime.hardwareThreadCount(),
                info = KggmlRuntime.info(),
            )
        }
    }
}

/** One row of the backend table. */
data class BackendRow(
    val backend: ComputeBackend,
    val label: String,
    val status: BackendStatus,
    val deviceCount: Int,
) {
    val testTag: String get() = "backend-row-${backend.name.lowercase()}"
}

enum class BackendStatus(val label: String) {
    AVAILABLE("Available"),
    NO_DEVICE("Compiled, no device"),
    NOT_COMPILED("Not compiled in this build"),
}

/** View state of the Hardware Backend Inspector. */
sealed interface BackendInspectorState {
    data object Loading : BackendInspectorState

    data class Ready(
        val rows: List<BackendRow>,
        val recommendedThreadCount: Int,
        val hardwareThreadCount: Int,
        val info: RuntimeInfo,
    ) : BackendInspectorState

    data class Failed(val message: String) : BackendInspectorState
}

/** Backends always listed by the inspector, in display order, even when the runtime does not report them. */
val INSPECTED_BACKENDS: List<ComputeBackend> = listOf(
    ComputeBackend.CPU_ONLY,
    ComputeBackend.GPU_METAL,
    ComputeBackend.GPU_VULKAN,
    ComputeBackend.GPU_OPENCL,
    ComputeBackend.NPU_COREML,
)

fun ComputeBackend.displayName(): String = when (this) {
    ComputeBackend.AUTO -> "Auto"
    ComputeBackend.CPU_ONLY -> "CPU"
    ComputeBackend.GPU_METAL -> "Metal"
    ComputeBackend.GPU_VULKAN -> "Vulkan"
    ComputeBackend.GPU_OPENCL -> "OpenCL"
    ComputeBackend.NPU_COREML -> "Core ML"
}

/** Pure mapping from a runtime snapshot to the inspector view state. */
fun RuntimeSnapshot.toInspectorState(): BackendInspectorState.Ready {
    val byBackend = backends.associateBy { it.backend }
    val rows = INSPECTED_BACKENDS.map { backend ->
        val info = byBackend[backend]
        val available = backend in availableBackends || info?.available == true
        val status = when {
            available -> BackendStatus.AVAILABLE
            info?.compiled == true -> BackendStatus.NO_DEVICE
            else -> BackendStatus.NOT_COMPILED
        }
        BackendRow(
            backend = backend,
            label = backend.displayName(),
            status = status,
            deviceCount = info?.deviceCount ?: 0,
        )
    }
    return BackendInspectorState.Ready(
        rows = rows,
        recommendedThreadCount = recommendedThreadCount,
        hardwareThreadCount = hardwareThreadCount,
        info = info,
    )
}

/**
 * Reads the runtime through [inspector] and maps the result (or failure) to a terminal view state.
 * Never throws except for coroutine cancellation.
 */
fun loadInspectorState(inspector: RuntimeInspector): BackendInspectorState =
    try {
        inspector.inspect().toInspectorState()
    } catch (e: CancellationException) {
        throw e
    } catch (t: Throwable) {
        BackendInspectorState.Failed(t.message ?: t::class.simpleName ?: "Unknown error")
    }
