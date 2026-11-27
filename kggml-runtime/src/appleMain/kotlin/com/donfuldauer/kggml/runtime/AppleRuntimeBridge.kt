package com.donfuldauer.kggml.runtime

import com.donfuldauer.kggml.runtime.cinterop.kggml_backend_info
import com.donfuldauer.kggml.runtime.cinterop.kggml_runtime_get_backends
import com.donfuldauer.kggml.runtime.cinterop.kggml_runtime_ggml_build_commit
import com.donfuldauer.kggml.runtime.cinterop.kggml_runtime_ggml_commit
import com.donfuldauer.kggml.runtime.cinterop.kggml_runtime_ggml_version
import com.donfuldauer.kggml.runtime.cinterop.kggml_runtime_hardware_thread_count
import com.donfuldauer.kggml.runtime.cinterop.kggml_runtime_has_coreml
import com.donfuldauer.kggml.runtime.cinterop.kggml_runtime_init
import com.donfuldauer.kggml.runtime.cinterop.kggml_runtime_is_initialized
import com.donfuldauer.kggml.runtime.cinterop.kggml_runtime_last_error
import com.donfuldauer.kggml.runtime.cinterop.kggml_runtime_performance_thread_count
import com.donfuldauer.kggml.runtime.cinterop.kggml_runtime_ref_count
import com.donfuldauer.kggml.runtime.cinterop.kggml_runtime_shutdown
import com.donfuldauer.kggml.runtime.cinterop.kggml_runtime_version
import com.donfuldauer.kggml.runtime.cinterop.kggml_status_name
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.convert
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.toKString
import kotlinx.cinterop.value

/**
 * [NativeRuntimeBridge] calling the `kggml_runtime.h` C ABI directly through Kotlin/Native cinterop.
 *
 * Every C entry point already catches C++ exceptions and returns a status code, so nothing can unwind into
 * Kotlin/Native frames. The library is linked dynamically (`GgmlRuntime.framework`, the iOS or macOS slice of
 * GgmlRuntime.xcframework), so there is nothing to load at runtime. Shared by iosArm64, iosSimulatorArm64 and
 * macosArm64.
 */
@OptIn(ExperimentalForeignApi::class)
internal object AppleRuntimeBridge : NativeRuntimeBridge {
    override fun init(): Int = kggml_runtime_init().toInt()

    override fun shutdown(): Int = kggml_runtime_shutdown().toInt()

    override fun isInitialized(): Boolean = kggml_runtime_is_initialized() != 0

    override fun refCount(): Int = kggml_runtime_ref_count()

    override fun getBackends(capacity: Int): RawBackendsResult = memScoped {
        val safeCapacity = capacity.coerceAtLeast(0)
        val count = alloc<IntVar>()
        val infos = if (safeCapacity > 0) allocArray<kggml_backend_info>(safeCapacity) else null
        val status = kggml_runtime_get_backends(infos, safeCapacity, count.ptr).toInt()
        if (status != RuntimeStatus.OK || infos == null) {
            return@memScoped RawBackendsResult(status, count.value, emptyList())
        }
        val entries = (0 until count.value.coerceIn(0, safeCapacity)).map { i ->
            val info = infos[i]
            RawBackendInfo(
                kind = info.kind,
                compiled = info.compiled != 0,
                available = info.available != 0,
                deviceCount = info.device_count,
                name = info.name.toKString(),
            )
        }
        RawBackendsResult(status, count.value, entries)
    }

    override fun performanceThreadCount(): RawIntResult = memScoped {
        val out = alloc<IntVar>()
        val status = kggml_runtime_performance_thread_count(out.ptr).toInt()
        RawIntResult(status, out.value)
    }

    override fun hardwareThreadCount(): Int = kggml_runtime_hardware_thread_count()

    override fun hasCoreMl(): Boolean = kggml_runtime_has_coreml() != 0

    override fun runtimeVersion(): String = kggml_runtime_version()?.toKString().orEmpty()

    override fun ggmlVersion(): String = kggml_runtime_ggml_version()?.toKString().orEmpty()

    override fun ggmlCommit(): String = kggml_runtime_ggml_commit()?.toKString().orEmpty()

    override fun ggmlBuildCommit(): String = kggml_runtime_ggml_build_commit()?.toKString().orEmpty()

    override fun statusName(status: Int): String =
        kggml_status_name(status.convert())?.toKString() ?: "KGGML_STATUS_$status"

    override fun lastError(): String = kggml_runtime_last_error()?.toKString().orEmpty()
}

internal actual fun createPlatformRuntimeBridge(): NativeRuntimeBridge = AppleRuntimeBridge
