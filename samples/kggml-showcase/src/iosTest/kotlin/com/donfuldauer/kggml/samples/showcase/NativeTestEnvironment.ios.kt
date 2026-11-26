package com.donfuldauer.kggml.samples.showcase

import kotlinx.cinterop.ExperimentalForeignApi
import platform.Foundation.NSFileManager
import platform.Foundation.NSProcessInfo
import platform.Foundation.NSTemporaryDirectory

@OptIn(ExperimentalForeignApi::class)
actual object NativeTestEnvironment {
    /** iOS test executables link the engine xcframeworks, so the engines are always present when tests run. */
    actual val requireNative: Boolean = true

    actual fun newTempDirectory(): String {
        val path = NSTemporaryDirectory() + "kggml-showcase-e2e-" + NSProcessInfo.processInfo.globallyUniqueString
        check(NSFileManager.defaultManager.createDirectoryAtPath(path, true, null, null)) { "cannot create $path" }
        return path
    }

    /** Unreachable: [requireNative] is always true here, so a missing library fails instead. */
    actual fun skip(reason: String): Nothing = throw AssertionError("iOS native tests cannot be skipped: $reason")
}
