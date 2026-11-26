package com.donfuldauer.kggml.samples.whisper

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.convert
import kotlinx.cinterop.usePinned
import platform.Foundation.NSData
import platform.Foundation.NSFileManager
import platform.Foundation.NSProcessInfo
import platform.Foundation.NSTemporaryDirectory
import platform.Foundation.dataWithBytes
import platform.Foundation.writeToFile

@OptIn(ExperimentalForeignApi::class)
actual object NativeTestEnvironment {
    /** iOS test executables link the xcframeworks, so the libraries are always present when tests run. */
    actual val requireNative: Boolean = true

    actual fun writeTempFile(name: String, bytes: ByteArray): String {
        val dir = NSTemporaryDirectory() + "kggml-minimal-" + NSProcessInfo.processInfo.globallyUniqueString
        check(NSFileManager.defaultManager.createDirectoryAtPath(dir, true, null, null)) { "cannot create $dir" }
        val path = "$dir/$name"
        val written = bytes.usePinned { NSData.dataWithBytes(it.addressOf(0), bytes.size.convert()).writeToFile(path, true) }
        check(written) { "cannot write $path" }
        return path
    }
}
