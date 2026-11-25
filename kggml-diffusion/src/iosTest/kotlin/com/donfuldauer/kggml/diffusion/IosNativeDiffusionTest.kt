package com.donfuldauer.kggml.diffusion

import kotlin.test.assertTrue
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

/**
 * [NativeDiffusionEngineContract] against KggmlDiffusion.framework on the iOS simulator (cinterop).
 * Only built when KggmlDiffusion.xcframework exists (macOS + Xcode).
 */
@OptIn(ExperimentalForeignApi::class)
class IosNativeDiffusionTest : NativeDiffusionEngineContract() {
    override fun writeTempFile(prefix: String, bytes: ByteArray): String {
        val path = NSTemporaryDirectory() + prefix + NSProcessInfo.processInfo.globallyUniqueString + ".gguf"
        val written = if (bytes.isEmpty()) {
            NSFileManager.defaultManager.createFileAtPath(path, null, null)
        } else {
            bytes.usePinned { NSData.dataWithBytes(it.addressOf(0), bytes.size.convert()).writeToFile(path, true) }
        }
        assertTrue(written, "could not write $path")
        return path
    }

    override fun deleteFile(path: String) {
        NSFileManager.defaultManager.removeItemAtPath(path, null)
    }

    override fun directoryPath(): String = NSTemporaryDirectory()
}
