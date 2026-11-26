package com.donfuldauer.kggml.whisper

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
 * [NativeWhisperEngineContract] against KggmlWhisper.framework on the iOS simulator: cinterop with `usePinned`
 * zero-copy PCM, `kggml_whisper_transcribe_with_state` with a `staticCFunction` new-segment callback and a `StableRef`
 * listener, pinned prompt tokens and the native streaming resampler. Only built when KggmlWhisper.xcframework exists
 * (macOS + Xcode).
 */
@OptIn(ExperimentalForeignApi::class)
class IosNativeWhisperTest : NativeWhisperEngineContract() {
    override fun writeTempFile(prefix: String, bytes: ByteArray): String {
        val path = NSTemporaryDirectory() + prefix + NSProcessInfo.processInfo.globallyUniqueString + ".bin"
        val written = bytes.usePinned { NSData.dataWithBytes(it.addressOf(0), bytes.size.convert()).writeToFile(path, true) }
        assertTrue(written, "could not write $path")
        return path
    }

    override fun deleteFile(path: String) {
        NSFileManager.defaultManager.removeItemAtPath(path, null)
    }

    override fun temporaryFileNames(): Set<String> =
        NSFileManager.defaultManager.contentsOfDirectoryAtPath(NSTemporaryDirectory(), null).orEmpty().map { it.toString() }.toSet()
}
