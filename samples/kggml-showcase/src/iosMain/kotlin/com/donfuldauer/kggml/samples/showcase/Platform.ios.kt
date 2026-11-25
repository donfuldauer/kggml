package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.toComposeImageBitmap
import com.donfuldauer.kggml.core.IosMemoryPressureObserver
import com.donfuldauer.kggml.core.MemoryPressureObserver
import com.donfuldauer.kggml.samples.showcase.llm.IosImageSource
import com.donfuldauer.kggml.samples.showcase.llm.ImageSource
import kotlinx.cinterop.BetaInteropApi
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import org.jetbrains.skia.ColorAlphaType
import org.jetbrains.skia.ColorType
import org.jetbrains.skia.ImageInfo
import platform.Foundation.NSApplicationSupportDirectory
import platform.Foundation.NSData
import platform.Foundation.NSDocumentDirectory
import platform.Foundation.NSFileManager
import platform.Foundation.NSProcessInfo
import platform.Foundation.NSSearchPathForDirectoriesInDomains
import platform.Foundation.NSUserDomainMask
import platform.Foundation.create
import platform.Foundation.dataWithContentsOfFile
import platform.Foundation.writeToFile
import platform.posix.memcpy
import org.jetbrains.skia.Image as SkiaImage

private fun userDirectory(directory: ULong): String =
    NSSearchPathForDirectoriesInDomains(directory, NSUserDomainMask, true).firstOrNull() as? String
        ?: error("No user directory $directory")

actual fun defaultModelDirectory(): String = userDirectory(NSApplicationSupportDirectory) + "/kggml/models"

actual fun defaultExportDirectory(): String = userDirectory(NSDocumentDirectory)

actual fun createMemoryPressureObserver(): MemoryPressureObserver = IosMemoryPressureObserver()

actual fun defaultResidentBudgetBytes(): Long = (NSProcessInfo.processInfo.physicalMemory.toLong() / 2).coerceAtLeast(512L shl 20)

actual fun platformImageSource(): ImageSource = IosImageSource()

@OptIn(ExperimentalForeignApi::class)
actual fun readLocalFile(path: String): ByteArray {
    val data = NSData.dataWithContentsOfFile(path) ?: throw IllegalArgumentException("Cannot read $path")
    return data.toByteArray()
}

@OptIn(ExperimentalForeignApi::class, BetaInteropApi::class)
actual fun writeLocalFile(path: String, bytes: ByteArray) {
    val parent = path.substringBeforeLast('/', "")
    if (parent.isNotEmpty()) {
        NSFileManager.defaultManager.createDirectoryAtPath(parent, withIntermediateDirectories = true, attributes = null, error = null)
    }
    val data = bytes.usePinned { NSData.create(bytes = it.addressOf(0), length = bytes.size.toULong()) }
    check(data.writeToFile(path, atomically = true)) { "Cannot write $path" }
}

actual fun pixelsToImageBitmap(width: Int, height: Int, channels: Int, pixels: ByteArray): ImageBitmap {
    val info = ImageInfo(width, height, ColorType.RGBA_8888, ColorAlphaType.UNPREMUL)
    return SkiaImage.makeRaster(info, toRgba(width, height, channels, pixels), width * 4).toComposeImageBitmap()
}

/** Copies [NSData] into a Kotlin [ByteArray]. */
@OptIn(ExperimentalForeignApi::class)
internal fun NSData.toByteArray(): ByteArray {
    val bytes = ByteArray(length.toInt())
    if (bytes.isNotEmpty()) bytes.usePinned { memcpy(it.addressOf(0), this.bytes, length) }
    return bytes
}
