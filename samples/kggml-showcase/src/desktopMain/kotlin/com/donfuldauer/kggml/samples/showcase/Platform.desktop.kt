package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.toComposeImageBitmap
import com.donfuldauer.kggml.core.ManualMemoryPressureObserver
import com.donfuldauer.kggml.core.MemoryPressureObserver
import com.donfuldauer.kggml.samples.showcase.llm.ImageSource
import com.donfuldauer.kggml.samples.showcase.llm.NoImageSource
import java.io.File
import org.jetbrains.skia.ColorAlphaType
import org.jetbrains.skia.ColorType
import org.jetbrains.skia.ImageInfo
import org.jetbrains.skia.Image as SkiaImage

actual fun defaultModelDirectory(): String =
    System.getProperty("kggml.models.dir")?.takeIf { it.isNotBlank() }
        ?: System.getenv("KGGML_MODELS_DIR")?.takeIf { it.isNotBlank() }
        ?: File(System.getProperty("user.home"), ".kggml/models").absolutePath

actual fun defaultExportDirectory(): String = File(System.getProperty("user.home"), "Pictures").absolutePath

/** Desktop OSes have no portable low-memory signal: eviction is manual (or simulated) only. */
actual fun createMemoryPressureObserver(): MemoryPressureObserver = ManualMemoryPressureObserver()

actual fun defaultResidentBudgetBytes(): Long {
    val total = (java.lang.management.ManagementFactory.getOperatingSystemMXBean() as? com.sun.management.OperatingSystemMXBean)
        ?.totalMemorySize ?: (16L shl 30)
    return (total / 2).coerceAtLeast(1L shl 30)
}

actual fun platformImageSource(): ImageSource = NoImageSource

actual fun readLocalFile(path: String): ByteArray = File(path).readBytes()

actual fun writeLocalFile(path: String, bytes: ByteArray) {
    val file = File(path)
    file.absoluteFile.parentFile?.mkdirs()
    file.writeBytes(bytes)
}

actual fun pixelsToImageBitmap(width: Int, height: Int, channels: Int, pixels: ByteArray): ImageBitmap {
    val info = ImageInfo(width, height, ColorType.RGBA_8888, ColorAlphaType.UNPREMUL)
    return SkiaImage.makeRaster(info, toRgba(width, height, channels, pixels), width * 4).toComposeImageBitmap()
}
