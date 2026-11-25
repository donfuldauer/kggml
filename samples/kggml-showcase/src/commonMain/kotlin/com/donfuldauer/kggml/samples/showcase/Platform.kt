package com.donfuldauer.kggml.samples.showcase

import androidx.compose.ui.graphics.ImageBitmap
import com.donfuldauer.kggml.core.MemoryPressureObserver
import com.donfuldauer.kggml.samples.showcase.llm.ImageSource

/**
 * Where the Model Manager keeps models: the app files directory on Android, Application Support on iOS,
 * `~/.kggml/models` on desktop (override with `-Dkggml.models.dir` / `KGGML_MODELS_DIR`).
 */
expect fun defaultModelDirectory(): String

/** Default directory of images exported by the Diffusion Image Studio. */
expect fun defaultExportDirectory(): String

/**
 * The OS low-memory signal source: `AndroidMemoryPressureObserver` (`onTrimMemory`), `IosMemoryPressureObserver`
 * (memory warnings), or a `ManualMemoryPressureObserver` on desktop (manual eviction only).
 */
expect fun createMemoryPressureObserver(): MemoryPressureObserver

/** RAM budget of the resident engine cache (about half the device's physical memory). */
expect fun defaultResidentBudgetBytes(): Long

/** Camera / photo library on mobile; [com.donfuldauer.kggml.samples.showcase.llm.NoImageSource] on desktop. */
expect fun platformImageSource(): ImageSource

/** Reads a whole local file into memory. */
expect fun readLocalFile(path: String): ByteArray

/** Writes [bytes] to a local file, creating its parent directories. */
expect fun writeLocalFile(path: String, bytes: ByteArray)

/** Wraps packed 8-bit pixels ([channels] = 1, 3 or 4 bytes per pixel) in an [ImageBitmap] for display. */
expect fun pixelsToImageBitmap(width: Int, height: Int, channels: Int, pixels: ByteArray): ImageBitmap

/** Expands packed 1/3/4-channel pixels to RGBA8888. */
fun toRgba(width: Int, height: Int, channels: Int, pixels: ByteArray): ByteArray {
    val count = width * height
    val out = ByteArray(count * 4)
    for (i in 0 until count) {
        val src = i * channels
        val dst = i * 4
        when (channels) {
            1, 2 -> {
                out[dst] = pixels[src]; out[dst + 1] = pixels[src]; out[dst + 2] = pixels[src]
                out[dst + 3] = if (channels == 2) pixels[src + 1] else 0xFF.toByte()
            }
            else -> {
                out[dst] = pixels[src]; out[dst + 1] = pixels[src + 1]; out[dst + 2] = pixels[src + 2]
                out[dst + 3] = if (channels == 4) pixels[src + 3] else 0xFF.toByte()
            }
        }
    }
    return out
}
