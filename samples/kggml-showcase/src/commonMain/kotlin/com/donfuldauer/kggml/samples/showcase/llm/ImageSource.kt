package com.donfuldauer.kggml.samples.showcase.llm

/**
 * Camera / photo-library access for the VLM panel. Images are returned as encoded bytes held in memory (JPEG / PNG);
 * implementations never write a temporary image file.
 */
interface ImageSource {
    val canPickFromGallery: Boolean
    val canCapture: Boolean

    /** Lets the user pick a photo; `null` if they cancelled. */
    suspend fun pickFromGallery(): ByteArray?

    /** Takes a photo with the camera; `null` if the user cancelled. */
    suspend fun capture(): ByteArray?
}

/** Platforms without a picker (desktop): images are loaded from a path instead. */
object NoImageSource : ImageSource {
    override val canPickFromGallery: Boolean = false
    override val canCapture: Boolean = false
    override suspend fun pickFromGallery(): ByteArray? = null
    override suspend fun capture(): ByteArray? = null
}

/** `PNG`, `JPEG`, `GIF`, `BMP`, `WebP` or `image` from the magic number of [bytes]. */
fun imageFormatOf(bytes: ByteArray): String {
    fun at(i: Int) = if (i < bytes.size) bytes[i].toInt() and 0xFF else -1
    return when {
        at(0) == 0x89 && at(1) == 0x50 && at(2) == 0x4E && at(3) == 0x47 -> "PNG"
        at(0) == 0xFF && at(1) == 0xD8 -> "JPEG"
        at(0) == 0x47 && at(1) == 0x49 && at(2) == 0x46 -> "GIF"
        at(0) == 0x42 && at(1) == 0x4D -> "BMP"
        at(0) == 0x52 && at(1) == 0x49 && at(8) == 0x57 && at(9) == 0x45 -> "WebP"
        else -> "image"
    }
}
