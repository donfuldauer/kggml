package com.donfuldauer.kggml.samples.showcase

import com.donfuldauer.kggml.samples.showcase.diffusion.PngEncoder
import com.donfuldauer.kggml.samples.showcase.diffusion.fakeImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * The Diffusion Image Studio's PNG export (pure-Kotlin [PngEncoder]) read back by an independent decoder
 * (javax.imageio), pixel for pixel, for RGB, RGBA and grayscale images.
 */
class PngExportImageIoDesktopTest {
    @Test
    fun rgbExportDecodesPixelForPixel() {
        val image = fakeImage(256, 256, 42)
        val decoded = assertNotNull(ImageIO.read(ByteArrayInputStream(PngEncoder.encode(image.width, image.height, 3, image.pixels))))
        assertEquals(256, decoded.width)
        assertEquals(256, decoded.height)
        for ((x, y) in listOf(0 to 0, 3 to 2, 255 to 255, 128 to 17)) {
            val o = (y * image.width + x) * 3
            val rgb = decoded.getRGB(x, y)
            assertEquals(image.pixels[o].toInt() and 0xFF, (rgb shr 16) and 0xFF, "R at $x,$y")
            assertEquals(image.pixels[o + 1].toInt() and 0xFF, (rgb shr 8) and 0xFF, "G at $x,$y")
            assertEquals(image.pixels[o + 2].toInt() and 0xFF, rgb and 0xFF, "B at $x,$y")
        }
    }

    @Test
    fun rgbaAndGrayExportsDecode() {
        val rgba = ByteArray(4 * 4 * 4) { (it * 7).toByte() }
        val decodedRgba = assertNotNull(ImageIO.read(ByteArrayInputStream(PngEncoder.encode(4, 4, 4, rgba))))
        val argb = decodedRgba.getRGB(1, 2)
        val o = (2 * 4 + 1) * 4
        assertEquals(rgba[o + 3].toInt() and 0xFF, (argb ushr 24) and 0xFF, "alpha")
        assertEquals(rgba[o].toInt() and 0xFF, (argb shr 16) and 0xFF, "red")

        val gray = ByteArray(16) { (it * 16).toByte() }
        val decodedGray = assertNotNull(ImageIO.read(ByteArrayInputStream(PngEncoder.encode(4, 4, 1, gray))))
        assertEquals(gray[5].toInt() and 0xFF, decodedGray.raster.getSample(1, 1, 0))
    }
}
