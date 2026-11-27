package com.donfuldauer.kggml.samples.showcase.llm

import kotlin.test.Test
import kotlin.test.assertEquals

/** [imageFormatOf]: the label of an attached image comes from its magic number, never from a file name. */
class ImageFormatTest {
    private fun bytes(vararg values: Int) = ByteArray(values.size) { values[it].toByte() }

    private fun ascii(text: String) = text.encodeToByteArray()

    @Test
    fun recognisesEverySupportedMagicNumber() {
        assertEquals("PNG", imageFormatOf(FakeImageSource.PNG_MAGIC))
        assertEquals("JPEG", imageFormatOf(FakeImageSource.JPEG_MAGIC))
        assertEquals("GIF", imageFormatOf(ascii("GIF89a") + ByteArray(4)))
        assertEquals("BMP", imageFormatOf(ascii("BM") + ByteArray(12)))
        // RIFF container: WebP only when the form type at offset 8 is "WEBP".
        assertEquals("WebP", imageFormatOf(ascii("RIFF") + bytes(0x24, 0, 0, 0) + ascii("WEBPVP8 ")))
    }

    @Test
    fun otherOrTruncatedDataIsAGenericImage() {
        assertEquals("image", imageFormatOf(ascii("RIFF") + bytes(0x24, 0, 0, 0) + ascii("WAVEfmt ")), "a RIFF/WAVE file is not WebP")
        assertEquals("image", imageFormatOf(ascii("RIFF")), "too short to carry the WebP form type")
        assertEquals("image", imageFormatOf(bytes(0xFF)), "a lone 0xFF is not a JPEG")
        assertEquals("image", imageFormatOf(bytes(0x89, 0x50, 0x4E)), "a truncated PNG signature")
        assertEquals("image", imageFormatOf(ascii("GI") + ByteArray(4)))
        assertEquals("image", imageFormatOf(ByteArray(0)))
    }
}
