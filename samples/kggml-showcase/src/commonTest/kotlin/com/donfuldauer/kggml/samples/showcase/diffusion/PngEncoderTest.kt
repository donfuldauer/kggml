package com.donfuldauer.kggml.samples.showcase.diffusion

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class PngEncoderTest {
    @Test
    fun knownChecksums() {
        assertEquals(0xCBF43926.toInt(), PngEncoder.crc32("123456789".encodeToByteArray()))
        assertEquals(0x11E60398, PngEncoder.adler32("Wikipedia".encodeToByteArray()))
    }

    @Test
    fun rgbImageRoundTripsThroughAStoredDeflateDecoder() {
        val pixels = ByteArray(5 * 3 * 3) { (it * 7).toByte() }
        val png = PngEncoder.encode(5, 3, 3, pixels)
        assertContentEquals(SIGNATURE, png.copyOf(8))
        val decoded = decodeStoredPng(png)
        assertEquals(5, decoded.width)
        assertEquals(3, decoded.height)
        assertEquals(2, decoded.colorType)
        assertContentEquals(pixels, decoded.pixels)
    }

    @Test
    fun largeImageSpansSeveralStoredBlocks() {
        // 200 x 200 RGBA = 160,200 filtered bytes: three 64 KiB stored blocks.
        val pixels = ByteArray(200 * 200 * 4) { (it % 253).toByte() }
        val decoded = decodeStoredPng(PngEncoder.encode(200, 200, 4, pixels))
        assertEquals(6, decoded.colorType)
        assertContentEquals(pixels, decoded.pixels)
        val gray = ByteArray(16) { it.toByte() }
        assertContentEquals(gray, decodeStoredPng(PngEncoder.encode(4, 4, 1, gray)).pixels)
    }

    @Test
    fun rejectsInvalidInput() {
        assertFailsWith<IllegalArgumentException> { PngEncoder.encode(0, 1, 3, ByteArray(0)) }
        assertFailsWith<IllegalArgumentException> { PngEncoder.encode(1, 1, 2, ByteArray(2)) }
        assertFailsWith<IllegalArgumentException> { PngEncoder.encode(2, 2, 3, ByteArray(3)) }
    }

    class Decoded(val width: Int, val height: Int, val colorType: Int, val pixels: ByteArray)

    companion object {
        val SIGNATURE = byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)

        private fun ByteArray.u32(at: Int): Int =
            ((this[at].toInt() and 0xFF) shl 24) or ((this[at + 1].toInt() and 0xFF) shl 16) or
                ((this[at + 2].toInt() and 0xFF) shl 8) or (this[at + 3].toInt() and 0xFF)

        /** Decodes PNGs written by [PngEncoder] (stored deflate, filter 0), checking every CRC and the Adler-32. */
        fun decodeStoredPng(png: ByteArray): Decoded {
            check(png.copyOf(8).contentEquals(SIGNATURE)) { "bad signature" }
            var at = 8
            var width = 0
            var height = 0
            var colorType = -1
            val idat = ArrayList<Byte>()
            while (true) {
                val length = png.u32(at)
                val type = png.copyOfRange(at + 4, at + 8).decodeToString()
                check(PngEncoder.crc32(png, at + 4, length + 4) == png.u32(at + 8 + length)) { "bad CRC in $type" }
                val data = png.copyOfRange(at + 8, at + 8 + length)
                when (type) {
                    "IHDR" -> {
                        width = data.u32(0)
                        height = data.u32(4)
                        check(data[8].toInt() == 8) { "bit depth" }
                        colorType = data[9].toInt()
                    }
                    "IDAT" -> idat.addAll(data.toList())
                    "IEND" -> break
                }
                at += 12 + length
            }
            val z = idat.toByteArray()
            check(z[0].toInt() == 0x78) { "zlib header" }
            val raw = ArrayList<Byte>()
            var p = 2
            while (true) {
                val final = z[p].toInt() and 1
                check((z[p].toInt() shr 1) and 3 == 0) { "only stored blocks expected" }
                val len = (z[p + 1].toInt() and 0xFF) or ((z[p + 2].toInt() and 0xFF) shl 8)
                val nlen = (z[p + 3].toInt() and 0xFF) or ((z[p + 4].toInt() and 0xFF) shl 8)
                check(len == nlen.inv() and 0xFFFF) { "LEN/NLEN mismatch" }
                for (i in 0 until len) raw += z[p + 5 + i]
                p += 5 + len
                if (final == 1) break
            }
            val rawBytes = raw.toByteArray()
            check(PngEncoder.adler32(rawBytes) == z.u32(p)) { "bad Adler-32" }
            val channels = when (colorType) {
                0 -> 1
                2 -> 3
                6 -> 4
                else -> error("color type $colorType")
            }
            val row = width * channels
            val pixels = ByteArray(row * height)
            for (y in 0 until height) {
                check(rawBytes[y * (row + 1)].toInt() == 0) { "filter" }
                rawBytes.copyInto(pixels, y * row, y * (row + 1) + 1, (y + 1) * (row + 1))
            }
            return Decoded(width, height, colorType, pixels)
        }
    }
}
