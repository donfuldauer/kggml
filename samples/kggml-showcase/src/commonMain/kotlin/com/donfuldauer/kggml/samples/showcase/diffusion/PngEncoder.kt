package com.donfuldauer.kggml.samples.showcase.diffusion

/**
 * Minimal pure-Kotlin PNG encoder for 8-bit RGB / RGBA / gray pixels: one IHDR, one IDAT holding an uncompressed
 * ("stored") zlib stream with filter type 0 rows, and IEND. Files are larger than a compressing encoder's output but
 * valid for every PNG decoder, and the encoder needs no platform API.
 */
object PngEncoder {
    private val SIGNATURE = byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)
    private const val MAX_STORED_BLOCK = 65_535

    /**
     * Encodes [pixels] (row-major, no padding, [channels] bytes per pixel: 1 = gray, 3 = RGB, 4 = RGBA).
     */
    fun encode(width: Int, height: Int, channels: Int, pixels: ByteArray): ByteArray {
        require(width > 0 && height > 0) { "width / height must be > 0" }
        require(channels == 1 || channels == 3 || channels == 4) { "channels must be 1, 3 or 4, was $channels" }
        require(pixels.size.toLong() == width.toLong() * height * channels) { "pixels must hold width * height * channels bytes" }
        val colorType = when (channels) {
            1 -> 0
            3 -> 2
            else -> 6
        }
        val ihdr = ByteBuilder(13).apply {
            int(width); int(height); byte(8); byte(colorType); byte(0); byte(0); byte(0)
        }.toByteArray()

        val rowBytes = width * channels
        val raw = ByteArray((rowBytes + 1) * height)
        for (y in 0 until height) {
            raw[y * (rowBytes + 1)] = 0 // filter: none
            pixels.copyInto(raw, y * (rowBytes + 1) + 1, y * rowBytes, (y + 1) * rowBytes)
        }

        val out = ByteBuilder(raw.size + raw.size / MAX_STORED_BLOCK * 5 + 64)
        out.bytes(SIGNATURE)
        out.chunk("IHDR", ihdr)
        out.chunk("IDAT", zlibStored(raw))
        out.chunk("IEND", ByteArray(0))
        return out.toByteArray()
    }

    /** A zlib stream of [data] in uncompressed deflate blocks. */
    private fun zlibStored(data: ByteArray): ByteArray {
        val blocks = maxOf(1, (data.size + MAX_STORED_BLOCK - 1) / MAX_STORED_BLOCK)
        val out = ByteBuilder(data.size + blocks * 5 + 6)
        out.byte(0x78); out.byte(0x01) // deflate, 32 KiB window, no dictionary, fastest
        var offset = 0
        do {
            val length = minOf(MAX_STORED_BLOCK, data.size - offset)
            val last = offset + length >= data.size
            out.byte(if (last) 1 else 0)
            out.byte(length and 0xFF); out.byte(length ushr 8)
            out.byte(length.inv() and 0xFF); out.byte((length.inv() ushr 8) and 0xFF)
            out.bytes(data, offset, length)
            offset += length
        } while (!last)
        out.int(adler32(data))
        return out.toByteArray()
    }

    fun adler32(data: ByteArray): Int {
        var a = 1L
        var b = 0L
        for (byte in data) {
            a = (a + (byte.toInt() and 0xFF)) % 65_521
            b = (b + a) % 65_521
        }
        return ((b shl 16) or a).toInt()
    }

    private val CRC_TABLE = IntArray(256) { n ->
        var c = n
        repeat(8) { c = if (c and 1 != 0) (0xEDB88320.toInt() xor (c ushr 1)) else c ushr 1 }
        c
    }

    fun crc32(data: ByteArray, offset: Int = 0, length: Int = data.size): Int {
        var c = -1
        for (i in offset until offset + length) c = CRC_TABLE[(c xor data[i].toInt()) and 0xFF] xor (c ushr 8)
        return c.inv()
    }

    private class ByteBuilder(capacity: Int) {
        private var buffer = ByteArray(maxOf(capacity, 16))
        private var size = 0

        private fun ensure(extra: Int) {
            if (size + extra > buffer.size) buffer = buffer.copyOf(maxOf(buffer.size * 2, size + extra))
        }

        fun byte(v: Int) {
            ensure(1)
            buffer[size++] = v.toByte()
        }

        fun int(v: Int) {
            byte(v ushr 24); byte(v ushr 16); byte(v ushr 8); byte(v)
        }

        fun bytes(src: ByteArray, offset: Int = 0, length: Int = src.size) {
            ensure(length)
            src.copyInto(buffer, size, offset, offset + length)
            size += length
        }

        fun chunk(type: String, data: ByteArray) {
            int(data.size)
            val typeAndData = ByteArray(4 + data.size)
            for (i in 0 until 4) typeAndData[i] = type[i].code.toByte()
            data.copyInto(typeAndData, 4)
            bytes(typeAndData)
            int(crc32(typeAndData))
        }

        fun toByteArray(): ByteArray = buffer.copyOf(size)
    }
}
