package com.donfuldauer.kggml.samples.whisper

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs

class PcmAudioTest {
    private class Wav {
        val bytes = mutableListOf<Byte>()
        fun ascii(s: String) = apply { s.forEach { bytes += it.code.toByte() } }
        fun u16(v: Int) = apply { bytes += (v and 0xFF).toByte(); bytes += ((v shr 8) and 0xFF).toByte() }
        fun u32(v: Int) = apply { u16(v and 0xFFFF); u16((v ushr 16) and 0xFFFF) }
        fun raw(vararg b: Byte) = apply { bytes.addAll(b.toList()) }
        fun build() = bytes.toByteArray()
    }

    private fun fmt(w: Wav, format: Int, channels: Int, rate: Int, bits: Int) = w.ascii("fmt ").u32(16)
        .u16(format).u16(channels).u32(rate).u32(rate * channels * bits / 8).u16(channels * bits / 8).u16(bits)

    private fun header(w: Wav = Wav()) = w.ascii("RIFF").u32(0).ascii("WAVE")

    @Test
    fun parsesPcm16Wav() {
        val w = fmt(header(), 1, 2, 44_100, 16).ascii("data").u32(8).u16(1).u16(0xFFFF).u16(0x7FFF).u16(0x8000)
        val audio = assertIs<PcmAudio.S16>(parsePcmFile(w.build()))
        assertEquals(44_100, audio.sampleRate)
        assertEquals(2, audio.channels)
        assertContentEquals(shortArrayOf(1, -1, Short.MAX_VALUE, Short.MIN_VALUE), audio.samples)
        assertEquals(0, audio.durationMs)
    }

    @Test
    fun parsesFloatWavAndSkipsUnknownChunks() {
        val w = header().ascii("LIST").u32(3).raw(1, 2, 3, 0) // odd size: padded to a word boundary
        fmt(w, 3, 1, 16_000, 32).ascii("data").u32(8).u32(0.5f.toBits()).u32((-1f).toBits())
        val audio = assertIs<PcmAudio.F32>(parsePcmFile(w.build()))
        assertContentEquals(floatArrayOf(0.5f, -1f), audio.samples)
        assertEquals(16_000, audio.sampleRate)
    }

    @Test
    fun parsesWaveFormatExtensible() {
        val w = header().ascii("fmt ").u32(40).u16(0xFFFE).u16(1).u32(16_000).u32(32_000).u16(2).u16(16)
            .u16(22).u16(16).u32(4).u16(1).raw(*ByteArray(14)).ascii("data").u32(2).u16(42)
        val audio = assertIs<PcmAudio.S16>(parsePcmFile(w.build()))
        assertContentEquals(shortArrayOf(42), audio.samples)
    }

    @Test
    fun clampsPlaceholderDataSizeAndDropsPartialFrames() {
        val w = fmt(header(), 1, 1, 16_000, 16).ascii("data").u32(-1).u16(7).u16(8).raw(9)
        val audio = assertIs<PcmAudio.S16>(parsePcmFile(w.build()))
        assertContentEquals(shortArrayOf(7, 8), audio.samples)
    }

    @Test
    fun treatsHeaderlessFilesAsRawS16() {
        val pcm = ByteArray(32_000 * 2) // 2 s at 16 kHz mono
        val audio = assertIs<PcmAudio.S16>(parsePcmFile(pcm))
        assertEquals(16_000, audio.sampleRate)
        assertEquals(1, audio.channels)
        assertEquals(2_000, audio.durationMs)
        assertEquals(1_000, assertIs<PcmAudio.S16>(parsePcmFile(pcm, rawSampleRate = 16_000, rawChannels = 2)).durationMs)
    }

    @Test
    fun rejectsMalformedFiles() {
        assertFailsWith<UnsupportedAudioException> { parsePcmFile(ByteArray(3)) }
        assertFailsWith<UnsupportedAudioException> { parsePcmFile(header().build()) }
        assertFailsWith<UnsupportedAudioException> { parsePcmFile(header().ascii("data").u32(0).build()) }
        assertFailsWith<UnsupportedAudioException> { parsePcmFile(fmt(header(), 1, 1, 16_000, 16).build()) }
        assertFailsWith<UnsupportedAudioException> { parsePcmFile(header().ascii("fmt ").u32(100).u16(1).build()) }
        assertFailsWith<UnsupportedAudioException> { parsePcmFile(header().ascii("fmt ").u32(8).u32(0).u32(0).build()) }
        assertFailsWith<UnsupportedAudioException> {
            parsePcmFile(fmt(header(), 1, 1, 16_000, 24).ascii("data").u32(3).raw(0, 0, 0).build())
        }
        assertFailsWith<UnsupportedAudioException> {
            parsePcmFile(fmt(header(), 1, 9, 16_000, 16).ascii("data").u32(0).build())
        }
        assertFailsWith<UnsupportedAudioException> {
            parsePcmFile(fmt(header(), 1, 1, 0, 16).ascii("data").u32(0).build())
        }
    }
}
