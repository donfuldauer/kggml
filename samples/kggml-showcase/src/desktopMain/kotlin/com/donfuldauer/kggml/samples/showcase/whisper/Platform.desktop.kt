package com.donfuldauer.kggml.samples.showcase.whisper

import com.donfuldauer.kggml.whisper.WhisperEngine
import java.io.File
import javax.sound.sampled.AudioSystem
import javax.sound.sampled.DataLine
import javax.sound.sampled.LineUnavailableException
import javax.sound.sampled.TargetDataLine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import javax.sound.sampled.AudioFormat as JavaAudioFormat

/**
 * Desktop microphone through `javax.sound.sampled`: 44.1 kHz, signed 16-bit little-endian, converted to float in
 * 100 ms chunks. Prefers stereo when the default input line supports it and otherwise uses mono.
 */
class DesktopAudioCapture(private val sampleRate: Int = 44_100, private val chunkMs: Int = 100) : AudioCapture {
    @Volatile private var format: JavaAudioFormat? = null

    override suspend fun prepare(): CaptureFormat {
        val chosen = listOf(1, 2).map(::pcm16).firstOrNull { AudioSystem.isLineSupported(DataLine.Info(TargetDataLine::class.java, it)) }
            ?: throw LineUnavailableException("No microphone supports $sampleRate Hz 16-bit PCM")
        format = chosen
        return CaptureFormat(sampleRate, chosen.channels)
    }

    override fun frames(): Flow<FloatArray> = flow {
        val audioFormat = format ?: error("prepare() must be called before frames()")
        val line = AudioSystem.getTargetDataLine(audioFormat)
        val chunkBytes = sampleRate * chunkMs / 1000 * audioFormat.frameSize
        line.open(audioFormat, chunkBytes * 4)
        try {
            line.start()
            val bytes = ByteArray(chunkBytes)
            while (currentCoroutineContext().isActive) {
                val read = line.read(bytes, 0, bytes.size)
                if (read <= 0) continue
                emit(FloatArray(read / 2) { i -> ((bytes[2 * i].toInt() and 0xFF) or (bytes[2 * i + 1].toInt() shl 8)) / 32768f })
            }
        } finally {
            line.stop()
            line.close()
        }
    }.flowOn(Dispatchers.IO)

    private fun pcm16(channels: Int) = JavaAudioFormat(sampleRate.toFloat(), 16, channels, true, false)
}

actual fun platformAudioCapture(): AudioCapture? = DesktopAudioCapture()

actual fun readAudioFile(path: String): ByteArray = File(path).readBytes()

actual fun defaultWhisperModelPath(): String =
    System.getProperty("kggml.whisper.model") ?: System.getenv("KGGML_WHISPER_MODEL") ?: ""

actual val supportsBackgroundStress: Boolean = false

actual fun platformBackgroundFallback(
    engine: WhisperEngine,
    scope: CoroutineScope,
    onError: (Throwable) -> Unit,
): AutoCloseable? = null
