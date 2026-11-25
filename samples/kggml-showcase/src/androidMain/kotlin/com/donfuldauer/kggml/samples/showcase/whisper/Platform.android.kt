package com.donfuldauer.kggml.samples.showcase.whisper

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.donfuldauer.kggml.whisper.WhisperEngine
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive

/**
 * Android microphone through `AudioRecord`: 48 kHz mono `ENCODING_PCM_FLOAT` in 100 ms chunks from the
 * `VOICE_RECOGNITION` source. Requires the `RECORD_AUDIO` runtime permission (requested by `MainActivity`).
 */
class AndroidAudioCapture(private val sampleRate: Int = 48_000, private val chunkMs: Int = 100) : AudioCapture {
    override suspend fun prepare(): CaptureFormat = CaptureFormat(sampleRate, 1)

    @SuppressLint("MissingPermission") // Checked by MainActivity; a denied permission fails the state check below.
    override fun frames(): Flow<FloatArray> = flow {
        val chunk = sampleRate * chunkMs / 1000
        val minBytes = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        check(minBytes > 0) { "AudioRecord does not support $sampleRate Hz float capture ($minBytes)" }
        val record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
            maxOf(minBytes, chunk * Float.SIZE_BYTES * 4),
        )
        try {
            check(record.state == AudioRecord.STATE_INITIALIZED) {
                "Microphone unavailable (is the RECORD_AUDIO permission granted?)"
            }
            record.startRecording()
            val buffer = FloatArray(chunk)
            while (currentCoroutineContext().isActive) {
                val read = record.read(buffer, 0, buffer.size, AudioRecord.READ_BLOCKING)
                check(read >= 0) { "AudioRecord.read failed with $read" }
                if (read > 0) emit(buffer.copyOf(read))
            }
        } finally {
            if (record.state == AudioRecord.STATE_INITIALIZED) record.stop()
            record.release()
        }
    }.flowOn(Dispatchers.IO)
}

actual fun platformAudioCapture(): AudioCapture? = AndroidAudioCapture()

actual fun readAudioFile(path: String): ByteArray = File(path).readBytes()

actual fun defaultWhisperModelPath(): String = ""

actual val supportsBackgroundStress: Boolean = false

actual fun platformBackgroundFallback(
    engine: WhisperEngine,
    scope: CoroutineScope,
    onError: (Throwable) -> Unit,
): AutoCloseable? = null
