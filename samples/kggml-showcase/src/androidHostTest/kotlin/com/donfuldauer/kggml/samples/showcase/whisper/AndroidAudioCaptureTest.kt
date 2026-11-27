package com.donfuldauer.kggml.samples.showcase.whisper

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.shadows.ShadowAudioRecord
import java.util.concurrent.atomic.AtomicReference
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * [AndroidAudioCapture] over Robolectric's `AudioRecord`, whose samples come from a scripted
 * `ShadowAudioRecord.AudioRecordSource` instead of a microphone.
 */
@RunWith(AndroidJUnit4::class)
class AndroidAudioCaptureTest {
    private val record = AtomicReference<AudioRecord>()

    /** Serves float reads from [read] and remembers the `AudioRecord` the capture created. */
    private fun microphone(read: (FloatArray, Int, Int) -> Int) {
        ShadowAudioRecord.setSourceProvider { audioRecord ->
            record.set(audioRecord)
            object : ShadowAudioRecord.AudioRecordSource {
                override fun readInFloatArray(audioData: FloatArray, offsetInFloats: Int, sizeInFloats: Int, isBlocking: Boolean): Int =
                    read(audioData, offsetInFloats, sizeInFloats)
            }
        }
    }

    @After
    fun clearMicrophone() {
        ShadowAudioRecord.clearSource()
    }

    @Test
    fun prepareReportsMonoAtTheConfiguredRate() = runBlocking {
        assertEquals(CaptureFormat(48_000, 1), AndroidAudioCapture().prepare())
        assertEquals(CaptureFormat(16_000, 1), AndroidAudioCapture(sampleRate = 16_000).prepare())
    }

    @Test
    fun framesAreHundredMillisecondFloatChunksFromTheVoiceRecognitionSource() = runBlocking {
        var next = 0f
        microphone { data, offset, size ->
            for (i in 0 until size) data[offset + i] = next++
            size
        }
        val chunks = AndroidAudioCapture().frames().take(2).toList()
        assertEquals(listOf(4_800, 4_800), chunks.map { it.size }, "100 ms at 48 kHz mono")
        assertEquals(0f, chunks[0].first())
        assertEquals(4_799f, chunks[0].last())
        assertEquals(4_800f, chunks[1].first(), "consecutive chunks continue the stream")

        val audioRecord = record.get()
        assertEquals(MediaRecorder.AudioSource.VOICE_RECOGNITION, audioRecord.audioSource)
        assertEquals(48_000, audioRecord.sampleRate)
        assertEquals(AudioFormat.ENCODING_PCM_FLOAT, audioRecord.audioFormat)
        assertEquals(1, audioRecord.channelCount)
        assertEquals(AudioRecord.STATE_UNINITIALIZED, audioRecord.state, "released once the collector stops")
    }

    @Test
    fun shortReadsAreEmittedAsIsAndEmptyReadsAreSkipped() = runBlocking {
        val reads = ArrayDeque(listOf(0, 160, 0, 80))
        microphone { data, offset, _ ->
            val n = reads.removeFirstOrNull() ?: 0
            for (i in 0 until n) data[offset + i] = 0.5f
            n
        }
        val chunks = AndroidAudioCapture(sampleRate = 16_000, chunkMs = 20).frames().take(2).toList()
        assertEquals(listOf(160, 80), chunks.map { it.size })
        assertContentEquals(FloatArray(80) { 0.5f }, chunks[1])
    }

    @Test
    fun readErrorFailsTheStreamAndReleasesTheRecorder() {
        microphone { _, _, _ -> AudioRecord.ERROR_INVALID_OPERATION }
        val e = assertFailsWith<IllegalStateException> { runBlocking { AndroidAudioCapture().frames().toList() } }
        assertEquals("AudioRecord.read failed with ${AudioRecord.ERROR_INVALID_OPERATION}", e.message)
        assertEquals(AudioRecord.STATE_UNINITIALIZED, record.get().state)
    }
}
