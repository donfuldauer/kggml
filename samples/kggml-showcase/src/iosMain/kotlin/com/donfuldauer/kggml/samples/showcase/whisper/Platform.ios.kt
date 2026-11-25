package com.donfuldauer.kggml.samples.showcase.whisper

import com.donfuldauer.kggml.whisper.WhisperEngine
import com.donfuldauer.kggml.whisper.observeAppLifecycle
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.get
import kotlinx.cinterop.usePinned
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import platform.AVFAudio.AVAudioEngine
import platform.AVFAudio.AVAudioPCMBuffer
import platform.AVFAudio.AVAudioSession
import platform.AVFAudio.AVAudioSessionCategoryRecord
import platform.AVFAudio.setActive
import platform.Foundation.NSBundle
import platform.Foundation.NSData
import platform.Foundation.dataWithContentsOfFile
import platform.posix.memcpy

/**
 * iOS microphone through an `AVAudioEngine` input-node tap at the hardware format (typically 48 kHz; mono or
 * stereo depending on the route). Requires `NSMicrophoneUsageDescription` in the host app's `Info.plist`; live
 * dictation in the background additionally needs the `audio` background mode.
 */
@OptIn(ExperimentalForeignApi::class)
class IosAudioCapture(private val tapFrames: UInt = 4096u) : AudioCapture {
    private val engine = AVAudioEngine()

    override suspend fun prepare(): CaptureFormat {
        val session = AVAudioSession.sharedInstance()
        check(session.setCategory(AVAudioSessionCategoryRecord, error = null)) { "AVAudioSession category rejected" }
        check(session.setActive(true, error = null)) { "AVAudioSession could not be activated" }
        val format = engine.inputNode.outputFormatForBus(0uL)
        val sampleRate = format.sampleRate.toInt()
        check(sampleRate > 0) { "No audio input available (microphone permission denied?)" }
        return CaptureFormat(sampleRate, format.channelCount.toInt())
    }

    override fun frames(): Flow<FloatArray> = callbackFlow {
        val input = engine.inputNode
        input.installTapOnBus(0uL, tapFrames, null) { buffer, _ ->
            if (buffer != null) trySend(buffer.toInterleaved())
        }
        engine.prepare()
        if (!engine.startAndReturnError(null)) {
            input.removeTapOnBus(0uL)
            throw IllegalStateException("AVAudioEngine failed to start")
        }
        awaitClose {
            input.removeTapOnBus(0uL)
            engine.stop()
        }
    }.buffer(Channel.UNLIMITED) // The tap cannot be back-pressured; never drop microphone audio.
}

/** Copies a (standard, deinterleaved) float buffer into one interleaved array. */
@OptIn(ExperimentalForeignApi::class)
private fun AVAudioPCMBuffer.toInterleaved(): FloatArray {
    val channels = format.channelCount.toInt()
    val frames = frameLength.toInt()
    val data = floatChannelData ?: return FloatArray(0)
    val step = stride.toInt().coerceAtLeast(1)
    val out = FloatArray(frames * channels)
    for (c in 0 until channels) {
        val channel = data[c] ?: continue
        for (i in 0 until frames) out[i * channels + c] = channel[i * step]
    }
    return out
}

actual fun platformAudioCapture(): AudioCapture? = IosAudioCapture()

@OptIn(ExperimentalForeignApi::class)
actual fun readAudioFile(path: String): ByteArray {
    val data = NSData.dataWithContentsOfFile(path) ?: throw IllegalArgumentException("Cannot read $path")
    val bytes = ByteArray(data.length.toInt())
    if (bytes.isNotEmpty()) bytes.usePinned { memcpy(it.addressOf(0), data.bytes, data.length) }
    return bytes
}

/** A `ggml-tiny.en.bin` bundled with the host app, if any. */
actual fun defaultWhisperModelPath(): String = NSBundle.mainBundle.pathForResource("ggml-tiny.en", "bin") ?: ""

actual val supportsBackgroundStress: Boolean = true

actual fun platformBackgroundFallback(
    engine: WhisperEngine,
    scope: CoroutineScope,
    onError: (Throwable) -> Unit,
): AutoCloseable? = engine.observeAppLifecycle(scope, onError = onError)
