package com.donfuldauer.kggml.docs

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.TranscriptionEvent
import com.donfuldauer.kggml.whisper.StreamingOptions
import com.donfuldauer.kggml.whisper.TranscriptionOptions
import com.donfuldauer.kggml.whisper.WhisperBackgroundFallback
import com.donfuldauer.kggml.whisper.WhisperEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.Flow

// README.md snippets for :kggml-whisper. Compiled by :docs-snippets; see build.gradle.kts.

internal suspend fun whisperOneShot(modelPath: String, pcm: FloatArray) {
    // snippet:start whisper-transcribe
    // `pcm`: mono float samples in [-1, 1]. Any rate / channel count works; 16 kHz mono is passed zero-copy.
    WhisperEngine(modelPath, BackendConfig.DEFAULT).use { engine ->
        val result = engine.transcribe(pcm, sampleRate = 16_000, options = TranscriptionOptions(language = "en"))
        println("${result.text} (${result.audioDurationMs} ms on ${result.backend})")
        result.segments.forEach { println("[${it.startMs}–${it.endMs} ms] ${it.text}") }
    }
    // snippet:end whisper-transcribe
}

internal suspend fun whisperStreaming(engine: WhisperEngine, microphone: Flow<ShortArray>) {
    // snippet:start whisper-stream
    // `microphone`: 16-bit PCM chunks straight from the platform recorder (AudioRecord, AVAudioEngine, javax.sound).
    engine.transcribeStream(microphone, sampleRate = 48_000, channels = 1, options = StreamingOptions.DEFAULT)
        .collect { event ->
            when (event) {
                is TranscriptionEvent.PartialHypothesis -> showDraft(event.text)  // may still change
                is TranscriptionEvent.CommittedSegment -> appendLine(event.text)  // final
                is TranscriptionEvent.Completed -> println("${event.segmentCount} segments")
                is TranscriptionEvent.Error -> throw event.exception
            }
        }
    // snippet:end whisper-stream
}

internal suspend fun whisperBackground(modelPath: String, appScope: CoroutineScope) {
    // snippet:start whisper-ios-background
    // Background-safe from the start: whisper.cpp is loaded with use_gpu = false (Accelerate / NEON only).
    val backgroundEngine = WhisperEngine(modelPath, BackendConfig.CPU)
    check(!backgroundEngine.usesGpu)

    // Or use Metal / Core ML in the foreground and drop to the CPU while the app is in the background.
    val engine = WhisperEngine(modelPath, BackendConfig(backend = ComputeBackend.NPU_COREML))
    val fallback = WhisperBackgroundFallback(engine, appScope)
    // Call these from your lifecycle callbacks. On iOS, `engine.observeAppLifecycle(appScope)` wires them to the
    // UIApplication notifications for you.
    fallback.onEnterBackground()  // aborts in-flight GPU work, reloads with ComputeBackend.CPU_ONLY
    fallback.onEnterForeground()  // restores the previous BackendConfig
    // snippet:end whisper-ios-background
    fallback.close()
    engine.close()
    backgroundEngine.close()
}

private fun showDraft(text: String) = println("… $text")

private fun appendLine(text: String) = println(text)
