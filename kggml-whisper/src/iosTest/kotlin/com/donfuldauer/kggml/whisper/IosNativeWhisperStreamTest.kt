package com.donfuldauer.kggml.whisper

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.TranscriptionEvent
import com.donfuldauer.kggml.core.TranscriptionEvent.Completed
import com.donfuldauer.kggml.core.TranscriptionEvent.CommittedSegment
import kotlin.concurrent.atomics.AtomicReference
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.convert
import kotlinx.cinterop.usePinned
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import platform.Foundation.NSData
import platform.Foundation.NSFileManager
import platform.Foundation.NSProcessInfo
import platform.Foundation.NSTemporaryDirectory
import platform.Foundation.dataWithBytes
import platform.Foundation.writeToFile

/**
 * [WhisperEngine.transcribeStream] on the iOS simulator: the cinterop path (`kggml_whisper_transcribe_with_state`
 * with a `staticCFunction` new-segment callback and a `StableRef` listener, pinned prompt tokens, the native
 * streaming resampler) against KggmlWhisper.framework and the synthetic [TinyWhisperModel].
 */
@OptIn(ExperimentalForeignApi::class)
class IosNativeWhisperStreamTest {
    private lateinit var modelPath: String

    @BeforeTest
    fun writeModel() {
        val bytes = TinyWhisperModel.bytes()
        modelPath = NSTemporaryDirectory() + "kggml-whisper-tiny-stream-" + NSProcessInfo.processInfo.globallyUniqueString + ".bin"
        val written = bytes.usePinned { NSData.dataWithBytes(it.addressOf(0), bytes.size.convert()).writeToFile(modelPath, true) }
        assertTrue(written, "could not write $modelPath")
    }

    @AfterTest
    fun deleteModel() {
        NSFileManager.defaultManager.removeItemAtPath(modelPath, null)
    }

    private fun speech(ms: Int) = FloatArray(ms * 16) { 0.1f * ((it % 100) / 50f - 1f) }
    private fun silence(ms: Int) = FloatArray(ms * 16)

    private fun chunked(vararg parts: FloatArray): Flow<FloatArray> = flow {
        for (p in parts) for (i in p.indices step 640) emit(p.copyOfRange(i, minOf(p.size, i + 640)))
    }

    private fun tempFiles(): Set<String> =
        NSFileManager.defaultManager.contentsOfDirectoryAtPath(NSTemporaryDirectory(), null).orEmpty().map { it.toString() }.toSet()

    @Test
    fun streamsCommittedSegmentsInOrderWithoutTemporaryFiles() = runBlocking {
        val before = tempFiles()
        WhisperEngine(modelPath, BackendConfig.CPU).use { engine ->
            val parts = buildList {
                repeat(4) {
                    add(speech(2_400))
                    add(silence(900))
                }
                add(speech(31_200)) // split at the 30 s window: prompt carry-over across windows
                add(silence(990))
            }.toTypedArray()
            val events = withTimeout(120.seconds) { engine.transcribeStream(chunked(*parts)).toList() }
            assertTrue(events.none { it is TranscriptionEvent.Error }, "$events")
            val completed = assertIs<Completed>(events.last())
            assertEquals(parts.sumOf { it.size } / 16L, completed.audioDurationMs)
            val committed = events.filterIsInstance<CommittedSegment>()
            assertEquals(completed.segmentCount, committed.size)
            assertEquals(committed.indices.toList(), committed.map { it.index })
            committed.zipWithNext().forEach { (a, b) -> assertTrue(a.endMs <= b.startMs, "$a then $b") }

            // int16 at 48 kHz stereo through the native streaming resampler.
            val pcm = ShortArray(48 * 2 * 2_000) { ((it / 2 % 300) - 150).toShort() }
            val events16 = engine.transcribeStream(flow { emit(pcm); emit(ShortArray(48 * 2 * 900)) }, sampleRate = 48_000, channels = 2).toList()
            assertEquals(2_900L, assertIs<Completed>(events16.last()).audioDurationMs)
        }
        assertEquals(emptySet(), tempFiles() - before, "no temporary audio files")
    }

    @Test
    fun cancellationAndCloseEndStreamsCleanly() = runBlocking {
        val engine = WhisperEngine(modelPath, BackendConfig.CPU.copy(cpuThreads = 1))
        val endless = flow { while (true) emit(speech(100)) }
        val first = AtomicReference<TranscriptionEvent?>(null)
        val job = launch(Dispatchers.Default) { engine.transcribeStream(endless).collect { first.compareAndSet(null, it) } }
        withTimeout(60.seconds) { while (first.load() == null) delay(1) }
        withTimeout(30.seconds) { job.cancelAndJoin() }

        // The engine is still usable; closing it ends a live stream with InferenceCancelledException.
        val last = AtomicReference<TranscriptionEvent?>(null)
        val second = launch(Dispatchers.Default) { engine.transcribeStream(endless).collect { last.store(it) } }
        withTimeout(60.seconds) { while (last.load() == null) delay(1) }
        engine.close()
        withTimeout(30.seconds) { second.join() }
        assertIs<InferenceCancelledException>(assertIs<TranscriptionEvent.Error>(last.load()).exception)
    }
}
