package com.donfuldauer.kggml.samples.showcase.llm

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceException
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.StopReason
import com.donfuldauer.kggml.core.TokenEvent
import com.donfuldauer.kggml.llama.GenerationOptions
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlin.math.sqrt

/** One recorded [FakeChatModel.generate] call. */
class GenerateCall(val prompt: String, val images: List<ByteArray>, val options: GenerationOptions)

/**
 * Deterministic LLM. Each generation streams [reply] (by default `"Echo: <last user line>"`) as one token per
 * word. With [tokenGate] set, every token waits for an element from the channel, so a test can observe streaming
 * and cancel mid-generation. JSON-schema (grammar) generations stream [jsonReply].
 */
class FakeChatModel(
    override val spec: ChatModelSpec,
    var reply: (String) -> String = { prompt -> "Echo: " + lastUserLine(prompt) },
    var jsonReply: String = """{"name": "Maria", "age": 34, "city": "Lisbon"}""",
    var streamError: InferenceException? = null,
    var tokenGate: Channel<Unit>? = null,
    /** Stop reason of the final [TokenEvent.Completed]; `null` ends the stream without one. */
    var stopReason: StopReason? = StopReason.END_OF_GENERATION,
) : ChatModel {
    val calls = mutableListOf<GenerateCall>()
    val restored = mutableListOf<ByteArray>()
    var sessionClears = 0
        private set
    var closed = false
        private set

    override val activeBackend: ComputeBackend
        get() = if (spec.config.backend == ComputeBackend.CPU_ONLY) ComputeBackend.CPU_ONLY else ComputeBackend.GPU_METAL
    override val usesGpu: Boolean get() = activeBackend != ComputeBackend.CPU_ONLY
    override val supportsVision: Boolean get() = spec.projectorPath != null
    override val contextLength: Int = 4096
    override val description: String = "fake-llm 0.5B Q4_K_M"
    override val estimatedBytes: Long = 1000

    override fun generate(prompt: String, images: List<ByteArray>, options: GenerationOptions): Flow<TokenEvent> = flow {
        calls += GenerateCall(prompt, images, options)
        if (images.isNotEmpty() && !supportsVision) {
            emit(TokenEvent.Error(InvalidInputException("no projector")))
            return@flow
        }
        val text = when {
            options.grammar != null -> jsonReply
            images.isNotEmpty() -> "An image of ${images.size} picture(s), ${images.first().size} bytes."
            else -> reply(prompt)
        }
        val pieces = text.split(" ").mapIndexed { i, word -> if (i == 0) word else " $word" }
        pieces.forEachIndexed { i, piece ->
            tokenGate?.receive()
            emit(TokenEvent.Token(i, 100 + i, piece))
            if (i == 0) streamError?.let {
                emit(TokenEvent.Error(it))
                return@flow
            }
        }
        stopReason?.let { emit(TokenEvent.Completed(pieces.size, it)) }
    }

    override suspend fun saveSession(): ByteArray = "session:${calls.size}".encodeToByteArray()

    override suspend fun restoreSession(session: ByteArray) {
        restored += session
    }

    override suspend fun clearSession() {
        sessionClears++
    }

    /** A bag-of-letters embedding: texts sharing letters are similar. */
    override suspend fun embed(text: String): FloatArray {
        val v = FloatArray(26)
        text.lowercase().forEach { c -> if (c in 'a'..'z') v[c - 'a'] += 1f }
        val norm = sqrt(v.sumOf { (it * it).toDouble() }).toFloat().takeIf { it > 0 } ?: 1f
        return FloatArray(26) { v[it] / norm }
    }

    override fun close() {
        closed = true
    }

    companion object {
        /** The last user turn of a ChatML prompt (or the whole prompt). */
        fun lastUserLine(prompt: String): String =
            prompt.substringAfterLast("<|im_start|>user\n", prompt).substringBefore("<|im_end|>").trim()
    }
}

/** Records every [open] and hands out [FakeChatModel]s (or throws [failure]). */
class FakeChatModelFactory(
    var failure: Throwable? = null,
    private val configure: (FakeChatModel) -> Unit = {},
) : ChatModelFactory {
    val opened = mutableListOf<FakeChatModel>()

    override suspend fun open(spec: ChatModelSpec): ChatModel {
        failure?.let { throw it }
        return FakeChatModel(spec).also(configure).also { opened += it }
    }
}

/** [ImageSource] fake returning fixed bytes. */
class FakeImageSource(
    var galleryImage: ByteArray? = PNG_MAGIC + ByteArray(12),
    var cameraImage: ByteArray? = JPEG_MAGIC + ByteArray(20),
) : ImageSource {
    override val canPickFromGallery: Boolean = true
    override val canCapture: Boolean = true
    override suspend fun pickFromGallery(): ByteArray? = galleryImage
    override suspend fun capture(): ByteArray? = cameraImage

    companion object {
        val PNG_MAGIC = byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)
        val JPEG_MAGIC = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 0xFF.toByte(), 0xE0.toByte())
    }
}
