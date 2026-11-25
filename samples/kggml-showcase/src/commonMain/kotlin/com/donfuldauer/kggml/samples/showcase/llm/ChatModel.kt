package com.donfuldauer.kggml.samples.showcase.llm

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.TokenEvent
import com.donfuldauer.kggml.llama.GenerationOptions
import com.donfuldauer.kggml.llama.LlamaEngine
import com.donfuldauer.kggml.llama.LlamaEngineOptions
import com.donfuldauer.kggml.llama.LlamaImage
import com.donfuldauer.kggml.models.ModelInspector
import com.donfuldauer.kggml.models.defaultModelFileSystem
import kotlinx.coroutines.flow.Flow

/** What to load: a GGUF model, an optional libmtmd projector, and the compute configuration. */
data class ChatModelSpec(val modelPath: String, val projectorPath: String?, val config: BackendConfig) {
    /** Resident-cache key: a different file or configuration is a different engine. */
    val key: String
        get() = "llm:$modelPath|${projectorPath.orEmpty()}|${config.backend}|gpu=${config.gpuLayers}|threads=${config.cpuThreads}"
}

/** A loaded LLM used by the chat screen ([LlamaChatModel] in the app, fakes in tests). */
interface ChatModel : AutoCloseable {
    val spec: ChatModelSpec

    /** The backend the model actually runs on (never [ComputeBackend.AUTO]). */
    val activeBackend: ComputeBackend
    val usesGpu: Boolean
    val supportsVision: Boolean
    val contextLength: Int
    val description: String

    /** Estimated resident bytes (header-based peak RAM estimate, or the file size). */
    val estimatedBytes: Long

    /** Streams the continuation of [prompt], with in-memory encoded [images] (PNG / JPEG bytes) for VLMs. */
    fun generate(prompt: String, images: List<ByteArray>, options: GenerationOptions): Flow<TokenEvent>

    /** Serializes the KV cache + token history. */
    suspend fun saveSession(): ByteArray

    /** Replaces the KV cache with a [saveSession] blob. */
    suspend fun restoreSession(session: ByteArray)

    suspend fun clearSession()

    /** L2-normalized embedding of [text]. */
    suspend fun embed(text: String): FloatArray
}

/** Opens [ChatModel]s; may block on native model loading (called on the I/O dispatcher). */
fun interface ChatModelFactory {
    suspend fun open(spec: ChatModelSpec): ChatModel
}

/** Production [ChatModelFactory] backed by `:kggml-llama`. */
object LlamaChatModelFactory : ChatModelFactory {
    override suspend fun open(spec: ChatModelSpec): ChatModel {
        val engine = LlamaEngine(
            modelPath = spec.modelPath,
            backendConfig = spec.config,
            options = LlamaEngineOptions(projectorPath = spec.projectorPath?.takeIf { it.isNotBlank() }),
        )
        val estimate = runCatching {
            ModelInspector.inspectFile(spec.modelPath).estimatedPeakRam.totalBytes
        }.getOrElse { defaultModelFileSystem().size(spec.modelPath).coerceAtLeast(0) }
        return LlamaChatModel(engine, spec, estimate)
    }
}

/** [ChatModel] over a [LlamaEngine]. Images are passed as [LlamaImage.Encoded] (decoded in memory by libmtmd). */
class LlamaChatModel(
    private val engine: LlamaEngine,
    override val spec: ChatModelSpec,
    override val estimatedBytes: Long,
) : ChatModel {
    override val activeBackend: ComputeBackend get() = engine.activeBackend
    override val usesGpu: Boolean get() = engine.usesGpu
    override val supportsVision: Boolean get() = engine.supportsVision
    override val contextLength: Int get() = engine.contextLength
    override val description: String get() = engine.modelInfo.description

    override fun generate(prompt: String, images: List<ByteArray>, options: GenerationOptions): Flow<TokenEvent> =
        if (images.isEmpty()) engine.generate(prompt, options) else engine.generate(prompt, images.map { LlamaImage.Encoded(it) }, options)

    override suspend fun saveSession(): ByteArray = engine.saveSession()

    override suspend fun restoreSession(session: ByteArray) = engine.restoreSession(session)

    override suspend fun clearSession() = engine.clearSession()

    override suspend fun embed(text: String): FloatArray = engine.embed(text, normalize = true)

    override fun close() = engine.close()
}
