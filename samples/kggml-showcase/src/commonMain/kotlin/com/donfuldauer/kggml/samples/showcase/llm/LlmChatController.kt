package com.donfuldauer.kggml.samples.showcase.llm

import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.ModelLoadException
import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.StopReason
import com.donfuldauer.kggml.core.TokenEvent
import com.donfuldauer.kggml.llama.GenerationOptions
import com.donfuldauer.kggml.llama.LlamaGrammar
import com.donfuldauer.kggml.samples.showcase.describe
import com.donfuldauer.kggml.samples.showcase.formatBytes
import com.donfuldauer.kggml.samples.showcase.models.ResidentModels
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.sqrt

/** What the chat screen is currently doing. */
enum class ChatActivity(val label: String) {
    IDLE("Idle"),
    LOADING_MODEL("Loading model…"),
    GENERATING("Generating…"),
    EXTRACTING("Extracting JSON…"),
    EMBEDDING("Computing embeddings…"),
    ANALYZING_IMAGE("Analyzing image…"),
    SNAPSHOT("Saving / restoring KV cache…"),
}

/** Properties of the loaded engine shown on screen. */
data class LoadedChatModel(
    val key: String,
    val backend: ComputeBackend,
    val usesGpu: Boolean,
    val supportsVision: Boolean,
    val contextLength: Int,
    val description: String,
    val config: BackendConfig,
)

/** A saved KV-cache session plus the conversation it belongs to. */
class SessionSnapshot(val id: Int, val modelKey: String, val messages: List<ChatMessage>, val bytes: ByteArray) {
    val label: String get() = "#$id · ${messages.size} message(s) · ${formatBytes(bytes.size.toLong())}"
}

/** An image attached to the VLM panel, held in memory only. */
class AttachedImage(val bytes: ByteArray, val source: String) {
    val label: String get() = "${imageFormatOf(bytes)} · ${formatBytes(bytes.size.toLong())} · $source"
}

/** View state of the LLM Chat, Structured Output & Vision screen. */
data class LlmChatState(
    val modelPath: String = "",
    val projectorPath: String = "",
    val template: PromptTemplate = PromptTemplate.CHATML,
    val systemPrompt: String = DEFAULT_SYSTEM_PROMPT,
    val backend: ComputeBackend = ComputeBackend.AUTO,
    /** [BackendConfig.AUTO_THREADS] or `1..`. */
    val cpuThreads: Int = BackendConfig.AUTO_THREADS,
    /** [BackendConfig.ALL_GPU_LAYERS] or `0..`. */
    val gpuLayers: Int = BackendConfig.ALL_GPU_LAYERS,
    val maxTokens: Int = 256,
    val loaded: LoadedChatModel? = null,
    val messages: List<ChatMessage> = emptyList(),
    /** Assistant text streaming in, `null` when no reply is being generated. */
    val streaming: String? = null,
    val draft: String = "",
    val snapshots: List<SessionSnapshot> = emptyList(),
    val schema: String = DEFAULT_SCHEMA,
    val extractionInput: String = DEFAULT_EXTRACTION_INPUT,
    val extractionOutput: String = "",
    val validation: JsonValidation = JsonValidation.Empty,
    val embedTextA: String = "The cat sat on the mat.",
    val embedTextB: String = "A kitten is resting on a rug.",
    val similarity: Float? = null,
    val embeddingDimensions: Int? = null,
    val imagePath: String = "",
    val image: AttachedImage? = null,
    val visionQuestion: String = "Describe this image in one sentence.",
    val visionAnswer: String = "",
    val activity: ChatActivity = ChatActivity.IDLE,
    val status: String? = null,
    val error: String? = null,
) {
    val isBusy: Boolean get() = activity != ChatActivity.IDLE
    val hasModel: Boolean get() = modelPath.isNotBlank()
    val canSend: Boolean get() = !isBusy && hasModel && draft.isNotBlank()
    val canExtract: Boolean get() = !isBusy && hasModel && schema.isNotBlank() && extractionInput.isNotBlank()
    val canEmbed: Boolean get() = !isBusy && hasModel && embedTextA.isNotBlank() && embedTextB.isNotBlank()
    val canAnalyzeImage: Boolean get() = !isBusy && hasModel && image != null && visionQuestion.isNotBlank()
    val canSaveSnapshot: Boolean get() = !isBusy && loaded != null && messages.isNotEmpty()
    val schemaError: String? get() = schemaProblem(schema)

    /** The configuration the next load uses. */
    val requestedConfig: BackendConfig
        get() = if (backend == ComputeBackend.CPU_ONLY) {
            BackendConfig(backend = backend, gpuLayers = 0, cpuThreads = cpuThreads)
        } else {
            BackendConfig(backend = backend, gpuLayers = gpuLayers, cpuThreads = cpuThreads)
        }

    /** The loaded engine was created with a different file or configuration than currently selected. */
    val needsReload: Boolean
        get() = loaded != null && loaded.key != ChatModelSpec(modelPath.trim(), projectorPath.trim().ifEmpty { null }, requestedConfig).key
}

const val DEFAULT_SYSTEM_PROMPT: String = "You are a concise, helpful assistant running fully on-device."

const val DEFAULT_SCHEMA: String = """{
  "type": "object",
  "properties": {
    "name": { "type": "string" },
    "age": { "type": "integer" },
    "city": { "type": "string" }
  },
  "required": ["name", "age", "city"]
}"""

const val DEFAULT_EXTRACTION_INPUT: String = "Maria is 34 years old and lives in Lisbon, where she works as an architect."

/**
 * State holder of the LLM screen: multi-turn streaming chat, compute controls, KV-cache snapshots, JSON-schema
 * constrained extraction with live validation, embedding similarity and VLM image analysis. All public methods must
 * be called on [scope]'s thread (the UI thread); model loading and file reads run on [ioDispatcher]. The engine is
 * leased from [residents] for every operation, so an OS low-memory signal can evict it while the screen is idle
 * (it is reloaded on the next use).
 */
class LlmChatController(
    private val scope: CoroutineScope,
    private val factory: ChatModelFactory,
    private val residents: ResidentModels,
    private val ioDispatcher: CoroutineDispatcher,
    private val imageSource: ImageSource = NoImageSource,
    private val readFile: (String) -> ByteArray = { error("file access unavailable") },
    initialModelPath: String = "",
    /** Sampler seed of chat replies; `null` (the app) draws a random one per reply, tests pin it. */
    private val chatSeed: Long? = null,
) : AutoCloseable {
    private val mutableState = MutableStateFlow(LlmChatState(modelPath = initialModelPath))
    val state: StateFlow<LlmChatState> = mutableState.asStateFlow()

    val canPickFromGallery: Boolean get() = imageSource.canPickFromGallery
    val canCapture: Boolean get() = imageSource.canCapture

    private var job: Job? = null
    private var nextSnapshotId = 1
    private var closed = false

    fun setModelPath(path: String) = mutableState.update { it.copy(modelPath = path) }
    fun setProjectorPath(path: String) = mutableState.update { it.copy(projectorPath = path) }
    fun setTemplate(template: PromptTemplate) = mutableState.update { it.copy(template = template) }
    fun setSystemPrompt(text: String) = mutableState.update { it.copy(systemPrompt = text) }
    fun setDraft(text: String) = mutableState.update { it.copy(draft = text) }
    fun setBackend(backend: ComputeBackend) = mutableState.update { it.copy(backend = backend) }

    /** Sets the CPU thread count ([BackendConfig.AUTO_THREADS] = automatic); takes effect on the next (re)load. */
    fun setCpuThreads(threads: Int) = mutableState.update {
        it.copy(cpuThreads = if (threads < 1) BackendConfig.AUTO_THREADS else threads.coerceAtMost(MAX_THREADS))
    }

    /** Sets the GPU-offloaded layers ([BackendConfig.ALL_GPU_LAYERS] = all); takes effect on the next (re)load. */
    fun setGpuLayers(layers: Int) = mutableState.update {
        it.copy(gpuLayers = if (layers < 0) BackendConfig.ALL_GPU_LAYERS else layers.coerceAtMost(MAX_GPU_LAYERS))
    }

    fun setMaxTokens(tokens: Int) = mutableState.update { it.copy(maxTokens = tokens.coerceIn(16, 4096)) }

    fun setSchema(schema: String) = mutableState.update { it.copy(schema = schema, validation = validateJson(it.extractionOutput, schema)) }
    fun setExtractionInput(text: String) = mutableState.update { it.copy(extractionInput = text) }

    /** Edits the extraction output by hand; it is validated live like streamed output. */
    fun setExtractionOutput(text: String) = mutableState.update { it.copy(extractionOutput = text, validation = validateJson(text, it.schema)) }

    fun setEmbedTextA(text: String) = mutableState.update { it.copy(embedTextA = text, similarity = null) }
    fun setEmbedTextB(text: String) = mutableState.update { it.copy(embedTextB = text, similarity = null) }
    fun setImagePath(path: String) = mutableState.update { it.copy(imagePath = path) }
    fun setVisionQuestion(text: String) = mutableState.update { it.copy(visionQuestion = text) }

    /**
     * Selects a model file (from the Model Manager or the model picker) with its own vision projector, if it has one.
     * A projector only works with the model it was made for (libmtmd rejects any other), so the previous one is never
     * carried over: without [projectorPath] the projector is cleared.
     */
    fun useModel(path: String, projectorPath: String? = null) = mutableState.update {
        it.copy(modelPath = path, projectorPath = projectorPath.orEmpty())
    }

    fun useProjector(path: String) = mutableState.update { it.copy(projectorPath = path) }

    /** Loads (or reloads with the current backend / threads / GPU layers) the model right away. */
    fun loadModel() {
        if (mutableState.value.isBusy || !mutableState.value.hasModel) return
        launchWork(ChatActivity.LOADING_MODEL) { withModel { } }
    }

    /** Sends the draft as a user turn and streams the assistant reply. */
    fun send() {
        val current = mutableState.value
        if (!current.canSend) return
        val text = current.draft.trim()
        mutableState.update { it.copy(draft = "", messages = it.messages + ChatMessage(ChatRole.USER, text), status = null) }
        launchWork(ChatActivity.GENERATING) {
            withModel { model ->
                val state = mutableState.value
                val prompt = state.template.format(state.systemPrompt, state.messages)
                val reply = StringBuilder()
                mutableState.update { it.copy(activity = ChatActivity.GENERATING, streaming = "") }
                try {
                    val result = model.generate(prompt, emptyList(), chatOptions(state)).streamInto(reply) { text ->
                        mutableState.update { it.copy(streaming = state.template.clean(text)) }
                    }
                    mutableState.update { it.copy(status = "${result.tokens} token(s), stopped: ${result.reason.label()}") }
                } finally {
                    val cleaned = state.template.clean(reply.toString())
                    mutableState.update {
                        it.copy(
                            streaming = null,
                            messages = if (cleaned.isEmpty()) it.messages else it.messages + ChatMessage(ChatRole.ASSISTANT, cleaned),
                        )
                    }
                }
            }
        }
    }

    /** Abandons the running operation (generation is aborted natively through the engine's abort callback). */
    fun cancel() {
        job?.cancel()
    }

    /** Clears the conversation and the engine's KV cache. */
    fun clearConversation() {
        if (mutableState.value.isBusy) return
        mutableState.update { it.copy(messages = emptyList(), streaming = null, status = "Conversation cleared", error = null) }
        if (mutableState.value.loaded == null) return
        launchWork(ChatActivity.SNAPSHOT) { withModel { it.clearSession() } }
    }

    /** Saves the KV cache of the conversation as an in-memory snapshot. */
    fun saveSnapshot() {
        if (!mutableState.value.canSaveSnapshot) return
        launchWork(ChatActivity.SNAPSHOT) {
            withModel { model ->
                val bytes = model.saveSession()
                val snapshot = SessionSnapshot(nextSnapshotId++, model.spec.key, mutableState.value.messages, bytes)
                mutableState.update { it.copy(snapshots = it.snapshots + snapshot, status = "Saved snapshot ${snapshot.label}") }
            }
        }
    }

    /** Restores snapshot [id]: the KV cache and the conversation it was taken from. */
    fun restoreSnapshot(id: Int) {
        val snapshot = mutableState.value.snapshots.firstOrNull { it.id == id } ?: return
        if (mutableState.value.isBusy) return
        launchWork(ChatActivity.SNAPSHOT) {
            withModel { model ->
                model.restoreSession(snapshot.bytes)
                mutableState.update { it.copy(messages = snapshot.messages, status = "Restored snapshot ${snapshot.label}") }
            }
        }
    }

    fun deleteSnapshot(id: Int) = mutableState.update { state -> state.copy(snapshots = state.snapshots.filterNot { it.id == id }) }

    /** Runs a JSON-schema constrained generation (GBNF compiled from the schema) over the extraction input. */
    fun extract() {
        val current = mutableState.value
        if (!current.canExtract) return
        current.schemaError?.let { problem ->
            mutableState.update { it.copy(error = problem) }
            return
        }
        val grammar = try {
            LlamaGrammar.jsonSchema(current.schema)
        } catch (e: IllegalArgumentException) {
            mutableState.update { it.copy(error = "Unsupported schema: ${e.message}") }
            return
        } catch (e: InvalidInputException) {
            // The schema-to-GBNF converter rejects valid JSON it cannot express (e.g. `pattern`) with this type.
            mutableState.update { it.copy(error = "Unsupported schema: ${e.message}") }
            return
        }
        mutableState.update { it.copy(extractionOutput = "", validation = JsonValidation.Empty) }
        launchWork(ChatActivity.EXTRACTING) {
            withModel { model ->
                mutableState.update { it.copy(activity = ChatActivity.EXTRACTING) }
                val instruction = "Extract the data described by this JSON schema from the text. Reply with JSON only.\n\n" +
                    "Schema:\n${current.schema}\n\nText:\n${current.extractionInput}"
                val prompt = current.template.format(EXTRACTION_SYSTEM_PROMPT, listOf(ChatMessage(ChatRole.USER, instruction)))
                val options = GenerationOptions(maxTokens = current.maxTokens, temperature = 0f, grammar = grammar)
                val output = StringBuilder()
                model.generate(prompt, emptyList(), options).streamInto(output) { text ->
                    mutableState.update { it.copy(extractionOutput = text, validation = validateJson(text, it.schema)) }
                }
                val validation = mutableState.value.validation
                mutableState.update { it.copy(status = "Extraction finished: ${validation.message}") }
            }
        }
    }

    /** Embeds both texts and shows their cosine similarity. */
    fun compareEmbeddings() {
        val current = mutableState.value
        if (!current.canEmbed) return
        launchWork(ChatActivity.EMBEDDING) {
            withModel { model ->
                mutableState.update { it.copy(activity = ChatActivity.EMBEDDING) }
                val a = model.embed(current.embedTextA)
                val b = model.embed(current.embedTextB)
                mutableState.update { it.copy(similarity = cosineSimilarity(a, b), embeddingDimensions = a.size) }
            }
        }
    }

    /** Reads the image at [LlmChatState.imagePath] into memory. */
    fun loadImageFromPath() {
        val path = mutableState.value.imagePath.trim()
        if (path.isEmpty() || mutableState.value.isBusy) return
        scope.launch {
            try {
                val bytes = withContext(ioDispatcher) { readFile(path) }
                attach(bytes, path.substringAfterLast('/').substringAfterLast('\\'))
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                mutableState.update { it.copy(error = "Cannot read image: ${t.describe()}") }
            }
        }
    }

    fun pickFromGallery() = acquireImage("Photo library") { imageSource.pickFromGallery() }

    fun captureFromCamera() = acquireImage("Camera") { imageSource.capture() }

    /** Attaches encoded image [bytes] (kept in memory only). */
    fun attachImage(bytes: ByteArray, source: String) = attach(bytes, source)

    fun clearImage() = mutableState.update { it.copy(image = null, visionAnswer = "") }

    /** Asks the VLM about the attached image (single turn; the image never touches the file system). */
    fun analyzeImage() {
        val current = mutableState.value
        val image = current.image ?: return
        if (!current.canAnalyzeImage) return
        mutableState.update { it.copy(visionAnswer = "") }
        launchWork(ChatActivity.ANALYZING_IMAGE) {
            withModel { model ->
                if (!model.supportsVision) {
                    throw IllegalStateException("This model has no vision projector: set the mmproj path and reload")
                }
                mutableState.update { it.copy(activity = ChatActivity.ANALYZING_IMAGE) }
                val prompt = current.template.format("", listOf(ChatMessage(ChatRole.USER, current.visionQuestion)), imageCount = 1)
                val answer = StringBuilder()
                val result = model.generate(prompt, listOf(image.bytes), chatOptions(current)).streamInto(answer) { text ->
                    mutableState.update { it.copy(visionAnswer = current.template.clean(text)) }
                }
                mutableState.update {
                    it.copy(visionAnswer = current.template.clean(answer.toString()), status = "Image analyzed: ${result.tokens} token(s)")
                }
            }
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        job?.cancel()
        mutableState.value.loaded?.let { residents.remove(it.key) }
    }

    // ----------------------------------------------------------------------------------------------------------

    private fun attach(bytes: ByteArray, source: String) {
        if (bytes.isEmpty()) {
            mutableState.update { it.copy(error = "The image is empty") }
            return
        }
        mutableState.update { it.copy(image = AttachedImage(bytes, source), visionAnswer = "", error = null) }
    }

    private fun acquireImage(source: String, pick: suspend () -> ByteArray?) {
        if (mutableState.value.isBusy) return
        scope.launch {
            try {
                pick()?.let { attach(it, source) }
            } catch (e: CancellationException) {
                throw e
            } catch (t: Throwable) {
                mutableState.update { it.copy(error = "$source unavailable: ${t.describe()}") }
            }
        }
    }

    private fun chatOptions(state: LlmChatState) = GenerationOptions(
        maxTokens = state.maxTokens,
        temperature = 0.7f,
        seed = chatSeed,
        stopSequences = state.template.stopSequences,
    )

    private class StreamResult(val tokens: Int, val reason: StopReason?)

    /** Collects a token stream into [sink], reporting the accumulated text; an in-band error is rethrown. */
    private suspend fun kotlinx.coroutines.flow.Flow<TokenEvent>.streamInto(
        sink: StringBuilder,
        onText: (String) -> Unit,
    ): StreamResult {
        var tokens = 0
        var reason: StopReason? = null
        collect { event ->
            when (event) {
                is TokenEvent.Token -> {
                    tokens++
                    sink.append(event.text)
                    onText(sink.toString())
                }
                is TokenEvent.Completed -> reason = event.stopReason
                is TokenEvent.Error -> throw event.exception
            }
        }
        return StreamResult(tokens, reason)
    }

    private fun launchWork(initial: ChatActivity, block: suspend () -> Unit) {
        if (closed) return
        mutableState.update { it.copy(activity = initial, error = null) }
        job = scope.launch {
            try {
                block()
            } catch (e: CancellationException) {
                mutableState.update { it.copy(status = "Cancelled") }
                throw e
            } catch (t: Throwable) {
                mutableState.update { it.copy(error = t.describe() + projectorHint(t, it.projectorPath)) }
            } finally {
                mutableState.update { it.copy(activity = ChatActivity.IDLE) }
            }
        }
    }

    /**
     * Leases the engine for the current file and configuration, loading it on [ioDispatcher] when it is not resident
     * (first use, a changed configuration, or after an eviction), and runs [block] with it.
     */
    private suspend fun withModel(block: suspend (ChatModel) -> Unit) {
        val state = mutableState.value
        val spec = ChatModelSpec(state.modelPath.trim(), state.projectorPath.trim().ifEmpty { null }, state.requestedConfig)
        val previous = state.loaded
        if (previous != null && previous.key != spec.key) {
            // Configuration changed (backend / threads / GPU layers / files): close the old engine before loading.
            residents.remove(previous.key)
            mutableState.update { it.copy(loaded = null) }
        }
        var reloaded = false
        val lease = residents.lease(spec.key) {
            reloaded = true
            mutableState.update { it.copy(activity = ChatActivity.LOADING_MODEL) }
            var opened: ChatModel? = null
            val model = try {
                withContext(ioDispatcher) { factory.open(spec).also { opened = it } }
            } catch (e: CancellationException) {
                opened?.close()
                throw e
            }
            if (closed) {
                model.close()
                throw CancellationException("LLM screen closed")
            }
            model to model.estimatedBytes
        }
        try {
            val model = lease.resource
            mutableState.update {
                it.copy(
                    loaded = LoadedChatModel(spec.key, model.activeBackend, model.usesGpu, model.supportsVision, model.contextLength, model.description, spec.config),
                    status = if (reloaded) {
                        if (previous?.key == spec.key) "Model was evicted from memory and has been reloaded" else "Loaded ${model.description}"
                    } else {
                        it.status
                    },
                )
            }
            block(model)
        } finally {
            lease.close()
            residents.refresh()
        }
    }

    companion object {
        const val MAX_THREADS: Int = 32
        const val MAX_GPU_LAYERS: Int = 200
        private const val EXTRACTION_SYSTEM_PROMPT = "You convert text into JSON that matches a given schema."
    }
}

/** Cosine similarity of two embeddings (`0` when either is all zeros). */
fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
    require(a.size == b.size) { "embedding sizes differ: ${a.size} != ${b.size}" }
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for (i in a.indices) {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    }
    if (na == 0.0 || nb == 0.0) return 0f
    return (dot / (sqrt(na) * sqrt(nb))).toFloat()
}

private fun StopReason?.label(): String = when (this) {
    StopReason.END_OF_GENERATION -> "end of generation"
    StopReason.MAX_TOKENS -> "max tokens"
    StopReason.STOP_SEQUENCE -> "stop sequence"
    StopReason.CONTEXT_FULL -> "context full"
    null -> "unknown"
}

/**
 * What to do about a vision projector that did not load: libmtmd refuses a projector made for another model (its
 * embedding width differs from the text model's), and the native message cannot tell that apart from a broken file.
 */
internal fun projectorHint(failure: Throwable, projectorPath: String): String =
    if (failure is ModelLoadException && projectorPath.isNotBlank() && failure.message.orEmpty().contains("projector")) {
        "\n\nA vision projector (mmproj) only works with the model it was made for, e.g. SmolVLM's projector with the " +
            "SmolVLM model. Pick the matching model (the picker fills in its projector), or clear the projector for " +
            "text-only chat. If they do match, the projector file may be incomplete: delete it and download it again."
    } else {
        ""
    }
