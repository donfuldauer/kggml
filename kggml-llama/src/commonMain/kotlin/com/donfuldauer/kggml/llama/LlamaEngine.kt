package com.donfuldauer.kggml.llama

import com.donfuldauer.kggml.core.BackendConfig
import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.InferenceCancelledException
import com.donfuldauer.kggml.core.InferenceEngine
import com.donfuldauer.kggml.core.InferenceException
import com.donfuldauer.kggml.core.InvalidInputException
import com.donfuldauer.kggml.core.SerialDispatcherFactory
import com.donfuldauer.kggml.core.SerialExecutor
import com.donfuldauer.kggml.core.StopReason
import com.donfuldauer.kggml.core.TokenEvent
import com.donfuldauer.kggml.runtime.KggmlRuntime
import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.AtomicLong
import kotlin.concurrent.atomics.AtomicReference
import kotlinx.coroutines.InternalCoroutinesApi
import kotlinx.coroutines.channels.ProducerScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.job

/**
 * Loads a GGUF model with llama.cpp and returns a ready [LlamaEngine].
 *
 * The model and its KV-cache context are created on the new engine's own dedicated thread. [backendConfig] is
 * enforced at load time:
 *
 * - [ComputeBackend.CPU_ONLY] hands llama.cpp an *empty* device list, so no GPU backend is ever created: weights,
 *   KV cache and compute stay on the CPU (Accelerate / NEON on Apple, legal in the iOS background).
 * - [ComputeBackend.GPU_METAL] / [ComputeBackend.GPU_VULKAN] offload [BackendConfig.gpuLayers] layers
 *   ([BackendConfig.ALL_GPU_LAYERS] = all) to the [LlamaEngineOptions.gpuDevice]-th device of that kind.
 * - [ComputeBackend.NPU_COREML] is not supported by llama.cpp; with [LlamaEngineOptions.allowBackendFallback] it
 *   falls back to Metal / CPU, otherwise loading fails with
 *   [com.donfuldauer.kggml.core.BackendUnavailableException].
 * - [ComputeBackend.AUTO] picks the first available of Metal, Vulkan, CPU.
 *
 * [BackendConfig.cpuThreads] sets the generation and prompt threads ([BackendConfig.AUTO_THREADS] = performance
 * cores), [BackendConfig.useFlashAttention] llama.cpp's flash attention, and [BackendConfig.useMmap] whether the
 * GGUF file is memory-mapped.
 *
 * @throws com.donfuldauer.kggml.core.ModelLoadException if the file is missing or not a loadable GGUF model.
 * @throws com.donfuldauer.kggml.core.BackendUnavailableException if the backend is unavailable and fallback is off.
 * @throws com.donfuldauer.kggml.core.OutOfMemoryException if the weights or KV cache cannot be allocated.
 * @throws com.donfuldauer.kggml.core.NativeFailureException if the native libraries cannot be loaded.
 */
public suspend fun LlamaEngine(
    modelPath: String,
    backendConfig: BackendConfig = BackendConfig.DEFAULT,
    contextConfig: LlamaContextConfig = LlamaContextConfig.DEFAULT,
    options: LlamaEngineOptions = LlamaEngineOptions(),
): LlamaEngine = LlamaEngine.open(
    modelPath = modelPath,
    backendConfig = backendConfig,
    contextConfig = contextConfig,
    options = options,
    bridges = LlamaBridgeHolder.platform,
    availableBackends = { platformAvailableBackends() },
    dispatcherFactory = SerialDispatcherFactory.Default,
)

/** Backends llama.cpp can use on this device: the runtime's GPU backends (llama.cpp has no Core ML path). */
internal fun platformAvailableBackends(): Set<ComputeBackend> = KggmlRuntime.availableBackends() - ComputeBackend.NPU_COREML

/**
 * llama.cpp text-generation engine: streaming generation, KV-cache sessions, grammar / JSON-schema constrained
 * decoding and embeddings.
 *
 * Create instances with the [LlamaEngine] factory function. Engines are ordinary objects — never singletons — and
 * any number of them (with the same or different models and backends) can coexist and run concurrently: each owns
 * its native model, KV-cache context and a dedicated single-threaded dispatcher ([SerialExecutor]) that serializes
 * loading, every inference call and the native release in [close].
 *
 * **Session.** The engine keeps one conversation in its KV cache: the tokens of the last prompt and of everything
 * generated after it. A new [generate] reuses the longest common token prefix with that session
 * ([GenerationOptions.reusePromptCache]), so continuing a conversation only evaluates the new text.
 * [saveSession] / [restoreSession] serialize that KV cache to a `ByteArray` and back, on this or another engine
 * with the same model.
 *
 * **Cancellation** is cooperative and prompt: cancelling the collector of [generate] sets the native abort flag
 * that llama.cpp polls inside every ggml graph computation (`llama_set_abort_callback`), so a running decode stops
 * within one graph step. The engine stays usable afterwards.
 */
public class LlamaEngine internal constructor(
    /** Path of the loaded GGUF model. */
    public val modelPath: String,
    override val backendConfig: BackendConfig,
    /** The KV-cache configuration this engine was created with. */
    public val contextConfig: LlamaContextConfig,
    /** LLM-specific construction options. */
    public val options: LlamaEngineOptions,
    private val bridges: LlamaBridgeHolder,
    private val availableBackends: () -> Set<ComputeBackend>,
    dispatcherFactory: SerialDispatcherFactory,
) : InferenceEngine {

    private class Loaded(val backend: ComputeBackend, val usesGpu: Boolean, val info: LlamaModelInfo, val contextLength: Int)

    private val executor = SerialExecutor("kggml-llama-${nextInstanceId.addAndFetch(1)}", dispatcherFactory)
    private val loaded = AtomicReference<Loaded?>(null)
    private val closed = AtomicBoolean(false)
    private val abortGate = AbortGate()
    private val sessionSize = AtomicInt(0)

    /** [generate] collections between entering and leaving the engine thread (see [close]). */
    private val activeGenerations = AtomicInt(0)

    // Native handles and session state: only touched on the executor thread.
    private var model: Long = 0L
    private var context: Long = 0L
    private var embeddingContext: Long = 0L

    /**
     * Sampler of the generation in progress, freed exactly once (via `exchange`) by the generation itself or by
     * [freeAll] if the engine closes first: after close the generation may resume off the engine thread.
     */
    private val activeSampler = AtomicLong(0L)
    private var fingerprint = SessionCodec.Fingerprint(0, 0, 0)

    /** Tokens currently held in sequence 0 of the KV cache, positions `0 until session.size`. */
    private var session = IntArray(16)
    private var sessionLength = 0

    private fun loadedState(): Loaded = loaded.load() ?: error("LlamaEngine is not loaded")

    /** The backend llama.cpp actually runs on (never [ComputeBackend.AUTO]). */
    override val activeBackend: ComputeBackend
        get() = loadedState().backend

    /** `true` if at least one GPU device was handed to llama.cpp; always `false` for [ComputeBackend.CPU_ONLY]. */
    public val usesGpu: Boolean
        get() = loadedState().usesGpu

    /** Static properties of the loaded model. */
    public val modelInfo: LlamaModelInfo
        get() = loadedState().info

    /** Effective context window in tokens (llama.cpp may round [LlamaContextConfig.contextLength] up). */
    public val contextLength: Int
        get() = loadedState().contextLength

    /** Number of tokens currently held in the session's KV cache. */
    public val sessionTokenCount: Int
        get() = sessionSize.load()

    override val isClosed: Boolean
        get() = closed.load()

    /** `true` while a generation or embedding is executing native code on this engine. */
    public val isGenerating: Boolean
        get() = abortGate.isArmed

    /**
     * Streams the continuation of [prompt] as a cold `Flow` of [TokenEvent]s. Each collection runs one generation on
     * the engine thread (collections and other calls on the same engine are serialized).
     *
     * Events, in order:
     * - [TokenEvent.Token] — one per generated token, with its incrementally decoded text. Multi-byte UTF-8 characters
     *   split across tokens are held back until complete, so a token's text may be empty. End-of-generation tokens
     *   are not emitted.
     * - [TokenEvent.Completed] — the last event, with the [StopReason].
     * - [TokenEvent.Error] — a typed [InferenceException] (invalid prompt or grammar, native failure,
     *   [abortActiveGeneration], or [close] of the engine, reported as [InferenceCancelledException]). It is terminal:
     *   no event follows it and the flow then **completes normally**.
     *
     * Cancelling the collector aborts a running decode through llama.cpp's abort callback and throws
     * `CancellationException` as usual; the session keeps every token that was fully evaluated.
     *
     * @throws IllegalStateException immediately if the engine is closed.
     */
    public fun generate(prompt: String, options: GenerationOptions = GenerationOptions.DEFAULT): Flow<TokenEvent> {
        check(!isClosed) { "LlamaEngine is closed" }
        return channelFlow {
            // A generation suspends on the engine thread (in send()) while the collector is slow. close() must not shut
            // that thread down underneath it: on Kotlin/Native the terminated worker would never resume the producer
            // and the flow would hang. So the last generation to leave shuts the executor down instead (see close()).
            activeGenerations.addAndFetch(1)
            try {
                serial { runGeneration(prompt, options, this@channelFlow) }
            } finally {
                if (activeGenerations.addAndFetch(-1) == 0 && isClosed) shutDownExecutor()
            }
        }
    }

    /**
     * Runs [generate] to completion and returns the generated text.
     *
     * @throws InferenceException for the failure a [TokenEvent.Error] would report.
     * @throws IllegalStateException if the engine is closed.
     */
    public suspend fun generateText(prompt: String, options: GenerationOptions = GenerationOptions.DEFAULT): GenerationResult {
        val text = StringBuilder()
        val tokens = ArrayList<Int>()
        var reason: StopReason? = null
        generate(prompt, options).collect { event ->
            when (event) {
                is TokenEvent.Token -> {
                    text.append(event.text)
                    tokens += event.tokenId
                }
                is TokenEvent.Completed -> reason = event.stopReason
                is TokenEvent.Error -> throw event.exception
            }
        }
        return GenerationResult(text.toString(), tokens.toIntArray(), reason ?: error("generation ended without completion"))
    }

    /**
     * Tokenizes UTF-8 [text] with the model's vocabulary.
     *
     * @param addSpecial add BOS/EOS as the model requires.
     * @param parseSpecial parse control tokens written as text.
     */
    public suspend fun tokenize(text: String, addSpecial: Boolean = false, parseSpecial: Boolean = false): IntArray =
        serial { tokenizeOnThread(text, addSpecial, parseSpecial) }

    /** Concatenates the pieces of [tokens] and decodes them as UTF-8. */
    public suspend fun detokenize(tokens: IntArray, renderSpecial: Boolean = false): String = serial {
        val bridge = bridges.get()
        val bytes = ArrayList<ByteArray>(tokens.size)
        for (token in tokens) bytes += piece(bridge, token, renderSpecial)
        val out = ByteArray(bytes.sumOf { it.size })
        var p = 0
        for (b in bytes) {
            b.copyInto(out, p)
            p += b.size
        }
        out.decodeToString()
    }

    /**
     * Computes the embedding of [text] on a dedicated embedding context (created on first use, separate from the
     * generation session, so embeddings never disturb it). The vector is pooled with
     * [LlamaContextConfig.embeddingPooling] (for causal LMs usually the last token's hidden state).
     *
     * @param normalize L2-normalize the vector (unit length), e.g. for cosine similarity.
     * @throws InvalidInputException if [text] is empty or longer than [LlamaContextConfig.embeddingContextLength]
     *   tokens.
     */
    public suspend fun embed(text: String, normalize: Boolean = true): FloatArray = serial {
        val bridge = bridges.get()
        val tokens = tokenizeOnThread(text, addSpecial = true, parseSpecial = false)
        if (tokens.isEmpty()) throw InvalidInputException("text produced no tokens")
        val limit = contextConfig.embeddingContextLength
        if (tokens.size > limit) {
            throw InvalidInputException("text is ${tokens.size} tokens, longer than embeddingContextLength ($limit)")
        }
        val ctx = ensureEmbeddingContext(bridge)
        bridge.check(bridge.memoryClear(ctx), "kggml_llama_memory_clear")
        val status = abortable(bridge, ctx) { bridge.decode(ctx, tokens, 0, tokens.size, 0, 0, true) }
        throwOnFailure(bridge, status, "kggml_llama_decode")
        val result = bridge.embeddings(ctx, 0, normalize)
        bridge.check(result.status, "kggml_llama_get_embeddings")
        result.values
    }

    /**
     * Serializes the session (token history + KV cache of the conversation) after all queued work.
     *
     * @return an opaque blob for [restoreSession] on an engine with the same model and a context at least as large.
     */
    public suspend fun saveSession(): ByteArray = serial {
        val bridge = bridges.get()
        val state = bridge.stateSave(context, SEQ)
        bridge.check(state.status, "kggml_llama_state_save")
        SessionCodec.encode(fingerprint, session.copyOf(sessionLength), state.bytes)
    }

    /**
     * Replaces the session with one saved by [saveSession]. A following [generate] whose prompt starts with the
     * restored tokens only evaluates the remainder.
     *
     * @throws InvalidInputException if [session] is corrupt, belongs to a different model, or does not fit this
     *   engine's context. A container rejected before reaching llama.cpp leaves the current session untouched; if
     *   llama.cpp rejects the KV-cache payload, the session is empty afterwards.
     */
    public suspend fun restoreSession(session: ByteArray): Unit = serial {
        val bridge = bridges.get()
        val decoded = SessionCodec.decode(session)
        if (decoded.fingerprint != fingerprint) {
            throw InvalidInputException("session was saved with a different model (${decoded.fingerprint} != $fingerprint)")
        }
        if (decoded.tokens.size >= contextLength) {
            throw InvalidInputException("session holds ${decoded.tokens.size} tokens; the context fits $contextLength")
        }
        clearSessionOnThread(bridge)
        try {
            bridge.check(bridge.stateLoad(context, SEQ, decoded.nativeState), "kggml_llama_state_load")
            val posMax = bridge.memorySeqPosMax(context, SEQ)
            bridge.check(posMax.status, "kggml_llama_memory_seq_pos_max")
            if (posMax.value + 1 != decoded.tokens.size) {
                throw InvalidInputException("session KV cache holds ${posMax.value + 1} positions, token history ${decoded.tokens.size}")
            }
            setSession(decoded.tokens, decoded.tokens.size)
        } catch (failure: Throwable) {
            runCatching { clearSessionOnThread(bridge) }
            throw failure
        }
    }

    /** Empties the session (KV cache and token history). */
    public suspend fun clearSession(): Unit = serial { clearSessionOnThread(bridges.get()) }

    /**
     * Requests the native abort of the generation or embedding currently executing, if any; a [generate] flow then
     * ends with a terminal [TokenEvent.Error] ([InferenceCancelledException]). Safe from any thread. Queued calls
     * are not affected.
     *
     * @return `true` if a running call was signalled.
     */
    public fun abortActiveGeneration(): Boolean = abortGate.fire { bridges.get().requestAbort(it) }

    /**
     * Closes the engine: aborts a running generation, rejects queued and future calls with [IllegalStateException],
     * frees the sampler, contexts and model on the engine thread, and stops that thread. Idempotent and safe from any
     * thread.
     */
    override fun close() {
        if (!closed.compareAndSet(expectedValue = false, newValue = true)) return
        abortActiveGeneration()
        // With a generation in flight, its collector shuts the executor down when the generation leaves the engine
        // thread (it observes the abort or isClosed at its next native call or suspension point).
        if (activeGenerations.load() == 0) shutDownExecutor()
    }

    /** Frees every native object on the engine thread after queued work, then stops it. Idempotent. */
    private fun shutDownExecutor() = executor.close(release = ::freeAll)

    /**
     * Runs [block] on the engine thread, serialized with every other call. Calls still queued when the engine closes
     * fail with [IllegalStateException] instead of touching native state that is about to be released.
     */
    private suspend fun <T> serial(block: suspend () -> T): T = executor.execute {
        check(!isClosed) { "LlamaEngine is closed" }
        block()
    }

    // ------------------------------------------------------------------------------------------------------
    // Executor-thread internals
    // ------------------------------------------------------------------------------------------------------

    internal suspend fun start() {
        try {
            executor.execute { load() }
        } catch (failure: Throwable) {
            close()
            throw failure
        }
    }

    private fun load() {
        val bridge = bridges.get()
        var backend = ComputeBackend.resolve(backendConfig.backend, availableBackends(), options.allowBackendFallback)
        while (true) {
            val result = bridge.modelLoad(
                modelPath,
                RawModelParams(
                    backend = LlamaBackendKind.of(backend),
                    gpuLayers = backendConfig.effectiveGpuLayers(backend),
                    gpuDevice = options.gpuDevice,
                    useMmap = backendConfig.useMmap,
                    useMlock = options.useMlock,
                ),
            )
            if (result.status == LlamaStatus.OK) {
                adopt(bridge, result.handle, backend)
                return
            }
            val next = backend.fallback
            if (result.status != LlamaStatus.BACKEND_UNAVAILABLE || !options.allowBackendFallback || next == null) {
                throw bridge.nativeError(result.status, "kggml_llama_model_load", backend)
            }
            backend = next
        }
    }

    private fun adopt(bridge: NativeLlamaBridge, newModel: Long, requested: ComputeBackend) {
        model = newModel
        try {
            val raw = bridge.modelInfo(newModel)
            bridge.check(raw.status, "kggml_llama_model_get_info", requested)
            if (!raw.hasDecoder) throw InvalidInputException("model cannot generate text (no decoder)")
            val info = LlamaModelInfo(
                description = bridge.modelDescription(newModel),
                vocabularySize = raw.vocabularySize,
                trainContextLength = raw.trainContextLength,
                embeddingSize = raw.embeddingSize,
                layerCount = raw.layerCount,
                parameterCount = raw.parameterCount,
                sizeBytes = raw.sizeBytes,
                bosToken = raw.bos,
                eosToken = raw.eos,
                hasEncoder = raw.hasEncoder,
            )
            val created = bridge.contextCreate(newModel, contextParams(contextConfig.contextLength, contextConfig.batchSize, embeddings = false))
            bridge.check(created.status, "kggml_llama_context_create", requested)
            context = created.handle
            val nCtx = bridge.contextSize(context)
            bridge.check(nCtx.status, "kggml_llama_context_n_ctx", requested)
            fingerprint = SessionCodec.Fingerprint(raw.vocabularySize, raw.embeddingSize, raw.layerCount)
            val backend = LlamaBackendKind.toComputeBackend(raw.backend) ?: requested
            loaded.store(Loaded(backend, raw.usesGpu, info, nCtx.value))
        } catch (failure: Throwable) {
            freeAll()
            throw failure
        }
    }

    private fun contextParams(length: Int, batch: Int, embeddings: Boolean): RawContextParams {
        val threads = if (backendConfig.cpuThreads == BackendConfig.AUTO_THREADS) 0 else backendConfig.cpuThreads
        return RawContextParams(
            contextLength = length,
            batchSize = batch,
            microBatchSize = if (embeddings) batch else contextConfig.microBatchSize,
            maxSequences = 1,
            threads = threads,
            batchThreads = threads,
            flashAttention = if (backendConfig.useFlashAttention) 1 else 0,
            embeddings = embeddings,
            pooling = if (embeddings) contextConfig.embeddingPooling.nativeValue else LlamaPooling.MODEL_DEFAULT.nativeValue,
        )
    }

    private fun ensureEmbeddingContext(bridge: NativeLlamaBridge): Long {
        if (embeddingContext != 0L) return embeddingContext
        val length = contextConfig.embeddingContextLength
        // n_batch = n_ubatch = the whole window: pooled (non-causal) embeddings need the input in one micro-batch.
        val created = bridge.contextCreate(model, contextParams(length, length, embeddings = true))
        bridge.check(created.status, "kggml_llama_context_create (embeddings)", activeBackend)
        embeddingContext = created.handle
        return embeddingContext
    }

    private fun freeAll() {
        // No abort may reach a context once it is freed (a generation suspended in `send` keeps the gate armed).
        abortGate.disarm()
        val bridge = runCatching { bridges.get() }.getOrNull() ?: return
        activeSampler.exchange(0L).let { if (it != 0L) bridge.samplerFree(it) }
        if (embeddingContext != 0L) bridge.contextFree(embeddingContext)
        if (context != 0L) bridge.contextFree(context)
        if (model != 0L) bridge.modelFree(model)
        embeddingContext = 0L
        context = 0L
        model = 0L
        setSession(IntArray(0), 0)
    }

    private fun tokenizeOnThread(text: String, addSpecial: Boolean, parseSpecial: Boolean): IntArray {
        val bridge = bridges.get()
        val result = bridge.tokenize(model, text.encodeToByteArray(), addSpecial, parseSpecial)
        bridge.check(result.status, "kggml_llama_tokenize")
        return result.tokens
    }

    private fun piece(bridge: NativeLlamaBridge, token: Int, special: Boolean): ByteArray {
        val result = bridge.tokenToPiece(model, token, special)
        bridge.check(result.status, "kggml_llama_token_to_piece")
        return result.bytes
    }

    private fun setSession(tokens: IntArray, length: Int) {
        session = tokens
        sessionLength = length
        sessionSize.store(length)
    }

    private fun appendSession(tokens: IntArray, from: Int, count: Int) {
        if (sessionLength + count > session.size) session = session.copyOf(maxOf(session.size * 2, sessionLength + count))
        tokens.copyInto(session, sessionLength, from, from + count)
        sessionLength += count
        sessionSize.store(sessionLength)
    }

    /**
     * Drops KV-cache positions `>= length` and the matching token history. Also used after a failed decode, whose
     * partially evaluated batch sits in the KV cache beyond the tracked session.
     */
    private fun truncateSession(bridge: NativeLlamaBridge, length: Int) {
        bridge.check(bridge.memorySeqRemove(context, SEQ, length, -1), "kggml_llama_memory_seq_rm")
        if (length < sessionLength) {
            sessionLength = length
            sessionSize.store(length)
        }
    }

    private fun clearSessionOnThread(bridge: NativeLlamaBridge) {
        bridge.check(bridge.memoryClear(context), "kggml_llama_memory_clear")
        setSession(session, 0)
    }

    private suspend fun runGeneration(prompt: String, options: GenerationOptions, out: ProducerScope<TokenEvent>) {
        val bridge = bridges.get()
        try {
            val promptTokens = tokenizeOnThread(prompt, options.addSpecialTokens, options.parseSpecialTokens)
            if (promptTokens.isEmpty()) throw InvalidInputException("prompt produced no tokens")
            val nCtx = contextLength
            if (promptTokens.size >= nCtx) {
                throw InvalidInputException("prompt is ${promptTokens.size} tokens; the context fits $nCtx (incl. output)")
            }
            val created = bridge.samplerCreate(model, options.toRaw())
            bridge.check(created.status, "kggml_llama_sampler_create")
            activeSampler.store(created.handle)
            val sampler = created.handle

            abortable(bridge, context) {
                evaluatePrompt(bridge, promptTokens, options)
                if (options.repeatPenalty != 1f) {
                    val from = maxOf(0, promptTokens.size - options.repeatLastN)
                    for (i in from until promptTokens.size) bridge.check(bridge.samplerAccept(sampler, promptTokens[i], false), "kggml_llama_sampler_accept")
                }
                generateTokens(bridge, sampler, options, nCtx, out)
            }
        } catch (e: InferenceException) {
            out.send(TokenEvent.Error(e))
        } finally {
            activeSampler.exchange(0L).let { if (it != 0L) bridge.samplerFree(it) }
        }
    }

    /** Evaluates the part of [tokens] not already in the session, leaving the logits of the last token readable. */
    private suspend fun evaluatePrompt(bridge: NativeLlamaBridge, tokens: IntArray, options: GenerationOptions) {
        var keep = 0
        if (options.reusePromptCache) {
            val limit = minOf(sessionLength, tokens.size)
            while (keep < limit && session[keep] == tokens[keep]) keep++
        }
        // At least the last prompt token is re-evaluated: its logits are not part of the KV cache.
        if (keep == tokens.size) keep--
        if (keep == 0) clearSessionOnThread(bridge) else if (keep < sessionLength) truncateSession(bridge, keep)
        val status = bridge.decode(context, tokens, keep, tokens.size - keep, keep, SEQ, true)
        if (status != LlamaStatus.OK) {
            runCatching { truncateSession(bridge, keep) } // drop a partially evaluated batch
            throwOnFailure(bridge, status, "kggml_llama_decode")
        }
        appendSession(tokens, keep, tokens.size - keep)
    }

    private suspend fun generateTokens(
        bridge: NativeLlamaBridge,
        sampler: Long,
        options: GenerationOptions,
        nCtx: Int,
        out: ProducerScope<TokenEvent>,
    ) {
        val decoder = Utf8StreamDecoder()
        val text = StringBuilder()
        val maxStop = options.stopSequences.maxOfOrNull { it.length } ?: 0
        val one = IntArray(1)
        var produced = 0
        while (true) {
            currentCoroutineContext().ensureActive()
            if (options.maxTokens != GenerationOptions.UNLIMITED && produced >= options.maxTokens) {
                return out.send(TokenEvent.Completed(produced, StopReason.MAX_TOKENS))
            }
            val sampled = bridge.samplerSample(sampler, context)
            bridge.check(sampled.status, "kggml_llama_sampler_sample")
            val token = sampled.value
            if (bridge.tokenIsEog(model, token)) return out.send(TokenEvent.Completed(produced, StopReason.END_OF_GENERATION))

            val pieceText = decoder.decode(piece(bridge, token, options.renderSpecialTokens))
            out.send(TokenEvent.Token(produced, token, pieceText))
            produced++
            // While `send` was suspended the engine thread was free: close() may have released every native handle.
            if (isClosed) throw InferenceCancelledException("LlamaEngine was closed during generation")
            if (maxStop > 0) {
                val searchFrom = maxOf(0, text.length - maxStop + 1)
                text.append(pieceText)
                if (options.stopSequences.any { text.indexOf(it, searchFrom) >= 0 }) {
                    return out.send(TokenEvent.Completed(produced, StopReason.STOP_SEQUENCE))
                }
                if (text.length > STOP_WINDOW + maxStop) text.deleteRange(0, text.length - maxStop)
            }
            if (sessionLength >= nCtx) return out.send(TokenEvent.Completed(produced, StopReason.CONTEXT_FULL))

            one[0] = token
            val status = bridge.decode(context, one, 0, 1, sessionLength, SEQ, true)
            if (status == LlamaStatus.BUFFER_TOO_SMALL) {
                runCatching { truncateSession(bridge, sessionLength) }
                return out.send(TokenEvent.Completed(produced, StopReason.CONTEXT_FULL))
            }
            if (status != LlamaStatus.OK) {
                runCatching { truncateSession(bridge, sessionLength) }
                throwOnFailure(bridge, status, "kggml_llama_decode")
            }
            appendSession(one, 0, 1)
        }
    }

    /**
     * Runs [call] on context [ctx] with the native abort wired to the calling coroutine's job: cancelling it (or
     * [abortActiveGeneration] / [close]) makes llama.cpp stop at its next abort poll. The abort latch is cleared once
     * on entry and stays armed for the whole call, so an abort landing between two decodes is not lost.
     */
    private suspend inline fun <T> abortable(bridge: NativeLlamaBridge, ctx: Long, call: () -> T): T {
        val job = currentCoroutineContext().job
        bridge.clearAbort(ctx)
        abortGate.arm(ctx)
        // onCancelling = true: the handler must run while this thread is still blocked in native code.
        @OptIn(InternalCoroutinesApi::class)
        val registration = job.invokeOnCompletion(onCancelling = true, invokeImmediately = true) { cause ->
            if (cause != null) abortActiveGeneration()
        }
        try {
            return call()
        } finally {
            registration.dispose()
            abortGate.disarm()
        }
    }

    /** Throws the right exception for a non-OK [status]. Must run right after the native call. */
    private suspend fun throwOnFailure(bridge: NativeLlamaBridge, status: Int, operation: String) {
        if (status == LlamaStatus.OK) return
        val error = bridge.nativeError(status, operation, activeBackend)
        if (status == LlamaStatus.CANCELLED) {
            currentCoroutineContext().ensureActive() // caller cancelled -> CancellationException
            if (isClosed) throw InferenceCancelledException("LlamaEngine was closed during generation", error)
        }
        throw error
    }

    public companion object {
        private const val SEQ = 0
        private const val STOP_WINDOW = 4096
        private val nextInstanceId = AtomicInt(0)

        /** Version information of the loaded native llama library. */
        public fun nativeInfo(): LlamaNativeInfo = nativeInfo(LlamaBridgeHolder.platform.get())

        internal fun nativeInfo(bridge: NativeLlamaBridge): LlamaNativeInfo = LlamaNativeInfo(
            moduleVersion = bridge.moduleVersion(),
            llamaCppVersion = bridge.upstreamVersion(),
            llamaCppCommit = bridge.upstreamCommit(),
            systemInfo = bridge.systemInfo(),
        )

        internal suspend fun open(
            modelPath: String,
            backendConfig: BackendConfig,
            contextConfig: LlamaContextConfig,
            options: LlamaEngineOptions,
            bridges: LlamaBridgeHolder,
            availableBackends: () -> Set<ComputeBackend>,
            dispatcherFactory: SerialDispatcherFactory,
        ): LlamaEngine {
            require(modelPath.isNotEmpty()) { "modelPath must not be empty" }
            val engine = LlamaEngine(modelPath, backendConfig, contextConfig, options, bridges, availableBackends, dispatcherFactory)
            engine.start()
            return engine
        }
    }
}
