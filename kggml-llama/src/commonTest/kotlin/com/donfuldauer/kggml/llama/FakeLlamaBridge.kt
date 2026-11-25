package com.donfuldauer.kggml.llama

import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.AtomicReference

/** Lock-free append-only log readable from any thread. */
internal class ConcurrentLog<T> {
    private val ref = AtomicReference<List<T>>(emptyList())

    fun add(item: T) {
        while (true) {
            val current = ref.load()
            if (ref.compareAndSet(current, current + item)) return
        }
    }

    val items: List<T> get() = ref.load()
}

internal data class DecodeCall(val context: Long, val tokens: List<Int>, val pos0: Int, val seqId: Int, val logitsLast: Boolean)

/**
 * Deterministic in-memory stand-in for the llama.cpp ABI.
 *
 * - Vocabulary: `BOS` = 1, `EOS` = 2, and token `b + BYTE_BASE` is the single byte `b` (so any UTF-8 text round-trips).
 * - Each context tracks the KV-cache length of sequence 0 and *validates* that decodes continue it contiguously,
 *   which checks the engine's session bookkeeping.
 * - Samplers emit [script] in order, then [EOS].
 * - Every handle is tracked so tests can assert that everything is freed exactly once, samplers / contexts first.
 */
internal class FakeLlamaBridge(
    var script: List<Int> = emptyList(),
    val contextSize: Int = 64,
    val vocab: Int = 300,
    val embd: Int = 4,
    val layers: Int = 1,
    /** Status returned by modelLoad per requested backend kind (default OK). */
    val loadStatus: (backendKind: Int) -> Int = { LlamaStatus.OK },
    /** Status returned by samplerCreate (e.g. INVALID_ARGUMENT for a bad grammar). */
    var samplerStatus: Int = LlamaStatus.OK,
    /** Invoked inside every decode before it completes (may block, e.g. until an abort is requested). */
    var onDecode: (FakeLlamaBridge, Long, Int) -> Int = { _, _, _ -> LlamaStatus.OK },
) : NativeLlamaBridge {
    private val nextHandle = AtomicInt(1000)
    private val liveRef = AtomicReference<Map<Long, String>>(emptyMap())
    private val abortRef = AtomicReference<Set<Long>>(emptySet())
    private val kv = HashMap<Long, Int>()
    private val samplerPos = HashMap<Long, Int>()

    val loads = ConcurrentLog<Pair<String, RawModelParams>>()
    val contexts = ConcurrentLog<RawContextParams>()
    val samplers = ConcurrentLog<RawSamplerParams>()
    val decodes = ConcurrentLog<DecodeCall>()
    val frees = ConcurrentLog<String>()
    val accepted = ConcurrentLog<Pair<Int, Boolean>>()
    val clears = AtomicInt(0)
    var lastErrorText = ""

    /** Live handles by kind ("model", "context", "sampler"). */
    val live: Map<Long, String> get() = liveRef.load()

    fun kvLength(context: Long): Int = kv[context] ?: 0

    private fun alloc(kind: String): Long {
        val h = nextHandle.addAndFetch(1).toLong()
        while (true) {
            val m = liveRef.load()
            if (liveRef.compareAndSet(m, m + (h to kind))) return h
        }
    }

    private fun release(h: Long, kind: String) {
        if (h == 0L) return
        val m = liveRef.load()
        check(m[h] == kind) { "freeing $kind $h which is not live (${m[h]})" }
        if (kind == "model") check(m.values.none { it == "context" || it == "sampler" }) { "model freed before its children" }
        liveRef.compareAndSet(m, m - h)
        frees.add(kind)
    }

    private fun checkLive(h: Long, kind: String) = check(live[h] == kind) { "$kind $h is not live" }

    override fun moduleVersion(): String = "1.0.0"
    override fun upstreamVersion(): String = "b8598"
    override fun upstreamCommit(): String = "9281dd135d6f175ceacca7d3a3ec440012264cfe"
    override fun systemInfo(): String = "FAKE"
    override fun statusName(status: Int): String = "STATUS_$status"
    override fun lastError(): String = lastErrorText

    override fun modelLoad(path: String, params: RawModelParams): RawHandleResult {
        loads.add(path to params)
        val status = loadStatus(params.backend)
        if (status != LlamaStatus.OK) {
            lastErrorText = "cannot load $path"
            return RawHandleResult(status, 0L)
        }
        return RawHandleResult(LlamaStatus.OK, alloc("model"))
    }

    override fun modelFree(model: Long) = release(model, "model")

    override fun modelInfo(model: Long): RawModelInfo {
        checkLive(model, "model")
        return RawModelInfo(
            status = LlamaStatus.OK, vocabularySize = vocab, trainContextLength = 512, embeddingSize = embd,
            layerCount = layers, hasDecoder = true, backend = loads.items.last().second.backend,
            usesGpu = loads.items.last().second.backend != LlamaBackendKind.CPU, bos = BOS, eos = EOS,
            sizeBytes = 1234, parameterCount = 99,
        )
    }

    override fun modelDescription(model: Long): String = "fake 1B F32"

    override fun tokenIsEog(model: Long, token: Int): Boolean = token == EOS

    override fun tokenize(model: Long, utf8: ByteArray, addSpecial: Boolean, parseSpecial: Boolean): RawTokensResult {
        checkLive(model, "model")
        val body = utf8.map { (it.toInt() and 0xFF) + BYTE_BASE }
        return RawTokensResult(LlamaStatus.OK, (if (addSpecial) listOf(BOS) + body else body).toIntArray())
    }

    override fun tokenToPiece(model: Long, token: Int, special: Boolean): RawBytesResult = when {
        token >= BYTE_BASE -> RawBytesResult(LlamaStatus.OK, byteArrayOf((token - BYTE_BASE).toByte()))
        special && token == BOS -> RawBytesResult(LlamaStatus.OK, "<s>".encodeToByteArray())
        else -> RawBytesResult(LlamaStatus.OK)
    }

    override fun contextCreate(model: Long, params: RawContextParams): RawHandleResult {
        checkLive(model, "model")
        contexts.add(params)
        val h = alloc("context")
        kv[h] = 0
        return RawHandleResult(LlamaStatus.OK, h)
    }

    override fun contextFree(context: Long) {
        release(context, "context")
        kv.remove(context)
    }

    override fun contextSize(context: Long): RawIntResult = RawIntResult(LlamaStatus.OK, contextSize)

    override fun requestAbort(context: Long) {
        while (true) {
            val s = abortRef.load()
            if (abortRef.compareAndSet(s, s + context)) return
        }
    }

    override fun clearAbort(context: Long) {
        while (true) {
            val s = abortRef.load()
            if (abortRef.compareAndSet(s, s - context)) return
        }
    }

    override fun abortRequested(context: Long): Boolean = context in abortRef.load()

    override fun decode(context: Long, tokens: IntArray, offset: Int, count: Int, pos0: Int, seqId: Int, logitsLast: Boolean): Int {
        checkLive(context, "context")
        val slice = tokens.copyOfRange(offset, offset + count).toList()
        decodes.add(DecodeCall(context, slice, pos0, seqId, logitsLast))
        check(pos0 == kvLength(context)) { "decode at $pos0 but the KV cache holds ${kvLength(context)}" }
        if (abortRequested(context)) return LlamaStatus.CANCELLED
        if (pos0 + count > contextSize) return LlamaStatus.BUFFER_TOO_SMALL
        kv[context] = pos0 + count // the batch lands in the cache before the hook may fail it (partial batch)
        val hook = onDecode(this, context, pos0)
        if (hook != LlamaStatus.OK) return hook
        if (abortRequested(context)) return LlamaStatus.CANCELLED
        return LlamaStatus.OK
    }

    override fun memorySeqRemove(context: Long, seqId: Int, p0: Int, p1: Int): Int {
        checkLive(context, "context")
        check(p1 < 0)
        kv[context] = minOf(kvLength(context), p0)
        return LlamaStatus.OK
    }

    override fun memoryClear(context: Long): Int {
        checkLive(context, "context")
        clears.addAndFetch(1)
        kv[context] = 0
        return LlamaStatus.OK
    }

    override fun memorySeqPosMax(context: Long, seqId: Int): RawIntResult = RawIntResult(LlamaStatus.OK, kvLength(context) - 1)

    override fun embeddings(context: Long, seqId: Int, normalize: Boolean): RawFloatsResult {
        checkLive(context, "context")
        val n = kvLength(context).toFloat()
        return RawFloatsResult(LlamaStatus.OK, if (normalize) floatArrayOf(1f, 0f, 0f, 0f) else floatArrayOf(n, 0f, 0f, 0f))
    }

    override fun stateSave(context: Long, seqId: Int): RawBytesResult {
        checkLive(context, "context")
        return RawBytesResult(LlamaStatus.OK, byteArrayOf(0x51, kvLength(context).toByte()))
    }

    override fun stateLoad(context: Long, seqId: Int, data: ByteArray): Int {
        checkLive(context, "context")
        if (data.size != 2 || data[0] != 0x51.toByte()) {
            lastErrorText = "corrupt state"
            return LlamaStatus.INVALID_ARGUMENT
        }
        kv[context] = data[1].toInt()
        return LlamaStatus.OK
    }

    override fun samplerCreate(model: Long, params: RawSamplerParams): RawHandleResult {
        checkLive(model, "model")
        samplers.add(params)
        if (samplerStatus != LlamaStatus.OK) {
            lastErrorText = "grammar does not parse"
            return RawHandleResult(samplerStatus, 0L)
        }
        val h = alloc("sampler")
        samplerPos[h] = 0
        return RawHandleResult(LlamaStatus.OK, h)
    }

    override fun samplerFree(sampler: Long) = release(sampler, "sampler")

    override fun samplerSample(sampler: Long, context: Long): RawIntResult {
        checkLive(sampler, "sampler")
        checkLive(context, "context")
        val i = samplerPos.getValue(sampler)
        samplerPos[sampler] = i + 1
        return RawIntResult(LlamaStatus.OK, script.getOrElse(i) { EOS })
    }

    override fun samplerAccept(sampler: Long, token: Int, applyGrammar: Boolean): Int {
        accepted.add(token to applyGrammar)
        return LlamaStatus.OK
    }

    override fun samplerReset(sampler: Long): Int = LlamaStatus.OK

    companion object {
        const val BOS = 1
        const val EOS = 2
        const val BYTE_BASE = 10

        /** Token ids spelling [text] in the fake vocabulary. */
        fun tokensOf(text: String): List<Int> = text.encodeToByteArray().map { (it.toInt() and 0xFF) + BYTE_BASE }
    }
}
