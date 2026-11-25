package com.donfuldauer.kggml.llama

import com.donfuldauer.kggml.core.ComputeBackend
import com.donfuldauer.kggml.core.StopReason
import com.donfuldauer.kggml.llama.grammar.JsonSchemaToGbnf

/**
 * KV-cache and batching configuration of a [LlamaEngine], fixed at construction time.
 *
 * @property contextLength context window in tokens (prompt + generated text kept in the KV cache).
 *   [MODEL_CONTEXT_LENGTH] (`0`) uses the model's training length, which can be very large (and memory hungry)
 *   for modern models.
 * @property batchSize maximum tokens per `llama_decode()` call during prompt processing; longer prompts are split.
 * @property microBatchSize physical micro-batch size, `0` = [batchSize].
 * @property embeddingPooling how [LlamaEngine.embed] pools token embeddings into one vector.
 * @property embeddingContextLength context window of the embedding context created lazily by
 *   [LlamaEngine.embed]. Inputs must fit in one batch of this size.
 */
public data class LlamaContextConfig(
    val contextLength: Int = DEFAULT_CONTEXT_LENGTH,
    val batchSize: Int = DEFAULT_BATCH_SIZE,
    val microBatchSize: Int = 0,
    val embeddingPooling: LlamaPooling = LlamaPooling.MODEL_DEFAULT,
    val embeddingContextLength: Int = DEFAULT_EMBEDDING_CONTEXT_LENGTH,
) {
    init {
        require(contextLength >= 0) { "contextLength must be >= 0 (0 = model training length), was $contextLength" }
        require(batchSize >= 1) { "batchSize must be >= 1, was $batchSize" }
        require(microBatchSize >= 0) { "microBatchSize must be >= 0 (0 = batchSize), was $microBatchSize" }
        require(embeddingContextLength >= 1) { "embeddingContextLength must be >= 1, was $embeddingContextLength" }
    }

    public companion object {
        /** [contextLength] sentinel: use the model's training context length. */
        public const val MODEL_CONTEXT_LENGTH: Int = 0
        public const val DEFAULT_CONTEXT_LENGTH: Int = 4096
        public const val DEFAULT_BATCH_SIZE: Int = 512
        public const val DEFAULT_EMBEDDING_CONTEXT_LENGTH: Int = 512

        public val DEFAULT: LlamaContextConfig = LlamaContextConfig()
    }
}

/** Pooling of token embeddings into a sequence embedding (`llama_pooling_type`). */
public enum class LlamaPooling(internal val nativeValue: Int) {
    /** The model's own default (e.g. MEAN or CLS for embedding models, NONE for causal LMs). */
    MODEL_DEFAULT(-1),

    /** No pooling: the embedding of the last input token. */
    NONE(0),
    MEAN(1),
    CLS(2),
    LAST(3),
}

/**
 * LLM-specific construction options of a [LlamaEngine].
 *
 * @property gpuDevice index among the GPU devices of the selected kind (Metal / Vulkan).
 * @property allowBackendFallback when the requested backend is unavailable, fall back along
 *   [ComputeBackend.fallback] (eventually [ComputeBackend.CPU_ONLY]) instead of failing.
 * @property useMlock lock the model weights in RAM (`mlock`).
 */
public data class LlamaEngineOptions(
    val gpuDevice: Int = 0,
    val allowBackendFallback: Boolean = true,
    val useMlock: Boolean = false,
) {
    init {
        require(gpuDevice >= 0) { "gpuDevice must be >= 0, was $gpuDevice" }
    }
}

/**
 * Options of one [LlamaEngine.generate] call.
 *
 * Sampling runs `[grammar] -> [repetition penalty] -> top-k -> top-p -> min-p -> temperature -> distribution`,
 * or greedy argmax when [temperature] is `<= 0`.
 *
 * @property maxTokens maximum number of generated tokens, or [UNLIMITED] (until end of generation or a full context).
 * @property temperature sampling temperature; `<= 0` selects greedy (deterministic) decoding.
 * @property topK keep the `k` most likely tokens, `<= 0` disables.
 * @property topP nucleus sampling threshold, `>= 1` disables.
 * @property minP minimum probability relative to the most likely token, `<= 0` disables.
 * @property repeatPenalty repetition penalty, `1` disables.
 * @property repeatLastN how many recent tokens the repetition penalty considers.
 * @property seed RNG seed of the final distribution sampler (`0..4294967294`), `null` for a random seed.
 * @property grammar constrains every sampled token (GBNF or JSON schema), `null` for unconstrained text.
 * @property stopSequences generation stops with [StopReason.STOP_SEQUENCE] once the generated text contains one of
 *   these. The token that completed the stop sequence is still emitted.
 * @property addSpecialTokens add BOS/EOS to the prompt as the model's tokenizer requires.
 * @property parseSpecialTokens parse control tokens written as text in the prompt (e.g. chat-template markers).
 * @property renderSpecialTokens render control tokens into [com.donfuldauer.kggml.core.TokenEvent.Token.text].
 * @property reusePromptCache reuse the KV cache of the longest common token prefix between this prompt and the
 *   tokens already in the engine's session (previous prompts, generations, or a restored session), so only the new
 *   suffix is evaluated. `false` clears the session first.
 */
public data class GenerationOptions(
    val maxTokens: Int = 256,
    val temperature: Float = 0.8f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.0f,
    val repeatLastN: Int = 64,
    val seed: Long? = null,
    val grammar: LlamaGrammar? = null,
    val stopSequences: List<String> = emptyList(),
    val addSpecialTokens: Boolean = true,
    val parseSpecialTokens: Boolean = true,
    val renderSpecialTokens: Boolean = false,
    val reusePromptCache: Boolean = true,
) {
    init {
        require(maxTokens == UNLIMITED || maxTokens >= 1) { "maxTokens must be >= 1 or UNLIMITED (-1), was $maxTokens" }
        require(!temperature.isNaN()) { "temperature must not be NaN" }
        require(!topP.isNaN() && topP >= 0f) { "topP must be >= 0, was $topP" }
        require(!minP.isNaN() && minP <= 1f) { "minP must be <= 1, was $minP" }
        require(!repeatPenalty.isNaN() && repeatPenalty > 0f) { "repeatPenalty must be > 0, was $repeatPenalty" }
        require(repeatLastN >= 0) { "repeatLastN must be >= 0, was $repeatLastN" }
        require(seed == null || seed in 0L..MAX_SEED) { "seed must be in 0..$MAX_SEED or null (random), was $seed" }
        require(stopSequences.none { it.isEmpty() }) { "stopSequences must not contain empty strings" }
    }

    /** `true` when [temperature] selects greedy decoding. */
    public val isGreedy: Boolean get() = temperature <= 0f

    internal fun toRaw(): RawSamplerParams = RawSamplerParams(
        temperature = temperature,
        topK = topK,
        topP = topP,
        minP = minP,
        repeatPenalty = repeatPenalty,
        repeatLastN = repeatLastN,
        seed = (seed ?: RANDOM_SEED).toInt(), // uint32 bits
        grammar = grammar?.gbnf,
        grammarRoot = grammar?.root,
    )

    public companion object {
        /** [maxTokens] sentinel: no limit other than end of generation or a full context. */
        public const val UNLIMITED: Int = -1

        /** Largest explicit [seed] (`0xFFFFFFFF` is llama.cpp's "random" sentinel). */
        public const val MAX_SEED: Long = 0xFFFFFFFEL

        private const val RANDOM_SEED: Long = 0xFFFFFFFFL

        public val DEFAULT: GenerationOptions = GenerationOptions()

        /** Deterministic greedy decoding. */
        public val GREEDY: GenerationOptions = GenerationOptions(temperature = 0f)
    }
}

/**
 * A GBNF grammar constraining generation (llama.cpp `llama_sampler_init_grammar`).
 *
 * Build one with [gbnf], [jsonSchema] or [json]. GBNF text is validated by llama.cpp when generation starts;
 * a grammar that does not parse fails the generation with [com.donfuldauer.kggml.core.InvalidInputException].
 *
 * @property gbnf the grammar text.
 * @property root the start symbol.
 */
public class LlamaGrammar private constructor(public val gbnf: String, public val root: String) {
    override fun equals(other: Any?): Boolean = other is LlamaGrammar && other.gbnf == gbnf && other.root == root

    override fun hashCode(): Int = gbnf.hashCode() * 31 + root.hashCode()

    override fun toString(): String = "LlamaGrammar(root=$root, gbnf=${gbnf.length} chars)"

    public companion object {
        /** A raw GBNF grammar starting at [root]. */
        public fun gbnf(gbnf: String, root: String = "root"): LlamaGrammar {
            require(gbnf.isNotBlank()) { "gbnf must not be blank" }
            require(root.isNotEmpty() && root.all { it in 'a'..'z' || it in 'A'..'Z' || it in '0'..'9' || it == '-' }) {
                "root must be a GBNF rule name, was '$root'"
            }
            return LlamaGrammar(gbnf, root)
        }

        /**
         * Constrains output to JSON documents valid against the JSON Schema [schema] (JSON text).
         *
         * Supported: `type` (incl. type arrays), `properties` / `required` (declared order, optional properties may be
         * omitted), `additionalProperties` on property-less objects, `items` / `prefixItems` / `minItems` /
         * `maxItems`, `minLength` / `maxLength`, `enum`, `const`, `anyOf` / `oneOf`, object `allOf`, and local
         * `$ref`s into `$defs` / `definitions` (recursion allowed). Objects with `properties` never produce
         * undeclared keys. Numeric bounds and `format` are not enforced.
         *
         * @throws com.donfuldauer.kggml.core.InvalidInputException if [schema] is not valid JSON or uses an
         *   unsupported keyword that changes the accepted language (e.g. `pattern`, remote `$ref`).
         */
        public fun jsonSchema(schema: String): LlamaGrammar = LlamaGrammar(JsonSchemaToGbnf.convert(schema), "root")

        /** Constrains output to any JSON object. */
        public fun json(): LlamaGrammar = LlamaGrammar(JsonSchemaToGbnf.convert("""{"type":"object"}"""), "root")
    }
}

/**
 * Static properties of the loaded model.
 *
 * @property description `llama_model_desc()`, e.g. `"llama 7B Q4_K - Medium"`.
 * @property bosToken beginning-of-sequence token id, `-1` if the vocabulary has none.
 * @property eosToken end-of-sequence token id, `-1` if the vocabulary has none.
 */
public data class LlamaModelInfo(
    val description: String,
    val vocabularySize: Int,
    val trainContextLength: Int,
    val embeddingSize: Int,
    val layerCount: Int,
    val parameterCount: Long,
    val sizeBytes: Long,
    val bosToken: Int,
    val eosToken: Int,
    val hasEncoder: Boolean,
)

/** The outcome of [LlamaEngine.generateText]. */
public data class GenerationResult(
    val text: String,
    val tokens: IntArray,
    val stopReason: StopReason,
) {
    override fun equals(other: Any?): Boolean =
        other is GenerationResult && other.text == text && other.tokens.contentEquals(tokens) && other.stopReason == stopReason

    override fun hashCode(): Int = (text.hashCode() * 31 + tokens.contentHashCode()) * 31 + stopReason.hashCode()
}

/**
 * Version information of the native llama library.
 *
 * @property moduleVersion Kggml llama module version, e.g. `"1.0.0"`.
 * @property llamaCppVersion pinned upstream llama.cpp release tag, e.g. `"b8598"`.
 * @property llamaCppCommit full upstream llama.cpp commit SHA (see `NOTICE`).
 * @property systemInfo `llama_print_system_info()` (CPU features of this build).
 */
public data class LlamaNativeInfo(
    val moduleVersion: String,
    val llamaCppVersion: String,
    val llamaCppCommit: String,
    val systemInfo: String,
)
