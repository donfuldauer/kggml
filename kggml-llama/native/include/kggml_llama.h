/*
 * kggml_llama.h - C ABI of the Kggml llama.cpp + libmtmd engine (:kggml-llama native layer).
 *
 * libkggml_llama (KggmlLlama.framework on iOS) contains llama.cpp (libllama), libmtmd (multimodal vision /
 * audio projector) and this wrapper, and links the shared ggml from :kggml-runtime dynamically. It never
 * contains a copy of ggml.
 *
 * Conventions (same as kggml_runtime.h / kggml_whisper.h):
 *   - Every function returning kggml_status is wrapped in a C++ try/catch. No exception ever crosses this
 *     boundary: std::bad_alloc -> KGGML_STATUS_OUT_OF_MEMORY, other std::exception ->
 *     KGGML_STATUS_NATIVE_FAILURE, anything else -> KGGML_STATUS_UNKNOWN.
 *   - On failure a diagnostic message is stored thread-locally; read it with kggml_llama_last_error().
 *   - Status codes 0..6 equal com.donfuldauer.kggml.core.NativeErrorCode.
 *   - Variable-size outputs use the BUFFER_TOO_SMALL protocol: when `capacity` is too small (or the buffer is
 *     NULL), KGGML_STATUS_BUFFER_TOO_SMALL is returned and the required size is stored in the `out_*` count.
 *
 * Object graph: kggml_llama_model -> kggml_llama_context / kggml_llama_sampler / kggml_llama_mtmd. Free every
 * context, sampler and mtmd object BEFORE the model it was created from.
 *
 * Threading: none of these objects is thread-safe. Callers serialize every call on a model and the objects
 * created from it on one thread (the Kotlin engines use a dedicated single-threaded dispatcher per instance).
 * The only exceptions are kggml_llama_request_abort() and kggml_llama_abort_requested(), which are lock-free
 * and may be called from any thread at any time while the context is alive. Different models are independent.
 */
#ifndef KGGML_LLAMA_H
#define KGGML_LLAMA_H

#include "kggml_runtime.h" /* kggml_status, kggml_backend_kind */

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(KGGML_LLAMA_BUILD)
#        define KGGML_LLAMA_API __declspec(dllexport)
#    elif defined(KGGML_LLAMA_SHARED)
#        define KGGML_LLAMA_API __declspec(dllimport)
#    else
#        define KGGML_LLAMA_API
#    endif
#else
#    define KGGML_LLAMA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t kggml_llama_token;

/* ------------------------------------------------------------------------------------------------------ */
/* Versions & errors                                                                                       */
/* ------------------------------------------------------------------------------------------------------ */

/** Kggml llama module version, e.g. "0.1.0". Static string. */
KGGML_LLAMA_API const char * kggml_llama_version(void);

/** Upstream llama.cpp release tag this build was pinned to, e.g. "b12816" (libmtmd comes from the same commit). */
KGGML_LLAMA_API const char * kggml_llama_upstream_version(void);

/** Full 40-char upstream llama.cpp commit SHA (see NOTICE). Static string. */
KGGML_LLAMA_API const char * kggml_llama_upstream_commit(void);

/** llama_print_system_info() of the linked llama.cpp + ggml (CPU features etc). Never NULL. */
KGGML_LLAMA_API const char * kggml_llama_system_info(void);

/**
 * Message describing the most recent failing kggml_llama_* call on the calling thread ("" after a success).
 * Valid until the next kggml_llama_* call on the same thread.
 */
KGGML_LLAMA_API const char * kggml_llama_last_error(void);

/** Clears the calling thread's llama last error. */
KGGML_LLAMA_API void kggml_llama_clear_last_error(void);

/* ------------------------------------------------------------------------------------------------------ */
/* Model                                                                                                   */
/* ------------------------------------------------------------------------------------------------------ */

typedef struct kggml_llama_model kggml_llama_model;

/**
 * Model-load progress callback (0.0 .. 1.0), invoked on the loading thread. Return 1 to continue, 0 to abort
 * the load; kggml_llama_model_load() then returns KGGML_STATUS_CANCELLED.
 */
typedef int32_t (*kggml_llama_progress_callback)(float progress, void * user_data);

typedef struct kggml_llama_model_params {
    /**
     * kggml_backend_kind. KGGML_BACKEND_CPU hands llama.cpp an EMPTY device list, so no GPU backend is ever
     * created for the model or its contexts (weights, KV cache and compute stay on the CPU / Accelerate /
     * NEON; on iOS this stays legal in the background). KGGML_BACKEND_METAL / KGGML_BACKEND_VULKAN offload to
     * the `gpu_device`-th GPU device of that kind. KGGML_BACKEND_COREML is not supported by llama.cpp and
     * returns KGGML_STATUS_BACKEND_UNAVAILABLE. Default: KGGML_BACKEND_CPU.
     */
    int32_t backend;
    int32_t gpu_layers;  /* layers offloaded to the GPU, -1 = all (ignored for CPU), default -1          */
    int32_t gpu_device;  /* index among the GPU devices of the requested kind, default 0                 */
    int32_t use_mmap;    /* 1 = mmap the GGUF file (default), 0 = read it into memory                    */
    int32_t use_mlock;   /* 1 = mlock the weights, default 0                                              */
    int32_t vocab_only;  /* 1 = load only the vocabulary (tokenizer), no weights, default 0               */
    kggml_llama_progress_callback progress_callback; /* optional, NULL = none                            */
    void * progress_user_data;
} kggml_llama_model_params;

/** Fills `params` with defaults (CPU, all layers if a GPU is requested, device 0, mmap on). */
KGGML_LLAMA_API kggml_status kggml_llama_model_default_params(kggml_llama_model_params * params);

/**
 * Loads a GGUF model from `model_path`. Returns KGGML_STATUS_MODEL_LOAD_FAILED when the file is missing or not
 * a loadable GGUF model, KGGML_STATUS_BACKEND_UNAVAILABLE when the requested backend is not compiled into the
 * runtime / not present on this device, KGGML_STATUS_CANCELLED when the progress callback aborted the load.
 * `params` may be NULL for defaults. Holds one kggml_runtime_init() reference until kggml_llama_model_free().
 */
KGGML_LLAMA_API kggml_status kggml_llama_model_load(const char * model_path, const kggml_llama_model_params * params,
                                                   kggml_llama_model ** out_model);

/** Frees a model. NULL is a no-op. Every context / sampler / mtmd object created from it must be freed first. */
KGGML_LLAMA_API void kggml_llama_model_free(kggml_llama_model * model);

typedef struct kggml_llama_model_info {
    int32_t  n_vocab;        /* vocabulary size                                     */
    int32_t  n_ctx_train;    /* training context length                             */
    int32_t  n_embd;         /* embedding width (output embedding size)             */
    int32_t  n_layer;        /* transformer blocks                                  */
    int32_t  has_encoder;    /* 1 for encoder-decoder models (T5 etc)               */
    int32_t  has_decoder;    /* 1 if the model can generate                         */
    int32_t  backend;        /* effective kggml_backend_kind                        */
    int32_t  uses_gpu;       /* 1 if at least one GPU device was handed to llama.cpp */
    uint64_t size_bytes;     /* total tensor size                                   */
    uint64_t n_params;       /* total parameter count                               */
    kggml_llama_token bos;   /* -1 if the vocabulary has none                       */
    kggml_llama_token eos;   /* -1 if the vocabulary has none                       */
} kggml_llama_model_info;

KGGML_LLAMA_API kggml_status kggml_llama_model_get_info(const kggml_llama_model * model, kggml_llama_model_info * out_info);

/** llama_model_desc() into `buffer` (NUL-terminated, truncated to `capacity`). `out_length` = full length. */
KGGML_LLAMA_API kggml_status kggml_llama_model_desc(const kggml_llama_model * model, char * buffer, size_t capacity,
                                                   size_t * out_length);

/** 1 if `token` is an end-of-generation token (EOS, EOT, ...), 0 otherwise (also for out-of-range ids). */
KGGML_LLAMA_API int32_t kggml_llama_token_is_eog(const kggml_llama_model * model, kggml_llama_token token);

/**
 * Tokenizes `text_len` bytes of UTF-8 `text`. `add_special` = 1 adds BOS/EOS as the model requires,
 * `parse_special` = 1 parses control tokens written as text. BUFFER_TOO_SMALL protocol on `out_count`.
 */
KGGML_LLAMA_API kggml_status kggml_llama_tokenize(const kggml_llama_model * model, const char * text, size_t text_len,
                                                 int32_t add_special, int32_t parse_special,
                                                 kggml_llama_token * out_tokens, int32_t capacity, int32_t * out_count);

/**
 * UTF-8 bytes of one token (NOT NUL-terminated; a piece may end in the middle of a multi-byte character, so
 * streaming callers must buffer incomplete sequences). `special` = 1 renders control tokens. BUFFER_TOO_SMALL
 * protocol on `out_length`.
 */
KGGML_LLAMA_API kggml_status kggml_llama_token_to_piece(const kggml_llama_model * model, kggml_llama_token token,
                                                       int32_t special, char * buffer, int32_t capacity,
                                                       int32_t * out_length);

/* ------------------------------------------------------------------------------------------------------ */
/* Context (KV cache + compute)                                                                            */
/* ------------------------------------------------------------------------------------------------------ */

typedef struct kggml_llama_context kggml_llama_context;

/** llama_pooling_type values for kggml_llama_context_params.pooling. */
#define KGGML_LLAMA_POOLING_UNSPECIFIED (-1)
#define KGGML_LLAMA_POOLING_NONE        0
#define KGGML_LLAMA_POOLING_MEAN        1
#define KGGML_LLAMA_POOLING_CLS         2
#define KGGML_LLAMA_POOLING_LAST        3

typedef struct kggml_llama_context_params {
    uint32_t n_ctx;           /* context length in tokens, 0 = model training length                   */
    uint32_t n_batch;         /* max tokens per llama_decode() call (kggml_llama_decode splits), def 512 */
    uint32_t n_ubatch;        /* physical micro-batch, 0 = n_batch                                     */
    uint32_t n_seq_max;       /* max distinct sequences (seq ids 0 .. n_seq_max-1), default 1          */
    int32_t  n_threads;       /* generation threads, 0 = kggml_runtime_performance_thread_count()     */
    int32_t  n_threads_batch; /* prompt-processing threads, 0 = n_threads                              */
    int32_t  flash_attn;      /* -1 = auto (default), 0 = off, 1 = on                                  */
    int32_t  embeddings;      /* 1 = embedding mode (kggml_llama_get_embeddings), default 0            */
    int32_t  pooling;         /* KGGML_LLAMA_POOLING_*, default UNSPECIFIED (model default)            */
} kggml_llama_context_params;

/** Fills `params` with defaults. */
KGGML_LLAMA_API kggml_status kggml_llama_context_default_params(kggml_llama_context_params * params);

/**
 * Creates a context on `model` (llama_init_from_model). The context's abort callback is installed here
 * (llama_set_abort_callback) and polls the flag set by kggml_llama_request_abort(). On a CPU model the KV cache
 * and every op stay on the CPU (offload_kqv / op_offload off). `params` may be NULL for defaults.
 */
KGGML_LLAMA_API kggml_status kggml_llama_context_create(kggml_llama_model * model,
                                                       const kggml_llama_context_params * params,
                                                       kggml_llama_context ** out_context);

/** Frees a context. NULL is a no-op. Must not race with any other call on the same context. */
KGGML_LLAMA_API void kggml_llama_context_free(kggml_llama_context * context);

/** Effective context length (may be rounded up by llama.cpp). */
KGGML_LLAMA_API kggml_status kggml_llama_context_n_ctx(const kggml_llama_context * context, uint32_t * out_n_ctx);

/* ------------------------------------------------------------------------------------------------------ */
/* Abort (any thread)                                                                                      */
/* ------------------------------------------------------------------------------------------------------ */

/**
 * Atomically requests cancellation of the decode / multimodal evaluation running (or next started) on
 * `context`. Lock-free and safe from any thread, e.g. a Kotlin Job.invokeOnCompletion handler. llama.cpp polls
 * it through llama_set_abort_callback() inside every ggml graph compute, kggml_llama_decode() also checks it
 * between batches, and the interrupted call returns KGGML_STATUS_CANCELLED. The request stays latched until
 * kggml_llama_clear_abort().
 */
KGGML_LLAMA_API void kggml_llama_request_abort(kggml_llama_context * context);

/** Clears a latched abort request. Call it before starting a new generation. */
KGGML_LLAMA_API void kggml_llama_clear_abort(kggml_llama_context * context);

/** 1 if an abort is currently requested, otherwise 0 (0 for NULL). */
KGGML_LLAMA_API int32_t kggml_llama_abort_requested(const kggml_llama_context * context);

/* ------------------------------------------------------------------------------------------------------ */
/* Decoding                                                                                                */
/* ------------------------------------------------------------------------------------------------------ */

/**
 * Evaluates `n_tokens` tokens of sequence `seq_id` at positions pos0 .. pos0 + n_tokens - 1 (zero-copy: the
 * token array is read in place), split into n_batch-sized llama_decode() calls. When `logits_last` = 1, logits
 * (or embeddings) are computed for the final token and readable with kggml_llama_get_logits() /
 * kggml_llama_get_embeddings(). Returns KGGML_STATUS_CANCELLED if an abort was requested before or during the
 * call (the KV cache then holds a partial batch: remove it with kggml_llama_memory_seq_rm()),
 * KGGML_STATUS_INVALID_ARGUMENT for out-of-vocabulary tokens / invalid positions or sequence ids, and
 * KGGML_STATUS_BUFFER_TOO_SMALL when the KV cache has no room left for the batch (context full).
 */
KGGML_LLAMA_API kggml_status kggml_llama_decode(kggml_llama_context * context, const kggml_llama_token * tokens,
                                               int32_t n_tokens, int32_t pos0, int32_t seq_id, int32_t logits_last);

/**
 * Logits of the last token evaluated with logits_last = 1: `*out_logits` points to n_vocab floats owned by the
 * context, valid until the next decode / free.
 */
KGGML_LLAMA_API kggml_status kggml_llama_get_logits(kggml_llama_context * context, const float ** out_logits,
                                                   int32_t * out_n_vocab);

/** Removes positions [p0, p1) of `seq_id` from the KV cache (p1 < 0 = to the end, seq_id < 0 = all sequences). */
KGGML_LLAMA_API kggml_status kggml_llama_memory_seq_rm(kggml_llama_context * context, int32_t seq_id, int32_t p0,
                                                      int32_t p1);

/** Clears the whole KV cache (all sequences). */
KGGML_LLAMA_API kggml_status kggml_llama_memory_clear(kggml_llama_context * context);

/** Largest position stored for `seq_id` in the KV cache, -1 when the sequence is empty. */
KGGML_LLAMA_API kggml_status kggml_llama_memory_seq_pos_max(kggml_llama_context * context, int32_t seq_id,
                                                           int32_t * out_pos);

/* ------------------------------------------------------------------------------------------------------ */
/* Embeddings                                                                                              */
/* ------------------------------------------------------------------------------------------------------ */

/**
 * Copies the embedding of sequence `seq_id` from the last decode of an embeddings = 1 context: the pooled
 * sequence embedding when pooling != NONE, otherwise the last token's embedding. `normalize` = 1 L2-normalizes
 * the copy. BUFFER_TOO_SMALL protocol on `out_count` (= n_embd).
 */
KGGML_LLAMA_API kggml_status kggml_llama_get_embeddings(kggml_llama_context * context, int32_t seq_id,
                                                       int32_t normalize, float * out, int32_t capacity,
                                                       int32_t * out_count);

/* ------------------------------------------------------------------------------------------------------ */
/* KV-cache state serialization (llama_state_* / llama_state_seq_*)                                       */
/* ------------------------------------------------------------------------------------------------------ */

/**
 * Bytes needed to serialize the context state: the whole context (KV cache of every sequence) when `seq_id` < 0,
 * otherwise only that sequence's KV cache.
 */
KGGML_LLAMA_API kggml_status kggml_llama_state_size(kggml_llama_context * context, int32_t seq_id, size_t * out_size);

/**
 * Serializes the state (see kggml_llama_state_size()) into `buffer`. If `capacity` is smaller than the current
 * size, returns BUFFER_TOO_SMALL with the required size in `out_written`.
 */
KGGML_LLAMA_API kggml_status kggml_llama_state_save(kggml_llama_context * context, int32_t seq_id, uint8_t * buffer,
                                                   size_t capacity, size_t * out_written);

/**
 * Restores a state written by kggml_llama_state_save() with the same `seq_id` scope (for a sequence, into
 * `seq_id`, which may differ from the saved one). The context must use the same model and a compatible
 * configuration. Logits / embeddings of the decode that preceded the save are not restored: decode the next token
 * (at position kggml_llama_memory_seq_pos_max() + 1) before sampling. Returns KGGML_STATUS_INVALID_ARGUMENT when
 * llama.cpp rejects the data.
 */
KGGML_LLAMA_API kggml_status kggml_llama_state_load(kggml_llama_context * context, int32_t seq_id,
                                                   const uint8_t * data, size_t size);

/* ------------------------------------------------------------------------------------------------------ */
/* Sampling (+ GBNF grammar constraints)                                                                   */
/* ------------------------------------------------------------------------------------------------------ */

typedef struct kggml_llama_sampler kggml_llama_sampler;

typedef struct kggml_llama_sampler_params {
    float        temperature;    /* <= 0 = greedy (argmax), default 0.8                                   */
    int32_t      top_k;          /* <= 0 = disabled, default 40                                           */
    float        top_p;          /* >= 1 = disabled, default 0.95                                         */
    float        min_p;          /* <= 0 = disabled, default 0.05                                         */
    float        repeat_penalty; /* 1 = disabled (default)                                                */
    int32_t      repeat_last_n;  /* tokens considered by repeat_penalty, default 64                       */
    uint32_t     seed;           /* RNG seed of the final distribution sampler, default 0xFFFFFFFF = random */
    const char * grammar;        /* GBNF grammar constraining every sampled token, NULL = none            */
    const char * grammar_root;   /* start symbol, NULL = "root"                                           */
} kggml_llama_sampler_params;

/** Fills `params` with defaults. */
KGGML_LLAMA_API kggml_status kggml_llama_sampler_default_params(kggml_llama_sampler_params * params);

/**
 * Builds a llama_sampler chain: [grammar] -> [penalties] -> top_k -> top_p -> min_p -> temperature -> dist
 * (or [grammar] -> [penalties] -> greedy). The grammar is copied. Returns KGGML_STATUS_INVALID_ARGUMENT when the
 * GBNF fails to parse or does not define the root symbol. `params` may be NULL for defaults.
 */
KGGML_LLAMA_API kggml_status kggml_llama_sampler_create(const kggml_llama_model * model,
                                                       const kggml_llama_sampler_params * params,
                                                       kggml_llama_sampler ** out_sampler);

/** Frees a sampler. NULL is a no-op. */
KGGML_LLAMA_API void kggml_llama_sampler_free(kggml_llama_sampler * sampler);

/**
 * Samples a token from the logits of output `idx` of the last decode (-1 = last output) and accepts it into
 * the chain (grammar state, repetition history). Returns KGGML_STATUS_NATIVE_FAILURE when a grammar leaves no
 * admissible token.
 */
KGGML_LLAMA_API kggml_status kggml_llama_sampler_sample(kggml_llama_sampler * sampler, kggml_llama_context * context,
                                                       int32_t idx, kggml_llama_token * out_token);

/**
 * Accepts `token` into the chain without sampling: always into the repetition history, and into the grammar
 * state when `apply_grammar` = 1 (use 0 for prompt tokens, 1 to force-feed generated text). Returns
 * KGGML_STATUS_INVALID_ARGUMENT, leaving every state unchanged, when the grammar does not admit `token`.
 */
KGGML_LLAMA_API kggml_status kggml_llama_sampler_accept(kggml_llama_sampler * sampler, kggml_llama_token token,
                                                       int32_t apply_grammar);

/** Resets grammar and repetition state (start of a new generation). */
KGGML_LLAMA_API kggml_status kggml_llama_sampler_reset(kggml_llama_sampler * sampler);

/* ------------------------------------------------------------------------------------------------------ */
/* Multimodal (libmtmd): in-memory images, no temporary files                                              */
/* ------------------------------------------------------------------------------------------------------ */

typedef struct kggml_llama_mtmd kggml_llama_mtmd;

typedef struct kggml_llama_mtmd_params {
    /**
     * kggml_backend_kind of the vision / audio encoder. KGGML_BACKEND_CPU sets mtmd_context_params.use_gpu =
     * false (no GPU backend is created); METAL / VULKAN let libmtmd use the first GPU device. Default: the
     * backend of the model.
     */
    int32_t backend;
    int32_t n_threads;        /* 0 = kggml_runtime_performance_thread_count()          */
    int32_t flash_attn;       /* -1 = auto (default), 0 = off, 1 = on                  */
    int32_t image_min_tokens; /* 0 = from the projector metadata                       */
    int32_t image_max_tokens; /* 0 = from the projector metadata                       */
    int32_t warmup;           /* 1 = run a warmup encode at init, default 0            */
} kggml_llama_mtmd_params;

/** Fills `params` with defaults; `backend` is KGGML_BACKEND_CPU here and resolved to the model's in init. */
KGGML_LLAMA_API kggml_status kggml_llama_mtmd_default_params(kggml_llama_mtmd_params * params);

/**
 * Loads a multimodal projector GGUF (mmproj) for `model` (mtmd_init_from_file). MODEL_LOAD_FAILED when the file
 * is missing or not a projector compatible with the model. `params` may be NULL (defaults, model backend).
 */
KGGML_LLAMA_API kggml_status kggml_llama_mtmd_init(kggml_llama_model * model, const char * mmproj_path,
                                                  const kggml_llama_mtmd_params * params,
                                                  kggml_llama_mtmd ** out_mtmd);

/** Frees a projector. NULL is a no-op. */
KGGML_LLAMA_API void kggml_llama_mtmd_free(kggml_llama_mtmd * mtmd);

/** Capabilities of a loaded projector: vision / audio input (1/0) and the audio sample rate (-1 if none). */
KGGML_LLAMA_API kggml_status kggml_llama_mtmd_get_caps(kggml_llama_mtmd * mtmd, int32_t * out_vision,
                                                      int32_t * out_audio, int32_t * out_audio_sample_rate);

/** The media marker ("<__media__>") that kggml_llama_mtmd_eval() replaces by each image, in order. Static. */
KGGML_LLAMA_API const char * kggml_llama_mtmd_default_marker(void);

#define KGGML_LLAMA_IMAGE_RGB     0 /* raw packed RGB888, width * height * 3 bytes            */
#define KGGML_LLAMA_IMAGE_ENCODED 1 /* PNG / JPEG / BMP / GIF / ... bytes, decoded in memory  */

typedef struct kggml_llama_image {
    const uint8_t * data;
    size_t          size;   /* bytes                                        */
    int32_t         format; /* KGGML_LLAMA_IMAGE_*                          */
    uint32_t        width;  /* RGB only                                     */
    uint32_t        height; /* RGB only                                     */
} kggml_llama_image;

/**
 * Tokenizes `prompt` (which contains one media marker per image) with libmtmd, encodes the images in memory
 * (no temporary files: RGB buffers are wrapped directly, encoded images are decoded from memory) and evaluates
 * all chunks into sequence `seq_id` of `context`, starting at position `n_past`. On success `out_n_past`
 * receives the next position; with `logits_last` = 1 the logits of the final prompt token are readable with
 * kggml_llama_get_logits(). Returns KGGML_STATUS_INVALID_ARGUMENT for undecodable images or a marker / image
 * count mismatch and KGGML_STATUS_CANCELLED when kggml_llama_request_abort() fired.
 */
KGGML_LLAMA_API kggml_status kggml_llama_mtmd_eval(kggml_llama_mtmd * mtmd, kggml_llama_context * context,
                                                  const char * prompt, const kggml_llama_image * images,
                                                  int32_t n_images, int32_t n_past, int32_t seq_id,
                                                  int32_t add_special, int32_t logits_last, int32_t * out_n_past);

/**
 * Tokenized multimodal prompt: text tokens plus the *preprocessed* images (the caller's image bytes are no longer
 * referenced once kggml_llama_mtmd_tokenize() returns, so bindings only need to pin them for that call). Owned by
 * the caller; free with kggml_llama_mtmd_chunks_free() before the projector it was created with.
 */
typedef struct kggml_llama_mtmd_chunks kggml_llama_mtmd_chunks;

/**
 * First half of kggml_llama_mtmd_eval(): splits `prompt` at its media markers and preprocesses `images` (wrapped or
 * decoded in memory, never written to disk). `parse_special` = 1 parses control tokens written as text. Returns
 * KGGML_STATUS_INVALID_ARGUMENT for an undecodable / malformed image or a marker / image count mismatch.
 */
KGGML_LLAMA_API kggml_status kggml_llama_mtmd_tokenize(kggml_llama_mtmd * mtmd, const char * prompt,
                                                      const kggml_llama_image * images, int32_t n_images,
                                                      int32_t add_special, int32_t parse_special,
                                                      kggml_llama_mtmd_chunks ** out_chunks);

/** Frees tokenized chunks. NULL is a no-op. */
KGGML_LLAMA_API void kggml_llama_mtmd_chunks_free(kggml_llama_mtmd_chunks * chunks);

/**
 * Sizes of tokenized chunks: total tokens (text + image embeddings), KV positions they occupy (smaller than the
 * token count for M-RoPE models) and the number of images. Every out pointer may be NULL.
 */
KGGML_LLAMA_API kggml_status kggml_llama_mtmd_chunks_info(const kggml_llama_mtmd_chunks * chunks, int32_t * out_n_tokens,
                                                         int32_t * out_n_pos, int32_t * out_n_images);

/**
 * Second half of kggml_llama_mtmd_eval(): encodes the images of `chunks` and evaluates every chunk into `seq_id` of
 * `context` from position `n_past`. KGGML_STATUS_BUFFER_TOO_SMALL (nothing evaluated) when the chunks do not fit
 * the context window; KGGML_STATUS_CANCELLED after kggml_llama_request_abort() (checked inside every text decode
 * and between chunks). `chunks` stays valid and can be evaluated again.
 */
KGGML_LLAMA_API kggml_status kggml_llama_mtmd_eval_chunks(kggml_llama_mtmd * mtmd, kggml_llama_context * context,
                                                         const kggml_llama_mtmd_chunks * chunks, int32_t n_past,
                                                         int32_t seq_id, int32_t logits_last, int32_t * out_n_past);

#ifdef __cplusplus
}
#endif

#endif /* KGGML_LLAMA_H */
