/*
 * kggml_diffusion.h - C ABI of the Kggml stable-diffusion.cpp engine (:kggml-diffusion native layer).
 *
 * libkggml_diffusion (KggmlDiffusion.framework on iOS) contains stable-diffusion.cpp and this wrapper, and links the
 * shared ggml from :kggml-runtime dynamically. It never contains a copy of ggml.
 *
 * Conventions (same as kggml_runtime.h / kggml_whisper.h / kggml_llama.h):
 *   - Every function returning kggml_status is wrapped in a C++ try/catch. No exception ever crosses this
 *     boundary: std::bad_alloc -> KGGML_STATUS_OUT_OF_MEMORY, other std::exception ->
 *     KGGML_STATUS_NATIVE_FAILURE, anything else -> KGGML_STATUS_UNKNOWN.
 *   - On failure a diagnostic message is stored thread-locally; read it with kggml_diffusion_last_error().
 *   - Status codes 0..6 equal com.donfuldauer.kggml.core.NativeErrorCode. In particular a generation stopped by
 *     the abort callback or kggml_diffusion_request_abort() returns KGGML_STATUS_CANCELLED, a model that cannot be
 *     opened / parsed returns KGGML_STATUS_MODEL_LOAD_FAILED.
 *
 * Threading: a context is not thread-safe. Callers serialize every call on one context on one thread (the Kotlin
 * DiffusionEngine uses a dedicated single-threaded dispatcher per instance). The exceptions are
 * kggml_diffusion_request_abort() / kggml_diffusion_abort_requested(), which are lock-free and may be called from
 * any thread while the context is alive. Different contexts may generate concurrently on different threads: the
 * progress / abort callbacks of one generation are only ever invoked for that generation (see
 * kggml_diffusion_generate()).
 *
 * Process-wide effects of stable-diffusion.cpp (documented limitations, not per-context):
 *   - Its log and progress hooks are global. This library installs its own dispatchers once per process; do not
 *     call sd_set_log_callback() / sd_set_progress_callback() yourself while kggml_diffusion is in use.
 *   - new_sd_ctx() replaces the process-wide ggml logger with ggml's stderr default; kggml_diffusion_context_create()
 *     saves and restores the previous ggml logger around it, so other engines' ggml log routing is preserved.
 */
#ifndef KGGML_DIFFUSION_H
#define KGGML_DIFFUSION_H

#include "kggml_runtime.h" /* kggml_status, kggml_backend_kind */

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(KGGML_DIFFUSION_BUILD)
#        define KGGML_DIFFUSION_API __declspec(dllexport)
#    elif defined(KGGML_DIFFUSION_SHARED)
#        define KGGML_DIFFUSION_API __declspec(dllimport)
#    else
#        define KGGML_DIFFUSION_API
#    endif
#else
#    define KGGML_DIFFUSION_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------------------------------------ */
/* Versions & errors                                                                                       */
/* ------------------------------------------------------------------------------------------------------ */

/** Kggml diffusion module version, e.g. "1.0.0". Static string. */
KGGML_DIFFUSION_API const char * kggml_diffusion_version(void);

/** Upstream stable-diffusion.cpp `git describe` of the pin, e.g. "master-612-d7ecbe1-2-g3b4d26f". Static string. */
KGGML_DIFFUSION_API const char * kggml_diffusion_upstream_version(void);

/** Full 40-char upstream stable-diffusion.cpp commit SHA (see NOTICE). Static string. */
KGGML_DIFFUSION_API const char * kggml_diffusion_upstream_commit(void);

/** sd_get_system_info() of the linked stable-diffusion.cpp + ggml (CPU features etc). Never NULL. */
KGGML_DIFFUSION_API const char * kggml_diffusion_system_info(void);

/**
 * Message describing the most recent failing kggml_diffusion_* call on the calling thread ("" after a success).
 * When stable-diffusion.cpp logged an error during the failing call, its first error line is appended. Valid until
 * the next kggml_diffusion_* call on the same thread.
 */
KGGML_DIFFUSION_API const char * kggml_diffusion_last_error(void);

/** Clears the calling thread's diffusion last error. */
KGGML_DIFFUSION_API void kggml_diffusion_clear_last_error(void);

/* ------------------------------------------------------------------------------------------------------ */
/* Context (model weights + backends)                                                                     */
/* ------------------------------------------------------------------------------------------------------ */

typedef struct kggml_diffusion_context kggml_diffusion_context;

typedef struct kggml_diffusion_context_params {
    /* Model files (UTF-8 paths; .gguf, .safetensors or .ckpt). Either `model_path` (a full checkpoint) or
       `diffusion_model_path` (a standalone diffusion model, e.g. Flux / SD3.5 / Z-Image, plus the component files
       below) is required. Unused components are NULL or "". */
    const char * model_path;
    const char * diffusion_model_path;
    const char * vae_path;       /* optional VAE override                                     */
    const char * taesd_path;     /* optional tiny autoencoder (faster, lower-quality decode)  */
    const char * clip_l_path;    /* optional text encoders (SD3 / Flux)                       */
    const char * clip_g_path;
    const char * t5xxl_path;
    const char * llm_path;       /* optional LLM text encoder (Qwen-Image, Z-Image, ...)      */

    /**
     * kggml_backend_kind (BackendConfig.backend). KGGML_BACKEND_CPU pins every stable-diffusion.cpp module to the
     * ggml CPU device, so no GPU backend is ever created (Accelerate / NEON only; on iOS this stays legal in the
     * background). KGGML_BACKEND_METAL / KGGML_BACKEND_VULKAN run on the `gpu_device`-th GPU device of that kind.
     * KGGML_BACKEND_COREML is not supported by stable-diffusion.cpp and returns KGGML_STATUS_BACKEND_UNAVAILABLE.
     * Default: KGGML_BACKEND_CPU.
     */
    int32_t backend;
    int32_t gpu_device;            /* index among the GPU devices of the requested kind, default 0              */
    int32_t n_threads;             /* CPU threads (BackendConfig.cpuThreads), <= 0 = performance cores, default 0 */
    int32_t use_mmap;              /* 1 = mmap weight files (BackendConfig.useMmap), default 1                    */
    int32_t flash_attn;            /* 1 = flash attention everywhere (BackendConfig.useFlashAttention), default 0 */
    /**
     * 1 = keep the weights in host memory and stream them to the GPU per graph (stable-diffusion.cpp
     * offload_params_to_cpu). The closest equivalent of BackendConfig.gpuLayers = 0 with a GPU backend; ignored
     * for CPU. Default 0.
     */
    int32_t offload_params_to_cpu;
    int32_t keep_clip_on_cpu;      /* 1 = run the text encoders on the CPU even with a GPU backend, default 0    */
    int32_t keep_vae_on_cpu;       /* 1 = run the VAE on the CPU even with a GPU backend, default 0             */
    int32_t vae_decode_only;       /* 1 = txt2img only (no VAE encoder weights), default 1                       */
    /** Weight type to convert to at load time (stable-diffusion.cpp sd_type_t / ggml_type), -1 = as stored (default). */
    int32_t weight_type;
} kggml_diffusion_context_params;

/** Fills `params` with defaults (CPU, performance-core threads, mmap on, flash attention off, txt2img only). */
KGGML_DIFFUSION_API kggml_status kggml_diffusion_context_default_params(kggml_diffusion_context_params * params);

/**
 * Loads the model files and creates a context (new_sd_ctx). Returns KGGML_STATUS_INVALID_ARGUMENT for NULL /
 * empty arguments or out-of-range values, KGGML_STATUS_MODEL_LOAD_FAILED when a model file is missing, empty,
 * not a supported format or not a loadable diffusion model, KGGML_STATUS_BACKEND_UNAVAILABLE when the requested
 * backend is not compiled into the runtime / not present on this device. Holds one kggml_runtime_init()
 * reference until kggml_diffusion_context_free().
 */
KGGML_DIFFUSION_API kggml_status kggml_diffusion_context_create(const kggml_diffusion_context_params * params,
                                                               kggml_diffusion_context ** out_context);

/** Frees a context and its weights. NULL is a no-op. Must not race a running kggml_diffusion_generate(). */
KGGML_DIFFUSION_API void kggml_diffusion_context_free(kggml_diffusion_context * context);

typedef struct kggml_diffusion_context_info {
    int32_t backend;                   /* effective kggml_backend_kind                                    */
    int32_t uses_gpu;                  /* 1 if a GPU device was handed to stable-diffusion.cpp            */
    int32_t n_threads;                 /* resolved CPU thread count                                       */
    int32_t supports_image_generation; /* sd_ctx_supports_image_generation()                             */
    int32_t default_sample_method;     /* sd_get_default_sample_method() (sample_method_t)               */
} kggml_diffusion_context_info;

KGGML_DIFFUSION_API kggml_status kggml_diffusion_context_get_info(const kggml_diffusion_context * context,
                                                                 kggml_diffusion_context_info * out_info);

/* ------------------------------------------------------------------------------------------------------ */
/* Abort                                                                                                   */
/* ------------------------------------------------------------------------------------------------------ */

/**
 * Latches an abort request on `context` (lock-free; any thread, any time while the context is alive; NULL is a
 * no-op). A running kggml_diffusion_generate() stops at its next step boundary and returns
 * KGGML_STATUS_CANCELLED; a later generate() returns KGGML_STATUS_CANCELLED immediately until
 * kggml_diffusion_clear_abort() is called.
 */
KGGML_DIFFUSION_API void kggml_diffusion_request_abort(kggml_diffusion_context * context);

/** Clears a latched abort request. Call it before starting a new generation. */
KGGML_DIFFUSION_API void kggml_diffusion_clear_abort(kggml_diffusion_context * context);

/** 1 if an abort is currently requested, otherwise 0 (0 for NULL). */
KGGML_DIFFUSION_API int32_t kggml_diffusion_abort_requested(const kggml_diffusion_context * context);

/* ------------------------------------------------------------------------------------------------------ */
/* Generation (txt2img)                                                                                    */
/* ------------------------------------------------------------------------------------------------------ */

/** Sampler / scheduler "let the model decide" value (stable-diffusion.cpp's per-model default). */
#define KGGML_DIFFUSION_DEFAULT (-1)

typedef struct kggml_diffusion_generate_params {
    const char * prompt;          /* required, UTF-8 (may be "")                                               */
    const char * negative_prompt; /* NULL = ""                                                                  */
    int32_t width;                /* pixels, > 0 and a multiple of 8, default 512                              */
    int32_t height;               /* pixels, > 0 and a multiple of 8, default 512                              */
    int32_t sample_steps;         /* > 0, default 20                                                            */
    float   cfg_scale;            /* classifier-free guidance (txt_cfg), default 7.0                           */
    float   distilled_guidance;   /* Flux-style distilled guidance, default 3.5                                */
    int32_t sample_method;        /* stable-diffusion.cpp sample_method_t, KGGML_DIFFUSION_DEFAULT = per model */
    int32_t scheduler;            /* stable-diffusion.cpp scheduler_t,    KGGML_DIFFUSION_DEFAULT = per model */
    int64_t seed;                 /* < 0 = random, default -1                                                  */
    int32_t clip_skip;            /* <= 0 = model default, default -1                                          */
    int32_t vae_tiling;           /* 1 = tiled VAE decode (lower peak memory), default 0                       */
} kggml_diffusion_generate_params;

/** Fills `params` with defaults (512x512, 20 steps, cfg 7.0, per-model sampler / scheduler, random seed). */
KGGML_DIFFUSION_API kggml_status kggml_diffusion_generate_default_params(kggml_diffusion_generate_params * params);

/**
 * Progress callback, invoked on the thread that called kggml_diffusion_generate(): `step` = 0 when sampling
 * starts, then 1..`total_steps` after each completed denoising step. With vae_tiling = 1 the VAE decode reports
 * its tiles the same way afterwards (a new 0..n sequence with a different `total_steps`).
 * `seconds_per_step` is upstream's timing estimate. Must not throw (a throwing callback is treated as an abort).
 */
typedef void (*kggml_diffusion_progress_callback)(int32_t step, int32_t total_steps, float seconds_per_step,
                                                  void * user_data);

/**
 * Abort callback (BackendConfig / coroutine cancellation): polled before generation starts and at every progress
 * point (each denoising step / VAE tile). Return non-zero to stop; kggml_diffusion_generate() then returns
 * KGGML_STATUS_CANCELLED. Invoked on the generating thread. Must not throw (a throwing callback aborts).
 */
typedef int32_t (*kggml_diffusion_abort_callback)(void * user_data);

/** An 8-bit RGB image (channels = 3, row-major, width * height * 3 bytes, no row padding). */
typedef struct kggml_diffusion_image {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint8_t * data; /* owned; release with kggml_diffusion_image_free() */
} kggml_diffusion_image;

/**
 * Generates one image from `params` (generate_image, txt2img). `progress_cb` / `abort_cb` may be NULL; both receive
 * `user_data` and are only ever invoked for THIS call, on the calling thread (concurrent generations on other
 * contexts never see them).
 *
 * Abort granularity is one denoising step (or one VAE tile): stable-diffusion.cpp has no per-graph abort hook at
 * the pinned commit, so the wrapper stops the sampler at its next progress point and unwinds it. The step that is
 * computing when the abort arrives finishes first.
 *
 * On KGGML_STATUS_OK `*out_image` holds the RGB8 result (free it with kggml_diffusion_image_free()); on any other
 * status it is zeroed. Returns KGGML_STATUS_INVALID_ARGUMENT for NULL / out-of-range arguments,
 * KGGML_STATUS_CANCELLED when aborted, KGGML_STATUS_NATIVE_FAILURE when stable-diffusion.cpp fails to generate.
 */
KGGML_DIFFUSION_API kggml_status kggml_diffusion_generate(kggml_diffusion_context * context,
                                                         const kggml_diffusion_generate_params * params,
                                                         kggml_diffusion_progress_callback progress_cb,
                                                         kggml_diffusion_abort_callback abort_cb, void * user_data,
                                                         kggml_diffusion_image * out_image);

/** Frees `image->data` and zeroes `*image`. NULL and already-freed (zeroed) images are no-ops. */
KGGML_DIFFUSION_API void kggml_diffusion_image_free(kggml_diffusion_image * image);

#ifdef __cplusplus
}
#endif

#endif /* KGGML_DIFFUSION_H */
