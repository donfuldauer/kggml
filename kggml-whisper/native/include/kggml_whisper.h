/*
 * kggml_whisper.h - C ABI of the Kggml whisper.cpp engine (:kggml-whisper native layer).
 *
 * libkggml_whisper (KggmlWhisper.framework on iOS) contains whisper.cpp plus this wrapper and links the
 * shared ggml from :kggml-runtime dynamically. It never contains a copy of ggml.
 *
 * Conventions (same as kggml_runtime.h):
 *   - Every function returning kggml_status is wrapped in a C++ try/catch. No exception ever crosses this
 *     boundary: std::bad_alloc -> KGGML_STATUS_OUT_OF_MEMORY, other std::exception ->
 *     KGGML_STATUS_NATIVE_FAILURE, anything else -> KGGML_STATUS_UNKNOWN.
 *   - On failure a diagnostic message is stored thread-locally; read it with kggml_whisper_last_error().
 *   - Status codes 0..6 equal com.donfuldauer.kggml.core.NativeErrorCode.
 *
 * Threading: a kggml_whisper_context is NOT thread-safe. Callers serialize init / transcribe / free on one
 * thread (the Kotlin engines use a dedicated single-threaded dispatcher per instance). The only exceptions
 * are kggml_whisper_request_abort() and kggml_whisper_abort_requested(), which are lock-free and may be
 * called from any thread at any time while the context is alive.
 *
 * Resamplers and the stateless PCM helpers have no shared state: each kggml_whisper_resampler must be
 * used by one thread at a time, different instances are independent.
 */
#ifndef KGGML_WHISPER_H
#define KGGML_WHISPER_H

#include "kggml_runtime.h" /* kggml_status, kggml_backend_kind */

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(KGGML_WHISPER_BUILD)
#        define KGGML_WHISPER_API __declspec(dllexport)
#    elif defined(KGGML_WHISPER_SHARED)
#        define KGGML_WHISPER_API __declspec(dllimport)
#    else
#        define KGGML_WHISPER_API
#    endif
#else
#    define KGGML_WHISPER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Sample rate whisper.cpp expects (WHISPER_SAMPLE_RATE). */
#define KGGML_WHISPER_SAMPLE_RATE 16000

/** Minimum audio length handed to whisper_full_with_state(); shorter input is zero-padded to it (1000 ms). */
#define KGGML_WHISPER_MIN_SAMPLES 16000

/* ------------------------------------------------------------------------------------------------------ */
/* Versions & capabilities                                                                                 */
/* ------------------------------------------------------------------------------------------------------ */

/** Kggml whisper module version, e.g. "1.0.0". Static string. */
KGGML_WHISPER_API const char * kggml_whisper_version(void);

/** Upstream whisper.cpp release this build was pinned to, e.g. "1.8.4". Static string. */
KGGML_WHISPER_API const char * kggml_whisper_upstream_version(void);

/** Full 40-char upstream whisper.cpp commit SHA (see NOTICE). Static string. */
KGGML_WHISPER_API const char * kggml_whisper_upstream_commit(void);

/** whisper_print_system_info() of the linked whisper.cpp + ggml (CPU features etc). Static string. */
KGGML_WHISPER_API const char * kggml_whisper_system_info(void);

/**
 * 1 if this build compiled the CoreML encoder (WHISPER_COREML, iOS / macOS only), otherwise 0. When 1, a
 * context created with backend KGGML_BACKEND_COREML loads "<model>-encoder.mlmodelc" next to the model
 * and runs the encoder on the Apple Neural Engine; when that bundle is missing it falls back to the GPU
 * (Metal) or CPU encoder automatically.
 */
KGGML_WHISPER_API int32_t kggml_whisper_has_coreml(void);

/* ------------------------------------------------------------------------------------------------------ */
/* Errors                                                                                                  */
/* ------------------------------------------------------------------------------------------------------ */

/**
 * Message describing the most recent failing kggml_whisper_* call on the calling thread ("" after a
 * success). Valid until the next kggml_whisper_* call on the same thread.
 */
KGGML_WHISPER_API const char * kggml_whisper_last_error(void);

/** Clears the calling thread's whisper last error. */
KGGML_WHISPER_API void kggml_whisper_clear_last_error(void);

/* ------------------------------------------------------------------------------------------------------ */
/* PCM conversion (16 kHz mono Float32 is the only format whisper.cpp accepts)                             */
/* ------------------------------------------------------------------------------------------------------ */

typedef enum kggml_pcm_format {
    KGGML_PCM_F32 = 0, /* interleaved float32, nominal range [-1, 1]                  */
    KGGML_PCM_S16 = 1  /* interleaved signed 16-bit, normalized as sample / 32768.0f  */
} kggml_pcm_format;

typedef enum kggml_resample_quality {
    KGGML_RESAMPLE_LINEAR = 0, /* 2-tap linear interpolation: cheapest, no anti-aliasing filter            */
    KGGML_RESAMPLE_SINC   = 1  /* Kaiser-windowed sinc low-pass (default): band-limited, alias-suppressing */
} kggml_resample_quality;

/** Converts `count` int16 samples to float32 (sample / 32768.0f, so the range is [-1, 1)). In-place unsafe. */
KGGML_WHISPER_API kggml_status kggml_whisper_pcm_s16_to_f32(const int16_t * in, size_t count, float * out);

/**
 * Averages `channels` interleaved float32 channels into mono. `out` receives `frames` samples and may alias
 * `in` (the write position never overtakes the read position). channels == 1 is a copy.
 */
KGGML_WHISPER_API kggml_status kggml_whisper_pcm_downmix_f32(const float * in, size_t frames, int32_t channels,
                                                             float * out);

/**
 * Exact number of output frames produced when `in_frames` frames at `in_rate` are fully converted
 * (process + flush) to `out_rate`: ceil(in_frames * out_rate / in_rate).
 */
KGGML_WHISPER_API kggml_status kggml_whisper_resampled_length(size_t in_frames, int32_t in_rate, int32_t out_rate,
                                                              size_t * out_frames);

/**
 * One-shot conversion of interleaved `format` PCM (`channels`, `sample_rate`) to 16 kHz mono float32.
 * `out_capacity` must be >= kggml_whisper_resampled_length(frames, sample_rate, 16000); otherwise
 * KGGML_STATUS_BUFFER_TOO_SMALL is returned and `out_frames` receives the required size (pass out = NULL,
 * out_capacity = 0 to query). Supported: 1 <= channels <= 8, 1000 <= sample_rate <= 384000.
 */
KGGML_WHISPER_API kggml_status kggml_whisper_pcm_to_whisper(const void * in, int32_t format, size_t frames,
                                                           int32_t channels, int32_t sample_rate, int32_t quality,
                                                           float * out, size_t out_capacity, size_t * out_frames);

/** Streaming resampler: chunked input produces the same samples as one-shot conversion of the whole stream. */
typedef struct kggml_whisper_resampler kggml_whisper_resampler;

/**
 * Creates a streaming converter from interleaved `format` PCM (`channels`, `in_rate`) to mono float32 at
 * `out_rate` (normally KGGML_WHISPER_SAMPLE_RATE). Free with kggml_whisper_resampler_free().
 */
KGGML_WHISPER_API kggml_status kggml_whisper_resampler_create(int32_t in_rate, int32_t out_rate, int32_t channels,
                                                             int32_t format, int32_t quality,
                                                             kggml_whisper_resampler ** out_resampler);

/** Frees a resampler. NULL is a no-op. */
KGGML_WHISPER_API void kggml_whisper_resampler_free(kggml_whisper_resampler * resampler);

/** Upper bound of frames the next process() call with `in_frames` input frames (or flush(), with 0) emits. */
KGGML_WHISPER_API kggml_status kggml_whisper_resampler_max_output(const kggml_whisper_resampler * resampler,
                                                                 size_t in_frames, size_t * out_frames);

/**
 * Feeds `in_frames` interleaved frames and writes every output sample that is now fully determined.
 * Output lags input by the filter half-width; call kggml_whisper_resampler_flush() at end of stream.
 * If `out_capacity` < kggml_whisper_resampler_max_output(in_frames), returns BUFFER_TOO_SMALL without
 * consuming anything and stores that bound in `out_frames`.
 */
KGGML_WHISPER_API kggml_status kggml_whisper_resampler_process(kggml_whisper_resampler * resampler, const void * in,
                                                              size_t in_frames, float * out, size_t out_capacity,
                                                              size_t * out_frames);

/** Emits the remaining tail (input treated as followed by silence) and resets the resampler for a new stream. */
KGGML_WHISPER_API kggml_status kggml_whisper_resampler_flush(kggml_whisper_resampler * resampler, float * out,
                                                            size_t out_capacity, size_t * out_frames);

/** Discards all buffered input and restarts at stream position 0. */
KGGML_WHISPER_API kggml_status kggml_whisper_resampler_reset(kggml_whisper_resampler * resampler);

/* ------------------------------------------------------------------------------------------------------ */
/* Context                                                                                                 */
/* ------------------------------------------------------------------------------------------------------ */

typedef struct kggml_whisper_context kggml_whisper_context;

typedef struct kggml_whisper_context_params {
    /**
     * kggml_backend_kind. KGGML_BACKEND_CPU sets whisper_context_params.use_gpu = false *before*
     * whisper_init_from_file_with_params_no_state(), so ggml never creates a GPU backend (on iOS this keeps
     * all work on Accelerate / NEON, which stays legal in the background). KGGML_BACKEND_METAL /
     * KGGML_BACKEND_VULKAN set use_gpu = true. KGGML_BACKEND_COREML additionally enables the CoreML
     * encoder (see kggml_whisper_has_coreml()); the decoder then uses the GPU when one is available.
     * Default: KGGML_BACKEND_CPU.
     */
    int32_t backend;
    int32_t gpu_device;  /* ggml GPU device index, default 0                       */
    int32_t flash_attn;  /* 1 = whisper flash attention (upstream default, all backends) */
} kggml_whisper_context_params;

/** Fills `params` with defaults (CPU, device 0, flash attention on). */
KGGML_WHISPER_API kggml_status kggml_whisper_context_default_params(kggml_whisper_context_params * params);

/**
 * Loads a whisper ggml model (.bin) from `model_path`. Returns KGGML_STATUS_MODEL_LOAD_FAILED when the file
 * is missing or invalid, KGGML_STATUS_BACKEND_UNAVAILABLE when the requested backend is not compiled into
 * the runtime / not present on this device (CoreML: not compiled into this module). `params` may be NULL
 * for defaults. Holds one kggml_runtime_init() reference until kggml_whisper_free().
 */
KGGML_WHISPER_API kggml_status kggml_whisper_init_from_file(const char * model_path,
                                                           const kggml_whisper_context_params * params,
                                                           kggml_whisper_context ** out_context);

/** Frees a context. NULL is a no-op. Must not race with any other call on the same context. */
KGGML_WHISPER_API void kggml_whisper_free(kggml_whisper_context * context);

/** Effective backend: kggml_backend_kind actually used (COREML only if the .mlmodelc encoder loaded). */
KGGML_WHISPER_API kggml_status kggml_whisper_get_backend(const kggml_whisper_context * context, int32_t * out_backend);

/** 1 when whisper_context_params.use_gpu was true for this context, otherwise 0. */
KGGML_WHISPER_API kggml_status kggml_whisper_uses_gpu(const kggml_whisper_context * context, int32_t * out_uses_gpu);

/** 1 if the model is multilingual. */
KGGML_WHISPER_API kggml_status kggml_whisper_is_multilingual(const kggml_whisper_context * context, int32_t * out);

/* ------------------------------------------------------------------------------------------------------ */
/* Abort (any thread)                                                                                      */
/* ------------------------------------------------------------------------------------------------------ */

/**
 * Atomically requests cancellation of the transcription running (or next started) on `context`. Lock-free
 * and safe from any thread, e.g. a Kotlin Job.invokeOnCompletion handler. whisper.cpp polls it through
 * whisper_full_params.abort_callback (inside every ggml graph compute) and encoder_begin_callback, and the
 * interrupted kggml_whisper_transcribe() returns KGGML_STATUS_CANCELLED. The request stays latched until
 * kggml_whisper_clear_abort().
 */
KGGML_WHISPER_API void kggml_whisper_request_abort(kggml_whisper_context * context);

/** Clears a latched abort request. Call it before starting a new transcription. */
KGGML_WHISPER_API void kggml_whisper_clear_abort(kggml_whisper_context * context);

/** 1 if an abort is currently requested, otherwise 0 (0 for NULL). */
KGGML_WHISPER_API int32_t kggml_whisper_abort_requested(const kggml_whisper_context * context);

/* ------------------------------------------------------------------------------------------------------ */
/* Transcription                                                                                           */
/* ------------------------------------------------------------------------------------------------------ */

typedef struct kggml_whisper_transcribe_params {
    const char * language;       /* ISO 639-1 code, "auto" or NULL (= "en" for English-only models, else auto) */
    const char * initial_prompt; /* optional decoder prompt (NULL = none)                                      */
    int32_t      n_threads;      /* 0 = kggml_runtime_performance_thread_count()                               */
    int32_t      translate;      /* 1 = translate to English                                                    */
    int32_t      no_context;     /* 1 = do not carry text context over from the previous call                  */
    int32_t      single_segment; /* 1 = force a single output segment (streaming)                              */
    int32_t      no_timestamps;  /* 1 = do not decode timestamp tokens                                          */
    int32_t      max_tokens;     /* max tokens per segment, 0 = no limit                                        */
    float        temperature;    /* initial sampling temperature, 0 = greedy                                    */
    float        temperature_inc;/* fallback increment, 0 disables temperature fallback                         */
} kggml_whisper_transcribe_params;

/** Fills `params` with defaults (auto language, greedy, temperature fallback 0.2, all flags 0). */
KGGML_WHISPER_API kggml_status kggml_whisper_transcribe_default_params(kggml_whisper_transcribe_params * params);

/**
 * Runs whisper_full_with_state() on `n_samples` 16 kHz mono float32 samples (zero-copy: `samples` is read
 * in place, except that input shorter than KGGML_WHISPER_MIN_SAMPLES is copied and zero-padded to 1000 ms).
 * `params` may be NULL for defaults. Returns KGGML_STATUS_CANCELLED if kggml_whisper_request_abort() was
 * called before or during the run; segments of a cancelled run are discarded (count 0).
 */
KGGML_WHISPER_API kggml_status kggml_whisper_transcribe(kggml_whisper_context * context, const float * samples,
                                                       size_t n_samples,
                                                       const kggml_whisper_transcribe_params * params);

typedef struct kggml_whisper_segment {
    int64_t      t0_ms;          /* segment start, milliseconds from the start of the transcribed buffer */
    int64_t      t1_ms;          /* segment end                                                         */
    float        no_speech_prob; /* whisper no-speech probability of the segment                       */
    const char * text;           /* UTF-8, NUL-terminated; valid until the next transcribe / free       */
} kggml_whisper_segment;

/** Number of segments produced by the last successful kggml_whisper_transcribe(). */
KGGML_WHISPER_API kggml_status kggml_whisper_segment_count(const kggml_whisper_context * context, int32_t * out_count);

/** Segment `index` (0 <= index < count) of the last transcription. */
KGGML_WHISPER_API kggml_status kggml_whisper_get_segment(const kggml_whisper_context * context, int32_t index,
                                                        kggml_whisper_segment * out_segment);

#ifdef __cplusplus
}
#endif

#endif /* KGGML_WHISPER_H */
