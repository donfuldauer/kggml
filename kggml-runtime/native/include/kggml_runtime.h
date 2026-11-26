/*
 * kggml_runtime.h - C ABI of the Kggml shared ggml runtime (:kggml-runtime).
 *
 * Every function is safe to call from any thread, never throws across the FFI boundary, and reports
 * failures through a kggml_status code plus a thread-local diagnostic message (kggml_runtime_last_error).
 *
 * Status codes 0..6 are numerically identical to com.donfuldauer.kggml.core.NativeErrorCode so the Kotlin
 * bridges can feed them straight into InferenceException.fromNativeError(). Runtime-specific codes live in
 * the 100+ range; Kotlin maps them to NativeFailureException, which preserves the raw code.
 */
#ifndef KGGML_RUNTIME_H
#define KGGML_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(KGGML_RUNTIME_BUILD)
#        define KGGML_API __declspec(dllexport)
#    elif defined(KGGML_RUNTIME_SHARED)
#        define KGGML_API __declspec(dllimport)
#    else
#        define KGGML_API
#    endif
#else
#    define KGGML_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------------------------------------ */
/* Status codes                                                                                            */
/* ------------------------------------------------------------------------------------------------------ */

typedef enum kggml_status {
    KGGML_STATUS_OK                  = 0,   /* NativeErrorCode.OK                                     */
    KGGML_STATUS_MODEL_LOAD_FAILED   = 1,   /* NativeErrorCode.MODEL_LOAD_FAILED  (engines only)      */
    KGGML_STATUS_INVALID_ARGUMENT    = 2,   /* NativeErrorCode.INVALID_INPUT                          */
    KGGML_STATUS_BACKEND_UNAVAILABLE = 3,   /* NativeErrorCode.BACKEND_UNAVAILABLE                    */
    KGGML_STATUS_OUT_OF_MEMORY       = 4,   /* NativeErrorCode.OUT_OF_MEMORY      (std::bad_alloc)    */
    KGGML_STATUS_CANCELLED           = 5,   /* NativeErrorCode.CANCELLED          (engines only)      */
    KGGML_STATUS_NATIVE_FAILURE      = 6,   /* NativeErrorCode.NATIVE_FAILURE     (std::exception)    */

    /* Runtime-specific codes -> NativeFailureException(nativeCode = code) on the Kotlin side. */
    KGGML_STATUS_NOT_INITIALIZED     = 100, /* kggml_runtime_init() has not been called (or ref count hit 0) */
    KGGML_STATUS_BUFFER_TOO_SMALL    = 101, /* caller buffer too small; required size is reported         */
    KGGML_STATUS_UNKNOWN             = 102  /* non-std::exception thrown (catch (...))                    */
} kggml_status;

/** Static, never-NULL, human-readable name of a status code ("KGGML_STATUS_OK", ...). */
KGGML_API const char * kggml_status_name(kggml_status status);

/* ------------------------------------------------------------------------------------------------------ */
/* Lifecycle (reference counted, idempotent)                                                               */
/* ------------------------------------------------------------------------------------------------------ */

/**
 * Initializes the runtime (ggml CPU tables, backend registry). Reference counted: every successful call
 * must be balanced by one kggml_runtime_shutdown(). Safe to call concurrently.
 */
KGGML_API kggml_status kggml_runtime_init(void);

/**
 * Releases one reference. Calling it when the runtime is not initialized is a no-op returning
 * KGGML_STATUS_OK (idempotent), so the ref count never goes negative.
 */
KGGML_API kggml_status kggml_runtime_shutdown(void);

/** 1 when the runtime holds at least one reference, otherwise 0. */
KGGML_API int32_t kggml_runtime_is_initialized(void);

/** Current reference count (diagnostics / tests). */
KGGML_API int32_t kggml_runtime_ref_count(void);

/* ------------------------------------------------------------------------------------------------------ */
/* Backend enumeration                                                                                     */
/* ------------------------------------------------------------------------------------------------------ */

/** Logical compute backends. Mirrors com.donfuldauer.kggml.core.ComputeBackend (minus AUTO). */
typedef enum kggml_backend_kind {
    KGGML_BACKEND_CPU    = 0, /* ComputeBackend.CPU_ONLY   - always compiled and available */
    KGGML_BACKEND_METAL  = 1, /* ComputeBackend.GPU_METAL  */
    KGGML_BACKEND_VULKAN = 2, /* ComputeBackend.GPU_VULKAN */
    KGGML_BACKEND_COREML = 3  /* ComputeBackend.NPU_COREML - compile-time capability flag only */
} kggml_backend_kind;

#define KGGML_BACKEND_KIND_COUNT 4
#define KGGML_BACKEND_NAME_MAX   32

typedef struct kggml_backend_info {
    int32_t kind;                          /* kggml_backend_kind                                       */
    int32_t compiled;                      /* 1 if support was compiled into this runtime build         */
    int32_t available;                     /* 1 if usable on this device right now (compiled && device) */
    int32_t device_count;                  /* ggml devices backing this kind (0 for COREML)             */
    char    name[KGGML_BACKEND_NAME_MAX];  /* NUL-terminated, e.g. "CPU", "Metal", "Vulkan", "CoreML"  */
} kggml_backend_info;

/**
 * Fills `out` with one entry per kggml_backend_kind (KGGML_BACKEND_KIND_COUNT entries, CPU first).
 *
 * `out_count` (required) always receives the number of entries needed. If `capacity` is smaller than that
 * number, nothing is written to `out` and KGGML_STATUS_BUFFER_TOO_SMALL is returned; pass `out = NULL`,
 * `capacity = 0` to query the size. Requires kggml_runtime_init().
 */
KGGML_API kggml_status kggml_runtime_get_backends(kggml_backend_info * out, int32_t capacity, int32_t * out_count);

/** Writes 1/0 to `out_available` for a single backend kind. Requires kggml_runtime_init(). */
KGGML_API kggml_status kggml_runtime_backend_available(int32_t kind, int32_t * out_available);

/** 1 if this build advertises the CoreML capability (compile-time KGGML_ENABLE_COREML), otherwise 0. */
KGGML_API int32_t kggml_runtime_has_coreml(void);

/* ------------------------------------------------------------------------------------------------------ */
/* CPU topology                                                                                            */
/* ------------------------------------------------------------------------------------------------------ */

/**
 * Number of performance cores, a good default for inference thread pools. Always in [1, hardware threads].
 * Linux/Android (SMT siblings counted once): with differing cpu_capacity (ARM phones), the cores with at least half
 * the highest capacity, i.e. prime + performance cores without the efficiency cores; otherwise the cores whose
 * cpuinfo_max_freq equals the highest one. Apple: hw.perflevel0.physicalcpu (fallback hw.physicalcpu); Windows: cores
 * of the highest EfficiencyClass. Falls back to std::thread::hardware_concurrency(). Does not require init.
 */
KGGML_API kggml_status kggml_runtime_performance_thread_count(int32_t * out_count);

/** Logical hardware threads (std::thread::hardware_concurrency(), at least 1). Does not require init. */
KGGML_API int32_t kggml_runtime_hardware_thread_count(void);

/* ------------------------------------------------------------------------------------------------------ */
/* Versions                                                                                                */
/* ------------------------------------------------------------------------------------------------------ */

/** Kggml runtime version, e.g. "1.0.0". Static string. */
KGGML_API const char * kggml_runtime_version(void);

/** Version reported by the linked ggml (ggml_version()), e.g. "0.9.9". Static string. */
KGGML_API const char * kggml_runtime_ggml_version(void);

/** Full 40-char upstream ggml commit SHA this runtime was pinned to (see NOTICE). Static string. */
KGGML_API const char * kggml_runtime_ggml_commit(void);

/** Commit string compiled into ggml itself (ggml_commit(); short SHA or "unknown"). Static string. */
KGGML_API const char * kggml_runtime_ggml_build_commit(void);

/* ------------------------------------------------------------------------------------------------------ */
/* Errors                                                                                                  */
/* ------------------------------------------------------------------------------------------------------ */

/**
 * Message describing the most recent failing call on the calling thread ("" if the last call succeeded).
 * The pointer stays valid until the next kggml_runtime_* call on the same thread; copy it immediately or
 * use kggml_runtime_copy_last_error().
 */
KGGML_API const char * kggml_runtime_last_error(void);

/**
 * Copies the thread-local last error message (NUL-terminated) into `buffer`. `out_length` (optional)
 * receives strlen(message). Returns KGGML_STATUS_BUFFER_TOO_SMALL if `capacity` <= strlen(message), in
 * which case a truncated, still NUL-terminated prefix is written (nothing when `capacity` is 0, which makes
 * `buffer = NULL, capacity = 0` a size query). `buffer = NULL` with `capacity > 0` is
 * KGGML_STATUS_INVALID_ARGUMENT. Does not modify the last error itself.
 */
KGGML_API kggml_status kggml_runtime_copy_last_error(char * buffer, size_t capacity, size_t * out_length);

/** Clears the calling thread's last error message. */
KGGML_API void kggml_runtime_clear_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* KGGML_RUNTIME_H */
