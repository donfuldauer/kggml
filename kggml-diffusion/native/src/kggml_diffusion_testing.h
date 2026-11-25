/*
 * kggml_diffusion_testing.h - INTERNAL fault-injection and introspection hooks for the kggml_diffusion test suite.
 *
 * Only compiled in when KGGML_DIFFUSION_TEST_HOOKS is ON (implied by KGGML_DIFFUSION_BUILD_TESTS). Release and
 * packaging builds never export these symbols. Not part of the public include/ directory.
 */
#ifndef KGGML_DIFFUSION_TESTING_H
#define KGGML_DIFFUSION_TESTING_H

#include "kggml_diffusion.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum kggml_diffusion_test_fault {
    KGGML_DIFFUSION_TEST_FAULT_NONE          = 0,
    KGGML_DIFFUSION_TEST_FAULT_BAD_ALLOC     = 1, /* std::bad_alloc     -> OUT_OF_MEMORY  */
    KGGML_DIFFUSION_TEST_FAULT_RUNTIME_ERROR = 2, /* std::runtime_error -> NATIVE_FAILURE */
    KGGML_DIFFUSION_TEST_FAULT_LOGIC_ERROR   = 3, /* std::out_of_range  -> NATIVE_FAILURE */
    KGGML_DIFFUSION_TEST_FAULT_NON_STD       = 4  /* throw 42           -> UNKNOWN        */
} kggml_diffusion_test_fault;

/** Arms a one-shot fault: the next guarded kggml_diffusion_* call on this thread throws it inside its guard. */
KGGML_DIFFUSION_API void kggml_diffusion_test_inject_fault(int32_t fault);

/**
 * Creates a context WITHOUT a model (no stable-diffusion.cpp sd_ctx, no runtime reference) so the abort flag and
 * kggml_diffusion_test_simulate_generate() can be exercised without weights. Free with
 * kggml_diffusion_context_free(). kggml_diffusion_generate() on it returns KGGML_STATUS_INVALID_ARGUMENT.
 */
KGGML_DIFFUSION_API kggml_status kggml_diffusion_test_context_create_stub(kggml_diffusion_context ** out_context);

/**
 * Runs the exact generation machinery of kggml_diffusion_generate() (argument checks, latched-abort check, the
 * per-thread progress / abort dispatch, the AbortSignal unwind and the guard) with stable-diffusion.cpp's sampler
 * replaced by a loop that reports steps 0..`steps` through upstream's own pretty_progress() -- i.e. through the
 * process-wide sd_set_progress_callback() hook exactly as a real sampler does. On success `*out_image` receives a
 * deterministic `width` x `height` RGB8 gradient (pixel (x, y) = {x & 255, y & 255, (x + y) & 255}).
 *
 * `fault_at_step` >= 0 with `fault` != NONE throws that fault from inside the progress dispatch when that step is
 * reported (it unwinds through upstream frames like an abort would). `ctx` may be a stub or a real context.
 */
KGGML_DIFFUSION_API kggml_status kggml_diffusion_test_simulate_generate(
    kggml_diffusion_context * context, int32_t steps, int32_t width, int32_t height,
    kggml_diffusion_progress_callback progress_cb, kggml_diffusion_abort_callback abort_cb, void * user_data,
    int32_t fault_at_step, int32_t fault, kggml_diffusion_image * out_image);

/**
 * Emits `text` through stable-diffusion.cpp's own logger (log_printf) at `level` (sd_log_level_t) on this thread,
 * so the test can observe the wrapper's log routing (ERROR lines are captured for last_error).
 */
KGGML_DIFFUSION_API void kggml_diffusion_test_emit_sd_log(int32_t level, const char * text);

/** The first stable-diffusion.cpp ERROR line captured on this thread since the last guarded kggml_diffusion_* call ("" if none). */
KGGML_DIFFUSION_API const char * kggml_diffusion_test_captured_sd_error(void);

/** Number of kggml_diffusion contexts currently alive in this process. */
KGGML_DIFFUSION_API int64_t kggml_diffusion_test_live_objects(void);

/**
 * Resolves the stable-diffusion.cpp backend spec kggml_diffusion_context_create() would pass for `params` into
 * `buffer` (NUL-terminated, e.g. "CPU" or "Vulkan0"), without loading anything. `out_uses_gpu` (optional) = 1 for
 * a GPU device. Requires nothing but the runtime (it takes and releases a runtime reference itself).
 */
KGGML_DIFFUSION_API kggml_status kggml_diffusion_test_resolve_backend(const kggml_diffusion_context_params * params,
                                                                     char * buffer, size_t capacity,
                                                                     int32_t * out_uses_gpu);

#ifdef __cplusplus
}
#endif

#endif /* KGGML_DIFFUSION_TESTING_H */
