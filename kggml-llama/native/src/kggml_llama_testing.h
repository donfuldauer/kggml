/*
 * kggml_llama_testing.h - INTERNAL fault-injection and introspection hooks for the kggml_llama test suite.
 *
 * Only compiled in when KGGML_LLAMA_TEST_HOOKS is ON (implied by KGGML_LLAMA_BUILD_TESTS). Release and
 * packaging builds never export these symbols. Not part of the public include/ directory.
 */
#ifndef KGGML_LLAMA_TESTING_H
#define KGGML_LLAMA_TESTING_H

#include "kggml_llama.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum kggml_llama_test_fault {
    KGGML_LLAMA_TEST_FAULT_NONE          = 0,
    KGGML_LLAMA_TEST_FAULT_BAD_ALLOC     = 1, /* std::bad_alloc            -> OUT_OF_MEMORY  */
    KGGML_LLAMA_TEST_FAULT_RUNTIME_ERROR = 2, /* std::runtime_error        -> NATIVE_FAILURE */
    KGGML_LLAMA_TEST_FAULT_LOGIC_ERROR   = 3, /* std::out_of_range         -> NATIVE_FAILURE */
    KGGML_LLAMA_TEST_FAULT_NON_STD       = 4  /* throw 42                  -> UNKNOWN        */
} kggml_llama_test_fault;

/** Arms a one-shot fault: the next guarded kggml_llama_* call on this thread throws it inside its guard. */
KGGML_LLAMA_API void kggml_llama_test_inject_fault(int32_t fault);

/**
 * Invokes the exact abort callback that kggml_llama_context_create() registered with llama_set_abort_callback()
 * and reports its result (1 = abort, 0 = continue).
 */
KGGML_LLAMA_API kggml_status kggml_llama_test_poll_abort(kggml_llama_context * context, int32_t * out_abort);

/**
 * Arms a test-only countdown: after `n` further abort-callback polls the context requests abort on itself,
 * simulating a cancellation that lands in the middle of a llama_decode() graph compute. -1 disarms.
 */
KGGML_LLAMA_API kggml_status kggml_llama_test_abort_after_polls(kggml_llama_context * context, int32_t n);

/** Number of abort-callback polls llama.cpp made on `context` since it was created (proves the plumbing). */
KGGML_LLAMA_API kggml_status kggml_llama_test_abort_poll_count(const kggml_llama_context * context,
                                                              int64_t * out_polls);

/** Number of kggml_llama models / contexts / samplers / mtmd objects currently alive in this process. */
KGGML_LLAMA_API int64_t kggml_llama_test_live_objects(void);

/**
 * Resolves the llama.cpp device list kggml_llama_model_load() would pass for `params`: `out_n_devices` = number
 * of GPU devices handed to llama.cpp (0 for CPU, which means "no offload at all"), `out_n_gpu_layers` the
 * llama_model_params.n_gpu_layers value.
 */
KGGML_LLAMA_API kggml_status kggml_llama_test_resolve_devices(const kggml_llama_model_params * params,
                                                             int32_t * out_n_devices, int32_t * out_n_gpu_layers);

#ifdef __cplusplus
}
#endif

#endif /* KGGML_LLAMA_TESTING_H */
