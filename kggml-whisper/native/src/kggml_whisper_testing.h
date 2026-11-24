/*
 * kggml_whisper_testing.h - INTERNAL fault-injection hooks for the kggml_whisper test suite.
 *
 * Only compiled in when KGGML_WHISPER_TEST_HOOKS is ON (implied by KGGML_WHISPER_BUILD_TESTS). Release and
 * packaging builds never export these symbols. Not part of the public include/ directory.
 */
#ifndef KGGML_WHISPER_TESTING_H
#define KGGML_WHISPER_TESTING_H

#include "kggml_whisper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum kggml_whisper_test_fault {
    KGGML_WHISPER_TEST_FAULT_NONE          = 0,
    KGGML_WHISPER_TEST_FAULT_BAD_ALLOC     = 1, /* std::bad_alloc            -> OUT_OF_MEMORY  */
    KGGML_WHISPER_TEST_FAULT_RUNTIME_ERROR = 2, /* std::runtime_error        -> NATIVE_FAILURE */
    KGGML_WHISPER_TEST_FAULT_LOGIC_ERROR   = 3, /* std::out_of_range         -> NATIVE_FAILURE */
    KGGML_WHISPER_TEST_FAULT_NON_STD       = 4  /* throw 42                  -> UNKNOWN        */
} kggml_whisper_test_fault;

/** Arms a one-shot fault: the next guarded kggml_whisper_* call on this thread throws it inside its guard. */
KGGML_WHISPER_API void kggml_whisper_test_inject_fault(int32_t fault);

/**
 * Invokes the exact abort callback and encoder-begin callback that kggml_whisper_transcribe() installs in
 * whisper_full_params for `context` and reports their results (1 = abort / 0 = continue for the abort
 * callback; 1 = continue / 0 = stop for encoder-begin).
 */
KGGML_WHISPER_API kggml_status kggml_whisper_test_poll_callbacks(kggml_whisper_context * context,
                                                                int32_t * out_abort, int32_t * out_encoder_continue);

/**
 * Arms a test-only countdown: after `n` further abort-callback polls during the next transcription the
 * context requests abort on itself, simulating a cancellation that lands mid-inference. -1 disarms.
 */
KGGML_WHISPER_API kggml_status kggml_whisper_test_abort_after_polls(kggml_whisper_context * context, int32_t n);

/** Number of abort-callback polls whisper.cpp made during the last transcription (proves plumbing). */
KGGML_WHISPER_API kggml_status kggml_whisper_test_abort_poll_count(const kggml_whisper_context * context,
                                                                  int64_t * out_polls);

/** whisper_context_params.use_gpu that kggml_whisper_init_from_file() would pass for `params`. */
KGGML_WHISPER_API kggml_status kggml_whisper_test_resolve_use_gpu(const kggml_whisper_context_params * params,
                                                                 int32_t * out_use_gpu, int32_t * out_coreml_allowed);

#ifdef __cplusplus
}
#endif

#endif /* KGGML_WHISPER_TESTING_H */
