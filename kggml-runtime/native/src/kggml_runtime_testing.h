/*
 * kggml_runtime_testing.h - INTERNAL fault-injection hooks for the kggml_runtime test suite.
 *
 * Only compiled into kggml_runtime when the KGGML_RUNTIME_TEST_HOOKS option is ON (implied by
 * KGGML_RUNTIME_BUILD_TESTS). Release / packaging presets build without it, so shipped binaries never
 * export these symbols. Not part of the public include/ directory.
 */
#ifndef KGGML_RUNTIME_TESTING_H
#define KGGML_RUNTIME_TESTING_H

#include "kggml_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum kggml_test_fault {
    KGGML_TEST_FAULT_NONE                = 0,
    KGGML_TEST_FAULT_BAD_ALLOC           = 1, /* throw std::bad_alloc                 -> OUT_OF_MEMORY  */
    KGGML_TEST_FAULT_BAD_ARRAY_NEW       = 2, /* throw std::bad_array_new_length      -> OUT_OF_MEMORY  */
    KGGML_TEST_FAULT_RUNTIME_ERROR       = 3, /* throw std::runtime_error             -> NATIVE_FAILURE */
    KGGML_TEST_FAULT_LOGIC_ERROR         = 4, /* throw std::invalid_argument          -> NATIVE_FAILURE */
    KGGML_TEST_FAULT_NON_STD             = 5  /* throw 42 (not a std::exception)      -> UNKNOWN        */
} kggml_test_fault;

/**
 * Arms a one-shot fault on the calling thread: the next guarded kggml_runtime_* entry point invoked on this
 * thread throws the requested exception *inside* its try/catch guard, then the fault disarms itself.
 */
KGGML_API void kggml_runtime_test_inject_fault(int32_t fault);

#ifdef __cplusplus
}
#endif

#endif /* KGGML_RUNTIME_TESTING_H */
