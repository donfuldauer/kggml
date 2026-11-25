// errors.cpp - thread-local last-error storage and (test builds only) fault injection for kggml_llama.
#include "guard.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

#if defined(KGGML_LLAMA_TEST_HOOKS)
#    include "kggml_llama_testing.h"
#endif

namespace kggml::llama {

namespace {

std::string & storage() noexcept {
    thread_local std::string message;
    return message;
}

#if defined(KGGML_LLAMA_TEST_HOOKS)
thread_local int32_t g_armed_fault = KGGML_LLAMA_TEST_FAULT_NONE;
#endif

} // namespace

void set_last_error(const char * message) noexcept {
    try {
        storage().assign(message != nullptr ? message : "");
    } catch (...) {
        storage().clear(); // OOM while recording the message: keep the status, drop the text
    }
}

void set_last_error(const char * where, const char * prefix, const char * detail) noexcept {
    try {
        std::string & s = storage();
        s.clear();
        if (where != nullptr) {
            s.append(where).append(": ");
        }
        if (prefix != nullptr) {
            s.append(prefix);
        }
        if (detail != nullptr) {
            s.append(detail);
        }
    } catch (...) {
        storage().clear();
    }
}

void clear_last_error() noexcept {
    storage().clear();
}

const char * last_error() noexcept {
    return storage().c_str();
}

#if defined(KGGML_LLAMA_TEST_HOOKS)
void maybe_throw_injected_fault() {
    const int32_t fault = g_armed_fault;
    g_armed_fault = KGGML_LLAMA_TEST_FAULT_NONE;
    switch (fault) {
        case KGGML_LLAMA_TEST_FAULT_BAD_ALLOC: throw std::bad_alloc();
        case KGGML_LLAMA_TEST_FAULT_RUNTIME_ERROR: throw std::runtime_error("injected runtime_error");
        case KGGML_LLAMA_TEST_FAULT_LOGIC_ERROR: throw std::out_of_range("injected out_of_range");
        case KGGML_LLAMA_TEST_FAULT_NON_STD: throw 42;
        default: break;
    }
}
#endif

} // namespace kggml::llama

extern "C" {

KGGML_LLAMA_API const char * kggml_llama_last_error(void) {
    return kggml::llama::last_error();
}

KGGML_LLAMA_API void kggml_llama_clear_last_error(void) {
    kggml::llama::clear_last_error();
}

#if defined(KGGML_LLAMA_TEST_HOOKS)
KGGML_LLAMA_API void kggml_llama_test_inject_fault(int32_t fault) {
    kggml::llama::g_armed_fault = fault;
}
#endif

} // extern "C"
