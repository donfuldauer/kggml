// errors.cpp - thread-local last-error storage and (test builds only) fault injection for kggml_diffusion.
#include "guard.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

#if defined(KGGML_DIFFUSION_TEST_HOOKS)
#    include "kggml_diffusion_testing.h"
#endif

namespace kggml::diffusion {

namespace {

std::string & storage() noexcept {
    thread_local std::string message;
    return message;
}

std::string & sd_storage() noexcept {
    thread_local std::string line;
    return line;
}

#if defined(KGGML_DIFFUSION_TEST_HOOKS)
thread_local int32_t g_armed_fault = KGGML_DIFFUSION_TEST_FAULT_NONE;
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
    sd_storage().clear();
}

const char * last_error() noexcept {
    return storage().c_str();
}

void record_sd_error_line(const char * text) noexcept {
    try {
        std::string & s = sd_storage();
        if (!s.empty()) {
            return; // keep the FIRST error of the failing call: it names the root cause, later ones are fallout
        }
        s.assign(text != nullptr ? text : "");
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) {
            s.pop_back();
        }
    } catch (...) {
        sd_storage().clear();
    }
}

const char * sd_error_line() noexcept {
    return sd_storage().c_str();
}

kggml_status fail_with_sd_log(kggml_status status, const char * message) noexcept {
    try {
        std::string s = message != nullptr ? message : "";
        if (!sd_storage().empty()) {
            s.append(" (stable-diffusion.cpp: ").append(sd_storage()).append(")");
        }
        set_last_error(s.c_str());
    } catch (...) {
        set_last_error(message);
    }
    return status;
}

#if defined(KGGML_DIFFUSION_TEST_HOOKS)
void throw_fault(int fault) {
    switch (fault) {
        case KGGML_DIFFUSION_TEST_FAULT_BAD_ALLOC: throw std::bad_alloc();
        case KGGML_DIFFUSION_TEST_FAULT_RUNTIME_ERROR: throw std::runtime_error("injected runtime_error");
        case KGGML_DIFFUSION_TEST_FAULT_LOGIC_ERROR: throw std::out_of_range("injected out_of_range");
        case KGGML_DIFFUSION_TEST_FAULT_NON_STD: throw 42;
        default: break;
    }
}

void maybe_throw_injected_fault() {
    const int32_t fault = g_armed_fault;
    g_armed_fault = KGGML_DIFFUSION_TEST_FAULT_NONE;
    throw_fault(fault);
}
#endif

} // namespace kggml::diffusion

extern "C" {

KGGML_DIFFUSION_API const char * kggml_diffusion_last_error(void) {
    return kggml::diffusion::last_error();
}

KGGML_DIFFUSION_API void kggml_diffusion_clear_last_error(void) {
    kggml::diffusion::clear_last_error();
}

#if defined(KGGML_DIFFUSION_TEST_HOOKS)
KGGML_DIFFUSION_API void kggml_diffusion_test_inject_fault(int32_t fault) {
    kggml::diffusion::g_armed_fault = fault;
}
#endif

} // extern "C"
