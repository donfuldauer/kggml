// guard.hpp - INTERNAL FFI boundary helpers for kggml_whisper.
//
// Every extern "C" entry point returning kggml_status runs its body through kggml::whisper::guarded(), which
// converts any C++ exception (from this wrapper, whisper.cpp or ggml) into a kggml_status and records a
// thread-local diagnostic message. No exception may escape into JNI or Kotlin/Native.
#pragma once

#include "kggml_whisper.h"

#include <exception>
#include <new>
#include <utility>

namespace kggml::whisper {

/// Records "<where>: <prefix><detail>" as the calling thread's last error. Every argument may be NULL.
void set_last_error(const char * where, const char * prefix, const char * detail) noexcept;
void set_last_error(const char * message) noexcept;
void clear_last_error() noexcept;
const char * last_error() noexcept;

/// Records `message` and returns `status`.
inline kggml_status fail(kggml_status status, const char * message) noexcept {
    set_last_error(message);
    return status;
}

#if defined(KGGML_WHISPER_TEST_HOOKS)
/// Throws the fault armed with kggml_whisper_test_inject_fault() on this thread (once), if any.
void maybe_throw_injected_fault();
#endif

/// Runs `body` (returning kggml_status) behind the FFI exception firewall.
template <typename Fn>
kggml_status guarded(const char * where, Fn && body) noexcept {
    try {
        clear_last_error();
#if defined(KGGML_WHISPER_TEST_HOOKS)
        maybe_throw_injected_fault();
#endif
        return std::forward<Fn>(body)();
    } catch (const std::bad_alloc & e) {
        set_last_error(where, "out of memory: ", e.what());
        return KGGML_STATUS_OUT_OF_MEMORY;
    } catch (const std::exception & e) {
        set_last_error(where, "", e.what());
        return KGGML_STATUS_NATIVE_FAILURE;
    } catch (...) {
        set_last_error(where, "", "unknown non-standard C++ exception");
        return KGGML_STATUS_UNKNOWN;
    }
}

} // namespace kggml::whisper
