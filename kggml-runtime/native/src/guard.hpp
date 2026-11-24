// guard.hpp - INTERNAL FFI boundary helpers for kggml_runtime.
//
// Every extern "C" entry point runs its body through kggml::detail::guarded(), which converts any C++
// exception into a kggml_status and records a thread-local diagnostic message. No exception may escape.
#pragma once

#include "kggml_runtime.h"

#include <exception>
#include <new>
#include <string>
#include <utility>

namespace kggml::detail {

/// Thread-local last-error storage (empty string == no error).
std::string & last_error_storage() noexcept;

/// Records `message` as the calling thread's last error. Never throws (drops the message on OOM).
void set_last_error(const char * message) noexcept;
/// Records "<where>: <prefix><detail>"; every argument may be NULL. Never throws.
void set_last_error(const char * where, const char * prefix, const char * detail) noexcept;
void clear_last_error() noexcept;

/// Convenience: records `message` and returns `status`.
inline kggml_status fail(kggml_status status, const char * message) noexcept {
    set_last_error(message);
    return status;
}

#if defined(KGGML_RUNTIME_TEST_HOOKS)
/// Throws the armed test fault (if any) for the calling thread and disarms it.
void maybe_throw_injected_fault();
#endif

/// Runs `body` (returning kggml_status) behind the FFI exception firewall.
template <typename Fn>
kggml_status guarded(const char * where, Fn && body) noexcept {
    try {
        clear_last_error();
#if defined(KGGML_RUNTIME_TEST_HOOKS)
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

} // namespace kggml::detail
