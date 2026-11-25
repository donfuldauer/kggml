// guard.hpp - INTERNAL FFI boundary helpers for kggml_diffusion.
//
// Every extern "C" entry point returning kggml_status runs its body through kggml::diffusion::guarded(), which
// converts any C++ exception (from this wrapper, stable-diffusion.cpp or ggml) into a kggml_status and records a
// thread-local diagnostic message. No exception may escape into JNI or Kotlin/Native.
#pragma once

#include "kggml_diffusion.h"

#include <exception>
#include <new>
#include <utility>

namespace kggml::diffusion {

/// Thrown by the progress dispatcher to unwind stable-diffusion.cpp's sampler when an abort is requested.
/// Deliberately NOT derived from std::exception, so upstream `catch (const std::exception &)` blocks never
/// swallow it. Caught by kggml_diffusion_generate() (-> KGGML_STATUS_CANCELLED) and, as a backstop, by guarded().
struct AbortSignal {};

/// Records "<where>: <prefix><detail>" as the calling thread's last error. Every argument may be NULL.
void set_last_error(const char * where, const char * prefix, const char * detail) noexcept;
void set_last_error(const char * message) noexcept;
/// Clears the last error AND the captured stable-diffusion.cpp error log line of the calling thread.
void clear_last_error() noexcept;
const char * last_error() noexcept;

/// Records the first stable-diffusion.cpp ERROR log line emitted on the calling thread since clear_last_error().
void record_sd_error_line(const char * text) noexcept;
/// That line ("" if none since the last clear_last_error()).
const char * sd_error_line() noexcept;

/// Records `message` and returns `status`.
inline kggml_status fail(kggml_status status, const char * message) noexcept {
    set_last_error(message);
    return status;
}

/// Records `message` plus " (stable-diffusion.cpp: <first error log line>)" when upstream logged one, returns `status`.
kggml_status fail_with_sd_log(kggml_status status, const char * message) noexcept;

#if defined(KGGML_DIFFUSION_TEST_HOOKS)
/// Throws the fault armed with kggml_diffusion_test_inject_fault() on this thread (once), if any.
void maybe_throw_injected_fault();
/// Throws `fault` (a kggml_diffusion_test_fault value); no-op for KGGML_DIFFUSION_TEST_FAULT_NONE.
void throw_fault(int fault);
#endif

/// Runs `body` (returning kggml_status) behind the FFI exception firewall.
template <typename Fn>
kggml_status guarded(const char * where, Fn && body) noexcept {
    try {
        clear_last_error();
#if defined(KGGML_DIFFUSION_TEST_HOOKS)
        maybe_throw_injected_fault();
#endif
        return std::forward<Fn>(body)();
    } catch (const AbortSignal &) {
        set_last_error(where, "", "generation aborted");
        return KGGML_STATUS_CANCELLED;
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

} // namespace kggml::diffusion
