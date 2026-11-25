// internal.hpp - INTERNAL object layouts shared by the kggml_diffusion translation units.
#pragma once

#include "kggml_diffusion.h"

#include "stable-diffusion.h"

#include <atomic>
#include <cstdint>
#include <string>

namespace kggml::diffusion {

#if defined(KGGML_DIFFUSION_TEST_HOOKS)
/// Live kggml_diffusion_context objects (leak / deterministic-free checks).
extern std::atomic<int64_t> g_live_objects;
#endif

/// Counts one live object for the lifetime of the owner (test builds only).
struct LiveCounter {
    LiveCounter() noexcept {
#if defined(KGGML_DIFFUSION_TEST_HOOKS)
        g_live_objects.fetch_add(1, std::memory_order_relaxed);
#endif
    }
    ~LiveCounter() {
#if defined(KGGML_DIFFUSION_TEST_HOOKS)
        g_live_objects.fetch_sub(1, std::memory_order_relaxed);
#endif
    }
    LiveCounter(const LiveCounter &) = delete;
    LiveCounter & operator=(const LiveCounter &) = delete;
};

} // namespace kggml::diffusion

struct kggml_diffusion_context {
    kggml::diffusion::LiveCounter live;
    sd_ctx_t * sd = nullptr;          // NULL only for test stubs
    bool holds_runtime_ref = false;
    int32_t backend = KGGML_BACKEND_CPU; // effective backend
    bool uses_gpu = false;
    int32_t n_threads = 1;
    std::string backend_spec;         // stable-diffusion.cpp backend assignment, e.g. "CPU" / "Vulkan0"

    std::atomic<bool> abort{false};   // latched by kggml_diffusion_request_abort() (any thread)

    kggml_diffusion_context() = default;
    kggml_diffusion_context(const kggml_diffusion_context &) = delete;
    kggml_diffusion_context & operator=(const kggml_diffusion_context &) = delete;
    ~kggml_diffusion_context();
};
