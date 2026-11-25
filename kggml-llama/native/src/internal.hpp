// internal.hpp - INTERNAL object layouts shared by the kggml_llama translation units.
#pragma once

#include "kggml_llama.h"

#include "llama.h"

#include <atomic>
#include <cstdint>
#include <vector>

namespace kggml::llama {

#if defined(KGGML_LLAMA_TEST_HOOKS)
/// Live kggml_llama_{model,context,sampler,mtmd} objects (leak / deterministic-free checks).
extern std::atomic<int64_t> g_live_objects;
#endif

/// Counts one live object for the lifetime of the owner (test builds only).
struct LiveCounter {
    LiveCounter() noexcept {
#if defined(KGGML_LLAMA_TEST_HOOKS)
        g_live_objects.fetch_add(1, std::memory_order_relaxed);
#endif
    }
    ~LiveCounter() {
#if defined(KGGML_LLAMA_TEST_HOOKS)
        g_live_objects.fetch_sub(1, std::memory_order_relaxed);
#endif
    }
    LiveCounter(const LiveCounter &) = delete;
    LiveCounter & operator=(const LiveCounter &) = delete;
};

int32_t resolve_threads(int32_t requested) noexcept;

} // namespace kggml::llama

struct kggml_llama_model {
    kggml::llama::LiveCounter live;
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    bool holds_runtime_ref = false;
    int32_t backend = KGGML_BACKEND_CPU; // effective backend
    bool uses_gpu = false;

    kggml_llama_model() = default;
    kggml_llama_model(const kggml_llama_model &) = delete;
    kggml_llama_model & operator=(const kggml_llama_model &) = delete;
    ~kggml_llama_model();
};

struct kggml_llama_context {
    kggml::llama::LiveCounter live;
    kggml_llama_model * owner = nullptr;
    llama_context * ctx = nullptr;
    uint32_t n_seq_max = 1;
    bool logits_valid = false; // the last decode computed outputs for its final token

    std::atomic<bool> abort{false};
    std::atomic<int64_t> abort_polls{0};
#if defined(KGGML_LLAMA_TEST_HOOKS)
    std::atomic<int64_t> abort_after_polls{-1};
#endif

    llama_batch batch{};
    bool batch_allocated = false;

    kggml_llama_context() = default;
    kggml_llama_context(const kggml_llama_context &) = delete;
    kggml_llama_context & operator=(const kggml_llama_context &) = delete;
    ~kggml_llama_context();
};

namespace kggml::llama {

/// The ggml_abort_callback registered with llama_set_abort_callback(). Polled from inside ggml graph compute,
/// possibly on ggml worker threads, so it only touches atomics.
bool abort_callback(void * user_data);

/// INVALID_ARGUMENT unless `context` is a live context.
kggml_status check_context(const kggml_llama_context * context) noexcept;

/// INVALID_ARGUMENT unless 0 <= seq_id < n_seq_max.
kggml_status check_seq(const kggml_llama_context * context, int32_t seq_id);

} // namespace kggml::llama
