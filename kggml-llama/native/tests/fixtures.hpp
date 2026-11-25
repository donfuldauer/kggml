// fixtures.hpp - RAII helpers shared by the kggml_llama tests.
#pragma once

#include "kggml_llama.h"
#include "tiny_model.hpp"

#if defined(KGGML_LLAMA_TEST_HOOKS)
#    include "kggml_llama_testing.h"
#endif

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace kggml::test {

struct ModelDeleter {
    void operator()(kggml_llama_model * m) const noexcept { kggml_llama_model_free(m); }
};
struct ContextDeleter {
    void operator()(kggml_llama_context * c) const noexcept { kggml_llama_context_free(c); }
};
struct SamplerDeleter {
    void operator()(kggml_llama_sampler * s) const noexcept { kggml_llama_sampler_free(s); }
};
using ModelPtr = std::unique_ptr<kggml_llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<kggml_llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<kggml_llama_sampler, SamplerDeleter>;

inline ModelPtr load_tiny_model() {
    kggml_llama_model * m = nullptr;
    const kggml_status s = kggml_llama_model_load(tiny_model_path().c_str(), nullptr, &m);
    EXPECT_EQ(s, KGGML_STATUS_OK) << kggml_llama_last_error();
    return ModelPtr(m);
}

inline kggml_llama_context_params small_context_params() {
    kggml_llama_context_params p{};
    kggml_llama_context_default_params(&p);
    p.n_ctx = 256;
    p.n_batch = 16;
    p.n_threads = 2;
    return p;
}

inline ContextPtr make_context(kggml_llama_model * model, const kggml_llama_context_params & p) {
    kggml_llama_context * c = nullptr;
    const kggml_status s = kggml_llama_context_create(model, &p, &c);
    EXPECT_EQ(s, KGGML_STATUS_OK) << kggml_llama_last_error();
    return ContextPtr(c);
}

/// Deterministic prompt of `n` letter tokens.
inline std::vector<kggml_llama_token> letter_prompt(int32_t n) {
    std::vector<kggml_llama_token> t{kTinyBos};
    for (int32_t i = 1; i < n; ++i) t.push_back(kTinyLetterBase + (i * 7) % 26);
    return t;
}

inline std::vector<float> last_logits(kggml_llama_context * c) {
    const float * logits = nullptr;
    int32_t n = 0;
    EXPECT_EQ(kggml_llama_get_logits(c, &logits, &n), KGGML_STATUS_OK) << kggml_llama_last_error();
    if (logits == nullptr) return {};
    return std::vector<float>(logits, logits + n);
}

} // namespace kggml::test
