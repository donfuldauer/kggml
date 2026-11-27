// Context lifecycle, backend toggle (use_gpu), model-load errors and real whisper.cpp inference on a tiny
// synthetic model.
#include "kggml_whisper.h"
#include "tiny_model.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(KGGML_WHISPER_TEST_HOOKS)
#    include "kggml_whisper_testing.h"
#endif

namespace {

using ContextPtr = std::unique_ptr<kggml_whisper_context, decltype(&kggml_whisper_free)>;

ContextPtr make_context(int32_t backend = KGGML_BACKEND_CPU) {
    kggml_whisper_context_params p{};
    EXPECT_EQ(kggml_whisper_context_default_params(&p), KGGML_STATUS_OK);
    p.backend = backend;
    kggml_whisper_context * ctx = nullptr;
    const auto s = kggml_whisper_init_from_file(kggml::test::tiny_model_path().c_str(), &p, &ctx);
    EXPECT_EQ(s, KGGML_STATUS_OK) << kggml_whisper_last_error();
    return ContextPtr(ctx, kggml_whisper_free);
}

kggml_whisper_transcribe_params fast_params() {
    kggml_whisper_transcribe_params p{};
    EXPECT_EQ(kggml_whisper_transcribe_default_params(&p), KGGML_STATUS_OK);
    p.n_threads = 2;
    p.temperature_inc = 0.0f; // no fallback re-decodes for the random model
    p.max_tokens = 8;
    return p;
}

} // namespace

TEST(Versions, ReportPinnedUpstream) {
    EXPECT_STREQ(kggml_whisper_version(), KGGML_TEST_EXPECT_WHISPER_VERSION);
    EXPECT_STREQ(kggml_whisper_upstream_version(), KGGML_TEST_EXPECT_UPSTREAM_VERSION);
    EXPECT_STREQ(kggml_whisper_upstream_commit(), KGGML_TEST_EXPECT_UPSTREAM_COMMIT);
    EXPECT_EQ(std::strlen(kggml_whisper_upstream_commit()), 40u);
    EXPECT_EQ(kggml_whisper_has_coreml(), KGGML_TEST_EXPECT_COREML ? 1 : 0);
    EXPECT_NE(std::string(kggml_whisper_system_info()).find("CPU"), std::string::npos);
}

TEST(Versions, LinksTheRuntimeGgml) {
    // whisper.cpp and kggml_runtime resolve ggml from the same shared library.
    EXPECT_STRNE(kggml_runtime_ggml_version(), "");
}

TEST(ContextParams, Defaults) {
    kggml_whisper_context_params p{};
    p.backend = 42;
    ASSERT_EQ(kggml_whisper_context_default_params(&p), KGGML_STATUS_OK);
    EXPECT_EQ(p.backend, KGGML_BACKEND_CPU);
    EXPECT_EQ(p.gpu_device, 0);
    EXPECT_EQ(p.flash_attn, 1);

    kggml_whisper_transcribe_params t{};
    ASSERT_EQ(kggml_whisper_transcribe_default_params(&t), KGGML_STATUS_OK);
    EXPECT_EQ(t.language, nullptr);
    EXPECT_EQ(t.n_threads, 0);
    EXPECT_FLOAT_EQ(t.temperature, 0.0f);
    EXPECT_FLOAT_EQ(t.temperature_inc, 0.2f);
}

#if defined(KGGML_WHISPER_TEST_HOOKS)
TEST(BackendToggle, CpuOnlyDisablesGpuAndCoreMl) {
    kggml_whisper_context_params p{};
    ASSERT_EQ(kggml_whisper_context_default_params(&p), KGGML_STATUS_OK);
    int32_t use_gpu = -1;
    int32_t coreml = -1;
    p.backend = KGGML_BACKEND_CPU;
    ASSERT_EQ(kggml_whisper_test_resolve_use_gpu(&p, &use_gpu, &coreml), KGGML_STATUS_OK);
    EXPECT_EQ(use_gpu, 0);
    EXPECT_EQ(coreml, 0);
    p.backend = KGGML_BACKEND_METAL;
    ASSERT_EQ(kggml_whisper_test_resolve_use_gpu(&p, &use_gpu, &coreml), KGGML_STATUS_OK);
    EXPECT_EQ(use_gpu, 1);
    EXPECT_EQ(coreml, 0);
    p.backend = KGGML_BACKEND_VULKAN;
    ASSERT_EQ(kggml_whisper_test_resolve_use_gpu(&p, &use_gpu, &coreml), KGGML_STATUS_OK);
    EXPECT_EQ(use_gpu, 1);
    EXPECT_EQ(coreml, 0);
    p.backend = KGGML_BACKEND_COREML;
    ASSERT_EQ(kggml_whisper_test_resolve_use_gpu(&p, &use_gpu, &coreml), KGGML_STATUS_OK);
    EXPECT_EQ(use_gpu, 1);
    EXPECT_EQ(coreml, 1);
    p.backend = 99;
    EXPECT_EQ(kggml_whisper_test_resolve_use_gpu(&p, &use_gpu, &coreml), KGGML_STATUS_INVALID_ARGUMENT);
}
#endif

TEST(Context, CpuOnlyContextReportsNoGpu) {
    auto ctx = make_context(KGGML_BACKEND_CPU);
    ASSERT_NE(ctx, nullptr);
    int32_t v = -1;
    ASSERT_EQ(kggml_whisper_uses_gpu(ctx.get(), &v), KGGML_STATUS_OK);
    EXPECT_EQ(v, 0);
    ASSERT_EQ(kggml_whisper_get_backend(ctx.get(), &v), KGGML_STATUS_OK);
    EXPECT_EQ(v, KGGML_BACKEND_CPU);
    ASSERT_EQ(kggml_whisper_is_multilingual(ctx.get(), &v), KGGML_STATUS_OK);
    EXPECT_EQ(v, 0); // 51864-token English vocabulary
}

TEST(Context, ContextHoldsARuntimeReference) {
    const int32_t before = kggml_runtime_ref_count();
    {
        auto ctx = make_context();
        ASSERT_NE(ctx, nullptr);
        EXPECT_EQ(kggml_runtime_ref_count(), before + 1);
    }
    EXPECT_EQ(kggml_runtime_ref_count(), before);
}

TEST(Context, MultipleIndependentInstances) {
    auto a = make_context();
    auto b = make_context();
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_NE(a.get(), b.get());
    kggml_whisper_request_abort(a.get());
    EXPECT_EQ(kggml_whisper_abort_requested(a.get()), 1);
    EXPECT_EQ(kggml_whisper_abort_requested(b.get()), 0);
}

TEST(Context, MissingModelIsModelLoadFailed) {
    kggml_whisper_context * ctx = reinterpret_cast<kggml_whisper_context *>(0x1);
    const int32_t before = kggml_runtime_ref_count();
    EXPECT_EQ(kggml_whisper_init_from_file("/definitely/not/here/ggml-tiny.bin", nullptr, &ctx),
              KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_NE(std::string(kggml_whisper_last_error()).find("cannot open model file"), std::string::npos);
    EXPECT_EQ(kggml_runtime_ref_count(), before); // failure paths release the runtime reference
}

TEST(Context, CorruptModelIsModelLoadFailed) {
    kggml_whisper_context * ctx = nullptr;
    EXPECT_EQ(kggml_whisper_init_from_file(kggml::test::garbage_model_path().c_str(), nullptr, &ctx),
              KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_NE(std::string(kggml_whisper_last_error()).find("could not load model"), std::string::npos);
}

TEST(Context, InvalidArguments) {
    kggml_whisper_context * ctx = nullptr;
    EXPECT_EQ(kggml_whisper_init_from_file("", nullptr, &ctx), KGGML_STATUS_INVALID_ARGUMENT);
    kggml_whisper_context_params p{};
    ASSERT_EQ(kggml_whisper_context_default_params(&p), KGGML_STATUS_OK);
    p.backend = 17;
    EXPECT_EQ(kggml_whisper_init_from_file(kggml::test::tiny_model_path().c_str(), &p, &ctx),
              KGGML_STATUS_INVALID_ARGUMENT);
    p.backend = KGGML_BACKEND_CPU;
    p.gpu_device = -1;
    EXPECT_EQ(kggml_whisper_init_from_file(kggml::test::tiny_model_path().c_str(), &p, &ctx),
              KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(Context, UnavailableBackendsAreRejected) {
    ASSERT_EQ(kggml_runtime_init(), KGGML_STATUS_OK);
    for (int32_t kind : {static_cast<int32_t>(KGGML_BACKEND_METAL), static_cast<int32_t>(KGGML_BACKEND_VULKAN),
                         static_cast<int32_t>(KGGML_BACKEND_OPENCL)}) {
        int32_t available = 0;
        ASSERT_EQ(kggml_runtime_backend_available(kind, &available), KGGML_STATUS_OK);
        kggml_whisper_context_params p{};
        ASSERT_EQ(kggml_whisper_context_default_params(&p), KGGML_STATUS_OK);
        p.backend = kind;
        kggml_whisper_context * ctx = nullptr;
        const auto s = kggml_whisper_init_from_file(kggml::test::tiny_model_path().c_str(), &p, &ctx);
        if (available == 1) {
            EXPECT_EQ(s, KGGML_STATUS_OK) << kggml_whisper_last_error();
            int32_t uses_gpu = 0;
            ASSERT_EQ(kggml_whisper_uses_gpu(ctx, &uses_gpu), KGGML_STATUS_OK);
            EXPECT_EQ(uses_gpu, 1);
            kggml_whisper_free(ctx);
        } else {
            EXPECT_EQ(s, KGGML_STATUS_BACKEND_UNAVAILABLE);
            EXPECT_EQ(ctx, nullptr);
        }
    }
    kggml_whisper_context_params p{};
    ASSERT_EQ(kggml_whisper_context_default_params(&p), KGGML_STATUS_OK);
    p.backend = KGGML_BACKEND_COREML;
    kggml_whisper_context * ctx = nullptr;
    const auto s = kggml_whisper_init_from_file(kggml::test::tiny_model_path().c_str(), &p, &ctx);
    if (kggml_whisper_has_coreml() == 1) {
        // No .mlmodelc next to the tiny model: whisper.cpp must fall back to the GPU / CPU encoder.
        ASSERT_EQ(s, KGGML_STATUS_OK) << kggml_whisper_last_error();
        int32_t backend = -1;
        ASSERT_EQ(kggml_whisper_get_backend(ctx, &backend), KGGML_STATUS_OK);
        EXPECT_NE(backend, KGGML_BACKEND_COREML);
        kggml_whisper_free(ctx);
    } else {
        EXPECT_EQ(s, KGGML_STATUS_BACKEND_UNAVAILABLE);
        EXPECT_NE(std::string(kggml_whisper_last_error()).find("CoreML"), std::string::npos);
    }
    kggml_runtime_shutdown();
}

TEST(Transcribe, RunsRealInferenceOnTinyModel) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    std::vector<float> pcm(16000 * 2, 0.0f);
    for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = 0.1f * static_cast<float>((i % 100) / 50.0 - 1.0);
    const auto p = fast_params();
    ASSERT_EQ(kggml_whisper_transcribe(ctx.get(), pcm.data(), pcm.size(), &p), KGGML_STATUS_OK)
        << kggml_whisper_last_error();
    int32_t n = -1;
    ASSERT_EQ(kggml_whisper_segment_count(ctx.get(), &n), KGGML_STATUS_OK);
    EXPECT_GE(n, 0);
    for (int32_t i = 0; i < n; ++i) {
        kggml_whisper_segment seg{};
        ASSERT_EQ(kggml_whisper_get_segment(ctx.get(), i, &seg), KGGML_STATUS_OK);
        EXPECT_NE(seg.text, nullptr);
        EXPECT_LE(seg.t0_ms, seg.t1_ms);
    }
    kggml_whisper_segment seg{};
    EXPECT_EQ(kggml_whisper_get_segment(ctx.get(), n, &seg), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_get_segment(ctx.get(), -1, &seg), KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(Transcribe, ShortAndEmptyInputArePaddedTo1000Ms) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    const auto p = fast_params();
    std::vector<float> tiny(800, 0.05f); // 50 ms: whisper.cpp alone would silently skip (< 100 ms)
    EXPECT_EQ(kggml_whisper_transcribe(ctx.get(), tiny.data(), tiny.size(), &p), KGGML_STATUS_OK)
        << kggml_whisper_last_error();
#if defined(KGGML_WHISPER_TEST_HOOKS)
    int64_t polls = 0;
    ASSERT_EQ(kggml_whisper_test_abort_poll_count(ctx.get(), &polls), KGGML_STATUS_OK);
    EXPECT_GT(polls, 0) << "padded input must reach the encoder";
#endif
    EXPECT_EQ(kggml_whisper_transcribe(ctx.get(), nullptr, 0, &p), KGGML_STATUS_OK) << kggml_whisper_last_error();
}

TEST(Transcribe, RejectsInvalidParameters) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    std::vector<float> pcm(16000, 0.0f);
    auto p = fast_params();
    p.language = "xx-not-a-language";
    EXPECT_EQ(kggml_whisper_transcribe(ctx.get(), pcm.data(), pcm.size(), &p), KGGML_STATUS_INVALID_ARGUMENT);
    p = fast_params();
    p.n_threads = -1;
    EXPECT_EQ(kggml_whisper_transcribe(ctx.get(), pcm.data(), pcm.size(), &p), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_transcribe(ctx.get(), nullptr, 10, &p), KGGML_STATUS_INVALID_ARGUMENT);
}
