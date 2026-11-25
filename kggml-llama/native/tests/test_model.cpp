// Model loading (error handling, backend selection, cancellation), vocabulary access and deterministic resource
// deallocation (runtime reference counting, live object accounting).
#include "fixtures.hpp"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kggml::test;

namespace {

kggml_status load(const std::string & path, const kggml_llama_model_params * p, kggml_llama_model ** m) {
    return kggml_llama_model_load(path.c_str(), p, m);
}

} // namespace

TEST(LlamaModel, MissingFileIsModelLoadFailed) {
    const int32_t refs = kggml_runtime_ref_count();
    kggml_llama_model * m = nullptr;
    EXPECT_EQ(load(::testing::TempDir() + "kggml-llama-does-not-exist.gguf", nullptr, &m), KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_EQ(m, nullptr);
    EXPECT_NE(std::string(kggml_llama_last_error()).find("cannot open model file"), std::string::npos);
    EXPECT_EQ(kggml_runtime_ref_count(), refs) << "a failed load must release its runtime reference";
}

TEST(LlamaModel, GarbageFileIsModelLoadFailed) {
    kggml_llama_model * m = nullptr;
    EXPECT_EQ(load(garbage_model_path(), nullptr, &m), KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_EQ(m, nullptr);
    EXPECT_NE(std::string(kggml_llama_last_error()).find("could not load model"), std::string::npos);
}

TEST(LlamaModel, TruncatedGgufIsModelLoadFailed) {
    kggml_llama_model * m = nullptr;
    EXPECT_EQ(load(truncated_model_path(), nullptr, &m), KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_EQ(m, nullptr);
}

TEST(LlamaModel, DirectoryIsModelLoadFailed) {
    kggml_llama_model * m = nullptr;
    const kggml_status s = load(::testing::TempDir(), nullptr, &m);
    EXPECT_EQ(s, KGGML_STATUS_MODEL_LOAD_FAILED) << kggml_llama_last_error();
    EXPECT_EQ(m, nullptr);
}

TEST(LlamaModel, LoadsTinyModelAndReportsInfo) {
    const int32_t refs = kggml_runtime_ref_count();
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    EXPECT_EQ(kggml_runtime_ref_count(), refs + 1);

    kggml_llama_model_info info{};
    ASSERT_EQ(kggml_llama_model_get_info(model.get(), &info), KGGML_STATUS_OK);
    EXPECT_EQ(info.n_vocab, kTinyVocab);
    EXPECT_EQ(info.n_embd, kTinyEmbd);
    EXPECT_EQ(info.n_ctx_train, kTinyCtxTrain);
    EXPECT_EQ(info.n_layer, 1);
    EXPECT_EQ(info.has_encoder, 0);
    EXPECT_EQ(info.has_decoder, 1);
    EXPECT_EQ(info.backend, KGGML_BACKEND_CPU);
    EXPECT_EQ(info.uses_gpu, 0);
    EXPECT_EQ(info.bos, kTinyBos);
    EXPECT_EQ(info.eos, kTinyEos);
    EXPECT_GT(info.n_params, 0u);
    EXPECT_GT(info.size_bytes, 0u);

    size_t len = 0;
    char desc[64] = {};
    ASSERT_EQ(kggml_llama_model_desc(model.get(), desc, sizeof(desc), &len), KGGML_STATUS_OK);
    EXPECT_GT(len, 0u);
    EXPECT_NE(std::string(desc).find("llama"), std::string::npos) << desc;
    char tiny[4] = {};
    ASSERT_EQ(kggml_llama_model_desc(model.get(), tiny, sizeof(tiny), &len), KGGML_STATUS_OK);
    EXPECT_EQ(std::string(tiny).size(), 3u) << "truncated and NUL-terminated";

    model.reset();
    EXPECT_EQ(kggml_runtime_ref_count(), refs) << "kggml_llama_model_free must release its runtime reference";
}

TEST(LlamaModel, LoadWithoutMmapAndVocabOnly) {
    kggml_llama_model_params p{};
    kggml_llama_model_default_params(&p);
    p.use_mmap = 0;
    kggml_llama_model * m = nullptr;
    ASSERT_EQ(load(tiny_model_path(), &p, &m), KGGML_STATUS_OK) << kggml_llama_last_error();
    kggml_llama_model_free(m);

    p.use_mmap = 1;
    p.vocab_only = 1;
    m = nullptr;
    ASSERT_EQ(load(tiny_model_path(), &p, &m), KGGML_STATUS_OK) << kggml_llama_last_error();
    ModelPtr vocab_only(m);
    int32_t n = 0;
    kggml_llama_token toks[8];
    EXPECT_EQ(kggml_llama_tokenize(m, "ab", 2, 1, 0, toks, 8, &n), KGGML_STATUS_OK);
    kggml_llama_context * c = nullptr;
    EXPECT_EQ(kggml_llama_context_create(m, nullptr, &c), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(c, nullptr);
}

TEST(LlamaModel, CoreMlIsUnavailable) {
    kggml_llama_model_params p{};
    kggml_llama_model_default_params(&p);
    p.backend = KGGML_BACKEND_COREML;
    kggml_llama_model * m = nullptr;
    const int32_t refs = kggml_runtime_ref_count();
    EXPECT_EQ(load(tiny_model_path(), &p, &m), KGGML_STATUS_BACKEND_UNAVAILABLE);
    EXPECT_EQ(m, nullptr);
    EXPECT_EQ(kggml_runtime_ref_count(), refs);
}

TEST(LlamaModel, MissingGpuBackendIsUnavailable) {
    ASSERT_EQ(kggml_runtime_init(), KGGML_STATUS_OK);
    for (int32_t kind : {KGGML_BACKEND_METAL, KGGML_BACKEND_VULKAN}) {
        int32_t available = 0;
        ASSERT_EQ(kggml_runtime_backend_available(kind, &available), KGGML_STATUS_OK);
        if (available == 1) continue; // exercised on hosts / CI runners that have this GPU backend
        kggml_llama_model_params p{};
        kggml_llama_model_default_params(&p);
        p.backend = kind;
        kggml_llama_model * m = nullptr;
        EXPECT_EQ(load(tiny_model_path(), &p, &m), KGGML_STATUS_BACKEND_UNAVAILABLE) << kind;
        EXPECT_EQ(m, nullptr);
    }
    kggml_runtime_shutdown();
}

#if defined(KGGML_LLAMA_TEST_HOOKS)
TEST(LlamaModel, CpuBackendHandsLlamaCppNoDevices) {
    kggml_llama_model_params p{};
    kggml_llama_model_default_params(&p);
    p.backend = KGGML_BACKEND_CPU;
    p.gpu_layers = 99; // ignored for CPU
    int32_t n_devices = -1;
    int32_t n_gpu_layers = -1;
    ASSERT_EQ(kggml_llama_test_resolve_devices(&p, &n_devices, &n_gpu_layers), KGGML_STATUS_OK);
    EXPECT_EQ(n_devices, 0) << "CPU_ONLY must never offer a GPU device to llama.cpp";
    EXPECT_EQ(n_gpu_layers, 0);
}

TEST(LlamaModel, GpuBackendHandsLlamaCppExactlyOneDevice) {
    kggml_llama_model_params p{};
    kggml_llama_model_default_params(&p);
    int32_t n_devices = -1;
    int32_t n_gpu_layers = -1;
    ASSERT_EQ(kggml_runtime_init(), KGGML_STATUS_OK);
    int32_t metal = 0;
    kggml_runtime_backend_available(KGGML_BACKEND_METAL, &metal);
    int32_t vulkan = 0;
    kggml_runtime_backend_available(KGGML_BACKEND_VULKAN, &vulkan);
    kggml_runtime_shutdown();
    const int32_t gpu_kind = metal == 1 ? KGGML_BACKEND_METAL : (vulkan == 1 ? KGGML_BACKEND_VULKAN : -1);
    if (gpu_kind < 0) {
        GTEST_SKIP() << "no GPU backend on this host";
    }
    p.backend = gpu_kind;
    p.gpu_layers = -1;
    ASSERT_EQ(kggml_llama_test_resolve_devices(&p, &n_devices, &n_gpu_layers), KGGML_STATUS_OK) << kggml_llama_last_error();
    EXPECT_EQ(n_devices, 1);
    EXPECT_EQ(n_gpu_layers, -1);
    p.gpu_device = 1000;
    EXPECT_EQ(kggml_llama_test_resolve_devices(&p, &n_devices, &n_gpu_layers), KGGML_STATUS_BACKEND_UNAVAILABLE);
}
#endif

namespace {

struct ProgressState {
    int calls = 0;
    int stop_after = -1;
    float last = -1.0f;
};

int32_t progress(float value, void * user_data) {
    auto * st = static_cast<ProgressState *>(user_data);
    ++st->calls;
    st->last = value;
    return st->stop_after >= 0 && st->calls > st->stop_after ? 0 : 1;
}

int32_t throwing_progress(float, void *) {
    throw std::runtime_error("callbacks must not throw, but the bridge survives it");
}

} // namespace

TEST(LlamaModel, ProgressCallbackReportsAndCanCancel) {
    kggml_llama_model_params p{};
    kggml_llama_model_default_params(&p);
    ProgressState st;
    p.progress_callback = progress;
    p.progress_user_data = &st;
    kggml_llama_model * m = nullptr;
    ASSERT_EQ(load(tiny_model_path(), &p, &m), KGGML_STATUS_OK) << kggml_llama_last_error();
    kggml_llama_model_free(m);
    EXPECT_GT(st.calls, 0);
    EXPECT_FLOAT_EQ(st.last, 1.0f);

    const int32_t refs = kggml_runtime_ref_count();
    ProgressState cancel;
    cancel.stop_after = 0; // abort on the first report
    p.progress_user_data = &cancel;
    m = nullptr;
    EXPECT_EQ(load(tiny_model_path(), &p, &m), KGGML_STATUS_CANCELLED);
    EXPECT_EQ(m, nullptr);
    EXPECT_EQ(kggml_runtime_ref_count(), refs);

    p.progress_callback = throwing_progress;
    EXPECT_EQ(load(tiny_model_path(), &p, &m), KGGML_STATUS_CANCELLED);
    EXPECT_EQ(m, nullptr);
}

TEST(LlamaVocab, TokenizeDetokenizeAndBufferProtocol) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    const std::string text = "ab z";
    int32_t n = 0;
    EXPECT_EQ(kggml_llama_tokenize(model.get(), text.data(), text.size(), 1, 0, nullptr, 0, &n),
              KGGML_STATUS_BUFFER_TOO_SMALL);
    ASSERT_GT(n, 0);
    std::vector<kggml_llama_token> toks(static_cast<size_t>(n));
    ASSERT_EQ(kggml_llama_tokenize(model.get(), text.data(), text.size(), 1, 0, toks.data(), n, &n), KGGML_STATUS_OK);
    ASSERT_GE(toks.size(), 3u);
    EXPECT_EQ(toks.front(), kTinyBos);
    EXPECT_EQ(toks.back(), kTinySpaceZ) << "SentencePiece merges \"▁\" + \"z\"";
    std::string round_trip;
    for (size_t i = 1; i < toks.size(); ++i) {
        int32_t len = 0;
        EXPECT_EQ(kggml_llama_token_to_piece(model.get(), toks[i], 0, nullptr, 0, &len), KGGML_STATUS_BUFFER_TOO_SMALL);
        std::string piece(static_cast<size_t>(len), '\0');
        ASSERT_EQ(kggml_llama_token_to_piece(model.get(), toks[i], 0, piece.data(), len, &len), KGGML_STATUS_OK);
        round_trip += piece;
    }
    EXPECT_EQ(round_trip, " ab z");

    // Byte fallback for characters outside the tiny vocabulary.
    ASSERT_EQ(kggml_llama_tokenize(model.get(), "\x01", 1, 0, 0, toks.data(), static_cast<int32_t>(toks.size()), &n),
              KGGML_STATUS_OK);
    EXPECT_GE(n, 1);

    int32_t len = 0;
    char buf[16];
    EXPECT_EQ(kggml_llama_token_to_piece(model.get(), kTinyVocab, 0, buf, 16, &len), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_token_to_piece(model.get(), -1, 0, buf, 16, &len), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_tokenize(model.get(), nullptr, 3, 1, 0, toks.data(), 3, &n),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_token_is_eog(model.get(), kTinyEos), 1);
    EXPECT_EQ(kggml_llama_token_is_eog(model.get(), kTinySpaceAb), 0);
    EXPECT_EQ(kggml_llama_token_is_eog(model.get(), kTinyVocab + 5), 0);
}

#if defined(KGGML_LLAMA_TEST_HOOKS)
TEST(LlamaResources, EveryObjectIsReleasedDeterministically) {
    const int64_t live = kggml_llama_test_live_objects();
    const int32_t refs = kggml_runtime_ref_count();
    for (int round = 0; round < 3; ++round) {
        auto model = load_tiny_model();
        ASSERT_TRUE(model);
        auto c1 = make_context(model.get(), small_context_params());
        auto c2 = make_context(model.get(), small_context_params());
        kggml_llama_sampler * s = nullptr;
        ASSERT_EQ(kggml_llama_sampler_create(model.get(), nullptr, &s), KGGML_STATUS_OK);
        SamplerPtr sampler(s);
        EXPECT_EQ(kggml_llama_test_live_objects(), live + 4);
        const auto prompt = letter_prompt(8);
        ASSERT_EQ(kggml_llama_decode(c1.get(), prompt.data(), 8, 0, 0, 1), KGGML_STATUS_OK);
        ASSERT_EQ(kggml_llama_decode(c2.get(), prompt.data(), 8, 0, 0, 1), KGGML_STATUS_OK);
        sampler.reset();
        c1.reset();
        c2.reset();
        EXPECT_EQ(kggml_llama_test_live_objects(), live + 1);
    }
    EXPECT_EQ(kggml_llama_test_live_objects(), live);
    EXPECT_EQ(kggml_runtime_ref_count(), refs);
}

TEST(LlamaResources, FailedCreationsLeakNothing) {
    const int64_t live = kggml_llama_test_live_objects();
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    kggml_llama_context_params p = small_context_params();
    p.n_seq_max = 0;
    kggml_llama_context * c = nullptr;
    EXPECT_EQ(kggml_llama_context_create(model.get(), &p, &c), KGGML_STATUS_INVALID_ARGUMENT);
    kggml_llama_sampler_params sp{};
    kggml_llama_sampler_default_params(&sp);
    sp.grammar = "root ::= (";
    kggml_llama_sampler * s = nullptr;
    EXPECT_EQ(kggml_llama_sampler_create(model.get(), &sp, &s), KGGML_STATUS_INVALID_ARGUMENT);
    kggml_llama_mtmd * mt = nullptr;
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), garbage_model_path().c_str(), nullptr, &mt), KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_EQ(kggml_llama_test_live_objects(), live + 1);
    model.reset();
    EXPECT_EQ(kggml_llama_test_live_objects(), live);
}
#endif

TEST(LlamaModel, IndependentModelsCoexist) {
    auto a = load_tiny_model();
    auto b = load_tiny_model();
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    auto ca = make_context(a.get(), small_context_params());
    auto cb = make_context(b.get(), small_context_params());
    const auto prompt = letter_prompt(6);
    ASSERT_EQ(kggml_llama_decode(ca.get(), prompt.data(), 6, 0, 0, 1), KGGML_STATUS_OK);
    ASSERT_EQ(kggml_llama_decode(cb.get(), prompt.data(), 6, 0, 0, 1), KGGML_STATUS_OK);
    EXPECT_EQ(last_logits(ca.get()), last_logits(cb.get())) << "same weights + same input = same logits";
}
