// libmtmd entry points: projector load error handling and argument validation (no projector weights are shipped;
// end-to-end vision inference is covered by the :kggml-llama VLM integration tests).
#include "fixtures.hpp"

#include <string>

using namespace kggml::test;

TEST(LlamaMtmd, DefaultParams) {
    kggml_llama_mtmd_params p{};
    ASSERT_EQ(kggml_llama_mtmd_default_params(&p), KGGML_STATUS_OK);
    EXPECT_EQ(p.backend, KGGML_BACKEND_CPU);
    EXPECT_EQ(p.n_threads, 0);
    EXPECT_EQ(p.flash_attn, -1);
    EXPECT_EQ(p.warmup, 0);
}

TEST(LlamaMtmd, ProjectorLoadErrors) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    kggml_llama_mtmd * mt = reinterpret_cast<kggml_llama_mtmd *>(0x1);

    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), nullptr, nullptr, &mt), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(mt, nullptr);
    EXPECT_EQ(kggml_llama_mtmd_init(nullptr, tiny_model_path().c_str(), nullptr, &mt), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), tiny_model_path().c_str(), nullptr, nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), (::testing::TempDir() + "kggml-no-mmproj.gguf").c_str(), nullptr, &mt),
              KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_NE(std::string(kggml_llama_last_error()).find("cannot open mmproj"), std::string::npos);
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), garbage_model_path().c_str(), nullptr, &mt), KGGML_STATUS_MODEL_LOAD_FAILED);
    // A valid GGUF that is a text model, not a projector: libmtmd's clip loader throws, the bridge reports it.
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), tiny_model_path().c_str(), nullptr, &mt), KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_NE(std::string(kggml_llama_last_error()).find("libmtmd could not load projector"), std::string::npos);
    EXPECT_EQ(mt, nullptr);
}

TEST(LlamaMtmd, ProjectorParamsAreValidated) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    kggml_llama_mtmd * mt = nullptr;
    kggml_llama_mtmd_params p{};
    kggml_llama_mtmd_default_params(&p);
    p.backend = KGGML_BACKEND_COREML;
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), tiny_model_path().c_str(), &p, &mt), KGGML_STATUS_BACKEND_UNAVAILABLE);
    p.backend = 17;
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), tiny_model_path().c_str(), &p, &mt), KGGML_STATUS_INVALID_ARGUMENT);
    p.backend = KGGML_BACKEND_CPU;
    p.n_threads = -2;
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), tiny_model_path().c_str(), &p, &mt), KGGML_STATUS_INVALID_ARGUMENT);
    p.n_threads = 0;
    p.flash_attn = 3;
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), tiny_model_path().c_str(), &p, &mt), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(mt, nullptr);
}
