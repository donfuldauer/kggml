// Context lifecycle, batched decoding, logits, KV-cache management, state serialization and embeddings.
#include "fixtures.hpp"

#include <cmath>
#include <string>
#include <vector>

using namespace kggml::test;

namespace {

class LlamaContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        model_ = load_tiny_model();
        ASSERT_TRUE(model_);
    }
    ModelPtr model_;
};

} // namespace

TEST_F(LlamaContextTest, CreateValidatesParameters) {
    kggml_llama_context * c = nullptr;
    kggml_llama_context_params p = small_context_params();
    p.n_batch = 0;
    EXPECT_EQ(kggml_llama_context_create(model_.get(), &p, &c), KGGML_STATUS_INVALID_ARGUMENT);
    p = small_context_params();
    p.n_seq_max = 100000;
    EXPECT_EQ(kggml_llama_context_create(model_.get(), &p, &c), KGGML_STATUS_INVALID_ARGUMENT);
    p = small_context_params();
    p.flash_attn = 7;
    EXPECT_EQ(kggml_llama_context_create(model_.get(), &p, &c), KGGML_STATUS_INVALID_ARGUMENT);
    p = small_context_params();
    p.pooling = 99;
    EXPECT_EQ(kggml_llama_context_create(model_.get(), &p, &c), KGGML_STATUS_INVALID_ARGUMENT);
    p = small_context_params();
    p.n_threads = -1;
    EXPECT_EQ(kggml_llama_context_create(model_.get(), &p, &c), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(c, nullptr);
    EXPECT_EQ(kggml_llama_context_create(model_.get(), nullptr, nullptr), KGGML_STATUS_INVALID_ARGUMENT);

    // NULL params = defaults (n_ctx = training length).
    ASSERT_EQ(kggml_llama_context_create(model_.get(), nullptr, &c), KGGML_STATUS_OK) << kggml_llama_last_error();
    ContextPtr ctx(c);
    uint32_t n_ctx = 0;
    ASSERT_EQ(kggml_llama_context_n_ctx(c, &n_ctx), KGGML_STATUS_OK);
    EXPECT_EQ(n_ctx, static_cast<uint32_t>(kTinyCtxTrain));
}

TEST_F(LlamaContextTest, DecodeProducesDeterministicFiniteLogits) {
    auto ctx = make_context(model_.get(), small_context_params());
    ASSERT_TRUE(ctx);
    const float * logits = nullptr;
    int32_t n_vocab = 0;
    EXPECT_EQ(kggml_llama_get_logits(ctx.get(), &logits, &n_vocab), KGGML_STATUS_INVALID_ARGUMENT)
        << "no decode yet";

    // 40 tokens with n_batch = 16: three llama_decode() calls.
    const auto prompt = letter_prompt(40);
    ASSERT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 40, 0, 0, 1), KGGML_STATUS_OK) << kggml_llama_last_error();
    const auto first = last_logits(ctx.get());
    ASSERT_EQ(first.size(), static_cast<size_t>(kTinyVocab));
    for (float v : first) ASSERT_TRUE(std::isfinite(v));
    int32_t pos_max = -2;
    ASSERT_EQ(kggml_llama_memory_seq_pos_max(ctx.get(), 0, &pos_max), KGGML_STATUS_OK);
    EXPECT_EQ(pos_max, 39);

    // Same input after clearing the cache, in a single batch layout, gives the same logits.
    ASSERT_EQ(kggml_llama_memory_clear(ctx.get()), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_llama_get_logits(ctx.get(), &logits, &n_vocab), KGGML_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(kggml_llama_memory_seq_pos_max(ctx.get(), 0, &pos_max), KGGML_STATUS_OK);
    EXPECT_EQ(pos_max, -1);
    ASSERT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 20, 0, 0, 0), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_llama_get_logits(ctx.get(), &logits, &n_vocab), KGGML_STATUS_INVALID_ARGUMENT)
        << "logits_last = 0 computes no outputs";
    ASSERT_EQ(kggml_llama_decode(ctx.get(), prompt.data() + 20, 20, 20, 0, 1), KGGML_STATUS_OK);
    const auto second = last_logits(ctx.get());
    ASSERT_EQ(second.size(), first.size());
    for (size_t i = 0; i < first.size(); ++i) EXPECT_NEAR(first[i], second[i], 1e-3f) << i;
}

TEST_F(LlamaContextTest, DecodeRejectsInvalidInput) {
    auto ctx = make_context(model_.get(), small_context_params());
    ASSERT_TRUE(ctx);
    const kggml_llama_token bad[] = {kTinyBos, kTinyVocab};
    EXPECT_EQ(kggml_llama_decode(ctx.get(), bad, 2, 0, 0, 1), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::string(kggml_llama_last_error()).find("out of range"), std::string::npos);
    const kggml_llama_token neg[] = {-3};
    EXPECT_EQ(kggml_llama_decode(ctx.get(), neg, 1, 0, 0, 1), KGGML_STATUS_INVALID_ARGUMENT);
    const kggml_llama_token ok[] = {kTinyBos};
    EXPECT_EQ(kggml_llama_decode(ctx.get(), ok, 0, 0, 0, 1), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_decode(ctx.get(), nullptr, 1, 0, 0, 1), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_decode(ctx.get(), ok, 1, -1, 0, 1), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_decode(ctx.get(), ok, 1, 0, 1, 1), KGGML_STATUS_INVALID_ARGUMENT) << "n_seq_max = 1";
    EXPECT_EQ(kggml_llama_decode(ctx.get(), ok, 1, 0, -1, 1), KGGML_STATUS_INVALID_ARGUMENT);
    // Still usable.
    EXPECT_EQ(kggml_llama_decode(ctx.get(), ok, 1, 0, 0, 1), KGGML_STATUS_OK) << kggml_llama_last_error();
}

TEST_F(LlamaContextTest, FullContextIsBufferTooSmall) {
    auto ctx = make_context(model_.get(), small_context_params());
    ASSERT_TRUE(ctx);
    uint32_t n_ctx = 0;
    ASSERT_EQ(kggml_llama_context_n_ctx(ctx.get(), &n_ctx), KGGML_STATUS_OK);
    const auto prompt = letter_prompt(static_cast<int32_t>(n_ctx));
    ASSERT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), static_cast<int32_t>(n_ctx), 0, 0, 1), KGGML_STATUS_OK)
        << kggml_llama_last_error();
    const kggml_llama_token one[] = {kTinyLetterBase};
    EXPECT_EQ(kggml_llama_decode(ctx.get(), one, 1, static_cast<int32_t>(n_ctx), 0, 1), KGGML_STATUS_BUFFER_TOO_SMALL)
        << kggml_llama_last_error();
    // Making room (context shift by removal) makes decoding possible again.
    ASSERT_EQ(kggml_llama_memory_seq_rm(ctx.get(), 0, static_cast<int32_t>(n_ctx) - 8, -1), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_llama_decode(ctx.get(), one, 1, static_cast<int32_t>(n_ctx) - 8, 0, 1), KGGML_STATUS_OK)
        << kggml_llama_last_error();
}

TEST_F(LlamaContextTest, WholeStateRoundTripRestoresTheKvCache) {
    auto ctx = make_context(model_.get(), small_context_params());
    ASSERT_TRUE(ctx);
    const auto prompt = letter_prompt(24);
    ASSERT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 24, 0, 0, 1), KGGML_STATUS_OK);
    size_t size = 0;
    ASSERT_EQ(kggml_llama_state_size(ctx.get(), -1, &size), KGGML_STATUS_OK);
    ASSERT_GT(size, 0u);
    size_t written = 0;
    std::vector<uint8_t> small(size / 2);
    EXPECT_EQ(kggml_llama_state_save(ctx.get(), -1, small.data(), small.size(), &written), KGGML_STATUS_BUFFER_TOO_SMALL);
    EXPECT_EQ(written, size);
    std::vector<uint8_t> state(size);
    ASSERT_EQ(kggml_llama_state_save(ctx.get(), -1, state.data(), state.size(), &written), KGGML_STATUS_OK);
    ASSERT_GT(written, 0u);
    state.resize(written);

    // Continue decoding from the original, then from a restored fresh context: identical results.
    const kggml_llama_token next[] = {kTinyLetterBase + 3};
    ASSERT_EQ(kggml_llama_decode(ctx.get(), next, 1, 24, 0, 1), KGGML_STATUS_OK);
    const auto expected = last_logits(ctx.get());

    auto restored = make_context(model_.get(), small_context_params());
    ASSERT_TRUE(restored);
    ASSERT_EQ(kggml_llama_state_load(restored.get(), -1, state.data(), state.size()), KGGML_STATUS_OK)
        << kggml_llama_last_error();
    int32_t pos_max = 0;
    ASSERT_EQ(kggml_llama_memory_seq_pos_max(restored.get(), 0, &pos_max), KGGML_STATUS_OK);
    EXPECT_EQ(pos_max, 23);
    const float * no_logits = nullptr;
    int32_t n_vocab = 0;
    EXPECT_EQ(kggml_llama_get_logits(restored.get(), &no_logits, &n_vocab), KGGML_STATUS_INVALID_ARGUMENT)
        << "a restored state carries the KV cache, not the outputs of the last decode";
    ASSERT_EQ(kggml_llama_decode(restored.get(), next, 1, 24, 0, 1), KGGML_STATUS_OK);
    const auto actual = last_logits(restored.get());
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-4f) << i;
}

TEST_F(LlamaContextTest, SequenceStateRoundTripAndCorruptDataIsRejected) {
    kggml_llama_context_params p = small_context_params();
    p.n_seq_max = 2;
    auto ctx = make_context(model_.get(), p);
    ASSERT_TRUE(ctx);
    const auto prompt = letter_prompt(12);
    ASSERT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 12, 0, 0, 1), KGGML_STATUS_OK);

    size_t size = 0;
    ASSERT_EQ(kggml_llama_state_size(ctx.get(), 0, &size), KGGML_STATUS_OK);
    ASSERT_GT(size, 0u);
    std::vector<uint8_t> state(size);
    size_t written = 0;
    ASSERT_EQ(kggml_llama_state_save(ctx.get(), 0, state.data(), state.size(), &written), KGGML_STATUS_OK);
    state.resize(written);

    // Reference: continue sequence 0 directly.
    const kggml_llama_token next[] = {kTinySpaceZ};
    ASSERT_EQ(kggml_llama_decode(ctx.get(), next, 1, 12, 0, 1), KGGML_STATUS_OK);
    const auto expected = last_logits(ctx.get());

    // Restore into sequence 1 and continue there.
    ASSERT_EQ(kggml_llama_state_load(ctx.get(), 1, state.data(), state.size()), KGGML_STATUS_OK) << kggml_llama_last_error();
    int32_t pos_max = 0;
    ASSERT_EQ(kggml_llama_memory_seq_pos_max(ctx.get(), 1, &pos_max), KGGML_STATUS_OK);
    EXPECT_EQ(pos_max, 11);
    ASSERT_EQ(kggml_llama_decode(ctx.get(), next, 1, 12, 1, 1), KGGML_STATUS_OK);
    const auto actual = last_logits(ctx.get());
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-4f) << i;

    // Corrupt / truncated data is rejected without crashing.
    std::vector<uint8_t> junk(16, 0xAB);
    EXPECT_EQ(kggml_llama_state_load(ctx.get(), 1, junk.data(), junk.size()), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_state_load(ctx.get(), 1, state.data(), state.size() / 3), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_state_load(ctx.get(), 5, state.data(), state.size()), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_state_load(ctx.get(), 0, nullptr, 4), KGGML_STATUS_INVALID_ARGUMENT);
}

TEST_F(LlamaContextTest, EmbeddingsArePooledAndNormalized) {
    kggml_llama_context_params p = small_context_params();
    p.embeddings = 1;
    p.pooling = KGGML_LLAMA_POOLING_MEAN;
    auto ctx = make_context(model_.get(), p);
    ASSERT_TRUE(ctx);
    const auto prompt = letter_prompt(10);
    ASSERT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 10, 0, 0, 1), KGGML_STATUS_OK) << kggml_llama_last_error();

    int32_t n = 0;
    EXPECT_EQ(kggml_llama_get_embeddings(ctx.get(), 0, 0, nullptr, 0, &n), KGGML_STATUS_BUFFER_TOO_SMALL);
    ASSERT_EQ(n, kTinyEmbd);
    std::vector<float> raw(static_cast<size_t>(n));
    ASSERT_EQ(kggml_llama_get_embeddings(ctx.get(), 0, 0, raw.data(), n, &n), KGGML_STATUS_OK) << kggml_llama_last_error();
    std::vector<float> unit(static_cast<size_t>(n));
    ASSERT_EQ(kggml_llama_get_embeddings(ctx.get(), 0, 1, unit.data(), n, &n), KGGML_STATUS_OK);
    double norm = 0.0;
    double raw_norm = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        ASSERT_TRUE(std::isfinite(raw[static_cast<size_t>(i)]));
        norm += static_cast<double>(unit[static_cast<size_t>(i)]) * unit[static_cast<size_t>(i)];
        raw_norm += static_cast<double>(raw[static_cast<size_t>(i)]) * raw[static_cast<size_t>(i)];
    }
    EXPECT_NEAR(norm, 1.0, 1e-4);
    EXPECT_GT(raw_norm, 0.0);
    EXPECT_EQ(kggml_llama_get_embeddings(ctx.get(), 1, 0, raw.data(), n, &n), KGGML_STATUS_INVALID_ARGUMENT);
}

TEST_F(LlamaContextTest, EmbeddingsRequireEmbeddingMode) {
    auto ctx = make_context(model_.get(), small_context_params());
    ASSERT_TRUE(ctx);
    const auto prompt = letter_prompt(4);
    ASSERT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 4, 0, 0, 1), KGGML_STATUS_OK);
    float out[kTinyEmbd];
    int32_t n = 0;
    EXPECT_EQ(kggml_llama_get_embeddings(ctx.get(), 0, 0, out, kTinyEmbd, &n), KGGML_STATUS_INVALID_ARGUMENT);
}

TEST_F(LlamaContextTest, LastTokenEmbeddingWithoutPooling) {
    kggml_llama_context_params p = small_context_params();
    p.embeddings = 1;
    p.pooling = KGGML_LLAMA_POOLING_NONE;
    auto ctx = make_context(model_.get(), p);
    ASSERT_TRUE(ctx);
    const auto prompt = letter_prompt(5);
    ASSERT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 5, 0, 0, 1), KGGML_STATUS_OK) << kggml_llama_last_error();
    std::vector<float> out(kTinyEmbd);
    int32_t n = 0;
    ASSERT_EQ(kggml_llama_get_embeddings(ctx.get(), 0, 1, out.data(), kTinyEmbd, &n), KGGML_STATUS_OK)
        << kggml_llama_last_error();
    EXPECT_EQ(n, kTinyEmbd);
}
