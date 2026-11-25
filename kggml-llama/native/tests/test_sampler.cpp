// Sampler chains and GBNF grammar constraints.
#include "fixtures.hpp"

#include <string>
#include <vector>

using namespace kggml::test;

namespace {

class LlamaSamplerTest : public ::testing::Test {
protected:
    void SetUp() override {
        model_ = load_tiny_model();
        ASSERT_TRUE(model_);
        ctx_ = make_context(model_.get(), small_context_params());
        ASSERT_TRUE(ctx_);
    }

    SamplerPtr make(const kggml_llama_sampler_params & p) {
        kggml_llama_sampler * s = nullptr;
        EXPECT_EQ(kggml_llama_sampler_create(model_.get(), &p, &s), KGGML_STATUS_OK) << kggml_llama_last_error();
        return SamplerPtr(s);
    }

    static kggml_llama_sampler_params greedy(const char * grammar = nullptr) {
        kggml_llama_sampler_params p{};
        kggml_llama_sampler_default_params(&p);
        p.temperature = 0.0f;
        p.grammar = grammar;
        return p;
    }

    std::string piece(kggml_llama_token t) {
        char buf[32];
        int32_t len = 0;
        EXPECT_EQ(kggml_llama_token_to_piece(model_.get(), t, 0, buf, sizeof(buf), &len), KGGML_STATUS_OK);
        return std::string(buf, static_cast<size_t>(len));
    }

    /// Greedy-generates up to `max` tokens (stopping at EOG) and returns the text.
    std::string generate(kggml_llama_sampler * s, int32_t max, bool * hit_eog = nullptr) {
        const auto prompt = letter_prompt(4);
        EXPECT_EQ(kggml_llama_memory_clear(ctx_.get()), KGGML_STATUS_OK);
        EXPECT_EQ(kggml_llama_decode(ctx_.get(), prompt.data(), 4, 0, 0, 1), KGGML_STATUS_OK);
        int32_t pos = 4;
        std::string out;
        if (hit_eog != nullptr) *hit_eog = false;
        for (int32_t i = 0; i < max; ++i) {
            kggml_llama_token t = -1;
            const kggml_status st = kggml_llama_sampler_sample(s, ctx_.get(), -1, &t);
            EXPECT_EQ(st, KGGML_STATUS_OK) << kggml_llama_last_error();
            if (st != KGGML_STATUS_OK) break;
            if (kggml_llama_token_is_eog(model_.get(), t) == 1) {
                if (hit_eog != nullptr) *hit_eog = true;
                break;
            }
            out += piece(t);
            EXPECT_EQ(kggml_llama_decode(ctx_.get(), &t, 1, pos++, 0, 1), KGGML_STATUS_OK);
        }
        return out;
    }

    ModelPtr model_;
    ContextPtr ctx_;
};

} // namespace

TEST_F(LlamaSamplerTest, SampleRequiresLogits) {
    auto s = make(greedy());
    kggml_llama_token t = 0;
    EXPECT_EQ(kggml_llama_sampler_sample(s.get(), ctx_.get(), -1, &t), KGGML_STATUS_INVALID_ARGUMENT);
    const auto prompt = letter_prompt(3);
    ASSERT_EQ(kggml_llama_decode(ctx_.get(), prompt.data(), 3, 0, 0, 1), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_llama_sampler_sample(s.get(), ctx_.get(), 0, &t), KGGML_STATUS_INVALID_ARGUMENT) << "idx != -1";
    EXPECT_EQ(kggml_llama_sampler_sample(s.get(), ctx_.get(), -1, nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(kggml_llama_sampler_sample(s.get(), ctx_.get(), -1, &t), KGGML_STATUS_OK);
    EXPECT_GE(t, 0);
    EXPECT_LT(t, kTinyVocab);
}

TEST_F(LlamaSamplerTest, GreedyIsArgmaxAndDeterministic) {
    auto s = make(greedy());
    const auto prompt = letter_prompt(5);
    ASSERT_EQ(kggml_llama_decode(ctx_.get(), prompt.data(), 5, 0, 0, 1), KGGML_STATUS_OK);
    const auto logits = last_logits(ctx_.get());
    kggml_llama_token best = 0;
    for (int32_t i = 1; i < kTinyVocab; ++i) {
        if (logits[static_cast<size_t>(i)] > logits[static_cast<size_t>(best)]) best = i;
    }
    kggml_llama_token t = -1;
    ASSERT_EQ(kggml_llama_sampler_sample(s.get(), ctx_.get(), -1, &t), KGGML_STATUS_OK);
    EXPECT_EQ(t, best);
}

TEST_F(LlamaSamplerTest, SeededSamplingIsReproducible) {
    kggml_llama_sampler_params p{};
    kggml_llama_sampler_default_params(&p);
    p.seed = 1234;
    p.temperature = 1.5f;
    p.repeat_penalty = 1.1f;
    auto a = make(p);
    auto b = make(p);
    EXPECT_EQ(generate(a.get(), 12), generate(b.get(), 12));
}

TEST_F(LlamaSamplerTest, GrammarConstrainsOutputToTheLanguage) {
    // A JSON-ish object over the tiny alphabet: {"<letters>"} followed by end of generation.
    const char * gbnf = R"(root ::= "{" "\"" [a-z]+ "\"" "}")";
    for (const float temperature : {0.0f, 0.9f}) {
        kggml_llama_sampler_params p = greedy(gbnf);
        p.temperature = temperature;
        p.seed = 7;
        auto s = make(p);
        bool eog = false;
        const std::string text = generate(s.get(), 40, &eog);
        ASSERT_GE(text.size(), 4u) << text;
        EXPECT_EQ(text.front(), '{') << text;
        EXPECT_EQ(text[1], '"') << text;
        if (eog) {
            EXPECT_EQ(text.substr(text.size() - 2), "\"}") << text;
        }
        for (size_t i = 2; i + 2 < text.size(); ++i) {
            EXPECT_TRUE(text[i] >= 'a' && text[i] <= 'z') << text;
        }
    }
}

TEST_F(LlamaSamplerTest, FiniteGrammarEndsWithEndOfGeneration) {
    auto s = make(greedy(R"(root ::= "ab")"));
    bool eog = false;
    EXPECT_EQ(generate(s.get(), 10, &eog), "ab");
    EXPECT_TRUE(eog) << "a completed grammar admits only end-of-generation tokens";

    // Reset restarts the grammar.
    ASSERT_EQ(kggml_llama_sampler_reset(s.get()), KGGML_STATUS_OK);
    EXPECT_EQ(generate(s.get(), 10, &eog), "ab");
}

TEST_F(LlamaSamplerTest, InvalidGrammarsAreRejected) {
    kggml_llama_sampler * s = nullptr;
    auto p = greedy("root ::= (");
    EXPECT_EQ(kggml_llama_sampler_create(model_.get(), &p, &s), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::string(kggml_llama_last_error()).find("GBNF"), std::string::npos);
    p = greedy(R"(start ::= "a")");
    EXPECT_EQ(kggml_llama_sampler_create(model_.get(), &p, &s), KGGML_STATUS_INVALID_ARGUMENT) << "no 'root' rule";
    p.grammar_root = "start";
    ASSERT_EQ(kggml_llama_sampler_create(model_.get(), &p, &s), KGGML_STATUS_OK) << kggml_llama_last_error();
    kggml_llama_sampler_free(s);
    s = nullptr;
    p = greedy();
    p.repeat_penalty = 0.0f;
    EXPECT_EQ(kggml_llama_sampler_create(model_.get(), &p, &s), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(s, nullptr);
}

TEST_F(LlamaSamplerTest, AcceptRejectsTokensOutsideTheGrammar) {
    auto s = make(greedy(R"(root ::= "ab")"));
    const kggml_llama_token a = kTinyLetterBase + 0;
    const kggml_llama_token b = kTinyLetterBase + 1;
    const kggml_llama_token z = kTinyLetterBase + 25;
    EXPECT_EQ(kggml_llama_sampler_accept(s.get(), z, 1), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_sampler_accept(s.get(), kTinyEos, 1), KGGML_STATUS_INVALID_ARGUMENT)
        << "EOG before the grammar is complete must be rejected, not abort the process";
    EXPECT_EQ(kggml_llama_sampler_accept(s.get(), z, 0), KGGML_STATUS_OK) << "prompt tokens bypass the grammar";
    EXPECT_EQ(kggml_llama_sampler_accept(s.get(), kTinyVocab, 0), KGGML_STATUS_INVALID_ARGUMENT);
    ASSERT_EQ(kggml_llama_sampler_accept(s.get(), a, 1), KGGML_STATUS_OK) << kggml_llama_last_error();
    ASSERT_EQ(kggml_llama_sampler_accept(s.get(), b, 1), KGGML_STATUS_OK) << kggml_llama_last_error();
    EXPECT_EQ(kggml_llama_sampler_accept(s.get(), a, 1), KGGML_STATUS_INVALID_ARGUMENT) << "grammar complete";

    // The grammar is now complete: sampling yields end of generation.
    const auto prompt = letter_prompt(3);
    ASSERT_EQ(kggml_llama_decode(ctx_.get(), prompt.data(), 3, 0, 0, 1), KGGML_STATUS_OK);
    kggml_llama_token t = -1;
    ASSERT_EQ(kggml_llama_sampler_sample(s.get(), ctx_.get(), -1, &t), KGGML_STATUS_OK) << kggml_llama_last_error();
    EXPECT_EQ(kggml_llama_token_is_eog(model_.get(), t), 1) << t;
}

TEST_F(LlamaSamplerTest, SamplerAndContextMustShareTheModel) {
    auto other = load_tiny_model();
    ASSERT_TRUE(other);
    kggml_llama_sampler * s = nullptr;
    const auto p = greedy();
    ASSERT_EQ(kggml_llama_sampler_create(other.get(), &p, &s), KGGML_STATUS_OK);
    SamplerPtr sampler(s);
    const auto prompt = letter_prompt(3);
    ASSERT_EQ(kggml_llama_decode(ctx_.get(), prompt.data(), 3, 0, 0, 1), KGGML_STATUS_OK);
    kggml_llama_token t = 0;
    EXPECT_EQ(kggml_llama_sampler_sample(s, ctx_.get(), -1, &t), KGGML_STATUS_INVALID_ARGUMENT);
}
