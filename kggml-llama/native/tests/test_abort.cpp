// Abort plumbing: kggml_llama_request_abort() -> llama_set_abort_callback() -> ggml graph compute -> CANCELLED.
#include "fixtures.hpp"

#include <atomic>
#include <chrono>
#include <thread>

using namespace kggml::test;

TEST(LlamaAbort, FlagIsLatchedUntilCleared) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto ctx = make_context(model.get(), small_context_params());
    ASSERT_TRUE(ctx);
    EXPECT_EQ(kggml_llama_abort_requested(ctx.get()), 0);
    kggml_llama_request_abort(ctx.get());
    EXPECT_EQ(kggml_llama_abort_requested(ctx.get()), 1);
    kggml_llama_request_abort(ctx.get()); // idempotent
    EXPECT_EQ(kggml_llama_abort_requested(ctx.get()), 1);
    kggml_llama_clear_abort(ctx.get());
    EXPECT_EQ(kggml_llama_abort_requested(ctx.get()), 0);
}

TEST(LlamaAbort, AbortRequestedBeforeDecodeCancelsAndCanBeCleared) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto ctx = make_context(model.get(), small_context_params());
    ASSERT_TRUE(ctx);
    const auto prompt = letter_prompt(8);

    kggml_llama_request_abort(ctx.get());
    EXPECT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 8, 0, 0, 1), KGGML_STATUS_CANCELLED);
    const float * logits = nullptr;
    int32_t n = 0;
    EXPECT_EQ(kggml_llama_get_logits(ctx.get(), &logits, &n), KGGML_STATUS_INVALID_ARGUMENT)
        << "a cancelled decode exposes no logits";

    kggml_llama_clear_abort(ctx.get());
    ASSERT_EQ(kggml_llama_memory_clear(ctx.get()), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 8, 0, 0, 1), KGGML_STATUS_OK) << kggml_llama_last_error();
}

#if defined(KGGML_LLAMA_TEST_HOOKS)

TEST(LlamaAbort, RegisteredCallbackReflectsTheFlag) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto ctx = make_context(model.get(), small_context_params());
    ASSERT_TRUE(ctx);
    int32_t abort = -1;
    ASSERT_EQ(kggml_llama_test_poll_abort(ctx.get(), &abort), KGGML_STATUS_OK);
    EXPECT_EQ(abort, 0);
    kggml_llama_request_abort(ctx.get());
    ASSERT_EQ(kggml_llama_test_poll_abort(ctx.get(), &abort), KGGML_STATUS_OK);
    EXPECT_EQ(abort, 1);
}

TEST(LlamaAbort, LlamaCppPollsTheCallbackDuringDecode) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto ctx = make_context(model.get(), small_context_params());
    ASSERT_TRUE(ctx);
    int64_t before = -1;
    ASSERT_EQ(kggml_llama_test_abort_poll_count(ctx.get(), &before), KGGML_STATUS_OK);
    const auto prompt = letter_prompt(8);
    ASSERT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 8, 0, 0, 1), KGGML_STATUS_OK);
    int64_t after = -1;
    ASSERT_EQ(kggml_llama_test_abort_poll_count(ctx.get(), &after), KGGML_STATUS_OK);
    EXPECT_GT(after, before) << "llama.cpp never polled the abort callback registered via llama_set_abort_callback";
}

TEST(LlamaAbort, AbortLandingMidComputeCancelsTheDecode) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto ctx = make_context(model.get(), small_context_params());
    ASSERT_TRUE(ctx);
    const auto prompt = letter_prompt(64); // 4 batches of 16

    // Fire inside the first graph compute (after a few node-level polls).
    ASSERT_EQ(kggml_llama_test_abort_after_polls(ctx.get(), 3), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 64, 0, 0, 1), KGGML_STATUS_CANCELLED)
        << kggml_llama_last_error();
    EXPECT_EQ(kggml_llama_abort_requested(ctx.get()), 1);
    int32_t pos_max = 0;
    ASSERT_EQ(kggml_llama_memory_seq_pos_max(ctx.get(), 0, &pos_max), KGGML_STATUS_OK);
    EXPECT_LT(pos_max, 63) << "the decode must stop before the whole prompt was processed";

    // Recovery: clear the flag, drop the partial sequence, decode again.
    ASSERT_EQ(kggml_llama_test_abort_after_polls(ctx.get(), -1), KGGML_STATUS_OK);
    kggml_llama_clear_abort(ctx.get());
    ASSERT_EQ(kggml_llama_memory_seq_rm(ctx.get(), 0, 0, -1), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 64, 0, 0, 1), KGGML_STATUS_OK) << kggml_llama_last_error();
}

TEST(LlamaAbort, AbortBetweenBatchesStopsBeforeTheNextBatch) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto ctx = make_context(model.get(), small_context_params());
    ASSERT_TRUE(ctx);
    const auto prompt = letter_prompt(16);
    ASSERT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 16, 0, 0, 1), KGGML_STATUS_OK);
    int64_t polls_per_batch = 0;
    ASSERT_EQ(kggml_llama_test_abort_poll_count(ctx.get(), &polls_per_batch), KGGML_STATUS_OK);
    ASSERT_GT(polls_per_batch, 0);

    ASSERT_EQ(kggml_llama_memory_clear(ctx.get()), KGGML_STATUS_OK);
    // Abort on the very first poll of the second batch: the first batch completes, nothing after it does.
    const auto longer = letter_prompt(48);
    ASSERT_EQ(kggml_llama_test_abort_after_polls(ctx.get(), static_cast<int32_t>(polls_per_batch) + 1), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_llama_decode(ctx.get(), longer.data(), 48, 0, 0, 1), KGGML_STATUS_CANCELLED);
    int32_t pos_max = 0;
    ASSERT_EQ(kggml_llama_memory_seq_pos_max(ctx.get(), 0, &pos_max), KGGML_STATUS_OK);
    EXPECT_GE(pos_max, 15);
    EXPECT_LT(pos_max, 47);
}

#endif // KGGML_LLAMA_TEST_HOOKS

TEST(LlamaAbort, AbortFromAnotherThreadCancelsALongDecode) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    kggml_llama_context_params p = small_context_params();
    p.n_ctx = 512;
    p.n_batch = 1; // one llama_decode() per token: plenty of abort checkpoints
    auto ctx = make_context(model.get(), p);
    ASSERT_TRUE(ctx);
    const auto prompt = letter_prompt(512);

    std::atomic<bool> started{false};
    std::thread canceller([&] {
        while (!started.load()) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        kggml_llama_request_abort(ctx.get());
    });
    started.store(true);
    const kggml_status s = kggml_llama_decode(ctx.get(), prompt.data(), 512, 0, 0, 1);
    canceller.join();
    // On a very fast host the whole decode may finish before the abort lands; both outcomes are legal, but a
    // cancelled decode must report CANCELLED and keep the flag latched.
    if (s == KGGML_STATUS_CANCELLED) {
        EXPECT_EQ(kggml_llama_abort_requested(ctx.get()), 1);
    } else {
        EXPECT_EQ(s, KGGML_STATUS_OK) << kggml_llama_last_error();
    }
    kggml_llama_clear_abort(ctx.get());
}
