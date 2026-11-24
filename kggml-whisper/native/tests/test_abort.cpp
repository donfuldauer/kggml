// Atomic abort callback plumbing: request_abort() from any thread reaches whisper.cpp's abort_callback /
// encoder_begin_callback and turns the running transcription into KGGML_STATUS_CANCELLED.
#include "kggml_whisper.h"
#include "tiny_model.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#if defined(KGGML_WHISPER_TEST_HOOKS)
#    include "kggml_whisper_testing.h"
#endif

namespace {

using ContextPtr = std::unique_ptr<kggml_whisper_context, decltype(&kggml_whisper_free)>;

ContextPtr make_context() {
    kggml_whisper_context * ctx = nullptr;
    EXPECT_EQ(kggml_whisper_init_from_file(kggml::test::tiny_model_path().c_str(), nullptr, &ctx), KGGML_STATUS_OK)
        << kggml_whisper_last_error();
    return ContextPtr(ctx, kggml_whisper_free);
}

kggml_whisper_transcribe_params fast_params() {
    kggml_whisper_transcribe_params p{};
    EXPECT_EQ(kggml_whisper_transcribe_default_params(&p), KGGML_STATUS_OK);
    p.n_threads = 2;
    p.temperature_inc = 0.0f;
    p.max_tokens = 8;
    return p;
}

} // namespace

TEST(Abort, LatchesUntilCleared) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(kggml_whisper_abort_requested(ctx.get()), 0);
    kggml_whisper_request_abort(ctx.get());
    kggml_whisper_request_abort(ctx.get()); // idempotent
    EXPECT_EQ(kggml_whisper_abort_requested(ctx.get()), 1);
    kggml_whisper_clear_abort(ctx.get());
    EXPECT_EQ(kggml_whisper_abort_requested(ctx.get()), 0);
}

TEST(Abort, RequestBeforeStartCancelsWithoutRunning) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    std::vector<float> pcm(16000, 0.0f);
    const auto p = fast_params();
    kggml_whisper_request_abort(ctx.get());
    EXPECT_EQ(kggml_whisper_transcribe(ctx.get(), pcm.data(), pcm.size(), &p), KGGML_STATUS_CANCELLED);
    int32_t n = -1;
    ASSERT_EQ(kggml_whisper_segment_count(ctx.get(), &n), KGGML_STATUS_OK);
    EXPECT_EQ(n, 0);
    kggml_whisper_clear_abort(ctx.get());
    EXPECT_EQ(kggml_whisper_transcribe(ctx.get(), pcm.data(), pcm.size(), &p), KGGML_STATUS_OK)
        << kggml_whisper_last_error();
}

TEST(Abort, RequestFromAnotherThreadIsVisible) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    std::thread([&] { kggml_whisper_request_abort(ctx.get()); }).join();
    EXPECT_EQ(kggml_whisper_abort_requested(ctx.get()), 1);
}

#if defined(KGGML_WHISPER_TEST_HOOKS)
TEST(Abort, CallbacksInstalledInWhisperParamsFollowTheFlag) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    int32_t abort = -1;
    int32_t encoder_continue = -1;
    ASSERT_EQ(kggml_whisper_test_poll_callbacks(ctx.get(), &abort, &encoder_continue), KGGML_STATUS_OK);
    EXPECT_EQ(abort, 0);
    EXPECT_EQ(encoder_continue, 1);
    kggml_whisper_request_abort(ctx.get());
    ASSERT_EQ(kggml_whisper_test_poll_callbacks(ctx.get(), &abort, &encoder_continue), KGGML_STATUS_OK);
    EXPECT_EQ(abort, 1);
    EXPECT_EQ(encoder_continue, 0);
}

TEST(Abort, WhisperPollsTheCallbackDuringInference) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    std::vector<float> pcm(16000, 0.01f);
    const auto p = fast_params();
    ASSERT_EQ(kggml_whisper_transcribe(ctx.get(), pcm.data(), pcm.size(), &p), KGGML_STATUS_OK)
        << kggml_whisper_last_error();
    int64_t polls = 0;
    ASSERT_EQ(kggml_whisper_test_abort_poll_count(ctx.get(), &polls), KGGML_STATUS_OK);
    EXPECT_GT(polls, 1); // encoder_begin + ggml graph computes (encoder and decoder)
}

TEST(Abort, MidInferenceAbortReturnsCancelled) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    // 40 s of audio = two 30 s encoder windows; abort fires on the 3rd poll, i.e. inside the first window.
    std::vector<float> pcm(16000 * 40, 0.01f);
    const auto p = fast_params();
    ASSERT_EQ(kggml_whisper_test_abort_after_polls(ctx.get(), 3), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_whisper_transcribe(ctx.get(), pcm.data(), pcm.size(), &p), KGGML_STATUS_CANCELLED);
    int64_t polls = 0;
    ASSERT_EQ(kggml_whisper_test_abort_poll_count(ctx.get(), &polls), KGGML_STATUS_OK);
    EXPECT_GE(polls, 3);
    int32_t n = -1;
    ASSERT_EQ(kggml_whisper_segment_count(ctx.get(), &n), KGGML_STATUS_OK);
    EXPECT_EQ(n, 0); // partial results of a cancelled run are discarded

    // The context stays usable after clearing.
    ASSERT_EQ(kggml_whisper_test_abort_after_polls(ctx.get(), -1), KGGML_STATUS_OK);
    kggml_whisper_clear_abort(ctx.get());
    std::vector<float> one_second(16000, 0.01f);
    EXPECT_EQ(kggml_whisper_transcribe(ctx.get(), one_second.data(), one_second.size(), &p), KGGML_STATUS_OK)
        << kggml_whisper_last_error();
}

TEST(Abort, ConcurrentRequestDuringLongRunCancels) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    // Long input so the run is still going when the other thread fires (10 windows of 30 s).
    std::vector<float> pcm(static_cast<size_t>(16000) * 30 * 10, 0.01f);
    const auto p = fast_params();
    std::atomic<bool> started{false};
    std::thread canceller([&] {
        while (!started.load()) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        kggml_whisper_request_abort(ctx.get());
    });
    started.store(true);
    const auto s = kggml_whisper_transcribe(ctx.get(), pcm.data(), pcm.size(), &p);
    canceller.join();
    EXPECT_EQ(s, KGGML_STATUS_CANCELLED) << kggml_whisper_last_error();
}
#endif
