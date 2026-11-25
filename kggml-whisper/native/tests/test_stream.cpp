// Streaming-state ABI: kggml_whisper_state_create/free (whisper_init_state / whisper_free_state),
// kggml_whisper_transcribe_with_state (whisper_full_with_state + new_segment_callback + prompt_tokens),
// short-buffer safety, abort and callback-driven stops, segment token read-back.
#include "kggml_whisper.h"
#include "tiny_model.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#if defined(KGGML_WHISPER_TEST_HOOKS)
#    include "kggml_whisper_testing.h"
#endif

namespace {

using ContextPtr = std::unique_ptr<kggml_whisper_context, decltype(&kggml_whisper_free)>;
using StatePtr = std::unique_ptr<kggml_whisper_state, decltype(&kggml_whisper_state_free)>;

/// Vocabulary layout of the tiny (English-only, 51864 ids) model: ids >= 50256 are special / timestamps.
constexpr int32_t kEnglishEot = 50256;
constexpr int32_t kTinyVocab = 51864;
constexpr int32_t kTinyTextCtx = 64;

ContextPtr make_context() {
    kggml_whisper_context * ctx = nullptr;
    EXPECT_EQ(kggml_whisper_init_from_file(kggml::test::tiny_model_path().c_str(), nullptr, &ctx), KGGML_STATUS_OK)
        << kggml_whisper_last_error();
    return ContextPtr(ctx, kggml_whisper_free);
}

StatePtr make_state(kggml_whisper_context * ctx) {
    kggml_whisper_state * st = nullptr;
    EXPECT_EQ(kggml_whisper_state_create(ctx, &st), KGGML_STATUS_OK) << kggml_whisper_last_error();
    return StatePtr(st, kggml_whisper_state_free);
}

kggml_whisper_transcribe_params fast_params() {
    kggml_whisper_transcribe_params p{};
    EXPECT_EQ(kggml_whisper_transcribe_default_params(&p), KGGML_STATUS_OK);
    p.n_threads = 2;
    p.temperature_inc = 0.0f;
    p.max_tokens = 8;
    p.no_context = 1;
    return p;
}

/// 2 s of a deterministic sawtooth; the tiny random model decodes text segments from it.
std::vector<float> speechlike(size_t n = 16000 * 2) {
    std::vector<float> pcm(n);
    for (size_t i = 0; i < n; ++i) pcm[i] = 0.1f * static_cast<float>((i % 100) / 50.0 - 1.0);
    return pcm;
}

struct CallbackLog {
    int calls = 0;
    int next_expected = 0;
    bool contiguous = true;
    bool readable = true;
    std::vector<int32_t> tokens;
    int32_t stop_with = KGGML_STATUS_OK;
};

int32_t record_segments(kggml_whisper_state * st, int32_t first, int32_t n_new, void * user_data) {
    auto * log = static_cast<CallbackLog *>(user_data);
    ++log->calls;
    if (first != log->next_expected || n_new <= 0) log->contiguous = false;
    log->next_expected = first + n_new;
    for (int32_t i = first; i < first + n_new; ++i) {
        kggml_whisper_segment seg{};
        if (kggml_whisper_state_get_segment(st, i, &seg) != KGGML_STATUS_OK || seg.text == nullptr) {
            log->readable = false;
        }
        int32_t count = 0;
        const auto q = kggml_whisper_state_get_segment_tokens(st, i, nullptr, 0, &count);
        if (q != KGGML_STATUS_OK && q != KGGML_STATUS_BUFFER_TOO_SMALL) log->readable = false;
        std::vector<int32_t> ids(static_cast<size_t>(count));
        if (kggml_whisper_state_get_segment_tokens(st, i, ids.data(), count, &count) != KGGML_STATUS_OK) {
            log->readable = false;
        }
        log->tokens.insert(log->tokens.end(), ids.begin(), ids.end());
    }
    return log->stop_with;
}

} // namespace

TEST(StreamState, CreateAndFreeAreBalancedAndValidated) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
#if defined(KGGML_WHISPER_TEST_HOOKS)
    const int64_t before = kggml_whisper_test_live_state_count();
#endif
    {
        auto a = make_state(ctx.get());
        auto b = make_state(ctx.get());
        ASSERT_NE(a, nullptr);
        ASSERT_NE(b, nullptr);
        EXPECT_NE(a.get(), b.get());
#if defined(KGGML_WHISPER_TEST_HOOKS)
        EXPECT_EQ(kggml_whisper_test_live_state_count(), before + 2);
#endif
    }
#if defined(KGGML_WHISPER_TEST_HOOKS)
    EXPECT_EQ(kggml_whisper_test_live_state_count(), before);
#endif
    kggml_whisper_state_free(nullptr); // no-op

    kggml_whisper_state * st = reinterpret_cast<kggml_whisper_state *>(0x1);
    EXPECT_EQ(kggml_whisper_state_create(nullptr, &st), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(st, nullptr);
    EXPECT_EQ(kggml_whisper_state_create(ctx.get(), nullptr), KGGML_STATUS_INVALID_ARGUMENT);

    int32_t n = -1;
    EXPECT_EQ(kggml_whisper_state_segment_count(nullptr, &n), KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(StreamState, MaxPromptTokensIsHalfTheTextContextMinusPrevMarker) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    int32_t n = 0;
    ASSERT_EQ(kggml_whisper_max_prompt_tokens(ctx.get(), &n), KGGML_STATUS_OK);
    EXPECT_EQ(n, kTinyTextCtx / 2 - 1);
    EXPECT_EQ(kggml_whisper_max_prompt_tokens(nullptr, &n), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_max_prompt_tokens(ctx.get(), nullptr), KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(StreamState, ShortAndEmptyBuffersArePaddedAndReachTheEncoder) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    auto st = make_state(ctx.get());
    ASSERT_NE(st, nullptr);
    const auto p = fast_params();
    for (size_t len : {size_t{1}, size_t{800}, size_t{1599}, size_t{15999}}) {
        std::vector<float> pcm(len, 0.05f);
        EXPECT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, nullptr, 0,
                                                      nullptr, nullptr),
                  KGGML_STATUS_OK)
            << len << ": " << kggml_whisper_last_error();
#if defined(KGGML_WHISPER_TEST_HOOKS)
        int64_t polls = 0;
        ASSERT_EQ(kggml_whisper_test_abort_poll_count(ctx.get(), &polls), KGGML_STATUS_OK);
        EXPECT_GT(polls, 0) << "padded " << len << "-sample input must reach the encoder";
#endif
    }
    EXPECT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), nullptr, 0, &p, nullptr, 0, nullptr, nullptr),
              KGGML_STATUS_OK)
        << kggml_whisper_last_error();
    EXPECT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), nullptr, 10, &p, nullptr, 0, nullptr, nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(StreamState, RejectsForeignStatesAndBadArguments) {
    auto a = make_context();
    auto b = make_context();
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    auto st = make_state(a.get());
    ASSERT_NE(st, nullptr);
    auto pcm = speechlike(16000);
    const auto p = fast_params();
    EXPECT_EQ(kggml_whisper_transcribe_with_state(b.get(), st.get(), pcm.data(), pcm.size(), &p, nullptr, 0, nullptr,
                                                  nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_transcribe_with_state(a.get(), nullptr, pcm.data(), pcm.size(), &p, nullptr, 0, nullptr,
                                                  nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_transcribe_with_state(nullptr, st.get(), pcm.data(), pcm.size(), &p, nullptr, 0, nullptr,
                                                  nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);
    auto bad = p;
    bad.language = "xx-not-a-language";
    EXPECT_EQ(kggml_whisper_transcribe_with_state(a.get(), st.get(), pcm.data(), pcm.size(), &bad, nullptr, 0, nullptr,
                                                  nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(StreamState, NewSegmentCallbackSeesEverySegmentInOrder) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    auto st = make_state(ctx.get());
    ASSERT_NE(st, nullptr);
    auto pcm = speechlike();
    const auto p = fast_params();
    CallbackLog log;
    ASSERT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, nullptr, 0,
                                                  record_segments, &log),
              KGGML_STATUS_OK)
        << kggml_whisper_last_error();
    int32_t n = -1;
    ASSERT_EQ(kggml_whisper_state_segment_count(st.get(), &n), KGGML_STATUS_OK);
    ASSERT_GT(n, 0) << "the tiny model is expected to decode at least one segment from this input";
    EXPECT_GE(log.calls, 1);
    EXPECT_TRUE(log.contiguous);
    EXPECT_TRUE(log.readable);
    EXPECT_EQ(log.next_expected, n) << "every finalized segment is announced exactly once";
    for (int32_t id : log.tokens) {
        EXPECT_GE(id, 0);
        EXPECT_LT(id, kEnglishEot) << "special / timestamp tokens must be filtered out";
    }

    // The same segments stay readable after the call, and out-of-range indices are rejected.
    kggml_whisper_segment seg{};
    for (int32_t i = 0; i < n; ++i) {
        ASSERT_EQ(kggml_whisper_state_get_segment(st.get(), i, &seg), KGGML_STATUS_OK);
        EXPECT_LE(seg.t0_ms, seg.t1_ms);
    }
    EXPECT_EQ(kggml_whisper_state_get_segment(st.get(), n, &seg), KGGML_STATUS_INVALID_ARGUMENT);
    int32_t count = 0;
    EXPECT_EQ(kggml_whisper_state_get_segment_tokens(st.get(), n, nullptr, 0, &count), KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(StreamState, SegmentTokensUseTheBufferTooSmallProtocol) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    auto st = make_state(ctx.get());
    ASSERT_NE(st, nullptr);
    auto pcm = speechlike();
    const auto p = fast_params();
    ASSERT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, nullptr, 0, nullptr,
                                                  nullptr),
              KGGML_STATUS_OK);
    int32_t n = 0;
    ASSERT_EQ(kggml_whisper_state_segment_count(st.get(), &n), KGGML_STATUS_OK);
    ASSERT_GT(n, 0);
    int32_t needed = -1;
    const auto q = kggml_whisper_state_get_segment_tokens(st.get(), 0, nullptr, 0, &needed);
    ASSERT_GE(needed, 0);
    if (needed > 0) {
        EXPECT_EQ(q, KGGML_STATUS_BUFFER_TOO_SMALL);
        std::vector<int32_t> small(static_cast<size_t>(needed - 1));
        int32_t got = 0;
        EXPECT_EQ(kggml_whisper_state_get_segment_tokens(st.get(), 0, small.data(), needed - 1, &got),
                  KGGML_STATUS_BUFFER_TOO_SMALL);
        EXPECT_EQ(got, needed);
    } else {
        EXPECT_EQ(q, KGGML_STATUS_OK);
    }
    std::vector<int32_t> ids(static_cast<size_t>(needed));
    int32_t got = -1;
    EXPECT_EQ(kggml_whisper_state_get_segment_tokens(st.get(), 0, ids.data(), needed, &got), KGGML_STATUS_OK);
    EXPECT_EQ(got, needed);
    EXPECT_EQ(kggml_whisper_state_get_segment_tokens(st.get(), 0, ids.data(), -1, &got), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_state_get_segment_tokens(st.get(), 0, ids.data(), needed, nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(StreamState, CallbackErrorStatusStopsDecodingAndIsReturned) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    auto st = make_state(ctx.get());
    ASSERT_NE(st, nullptr);
    auto pcm = speechlike();
    const auto p = fast_params();
    CallbackLog log;
    log.stop_with = KGGML_STATUS_NATIVE_FAILURE;
    EXPECT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, nullptr, 0,
                                                  record_segments, &log),
              KGGML_STATUS_NATIVE_FAILURE);
    EXPECT_EQ(log.calls, 1) << "no further callbacks after a stop request";
    int32_t n = -1;
    ASSERT_EQ(kggml_whisper_state_segment_count(st.get(), &n), KGGML_STATUS_OK);
    EXPECT_EQ(n, 0) << "segments of a stopped run are discarded";

    // The state is reusable afterwards.
    log = CallbackLog{};
    EXPECT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, nullptr, 0,
                                                  record_segments, &log),
              KGGML_STATUS_OK)
        << kggml_whisper_last_error();
}

TEST(StreamState, PromptTokensAreValidatedAndCarriedForward) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    auto st = make_state(ctx.get());
    ASSERT_NE(st, nullptr);
    auto pcm = speechlike();
    const auto p = fast_params();

    // Window 1: no history. Collect its text tokens through the callback.
    CallbackLog first;
    ASSERT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, nullptr, 0,
                                                  record_segments, &first),
              KGGML_STATUS_OK)
        << kggml_whisper_last_error();
    std::vector<int32_t> history = first.tokens;
    if (history.empty()) history = {1, 2, 3}; // the random model may decode timestamps only: any text ids do

    // Window 2: conditioned on window 1's tokens.
    CallbackLog second;
    ASSERT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, history.data(),
                                                  static_cast<int32_t>(history.size()), record_segments, &second),
              KGGML_STATUS_OK)
        << kggml_whisper_last_error();
#if defined(KGGML_WHISPER_TEST_HOOKS)
    int32_t count = 0;
    std::vector<int32_t> seen(history.size() + 4);
    ASSERT_EQ(kggml_whisper_test_state_last_prompt(st.get(), seen.data(), static_cast<int32_t>(seen.size()), &count),
              KGGML_STATUS_OK);
    seen.resize(static_cast<size_t>(count));
    EXPECT_EQ(seen, history) << "prompt_tokens must reach whisper_full_params unchanged";
#endif

    // Longer than whisper's prompt budget: accepted (whisper.cpp keeps the newest tokens).
    std::vector<int32_t> lots(200);
    for (size_t i = 0; i < lots.size(); ++i) lots[i] = static_cast<int32_t>(100 + i);
    EXPECT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, lots.data(),
                                                  static_cast<int32_t>(lots.size()), nullptr, nullptr),
              KGGML_STATUS_OK)
        << kggml_whisper_last_error();

    // Out-of-vocabulary ids and inconsistent pointers are rejected before whisper.cpp could index with them.
    const int32_t bad_high[] = {1, kTinyVocab};
    const int32_t bad_low[] = {-1};
    EXPECT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, bad_high, 2,
                                                  nullptr, nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, bad_low, 1, nullptr,
                                                  nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, nullptr, 3, nullptr,
                                                  nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, bad_low, -1,
                                                  nullptr, nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(StreamState, PromptedDecodesAreDeterministic) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    auto st = make_state(ctx.get());
    ASSERT_NE(st, nullptr);
    auto pcm = speechlike();
    const auto p = fast_params();
    const int32_t prompt[] = {400, 401, 402, 403};
    CallbackLog a;
    CallbackLog b;
    ASSERT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, prompt, 4,
                                                  record_segments, &a),
              KGGML_STATUS_OK);
    ASSERT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, prompt, 4,
                                                  record_segments, &b),
              KGGML_STATUS_OK);
    // no_context = 1: the explicit prompt is the only history, so a repeated call reproduces the same tokens.
    EXPECT_EQ(a.tokens, b.tokens);
}

TEST(StreamState, StatesAreIndependentOfTheContextResults) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    auto st = make_state(ctx.get());
    ASSERT_NE(st, nullptr);
    auto pcm = speechlike();
    const auto p = fast_params();
    ASSERT_EQ(kggml_whisper_transcribe(ctx.get(), pcm.data(), pcm.size(), &p), KGGML_STATUS_OK);
    int32_t ctx_count = -1;
    ASSERT_EQ(kggml_whisper_segment_count(ctx.get(), &ctx_count), KGGML_STATUS_OK);

    std::vector<float> silence(16000, 0.0f);
    ASSERT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), silence.data(), silence.size(), &p, nullptr, 0,
                                                  nullptr, nullptr),
              KGGML_STATUS_OK);
    int32_t after = -1;
    ASSERT_EQ(kggml_whisper_segment_count(ctx.get(), &after), KGGML_STATUS_OK);
    EXPECT_EQ(after, ctx_count) << "a stream state must not clobber the context's own results";
}

TEST(StreamState, AbortRequestCancelsStateTranscriptions) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    auto st = make_state(ctx.get());
    ASSERT_NE(st, nullptr);
    auto pcm = speechlike();
    const auto p = fast_params();
    kggml_whisper_request_abort(ctx.get());
    CallbackLog log;
    EXPECT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, nullptr, 0,
                                                  record_segments, &log),
              KGGML_STATUS_CANCELLED);
    EXPECT_EQ(log.calls, 0);
    kggml_whisper_clear_abort(ctx.get());

#if defined(KGGML_WHISPER_TEST_HOOKS)
    // Mid-inference: abort lands on the 3rd poll inside the first of two 30 s windows.
    std::vector<float> long_pcm(16000 * 40, 0.01f);
    ASSERT_EQ(kggml_whisper_test_abort_after_polls(ctx.get(), 3), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), long_pcm.data(), long_pcm.size(), &p, nullptr,
                                                  0, record_segments, &log),
              KGGML_STATUS_CANCELLED);
    int32_t n = -1;
    ASSERT_EQ(kggml_whisper_state_segment_count(st.get(), &n), KGGML_STATUS_OK);
    EXPECT_EQ(n, 0);
    ASSERT_EQ(kggml_whisper_test_abort_after_polls(ctx.get(), -1), KGGML_STATUS_OK);
    kggml_whisper_clear_abort(ctx.get());
#endif
    EXPECT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, nullptr, 0, nullptr,
                                                  nullptr),
              KGGML_STATUS_OK)
        << kggml_whisper_last_error();
}

TEST(StreamState, MultiWindowInputCallsBackAcrossEncoderWindows) {
    auto ctx = make_context();
    ASSERT_NE(ctx, nullptr);
    auto st = make_state(ctx.get());
    ASSERT_NE(st, nullptr);
    auto pcm = speechlike(16000 * 45); // two 30 s encoder windows
    const auto p = fast_params();
    CallbackLog log;
    ASSERT_EQ(kggml_whisper_transcribe_with_state(ctx.get(), st.get(), pcm.data(), pcm.size(), &p, nullptr, 0,
                                                  record_segments, &log),
              KGGML_STATUS_OK)
        << kggml_whisper_last_error();
    int32_t n = -1;
    ASSERT_EQ(kggml_whisper_state_segment_count(st.get(), &n), KGGML_STATUS_OK);
    EXPECT_TRUE(log.contiguous);
    EXPECT_EQ(log.next_expected, n);
    int64_t last_t0 = -1;
    for (int32_t i = 0; i < n; ++i) {
        kggml_whisper_segment seg{};
        ASSERT_EQ(kggml_whisper_state_get_segment(st.get(), i, &seg), KGGML_STATUS_OK);
        EXPECT_GE(seg.t0_ms, last_t0) << "segment start times are monotonic across windows";
        last_t0 = seg.t0_ms;
    }
}
