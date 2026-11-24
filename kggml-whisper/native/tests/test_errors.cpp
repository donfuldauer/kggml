// extern "C" exception -> structured error code mapping (C++ exceptions never cross the FFI boundary).
#include "kggml_whisper.h"

#include <gtest/gtest.h>

#include <ostream>
#include <string>
#include <thread>
#include <vector>

#if defined(KGGML_WHISPER_TEST_HOOKS)
#    include "kggml_whisper_testing.h"

namespace {

struct FaultCase {
    int32_t fault;
    kggml_status expected;
    const char * message_part;
};

void PrintTo(const FaultCase & c, std::ostream * os) {
    *os << "fault=" << c.fault << " expected=" << kggml_status_name(c.expected);
}

class FaultMapping : public ::testing::TestWithParam<FaultCase> {};

} // namespace

TEST_P(FaultMapping, PcmEntryPointMapsInjectedException) {
    const auto c = GetParam();
    kggml_whisper_test_inject_fault(c.fault);
    const int16_t in = 1;
    float out = 0.0f;
    EXPECT_EQ(kggml_whisper_pcm_s16_to_f32(&in, 1, &out), c.expected);
    EXPECT_NE(std::string(kggml_whisper_last_error()).find(c.message_part), std::string::npos)
        << kggml_whisper_last_error();
    EXPECT_NE(std::string(kggml_whisper_last_error()).find("kggml_whisper_pcm_s16_to_f32"), std::string::npos);
    // One-shot: the next call succeeds and clears the error.
    EXPECT_EQ(kggml_whisper_pcm_s16_to_f32(&in, 1, &out), KGGML_STATUS_OK);
    EXPECT_STREQ(kggml_whisper_last_error(), "");
}

TEST_P(FaultMapping, ContextEntryPointMapsInjectedExceptionAndReturnsNoContext) {
    const auto c = GetParam();
    kggml_whisper_context * ctx = reinterpret_cast<kggml_whisper_context *>(0x1);
    kggml_whisper_test_inject_fault(c.fault);
    EXPECT_EQ(kggml_whisper_init_from_file("does-not-matter.bin", nullptr, &ctx), c.expected);
    // The fault fires before the body runs, so out_context is untouched; callers must only use it on OK.
    EXPECT_NE(std::string(kggml_whisper_last_error()).find(c.message_part), std::string::npos);
}

TEST_P(FaultMapping, ResamplerEntryPointMapsInjectedException) {
    const auto c = GetParam();
    kggml_whisper_resampler * r = nullptr;
    kggml_whisper_test_inject_fault(c.fault);
    EXPECT_EQ(kggml_whisper_resampler_create(48000, 16000, 1, KGGML_PCM_F32, KGGML_RESAMPLE_SINC, &r), c.expected);
    EXPECT_EQ(r, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    AllFaults, FaultMapping,
    ::testing::Values(FaultCase{KGGML_WHISPER_TEST_FAULT_BAD_ALLOC, KGGML_STATUS_OUT_OF_MEMORY, "out of memory"},
                      FaultCase{KGGML_WHISPER_TEST_FAULT_RUNTIME_ERROR, KGGML_STATUS_NATIVE_FAILURE, "injected runtime_error"},
                      FaultCase{KGGML_WHISPER_TEST_FAULT_LOGIC_ERROR, KGGML_STATUS_NATIVE_FAILURE, "injected out_of_range"},
                      FaultCase{KGGML_WHISPER_TEST_FAULT_NON_STD, KGGML_STATUS_UNKNOWN, "non-standard"}),
    [](const auto & info) {
        switch (info.param.fault) {
            case KGGML_WHISPER_TEST_FAULT_BAD_ALLOC: return std::string("BadAlloc");
            case KGGML_WHISPER_TEST_FAULT_RUNTIME_ERROR: return std::string("RuntimeError");
            case KGGML_WHISPER_TEST_FAULT_LOGIC_ERROR: return std::string("LogicError");
            default: return std::string("NonStd");
        }
    });

TEST(FaultInjection, IsThreadLocal) {
    kggml_whisper_test_inject_fault(KGGML_WHISPER_TEST_FAULT_RUNTIME_ERROR);
    kggml_status other = KGGML_STATUS_UNKNOWN;
    std::thread([&] {
        const int16_t in = 1;
        float out = 0.0f;
        other = kggml_whisper_pcm_s16_to_f32(&in, 1, &out);
    }).join();
    EXPECT_EQ(other, KGGML_STATUS_OK);
    kggml_whisper_test_inject_fault(KGGML_WHISPER_TEST_FAULT_NONE); // disarm for later tests
}
#endif

TEST(Errors, FrameCountOverflowIsRejected) {
    // A frame count whose float buffer could never be addressed is rejected before any arithmetic overflows.
    size_t n = 0;
    EXPECT_EQ(kggml_whisper_resampled_length(static_cast<size_t>(-1), 48000, 16000, &n), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::string(kggml_whisper_last_error()).find("overflow"), std::string::npos);
}

TEST(Errors, LastErrorIsPerThreadAndClearable) {
    EXPECT_EQ(kggml_whisper_pcm_downmix_f32(nullptr, 1, 0, nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    const std::string mine = kggml_whisper_last_error();
    EXPECT_FALSE(mine.empty());
    std::string theirs = "unset";
    std::thread([&] { theirs = kggml_whisper_last_error(); }).join();
    EXPECT_EQ(theirs, "");
    kggml_whisper_clear_last_error();
    EXPECT_STREQ(kggml_whisper_last_error(), "");
}

TEST(Errors, NullPointersNeverCrash) {
    EXPECT_EQ(kggml_whisper_context_default_params(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_transcribe_default_params(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_init_from_file(nullptr, nullptr, nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_transcribe(nullptr, nullptr, 0, nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    int32_t i = 0;
    kggml_whisper_segment seg{};
    EXPECT_EQ(kggml_whisper_segment_count(nullptr, &i), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_get_segment(nullptr, 0, &seg), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_get_backend(nullptr, &i), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_uses_gpu(nullptr, &i), KGGML_STATUS_INVALID_ARGUMENT);
    kggml_whisper_request_abort(nullptr);
    kggml_whisper_clear_abort(nullptr);
    EXPECT_EQ(kggml_whisper_abort_requested(nullptr), 0);
    kggml_whisper_free(nullptr);
}
