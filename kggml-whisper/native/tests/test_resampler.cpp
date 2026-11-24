// Resampling accuracy: 44.1 / 48 kHz mono / stereo Float32 / Int16 -> 16 kHz mono Float32.
#include "kggml_whisper.h"
#include "resampler.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<float> sine(int32_t rate, double freq, size_t frames, int32_t channels, double amp = 0.5) {
    std::vector<float> v(frames * static_cast<size_t>(channels));
    for (size_t i = 0; i < frames; ++i) {
        const auto s = static_cast<float>(amp * std::sin(2.0 * kPi * freq * static_cast<double>(i) / rate));
        for (int32_t c = 0; c < channels; ++c) v[i * static_cast<size_t>(channels) + static_cast<size_t>(c)] = s;
    }
    return v;
}

std::vector<int16_t> to_s16(const std::vector<float> & f) {
    std::vector<int16_t> s(f.size());
    for (size_t i = 0; i < f.size(); ++i) {
        s[i] = static_cast<int16_t>(std::lround(std::clamp(f[i], -1.0f, 1.0f) * 32767.0f));
    }
    return s;
}

std::vector<float> convert(const void * in, int32_t format, size_t frames, int32_t channels, int32_t rate,
                           int32_t quality) {
    size_t needed = 0;
    EXPECT_EQ(kggml_whisper_resampled_length(frames, rate, 16000, &needed), KGGML_STATUS_OK);
    std::vector<float> out(needed);
    size_t written = 0;
    EXPECT_EQ(kggml_whisper_pcm_to_whisper(in, format, frames, channels, rate, quality, out.data(), out.size(),
                                           &written),
              KGGML_STATUS_OK)
        << kggml_whisper_last_error();
    EXPECT_EQ(written, needed);
    return out;
}

/// Max abs error vs the ideal 16 kHz sine, ignoring `edge` samples at each end (filter warm-up / tail).
double max_error(const std::vector<float> & out, double freq, double amp, size_t edge) {
    double err = 0.0;
    for (size_t n = edge; n + edge < out.size(); ++n) {
        const double ideal = amp * std::sin(2.0 * kPi * freq * static_cast<double>(n) / 16000.0);
        err = std::max(err, std::fabs(out[n] - ideal));
    }
    return err;
}

double rms(const std::vector<float> & v, size_t edge) {
    double acc = 0.0;
    size_t n = 0;
    for (size_t i = edge; i + edge < v.size(); ++i, ++n) acc += static_cast<double>(v[i]) * v[i];
    return std::sqrt(acc / static_cast<double>(std::max<size_t>(n, 1)));
}

} // namespace

// (input rate, channels, format)
class ResampleAccuracy : public ::testing::TestWithParam<std::tuple<int32_t, int32_t, int32_t>> {};

TEST_P(ResampleAccuracy, SincMatchesIdealSineAt16k) {
    const auto [rate, channels, format] = GetParam();
    const size_t frames = static_cast<size_t>(rate); // 1 s
    for (double freq : {440.0, 1000.0, 3000.0, 6000.0}) {
        const auto f = sine(rate, freq, frames, channels);
        std::vector<float> out;
        if (format == KGGML_PCM_F32) {
            out = convert(f.data(), format, frames, channels, rate, KGGML_RESAMPLE_SINC);
        } else {
            const auto s = to_s16(f);
            out = convert(s.data(), format, frames, channels, rate, KGGML_RESAMPLE_SINC);
        }
        ASSERT_EQ(out.size(), 16000u);
        // Float: limited by the Kaiser window / passband ripple. Int16: plus 16-bit quantization.
        const double tol = format == KGGML_PCM_F32 ? 2e-3 : 3e-3;
        EXPECT_LT(max_error(out, freq, 0.5 * 32767.0 / 32768.0, 64), tol) << "rate=" << rate << " freq=" << freq;
    }
}

TEST_P(ResampleAccuracy, LinearIsReasonableForLowFrequencies) {
    const auto [rate, channels, format] = GetParam();
    const size_t frames = static_cast<size_t>(rate) / 2;
    const auto f = sine(rate, 440.0, frames, channels);
    std::vector<float> out;
    if (format == KGGML_PCM_F32) {
        out = convert(f.data(), format, frames, channels, rate, KGGML_RESAMPLE_LINEAR);
    } else {
        const auto s = to_s16(f);
        out = convert(s.data(), format, frames, channels, rate, KGGML_RESAMPLE_LINEAR);
    }
    EXPECT_LT(max_error(out, 440.0, 0.5, 4), 5e-3);
}

INSTANTIATE_TEST_SUITE_P(
    CommonRates, ResampleAccuracy,
    ::testing::Combine(::testing::Values(44100, 48000), ::testing::Values(1, 2),
                       ::testing::Values(static_cast<int32_t>(KGGML_PCM_F32), static_cast<int32_t>(KGGML_PCM_S16))),
    [](const auto & info) {
        return std::to_string(std::get<0>(info.param)) + "Hz_" + std::to_string(std::get<1>(info.param)) + "ch_" +
               (std::get<2>(info.param) == KGGML_PCM_F32 ? "f32" : "s16");
    });

TEST(Resampler, OutputLengthIsCeilOfRatio) {
    size_t n = 0;
    ASSERT_EQ(kggml_whisper_resampled_length(48000, 48000, 16000, &n), KGGML_STATUS_OK);
    EXPECT_EQ(n, 16000u);
    ASSERT_EQ(kggml_whisper_resampled_length(44100, 44100, 16000, &n), KGGML_STATUS_OK);
    EXPECT_EQ(n, 16000u);
    ASSERT_EQ(kggml_whisper_resampled_length(1, 44100, 16000, &n), KGGML_STATUS_OK);
    EXPECT_EQ(n, 1u);
    ASSERT_EQ(kggml_whisper_resampled_length(442, 44100, 16000, &n), KGGML_STATUS_OK);
    EXPECT_EQ(n, 161u); // 442 * 160 / 441 = 160.36
    ASSERT_EQ(kggml_whisper_resampled_length(0, 48000, 16000, &n), KGGML_STATUS_OK);
    EXPECT_EQ(n, 0u);
    ASSERT_EQ(kggml_whisper_resampled_length(100, 8000, 16000, &n), KGGML_STATUS_OK);
    EXPECT_EQ(n, 200u);
    EXPECT_EQ(kggml_whisper_resampled_length(1, 999, 16000, &n), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_resampled_length(1, 48000, 16000, nullptr), KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(Resampler, SincSuppressesAliasingWhenDownsampling) {
    // 12 kHz at 48 kHz is above the 8 kHz output Nyquist and would alias to 4 kHz without a low-pass.
    const auto f = sine(48000, 12000.0, 48000, 1);
    const auto sinc = convert(f.data(), KGGML_PCM_F32, 48000, 1, 48000, KGGML_RESAMPLE_SINC);
    EXPECT_LT(rms(sinc, 64), 1e-3);
    const auto lin = convert(f.data(), KGGML_PCM_F32, 48000, 1, 48000, KGGML_RESAMPLE_LINEAR);
    EXPECT_GT(rms(lin, 4), 0.1); // linear has no anti-aliasing filter; documents the trade-off
}

TEST(Resampler, DcGainIsUnity) {
    std::vector<float> dc(44100, 0.25f);
    const auto out = convert(dc.data(), KGGML_PCM_F32, dc.size(), 1, 44100, KGGML_RESAMPLE_SINC);
    for (size_t i = 64; i + 64 < out.size(); ++i) {
        ASSERT_NEAR(out[i], 0.25f, 1e-5f) << i;
    }
}

TEST(Resampler, SameRateIsExactPassthrough) {
    const auto f = sine(16000, 1234.0, 1000, 2);
    const auto out = convert(f.data(), KGGML_PCM_F32, 1000, 2, 16000, KGGML_RESAMPLE_SINC);
    ASSERT_EQ(out.size(), 1000u);
    for (size_t i = 0; i < out.size(); ++i) {
        ASSERT_EQ(out[i], f[2 * i]);
    }
}

TEST(Resampler, UpsamplingFrom8kIsAccurate) {
    const auto f = sine(8000, 1000.0, 8000, 1);
    const auto out = convert(f.data(), KGGML_PCM_F32, 8000, 1, 8000, KGGML_RESAMPLE_SINC);
    ASSERT_EQ(out.size(), 16000u);
    EXPECT_LT(max_error(out, 1000.0, 0.5, 64), 2e-3);
}

TEST(Resampler, ChunkedStreamingIsBitExactWithOneShot) {
    for (int32_t rate : {44100, 48000, 22050, 8000}) {
        const size_t frames = static_cast<size_t>(rate) * 3 / 2;
        const auto f = sine(rate, 777.0, frames, 2);
        const auto one_shot = convert(f.data(), KGGML_PCM_F32, frames, 2, rate, KGGML_RESAMPLE_SINC);

        kggml_whisper_resampler * r = nullptr;
        ASSERT_EQ(kggml_whisper_resampler_create(rate, 16000, 2, KGGML_PCM_F32, KGGML_RESAMPLE_SINC, &r),
                  KGGML_STATUS_OK);
        std::unique_ptr<kggml_whisper_resampler, decltype(&kggml_whisper_resampler_free)> guard(
            r, kggml_whisper_resampler_free);
        std::vector<float> streamed;
        size_t pos = 0;
        size_t chunk = 1;
        while (pos < frames) {
            const size_t n = std::min(chunk, frames - pos);
            size_t bound = 0;
            ASSERT_EQ(kggml_whisper_resampler_max_output(r, n, &bound), KGGML_STATUS_OK);
            std::vector<float> out(bound);
            size_t written = 0;
            ASSERT_EQ(kggml_whisper_resampler_process(r, f.data() + pos * 2, n, out.data(), out.size(), &written),
                      KGGML_STATUS_OK);
            ASSERT_LE(written, bound);
            streamed.insert(streamed.end(), out.begin(), out.begin() + static_cast<std::ptrdiff_t>(written));
            pos += n;
            chunk = chunk * 3 + 7; // 1, 10, 37, 118, ... irregular chunk sizes
            if (chunk > 5000) chunk = 3;
        }
        size_t bound = 0;
        ASSERT_EQ(kggml_whisper_resampler_max_output(r, 0, &bound), KGGML_STATUS_OK);
        std::vector<float> tail(bound);
        size_t written = 0;
        ASSERT_EQ(kggml_whisper_resampler_flush(r, tail.data(), tail.size(), &written), KGGML_STATUS_OK);
        EXPECT_EQ(written, bound);
        streamed.insert(streamed.end(), tail.begin(), tail.begin() + static_cast<std::ptrdiff_t>(written));
        ASSERT_EQ(streamed.size(), one_shot.size()) << rate;
        for (size_t i = 0; i < streamed.size(); ++i) {
            ASSERT_EQ(streamed[i], one_shot[i]) << "rate=" << rate << " i=" << i;
        }
    }
}

TEST(Resampler, FlushResetsForANewStream) {
    kggml_whisper_resampler * r = nullptr;
    ASSERT_EQ(kggml_whisper_resampler_create(48000, 16000, 1, KGGML_PCM_S16, KGGML_RESAMPLE_SINC, &r), KGGML_STATUS_OK);
    std::vector<int16_t> in(4800, 1000);
    std::vector<float> out(2000);
    size_t a = 0;
    size_t b = 0;
    ASSERT_EQ(kggml_whisper_resampler_process(r, in.data(), in.size(), out.data(), out.size(), &a), KGGML_STATUS_OK);
    ASSERT_EQ(kggml_whisper_resampler_flush(r, out.data() + a, out.size() - a, &b), KGGML_STATUS_OK);
    EXPECT_EQ(a + b, 1600u);
    ASSERT_EQ(kggml_whisper_resampler_process(r, in.data(), in.size(), out.data(), out.size(), &a), KGGML_STATUS_OK);
    ASSERT_EQ(kggml_whisper_resampler_flush(r, out.data() + a, out.size() - a, &b), KGGML_STATUS_OK);
    EXPECT_EQ(a + b, 1600u);
    ASSERT_EQ(kggml_whisper_resampler_reset(r), KGGML_STATUS_OK);
    kggml_whisper_resampler_free(r);
}

TEST(Resampler, BufferTooSmallConsumesNothing) {
    kggml_whisper_resampler * r = nullptr;
    ASSERT_EQ(kggml_whisper_resampler_create(44100, 16000, 1, KGGML_PCM_F32, KGGML_RESAMPLE_SINC, &r), KGGML_STATUS_OK);
    std::vector<float> in(4410, 0.1f);
    std::vector<float> out(10);
    size_t written = 0;
    EXPECT_EQ(kggml_whisper_resampler_process(r, in.data(), in.size(), out.data(), out.size(), &written),
              KGGML_STATUS_BUFFER_TOO_SMALL);
    EXPECT_GE(written, 1600u); // reports the bound
    size_t bound = 0;
    ASSERT_EQ(kggml_whisper_resampler_max_output(r, 0, &bound), KGGML_STATUS_OK);
    EXPECT_EQ(bound, 0u); // nothing was consumed
    kggml_whisper_resampler_free(r);
}

TEST(Resampler, RejectsInvalidParameters) {
    kggml_whisper_resampler * r = reinterpret_cast<kggml_whisper_resampler *>(0x1);
    EXPECT_EQ(kggml_whisper_resampler_create(0, 16000, 1, KGGML_PCM_F32, KGGML_RESAMPLE_SINC, &r),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(r, nullptr);
    EXPECT_NE(std::string(kggml_whisper_last_error()).find("sample rates"), std::string::npos);
    EXPECT_EQ(kggml_whisper_resampler_create(48000, 16000, 0, KGGML_PCM_F32, KGGML_RESAMPLE_SINC, &r),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_resampler_create(48000, 16000, 1, 7, KGGML_RESAMPLE_SINC, &r), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_resampler_create(48000, 16000, 1, KGGML_PCM_F32, 9, &r), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_resampler_create(48000, 16000, 1, KGGML_PCM_F32, KGGML_RESAMPLE_SINC, nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);
    size_t n = 0;
    EXPECT_EQ(kggml_whisper_resampler_process(nullptr, nullptr, 0, nullptr, 0, &n), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_resampler_flush(nullptr, nullptr, 0, &n), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_resampler_reset(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    kggml_whisper_resampler_free(nullptr); // no-op
}

TEST(Resampler, UnusualRatePairUsesOnTheFlyKernel) {
    // 44101 / 16000 are coprime: L = 16000 phases x 2H taps exceeds the table budget.
    const auto f = sine(44101, 500.0, 44101, 1);
    const auto out = convert(f.data(), KGGML_PCM_F32, 44101, 1, 44101, KGGML_RESAMPLE_SINC);
    ASSERT_EQ(out.size(), 16000u);
    EXPECT_LT(max_error(out, 500.0, 0.5, 64), 2e-3);
}

TEST(PadToMinLength, PadsShortAndPassesLongInputThrough) {
    std::vector<float> scratch;
    size_t n = 0;
    const std::vector<float> shortbuf(100, 0.5f);
    const float * p = kggml::whisper::pad_to_min_length(shortbuf.data(), shortbuf.size(), 16000, scratch, n);
    ASSERT_EQ(n, 16000u);
    EXPECT_EQ(p, scratch.data());
    EXPECT_EQ(p[99], 0.5f);
    EXPECT_EQ(p[100], 0.0f);
    EXPECT_EQ(p[15999], 0.0f);

    const std::vector<float> longbuf(16000, 0.25f);
    p = kggml::whisper::pad_to_min_length(longbuf.data(), longbuf.size(), 16000, scratch, n);
    EXPECT_EQ(p, longbuf.data()); // zero-copy for >= 1000 ms
    EXPECT_EQ(n, 16000u);

    p = kggml::whisper::pad_to_min_length(nullptr, 0, 16000, scratch, n);
    EXPECT_EQ(n, 16000u);
    EXPECT_EQ(p[0], 0.0f);
}
