// Int16 -> Float32 normalization and stereo/multichannel -> mono downmixing.
#include "kggml_whisper.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

TEST(PcmS16ToF32, NormalizesFullScaleRange) {
    const std::vector<int16_t> in = {0, 1, -1, 16384, -16384, std::numeric_limits<int16_t>::max(),
                                     std::numeric_limits<int16_t>::min()};
    std::vector<float> out(in.size(), 99.0f);
    ASSERT_EQ(kggml_whisper_pcm_s16_to_f32(in.data(), in.size(), out.data()), KGGML_STATUS_OK);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 1.0f / 32768.0f);
    EXPECT_FLOAT_EQ(out[2], -1.0f / 32768.0f);
    EXPECT_FLOAT_EQ(out[3], 0.5f);
    EXPECT_FLOAT_EQ(out[4], -0.5f);
    EXPECT_FLOAT_EQ(out[5], 32767.0f / 32768.0f);
    EXPECT_FLOAT_EQ(out[6], -1.0f);
    for (float v : out) {
        EXPECT_GE(v, -1.0f);
        EXPECT_LT(v, 1.0f);
    }
}

TEST(PcmS16ToF32, EmptyInputAndNullChecks) {
    EXPECT_EQ(kggml_whisper_pcm_s16_to_f32(nullptr, 0, nullptr), KGGML_STATUS_OK);
    float out = 0.0f;
    EXPECT_EQ(kggml_whisper_pcm_s16_to_f32(nullptr, 1, &out), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_STRNE(kggml_whisper_last_error(), "");
    const int16_t in = 1;
    EXPECT_EQ(kggml_whisper_pcm_s16_to_f32(&in, 1, nullptr), KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(PcmDownmix, StereoAveragesChannels) {
    const std::vector<float> stereo = {1.0f, 0.0f, 0.5f, 0.5f, -1.0f, 1.0f, 0.25f, -0.75f};
    std::vector<float> mono(4);
    ASSERT_EQ(kggml_whisper_pcm_downmix_f32(stereo.data(), 4, 2, mono.data()), KGGML_STATUS_OK);
    EXPECT_FLOAT_EQ(mono[0], 0.5f);
    EXPECT_FLOAT_EQ(mono[1], 0.5f);
    EXPECT_FLOAT_EQ(mono[2], 0.0f);
    EXPECT_FLOAT_EQ(mono[3], -0.25f);
}

TEST(PcmDownmix, InPlaceAliasingIsSafe) {
    std::vector<float> buf = {0.2f, 0.4f, 0.6f, 0.0f, 0.3f, 0.3f, 0.3f, 0.9f, 0.0f};
    ASSERT_EQ(kggml_whisper_pcm_downmix_f32(buf.data(), 3, 3, buf.data()), KGGML_STATUS_OK);
    EXPECT_NEAR(buf[0], 0.4f, 1e-6f);
    EXPECT_NEAR(buf[1], 0.2f, 1e-6f);
    EXPECT_NEAR(buf[2], 0.4f, 1e-6f);
}

TEST(PcmDownmix, MonoIsCopyAndChannelBoundsAreChecked) {
    const std::vector<float> in = {0.1f, -0.2f, 0.3f};
    std::vector<float> out(3);
    ASSERT_EQ(kggml_whisper_pcm_downmix_f32(in.data(), 3, 1, out.data()), KGGML_STATUS_OK);
    EXPECT_EQ(out, in);
    EXPECT_EQ(kggml_whisper_pcm_downmix_f32(in.data(), 3, 0, out.data()), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_pcm_downmix_f32(in.data(), 1, 9, out.data()), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_whisper_pcm_downmix_f32(nullptr, 1, 1, out.data()), KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(PcmToWhisper, StereoInt16AntiPhaseCancelsToSilence) {
    // L = x, R = -x  ->  mono is exactly 0 regardless of the resampler.
    std::vector<int16_t> stereo(4410 * 2);
    for (size_t i = 0; i < 4410; ++i) {
        const auto v = static_cast<int16_t>((i * 37) % 20000 - 10000);
        stereo[2 * i] = v;
        stereo[2 * i + 1] = static_cast<int16_t>(-v);
    }
    size_t needed = 0;
    EXPECT_EQ(kggml_whisper_pcm_to_whisper(stereo.data(), KGGML_PCM_S16, 4410, 2, 44100, KGGML_RESAMPLE_SINC, nullptr,
                                           0, &needed),
              KGGML_STATUS_BUFFER_TOO_SMALL);
    EXPECT_EQ(needed, 1600u);
    std::vector<float> out(needed);
    size_t written = 0;
    ASSERT_EQ(kggml_whisper_pcm_to_whisper(stereo.data(), KGGML_PCM_S16, 4410, 2, 44100, KGGML_RESAMPLE_SINC,
                                           out.data(), out.size(), &written),
              KGGML_STATUS_OK);
    ASSERT_EQ(written, 1600u);
    for (float v : out) {
        EXPECT_EQ(v, 0.0f);
    }
}
