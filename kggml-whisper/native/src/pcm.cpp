// pcm.cpp - extern "C" PCM conversion API (Int16 normalization, downmix, streaming resampler).
#include "kggml_whisper.h"

#include "guard.hpp"
#include "resampler.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

struct kggml_whisper_resampler {
    explicit kggml_whisper_resampler(kggml::whisper::Resampler r) : impl(std::move(r)) {}
    kggml::whisper::Resampler impl;
};

using kggml::whisper::fail;
using kggml::whisper::guarded;

namespace {

/// Resampler parameter validation errors are caller mistakes, not native failures.
template <typename Fn>
kggml_status with_invalid_argument(Fn && fn) {
    try {
        return fn();
    } catch (const std::invalid_argument & e) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, e.what());
    }
}

kggml_status buffer_too_small(size_t required, size_t * out_frames) {
    if (out_frames != nullptr) {
        *out_frames = required;
    }
    return fail(KGGML_STATUS_BUFFER_TOO_SMALL,
                ("output buffer too small, need " + std::to_string(required) + " frames").c_str());
}

constexpr size_t kMaxFrames = std::numeric_limits<size_t>::max() / (sizeof(float) * 8);

} // namespace

extern "C" {

KGGML_WHISPER_API kggml_status kggml_whisper_pcm_s16_to_f32(const int16_t * in, size_t count, float * out) {
    return guarded(__func__, [&] {
        if (count > 0 && (in == nullptr || out == nullptr)) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "in/out is NULL");
        }
        kggml::whisper::s16_to_f32(in, count, out);
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_pcm_downmix_f32(const float * in, size_t frames, int32_t channels,
                                                             float * out) {
    return guarded(__func__, [&] {
        if (channels < 1 || channels > kggml::whisper::Resampler::kMaxChannels) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "channels must be within [1, 8]");
        }
        if (frames > 0 && (in == nullptr || out == nullptr)) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "in/out is NULL");
        }
        if (frames > kMaxFrames) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "frame count overflows");
        }
        kggml::whisper::downmix_f32(in, frames, channels, out);
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_resampled_length(size_t in_frames, int32_t in_rate, int32_t out_rate,
                                                              size_t * out_frames) {
    return guarded(__func__, [&] {
        if (out_frames == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_frames is NULL");
        }
        using R = kggml::whisper::Resampler;
        if (in_rate < R::kMinRate || in_rate > R::kMaxRate || out_rate < R::kMinRate || out_rate > R::kMaxRate) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "sample rate out of range [1000, 384000]");
        }
        if (in_frames > kMaxFrames) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "frame count overflows");
        }
        *out_frames = static_cast<size_t>(R::output_length(in_frames, in_rate, out_rate));
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_pcm_to_whisper(const void * in, int32_t format, size_t frames,
                                                           int32_t channels, int32_t sample_rate, int32_t quality,
                                                           float * out, size_t out_capacity, size_t * out_frames) {
    return guarded(__func__, [&] {
        return with_invalid_argument([&] {
            if (out_frames == nullptr) {
                return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_frames is NULL");
            }
            *out_frames = 0;
            if (frames > kMaxFrames) {
                return fail(KGGML_STATUS_INVALID_ARGUMENT, "frame count overflows");
            }
            kggml::whisper::Resampler r(sample_rate, KGGML_WHISPER_SAMPLE_RATE, channels,
                                        static_cast<kggml::whisper::SampleFormat>(format),
                                        static_cast<kggml::whisper::Quality>(quality));
            const auto required =
                static_cast<size_t>(kggml::whisper::Resampler::output_length(frames, sample_rate, KGGML_WHISPER_SAMPLE_RATE));
            if (out_capacity < required || (required > 0 && out == nullptr)) {
                return buffer_too_small(required, out_frames);
            }
            if (frames > 0 && in == nullptr) {
                return fail(KGGML_STATUS_INVALID_ARGUMENT, "in is NULL");
            }
            size_t written = r.process(in, frames, out);
            written += r.flush(out + written);
            *out_frames = written;
            return KGGML_STATUS_OK;
        });
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_resampler_create(int32_t in_rate, int32_t out_rate, int32_t channels,
                                                             int32_t format, int32_t quality,
                                                             kggml_whisper_resampler ** out_resampler) {
    return guarded(__func__, [&] {
        return with_invalid_argument([&] {
            if (out_resampler == nullptr) {
                return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_resampler is NULL");
            }
            *out_resampler = nullptr;
            auto r = std::make_unique<kggml_whisper_resampler>(kggml::whisper::Resampler(
                in_rate, out_rate, channels, static_cast<kggml::whisper::SampleFormat>(format),
                static_cast<kggml::whisper::Quality>(quality)));
            *out_resampler = r.release();
            return KGGML_STATUS_OK;
        });
    });
}

KGGML_WHISPER_API void kggml_whisper_resampler_free(kggml_whisper_resampler * resampler) {
    delete resampler;
}

KGGML_WHISPER_API kggml_status kggml_whisper_resampler_max_output(const kggml_whisper_resampler * resampler,
                                                                 size_t in_frames, size_t * out_frames) {
    return guarded(__func__, [&] {
        if (resampler == nullptr || out_frames == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "resampler/out_frames is NULL");
        }
        if (in_frames > kMaxFrames) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "frame count overflows");
        }
        *out_frames = resampler->impl.max_output(in_frames);
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_resampler_process(kggml_whisper_resampler * resampler, const void * in,
                                                              size_t in_frames, float * out, size_t out_capacity,
                                                              size_t * out_frames) {
    return guarded(__func__, [&] {
        if (resampler == nullptr || out_frames == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "resampler/out_frames is NULL");
        }
        *out_frames = 0;
        if (in_frames > kMaxFrames) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "frame count overflows");
        }
        if (in_frames > 0 && in == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "in is NULL");
        }
        const size_t bound = resampler->impl.max_output(in_frames);
        if (out_capacity < bound || (bound > 0 && out == nullptr)) {
            return buffer_too_small(bound, out_frames);
        }
        *out_frames = resampler->impl.process(in, in_frames, out);
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_resampler_flush(kggml_whisper_resampler * resampler, float * out,
                                                            size_t out_capacity, size_t * out_frames) {
    return guarded(__func__, [&] {
        if (resampler == nullptr || out_frames == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "resampler/out_frames is NULL");
        }
        *out_frames = 0;
        const size_t bound = resampler->impl.max_output(0);
        if (out_capacity < bound || (bound > 0 && out == nullptr)) {
            return buffer_too_small(bound, out_frames);
        }
        *out_frames = resampler->impl.flush(out);
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_resampler_reset(kggml_whisper_resampler * resampler) {
    return guarded(__func__, [&] {
        if (resampler == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "resampler is NULL");
        }
        resampler->impl.reset();
        return KGGML_STATUS_OK;
    });
}

} // extern "C"
