// resampler.cpp - INTERNAL streaming PCM converter. See resampler.hpp for the math.
#include "resampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

namespace kggml::whisper {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr size_t kMaxTableEntries = size_t{1} << 18; // 1 MiB of float taps

/// Modified Bessel function of the first kind, order 0 (power series; converges fast for beta < 20).
double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    const double half_sq = (x * 0.5) * (x * 0.5);
    for (int k = 1; k < 64; ++k) {
        term *= half_sq / (static_cast<double>(k) * static_cast<double>(k));
        sum += term;
        if (term < sum * 1e-17) {
            break;
        }
    }
    return sum;
}

} // namespace

Resampler::Resampler(int32_t in_rate, int32_t out_rate, int32_t channels, SampleFormat format, Quality quality)
    : in_rate_(in_rate), out_rate_(out_rate), channels_(channels), format_(format), quality_(quality) {
    if (in_rate < kMinRate || in_rate > kMaxRate || out_rate < kMinRate || out_rate > kMaxRate) {
        throw std::invalid_argument("sample rates must be within [" + std::to_string(kMinRate) + ", " +
                                    std::to_string(kMaxRate) + "] Hz, got " + std::to_string(in_rate) + " -> " +
                                    std::to_string(out_rate));
    }
    if (channels < 1 || channels > kMaxChannels) {
        throw std::invalid_argument("channel count must be within [1, " + std::to_string(kMaxChannels) + "], got " +
                                    std::to_string(channels));
    }
    if (format != SampleFormat::F32 && format != SampleFormat::S16) {
        throw std::invalid_argument("unknown PCM format " + std::to_string(static_cast<int32_t>(format)));
    }
    if (quality != Quality::Linear && quality != Quality::Sinc) {
        throw std::invalid_argument("unknown resample quality " + std::to_string(static_cast<int32_t>(quality)));
    }

    const auto g = std::gcd(static_cast<uint64_t>(in_rate), static_cast<uint64_t>(out_rate));
    up_ = static_cast<uint64_t>(out_rate) / g;
    down_ = static_cast<uint64_t>(in_rate) / g;
    passthrough_ = in_rate == out_rate;

    const double scale = std::min(1.0, static_cast<double>(out_rate) / static_cast<double>(in_rate));
    if (quality == Quality::Linear) {
        half_width_ = 1;
        cutoff_ = 0.5;
    } else {
        half_width_ = static_cast<int32_t>(std::ceil(kSincZeroCrossings / scale));
        cutoff_ = 0.5 * scale * kSincRolloff;
    }

    if (!passthrough_) {
        const size_t taps = static_cast<size_t>(half_width_) * 2;
        if (up_ * taps <= kMaxTableEntries) {
            table_.resize(static_cast<size_t>(up_) * taps);
            for (uint64_t p = 0; p < up_; ++p) {
                const double frac = static_cast<double>(p) / static_cast<double>(up_);
                double sum = 0.0;
                float * row = table_.data() + p * taps;
                for (size_t j = 0; j < taps; ++j) {
                    const double d = frac + static_cast<double>(half_width_ - 1) - static_cast<double>(j);
                    const double w = kernel(d);
                    row[j] = static_cast<float>(w);
                    sum += w;
                }
                for (size_t j = 0; j < taps; ++j) {
                    row[j] = static_cast<float>(row[j] / sum);
                }
            }
        }
    }
    buffer_.reserve(static_cast<size_t>(half_width_) * 4 + 4096);
}

double Resampler::kernel(double d) const {
    const double ad = std::fabs(d);
    if (quality_ == Quality::Linear) {
        return ad >= 1.0 ? 0.0 : 1.0 - ad;
    }
    const double h = static_cast<double>(half_width_);
    if (ad >= h) {
        return 0.0;
    }
    const double x = 2.0 * cutoff_ * d;
    const double sinc = x == 0.0 ? 1.0 : std::sin(kPi * x) / (kPi * x);
    const double r = d / h;
    const double window = bessel_i0(kKaiserBeta * std::sqrt(std::max(0.0, 1.0 - r * r))) / bessel_i0(kKaiserBeta);
    return 2.0 * cutoff_ * sinc * window;
}

uint64_t Resampler::output_length(uint64_t in_frames, int32_t in_rate, int32_t out_rate) {
    const auto g = std::gcd(static_cast<uint64_t>(in_rate), static_cast<uint64_t>(out_rate));
    const uint64_t up = static_cast<uint64_t>(out_rate) / g;
    const uint64_t down = static_cast<uint64_t>(in_rate) / g;
    return (in_frames * up + down - 1) / down;
}

size_t Resampler::max_output(size_t in_frames) const {
    if (passthrough_) {
        return in_frames;
    }
    const uint64_t total_in = base_ + buffer_.size() + in_frames;
    const uint64_t total_out = (total_in * up_ + down_ - 1) / down_;
    return total_out > next_out_ ? static_cast<size_t>(total_out - next_out_) : 0;
}

float Resampler::sample_at(uint64_t index) const {
    if (index < base_) {
        return 0.0f; // only reachable for the implicit silence before the stream start
    }
    const uint64_t offset = index - base_;
    return offset < buffer_.size() ? buffer_[static_cast<size_t>(offset)] : 0.0f;
}

float Resampler::compute(uint64_t n) const {
    const uint64_t pos = n * down_;
    const uint64_t i = pos / up_;
    const uint64_t phase = pos % up_;
    const auto h = static_cast<int64_t>(half_width_);
    const size_t taps = static_cast<size_t>(half_width_) * 2;
    const int64_t first = static_cast<int64_t>(i) - h + 1;

    double acc = 0.0;
    if (!table_.empty()) {
        const float * row = table_.data() + phase * taps;
        for (size_t j = 0; j < taps; ++j) {
            const int64_t k = first + static_cast<int64_t>(j);
            if (k >= 0) {
                acc += static_cast<double>(row[j]) * sample_at(static_cast<uint64_t>(k));
            }
        }
        return static_cast<float>(acc);
    }

    // Large L (unusual rate pairs): evaluate and normalize the kernel for this phase on the fly.
    const double frac = static_cast<double>(phase) / static_cast<double>(up_);
    double norm = 0.0;
    for (size_t j = 0; j < taps; ++j) {
        const double w = kernel(frac + static_cast<double>(h - 1) - static_cast<double>(j));
        norm += w;
        const int64_t k = first + static_cast<int64_t>(j);
        if (k >= 0) {
            acc += w * sample_at(static_cast<uint64_t>(k));
        }
    }
    return static_cast<float>(acc / norm);
}

void Resampler::trim() {
    const uint64_t i_next = (next_out_ * down_) / up_;
    const uint64_t h = static_cast<uint64_t>(half_width_);
    const uint64_t keep_from = i_next + 1 > h ? i_next + 1 - h : 0;
    if (keep_from > base_) {
        const auto drop = static_cast<size_t>(std::min<uint64_t>(keep_from - base_, buffer_.size()));
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(drop));
        base_ += drop;
    }
}

size_t Resampler::process(const void * in, size_t in_frames, float * out) {
    if (in_frames == 0) {
        return 0;
    }
    const size_t old_size = buffer_.size();
    if (passthrough_) {
        // Same rate: format conversion + downmix straight into the caller buffer.
        buffer_.clear();
    }
    float * dst;
    if (passthrough_) {
        dst = out;
    } else {
        buffer_.resize(old_size + in_frames);
        dst = buffer_.data() + old_size;
    }

    const auto ch = static_cast<size_t>(channels_);
    if (format_ == SampleFormat::F32) {
        const auto * src = static_cast<const float *>(in);
        if (ch == 1) {
            std::memcpy(dst, src, in_frames * sizeof(float));
        } else {
            downmix_f32(src, in_frames, channels_, dst);
        }
    } else {
        const auto * src = static_cast<const int16_t *>(in);
        const float norm = 1.0f / (32768.0f * static_cast<float>(ch));
        for (size_t f = 0; f < in_frames; ++f) {
            int32_t sum = 0;
            for (size_t c = 0; c < ch; ++c) {
                sum += src[f * ch + c];
            }
            dst[f] = static_cast<float>(sum) * norm;
        }
    }

    if (passthrough_) {
        next_out_ += in_frames;
        base_ += in_frames;
        return in_frames;
    }

    const uint64_t total_in = base_ + buffer_.size();
    const uint64_t h = static_cast<uint64_t>(half_width_);
    size_t written = 0;
    // Output n is final once its right-most tap (index i_n + H) has arrived.
    while (true) {
        const uint64_t i = (next_out_ * down_) / up_;
        if (i + h >= total_in) {
            break;
        }
        out[written++] = compute(next_out_);
        ++next_out_;
    }
    trim();
    return written;
}

size_t Resampler::flush(float * out) {
    size_t written = 0;
    if (!passthrough_) {
        const uint64_t total_in = base_ + buffer_.size();
        const uint64_t total_out = (total_in * up_ + down_ - 1) / down_;
        while (next_out_ < total_out) {
            out[written++] = compute(next_out_);
            ++next_out_;
        }
    }
    reset();
    return written;
}

void Resampler::reset() {
    buffer_.clear();
    base_ = 0;
    next_out_ = 0;
}

void s16_to_f32(const int16_t * in, size_t count, float * out) noexcept {
    constexpr float norm = 1.0f / 32768.0f;
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<float>(in[i]) * norm;
    }
}

void downmix_f32(const float * in, size_t frames, int32_t channels, float * out) noexcept {
    const auto ch = static_cast<size_t>(channels);
    if (ch == 1) {
        if (out != in) {
            std::memmove(out, in, frames * sizeof(float));
        }
        return;
    }
    const float norm = 1.0f / static_cast<float>(ch);
    for (size_t f = 0; f < frames; ++f) {
        float sum = 0.0f;
        for (size_t c = 0; c < ch; ++c) {
            sum += in[f * ch + c];
        }
        out[f] = sum * norm; // out[f] is written after in[f*ch .. f*ch+ch-1] was read, so aliasing is safe
    }
}

const float * pad_to_min_length(const float * samples, size_t n_samples, size_t min_samples,
                                std::vector<float> & scratch, size_t & out_n) {
    if (n_samples >= min_samples) {
        out_n = n_samples;
        return samples;
    }
    scratch.assign(min_samples, 0.0f);
    if (n_samples > 0) {
        std::memcpy(scratch.data(), samples, n_samples * sizeof(float));
    }
    out_n = min_samples;
    return scratch.data();
}

} // namespace kggml::whisper
