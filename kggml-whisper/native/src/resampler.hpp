// resampler.hpp - INTERNAL streaming PCM converter (format -> mono float32, any rate -> any rate).
//
// Rational polyphase resampler: with g = gcd(in_rate, out_rate), L = out_rate / g and M = in_rate / g, output
// sample n sits at input position n * M / L. Each output is a dot product of the 2H input samples around that
// position with a kernel evaluated at the fractional phase (n * M mod L) / L:
//
//   SINC:   Kaiser-windowed sinc low-pass with cutoff 0.5 * min(1, out/in) * rolloff, i.e. just below the
//           output Nyquist when downsampling (alias suppression) and the input Nyquist when upsampling.
//           H = ceil(zero_crossings / min(1, out/in)) input samples per side. Each phase is normalized to
//           unity DC gain. Phases are precomputed when L * 2H is small (all common audio rates).
//   LINEAR: H = 1, triangle kernel (classic linear interpolation, no low-pass).
//
// Input before the first sample and after the end of stream (flush) is treated as silence, so a stream of
// N input frames produces exactly ceil(N * out_rate / in_rate) output frames, whether it is fed in one
// call or in arbitrary chunks (chunking is bit-exact with one-shot conversion).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kggml::whisper {

enum class SampleFormat : int32_t { F32 = 0, S16 = 1 };
enum class Quality : int32_t { Linear = 0, Sinc = 1 };

class Resampler {
public:
    static constexpr int32_t kMinRate = 1000;
    static constexpr int32_t kMaxRate = 384000;
    static constexpr int32_t kMaxChannels = 8;
    static constexpr int32_t kSincZeroCrossings = 16;
    static constexpr double kSincRolloff = 0.945;
    static constexpr double kKaiserBeta = 8.6;

    /// Throws std::invalid_argument on unsupported parameters.
    Resampler(int32_t in_rate, int32_t out_rate, int32_t channels, SampleFormat format, Quality quality);

    /// Exact output length of a complete conversion of `in_frames` frames.
    static uint64_t output_length(uint64_t in_frames, int32_t in_rate, int32_t out_rate);

    /// Upper bound of frames emitted by process(in_frames) or flush() (in_frames = 0) from the current state.
    size_t max_output(size_t in_frames) const;

    /// Consumes interleaved frames, writes the newly determined outputs; returns the number written.
    /// `out` must hold max_output(in_frames) samples.
    size_t process(const void * in, size_t in_frames, float * out);

    /// Writes the tail (zero-padded input) and resets. `out` must hold max_output(0) samples.
    size_t flush(float * out);

    void reset();

    int32_t half_width() const { return half_width_; }
    int32_t in_rate() const { return in_rate_; }
    int32_t out_rate() const { return out_rate_; }

private:
    double kernel(double d) const;
    float compute(uint64_t n) const;
    void trim();
    float sample_at(uint64_t index) const; // 0 outside [base_, base_ + buffer_.size())

    int32_t in_rate_;
    int32_t out_rate_;
    int32_t channels_;
    SampleFormat format_;
    Quality quality_;
    bool passthrough_;

    uint64_t up_;   // L
    uint64_t down_; // M
    int32_t half_width_;
    double cutoff_; // normalized to the input rate, cycles per input sample

    std::vector<float> table_; // up_ phases x (2 * half_width_) taps, empty = compute on the fly

    std::vector<float> buffer_; // mono input samples with absolute indices [base_, base_ + size)
    uint64_t base_ = 0;
    uint64_t next_out_ = 0;    // absolute index of the next output sample
};

/// int16 -> float32 (x / 32768).
void s16_to_f32(const int16_t * in, size_t count, float * out) noexcept;

/// Interleaved float32 -> mono average. `out` may alias `in`.
void downmix_f32(const float * in, size_t frames, int32_t channels, float * out) noexcept;

/// Minimum-length guard for whisper_full_with_state(): copies `samples` into `scratch` zero-padded to
/// `min_samples` when shorter and returns the buffer to feed whisper (either `samples` itself or `scratch`).
const float * pad_to_min_length(const float * samples, size_t n_samples, size_t min_samples,
                                std::vector<float> & scratch, size_t & out_n);

} // namespace kggml::whisper
