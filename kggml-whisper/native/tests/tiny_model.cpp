// tiny_model.cpp - see tiny_model.hpp.
#include "tiny_model.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <process.h>
#    define KGGML_GETPID _getpid
#else
#    include <unistd.h>
#    define KGGML_GETPID getpid
#endif

namespace kggml::test {
namespace {

constexpr int32_t kVocab = 51864; // English-only vocabulary size
constexpr int32_t kAudioCtx = 1500;
constexpr int32_t kState = 16;
constexpr int32_t kHeads = 1;
constexpr int32_t kLayers = 1;
constexpr int32_t kTextCtx = 64;
constexpr int32_t kMels = 80;
constexpr int32_t kFft = 201;
constexpr uint32_t kGgmlMagic = 0x67676d6c; // "ggml"

/// IEEE-754 binary32 -> binary16, round-to-nearest-even; enough for the small normal values written here.
uint16_t to_f16(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign); // flush tiny values to zero
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    const uint32_t rest = mant & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) {
        ++h;
    }
    return static_cast<uint16_t>(h);
}

class Writer {
public:
    explicit Writer(const std::string & path) : out_(path, std::ios::binary | std::ios::trunc) {
        if (!out_) throw std::runtime_error("cannot create " + path);
    }
    void i32(int32_t v) { out_.write(reinterpret_cast<const char *>(&v), sizeof v); }
    void u32(uint32_t v) { out_.write(reinterpret_cast<const char *>(&v), sizeof v); }
    void f32s(const std::vector<float> & v) {
        out_.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    }

    // Deterministic pseudo-random weights in [-scale, scale]; layer-norm weights are 1 (`ones`).
    // F16 is used exactly where whisper.cpp expects it for ftype = 1 (matrices and conv kernels).
    void tensor(const std::string & name, std::vector<int32_t> ne, bool ones = false, bool f16 = false) {
        constexpr float scale = 0.05f;
        i32(static_cast<int32_t>(ne.size()));
        i32(static_cast<int32_t>(name.size()));
        i32(f16 ? 1 : 0); // GGML_TYPE_F16 : GGML_TYPE_F32
        size_t n = 1;
        for (int32_t d : ne) {
            i32(d);
            n *= static_cast<size_t>(d);
        }
        out_.write(name.data(), static_cast<std::streamsize>(name.size()));
        std::vector<float> data(n);
        for (auto & x : data) {
            state_ = state_ * 1664525u + 1013904223u;
            x = ones ? 1.0f : scale * (static_cast<float>(state_ >> 8) / 8388608.0f - 1.0f);
        }
        if (f16) {
            std::vector<uint16_t> half(n);
            for (size_t i = 0; i < n; ++i) half[i] = to_f16(data[i]);
            out_.write(reinterpret_cast<const char *>(half.data()), static_cast<std::streamsize>(n * sizeof(uint16_t)));
        } else {
            f32s(data);
        }
    }

    /// Matrix / conv kernel (F16).
    void matrix(const std::string & name, std::vector<int32_t> ne) { tensor(name, std::move(ne), false, true); }

    void vocab(int32_t n) {
        i32(n);
        for (int32_t id = 0; id < n; ++id) {
            const std::string word = id == 220 ? std::string(" ") : "tok" + std::to_string(id);
            u32(static_cast<uint32_t>(word.size()));
            out_.write(word.data(), static_cast<std::streamsize>(word.size()));
        }
    }

    void close() {
        out_.close();
        if (!out_) throw std::runtime_error("write failed");
    }

private:
    std::ofstream out_;
    uint32_t state_ = 12345u;
};

void block(Writer & w, const std::string & p, bool cross) {
    const int32_t S = kState;
    w.tensor(p + "mlp_ln.weight", {S}, true);
    w.tensor(p + "mlp_ln.bias", {S});
    w.matrix(p + "mlp.0.weight", {S, 4 * S});
    w.tensor(p + "mlp.0.bias", {4 * S});
    w.matrix(p + "mlp.2.weight", {4 * S, S});
    w.tensor(p + "mlp.2.bias", {S});
    w.tensor(p + "attn_ln.weight", {S}, true);
    w.tensor(p + "attn_ln.bias", {S});
    w.matrix(p + "attn.query.weight", {S, S});
    w.tensor(p + "attn.query.bias", {S});
    w.matrix(p + "attn.key.weight", {S, S});
    w.matrix(p + "attn.value.weight", {S, S});
    w.tensor(p + "attn.value.bias", {S});
    w.matrix(p + "attn.out.weight", {S, S});
    w.tensor(p + "attn.out.bias", {S});
    if (cross) {
        w.tensor(p + "cross_attn_ln.weight", {S}, true);
        w.tensor(p + "cross_attn_ln.bias", {S});
        w.matrix(p + "cross_attn.query.weight", {S, S});
        w.tensor(p + "cross_attn.query.bias", {S});
        w.matrix(p + "cross_attn.key.weight", {S, S});
        w.matrix(p + "cross_attn.value.weight", {S, S});
        w.tensor(p + "cross_attn.value.bias", {S});
        w.matrix(p + "cross_attn.out.weight", {S, S});
        w.tensor(p + "cross_attn.out.bias", {S});
    }
}

std::string write_tiny_model() {
    const std::string path = ::testing::TempDir() + "kggml-whisper-tiny-" + std::to_string(KGGML_GETPID()) + ".bin";
    Writer w(path);
    w.u32(kGgmlMagic);
    for (int32_t v : {kVocab, kAudioCtx, kState, kHeads, kLayers, kTextCtx, kState, kHeads, kLayers, kMels, 1 /*ftype F16*/}) {
        w.i32(v);
    }
    // Mel filter bank: a simple bin-to-band mapping is enough for a structurally valid spectrogram.
    w.i32(kMels);
    w.i32(kFft);
    std::vector<float> filters(static_cast<size_t>(kMels) * kFft, 0.0f);
    for (int32_t m = 0; m < kMels; ++m) {
        filters[static_cast<size_t>(m) * kFft + static_cast<size_t>(m * 2 + 1)] = 1.0f;
    }
    w.f32s(filters);
    // Vocabulary: 256 explicit tokens (id 220 is " ", which whisper.cpp's logit filter looks up by text);
    // whisper.cpp synthesizes names for the remaining ids up to n_vocab.
    w.vocab(256);

    const int32_t S = kState;
    w.tensor("encoder.positional_embedding", {S, kAudioCtx});
    w.matrix("encoder.conv1.weight", {3, kMels, S});
    w.tensor("encoder.conv1.bias", {1, S});
    w.matrix("encoder.conv2.weight", {3, S, S});
    w.tensor("encoder.conv2.bias", {1, S});
    w.tensor("encoder.ln_post.weight", {S}, true);
    w.tensor("encoder.ln_post.bias", {S});
    for (int32_t i = 0; i < kLayers; ++i) {
        block(w, "encoder.blocks." + std::to_string(i) + ".", false);
    }
    w.tensor("decoder.positional_embedding", {S, kTextCtx});
    w.matrix("decoder.token_embedding.weight", {S, kVocab});
    w.tensor("decoder.ln.weight", {S}, true);
    w.tensor("decoder.ln.bias", {S});
    for (int32_t i = 0; i < kLayers; ++i) {
        block(w, "decoder.blocks." + std::to_string(i) + ".", true);
    }
    w.close();
    return path;
}

} // namespace

const std::string & tiny_model_path() {
    static const std::string path = write_tiny_model();
    return path;
}

const std::string & garbage_model_path() {
    static const std::string path = [] {
        const std::string p = ::testing::TempDir() + "kggml-whisper-garbage-" + std::to_string(KGGML_GETPID()) + ".bin";
        std::ofstream out(p, std::ios::binary | std::ios::trunc);
        out << "this is not a whisper model, just some bytes with a bad magic number";
        return p;
    }();
    return path;
}

} // namespace kggml::test
