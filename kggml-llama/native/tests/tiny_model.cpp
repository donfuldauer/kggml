// tiny_model.cpp - see tiny_model.hpp.
#include "tiny_model.hpp"

#include "ggml.h"
#include "gguf.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
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

constexpr int32_t kHeads = 2;
constexpr int32_t kHeadsKv = 2;
constexpr int32_t kFf = 64;
constexpr int32_t kLayers = 1;

// llama_token_type (llama.h)
constexpr int32_t kTypeNormal = 1;
constexpr int32_t kTypeUnknown = 2;
constexpr int32_t kTypeControl = 3;
constexpr int32_t kTypeByte = 6;

std::string temp_path(const char * stem, const char * ext) {
    return ::testing::TempDir() + stem + std::to_string(KGGML_GETPID()) + ext;
}

/// Deterministic xorshift weights in [-scale, scale].
struct Rng {
    uint32_t s;
    float next(float scale) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return (static_cast<float>(s & 0xFFFFFF) / static_cast<float>(0xFFFFFF) * 2.0f - 1.0f) * scale;
    }
};

std::string write_tiny_model() {
    const std::string path = temp_path("kggml-llama-tiny-", ".gguf");

    gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.architecture", "llama");
    gguf_set_val_str(gguf, "general.name", "kggml-tiny-llama");
    gguf_set_val_u32(gguf, "general.file_type", 0); // ALL_F32
    gguf_set_val_u32(gguf, "llama.context_length", kTinyCtxTrain);
    gguf_set_val_u32(gguf, "llama.embedding_length", kTinyEmbd);
    gguf_set_val_u32(gguf, "llama.block_count", kLayers);
    gguf_set_val_u32(gguf, "llama.feed_forward_length", kFf);
    gguf_set_val_u32(gguf, "llama.attention.head_count", kHeads);
    gguf_set_val_u32(gguf, "llama.attention.head_count_kv", kHeadsKv);
    gguf_set_val_f32(gguf, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(gguf, "llama.rope.dimension_count", kTinyEmbd / kHeads);
    gguf_set_val_f32(gguf, "llama.rope.freq_base", 10000.0f);
    gguf_set_val_u32(gguf, "llama.vocab_size", kTinyVocab);

    // SentencePiece vocabulary with byte fallback.
    std::vector<std::string> tokens;
    std::vector<float> scores;
    std::vector<int32_t> types;
    auto add = [&](std::string piece, float score, int32_t type) {
        tokens.push_back(std::move(piece));
        scores.push_back(score);
        types.push_back(type);
    };
    add("<unk>", 0.0f, kTypeUnknown);
    add("<s>", 0.0f, kTypeControl);
    add("</s>", 0.0f, kTypeControl);
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
        add(buf, 0.0f, kTypeByte);
    }
    add("\xE2\x96\x81", -1.0f, kTypeNormal); // "▁" (U+2581)
    for (char c = 'a'; c <= 'z'; ++c) {
        add(std::string(1, c), -2.0f, kTypeNormal);
    }
    add("\xE2\x96\x81" "ab", -0.5f, kTypeNormal);
    add("\xE2\x96\x81" "z", -0.5f, kTypeNormal);
    if (static_cast<int32_t>(tokens.size()) != kTinyVocab) throw std::logic_error("tiny vocab size mismatch");

    std::vector<const char *> token_ptrs;
    for (const auto & t : tokens) token_ptrs.push_back(t.c_str());
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(gguf, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(gguf, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_val_u32(gguf, "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_u32(gguf, "tokenizer.ggml.bos_token_id", kTinyBos);
    gguf_set_val_u32(gguf, "tokenizer.ggml.eos_token_id", kTinyEos);
    gguf_set_val_bool(gguf, "tokenizer.ggml.add_bos_token", true);
    gguf_set_val_bool(gguf, "tokenizer.ggml.add_eos_token", false);

    // Tensors (F32). ggml ne0 is the input dimension.
    const int32_t n_embd_kv = kTinyEmbd / kHeads * kHeadsKv;
    ggml_init_params ip{};
    ip.mem_size = 4 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = false;
    ggml_context * ctx = ggml_init(ip);
    Rng rng{0x9E3779B9u};
    auto tensor = [&](const char * name, int64_t ne0, int64_t ne1, bool norm) {
        ggml_tensor * t = ne1 > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1) : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
        ggml_set_name(t, name);
        auto * d = static_cast<float *>(t->data);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = norm ? 1.0f : rng.next(0.5f);
        gguf_add_tensor(gguf, t);
    };
    tensor("token_embd.weight", kTinyEmbd, kTinyVocab, false);
    tensor("output_norm.weight", kTinyEmbd, 0, true);
    tensor("output.weight", kTinyEmbd, kTinyVocab, false);
    tensor("blk.0.attn_norm.weight", kTinyEmbd, 0, true);
    tensor("blk.0.attn_q.weight", kTinyEmbd, kTinyEmbd, false);
    tensor("blk.0.attn_k.weight", kTinyEmbd, n_embd_kv, false);
    tensor("blk.0.attn_v.weight", kTinyEmbd, n_embd_kv, false);
    tensor("blk.0.attn_output.weight", kTinyEmbd, kTinyEmbd, false);
    tensor("blk.0.ffn_norm.weight", kTinyEmbd, 0, true);
    tensor("blk.0.ffn_gate.weight", kTinyEmbd, kFf, false);
    tensor("blk.0.ffn_down.weight", kFf, kTinyEmbd, false);
    tensor("blk.0.ffn_up.weight", kTinyEmbd, kFf, false);

    const bool ok = gguf_write_to_file(gguf, path.c_str(), false);
    ggml_free(ctx);
    gguf_free(gguf);
    if (!ok) throw std::runtime_error("could not write " + path);
    return path;
}

} // namespace

const std::string & tiny_model_path() {
    static const std::string path = write_tiny_model();
    return path;
}

const std::string & truncated_model_path() {
    static const std::string path = [] {
        std::ifstream in(tiny_model_path(), std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::string p = temp_path("kggml-llama-truncated-", ".gguf");
        std::ofstream out(p, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size() / 2));
        return p;
    }();
    return path;
}

const std::string & garbage_model_path() {
    static const std::string path = [] {
        const std::string p = temp_path("kggml-llama-garbage-", ".gguf");
        std::ofstream out(p, std::ios::binary | std::ios::trunc);
        out << "this is not a GGUF model, just some bytes with a bad magic number";
        return p;
    }();
    return path;
}

} // namespace kggml::test
