// tiny_model.hpp - writes structurally valid, deterministically initialized GGUF fixtures for the kggml_llama tests.
//
// The text model is a real llama.cpp "llama" architecture (RMSNorm, RoPE, SwiGLU, GQA-capable attention) with a
// 32-wide state, one block and a 288-token SentencePiece vocabulary (<unk>, <s>, </s>, the 256 byte-fallback
// tokens, "▁", a..z, "▁ab" and "▁z"), so it is ~100 KB and a decode runs in microseconds. Its output is meaningless;
// tests use it to drive the real llama.cpp load / decode / sampling / state paths without shipping model weights.
// Everything is generated at test time through the gguf_* API of the :kggml-runtime ggml; nothing is committed.
#pragma once

#include <cstdint>
#include <string>

namespace kggml::test {

constexpr int32_t kTinyVocab = 288;
constexpr int32_t kTinyEmbd = 32;
constexpr int32_t kTinyCtxTrain = 512;
constexpr int32_t kTinyBos = 1;
constexpr int32_t kTinyEos = 2;
constexpr int32_t kTinyByteBase = 3;      // <0x00> .. <0xFF> are ids 3 .. 258
constexpr int32_t kTinySpace = 259;       // "▁"
constexpr int32_t kTinyLetterBase = 260;  // "a" .. "z" are ids 260 .. 285
constexpr int32_t kTinySpaceAb = 286;     // "▁ab"
constexpr int32_t kTinySpaceZ = 287;      // "▁z"

// Tiny LLaVA-style ("mlp") vision projector for the tiny model: a 2-block ViT with a 16-wide state over 28x28 images
// in 14x14 patches (so every image becomes 4 embeddings of kTinyEmbd), deterministic weights, ~40 KB.
constexpr int32_t kTinyMmprojImageSize = 28;
constexpr int32_t kTinyMmprojImageTokens = 4;

/// Path of the tiny llama GGUF, generated once per test process under the GoogleTest temp dir.
const std::string & tiny_model_path();

/// Path of the tiny "mlp" projector GGUF (mmproj) matching the tiny model, generated once per test process.
const std::string & tiny_mmproj_path();

/// The first half of the tiny model's bytes (valid GGUF header, truncated tensor data).
const std::string & truncated_model_path();

/// Path of a file that is not a GGUF file at all (bad magic).
const std::string & garbage_model_path();

} // namespace kggml::test
