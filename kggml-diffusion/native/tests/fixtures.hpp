// fixtures.hpp - RAII helpers and generated (never committed) model-file fixtures for the kggml_diffusion tests.
#pragma once

#include "kggml_diffusion.h"
#include "kggml_diffusion_testing.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

namespace kggml::test {

struct ContextDeleter {
    void operator()(kggml_diffusion_context * c) const noexcept { kggml_diffusion_context_free(c); }
};
using ContextPtr = std::unique_ptr<kggml_diffusion_context, ContextDeleter>;

/// Owns a kggml_diffusion_image (freed with kggml_diffusion_image_free()).
struct Image {
    kggml_diffusion_image img{};
    Image() = default;
    Image(const Image &) = delete;
    Image & operator=(const Image &) = delete;
    ~Image() { kggml_diffusion_image_free(&img); }
    kggml_diffusion_image * out() { return &img; }
};

inline ContextPtr make_stub() {
    kggml_diffusion_context * c = nullptr;
    EXPECT_EQ(kggml_diffusion_test_context_create_stub(&c), KGGML_STATUS_OK) << kggml_diffusion_last_error();
    return ContextPtr(c);
}

inline kggml_diffusion_context_params cpu_params(const std::string & model_path) {
    kggml_diffusion_context_params p{};
    EXPECT_EQ(kggml_diffusion_context_default_params(&p), KGGML_STATUS_OK);
    p.model_path = model_path.c_str();
    p.n_threads = 2;
    return p;
}

/// <gtest temp dir>/<stem><pid>-<n><ext>, unique per call.
std::string temp_path(const char * stem, const char * ext);

/// Writes `bytes` verbatim to a fresh temp file and returns its path.
std::string write_file(const char * stem, const char * ext, const std::string & bytes);

/// Writes a valid GGUF (via the ggml gguf_* API of the shared runtime) holding one F32 tensor of `n_floats`
/// deterministic values named `tensor_name`, with general.architecture = "kggml-test". It is a well-formed file
/// but not a diffusion model.
std::string write_gguf(const char * stem, const std::string & tensor_name, int64_t n_floats);

/// write_gguf() cut to half its size (header intact, tensor data missing).
std::string write_truncated_gguf();

/// Reads a whole file.
std::string read_file(const std::string & path);

} // namespace kggml::test
