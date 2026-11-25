// fixtures.cpp - see fixtures.hpp.
#include "fixtures.hpp"

#include "ggml.h"
#include "gguf.h"

#include <atomic>
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

std::string temp_path(const char * stem, const char * ext) {
    static std::atomic<int> counter{0};
    return ::testing::TempDir() + stem + std::to_string(KGGML_GETPID()) + "-" + std::to_string(counter.fetch_add(1)) +
           ext;
}

std::string write_file(const char * stem, const char * ext, const std::string & bytes) {
    const std::string path = temp_path(stem, ext);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("cannot write " + path);
    return path;
}

std::string write_gguf(const char * stem, const std::string & tensor_name, int64_t n_floats) {
    const std::string path = temp_path(stem, ".gguf");

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4 + static_cast<size_t>(n_floats) * sizeof(float) + 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = false;
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) throw std::runtime_error("ggml_init failed");
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_floats);
    ggml_set_name(t, tensor_name.c_str());
    auto * data = static_cast<float *>(t->data);
    uint32_t s = 0x9E3779B9u;
    for (int64_t i = 0; i < n_floats; ++i) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        data[i] = static_cast<float>(s & 0xFFFF) / 65535.0f - 0.5f;
    }

    gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.architecture", "kggml-test");
    gguf_set_val_str(gguf, "general.name", "kggml-diffusion-not-a-model");
    gguf_add_tensor(gguf, t);
    const bool ok = gguf_write_to_file(gguf, path.c_str(), false);
    gguf_free(gguf);
    ggml_free(ctx);
    if (!ok) throw std::runtime_error("gguf_write_to_file failed for " + path);
    return path;
}

std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string write_truncated_gguf() {
    const std::string full = read_file(write_gguf("kggml-diffusion-full-", "model.diffusion_model.input_blocks.0.0.weight", 4096));
    return write_file("kggml-diffusion-truncated-", ".gguf", full.substr(0, full.size() / 2));
}

} // namespace kggml::test
