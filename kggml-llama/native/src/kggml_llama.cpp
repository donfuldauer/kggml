// kggml_llama.cpp - extern "C" llama.cpp wrapper: model / context lifecycle, backend selection, abort plumbing,
// decoding, embeddings, KV-cache state serialization and GBNF-constrained sampling. Every entry point that can
// fail runs behind kggml::llama::guarded().
#include "kggml_llama.h"

#include "guard.hpp"
#include "internal.hpp"

#if defined(KGGML_LLAMA_TEST_HOOKS)
#    include "kggml_llama_testing.h"
#endif

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifndef KGGML_LLAMA_VERSION_STRING
#    define KGGML_LLAMA_VERSION_STRING "0.0.0"
#endif
#ifndef KGGML_LLAMA_UPSTREAM_VERSION
#    define KGGML_LLAMA_UPSTREAM_VERSION "unknown"
#endif
#ifndef KGGML_LLAMA_UPSTREAM_COMMIT
#    define KGGML_LLAMA_UPSTREAM_COMMIT "unknown"
#endif

namespace kggml::llama {
#if defined(KGGML_LLAMA_TEST_HOOKS)
std::atomic<int64_t> g_live_objects{0};
#endif
} // namespace kggml::llama

kggml_llama_model::~kggml_llama_model() {
    if (model != nullptr) {
        llama_model_free(model);
    }
    if (holds_runtime_ref) {
        kggml_runtime_shutdown();
    }
}

kggml_llama_context::~kggml_llama_context() {
    if (batch_allocated) {
        llama_batch_free(batch);
    }
    if (ctx != nullptr) {
        llama_free(ctx);
    }
}

struct kggml_llama_sampler {
    kggml::llama::LiveCounter live;
    const llama_vocab * vocab = nullptr;
    llama_sampler * grammar = nullptr; // applied first, separately, so an empty admissible set is detectable
    llama_sampler * chain = nullptr;
    std::vector<llama_token_data> candidates;

    kggml_llama_sampler() = default;
    kggml_llama_sampler(const kggml_llama_sampler &) = delete;
    kggml_llama_sampler & operator=(const kggml_llama_sampler &) = delete;
    ~kggml_llama_sampler() {
        if (grammar != nullptr) {
            llama_sampler_free(grammar);
        }
        if (chain != nullptr) {
            llama_sampler_free(chain);
        }
    }
};

namespace kggml::llama {

int32_t resolve_threads(int32_t requested) noexcept {
    if (requested > 0) {
        return requested;
    }
    int32_t n = 0;
    if (kggml_runtime_performance_thread_count(&n) != KGGML_STATUS_OK || n <= 0) {
        n = 4;
    }
    return n;
}

bool abort_callback(void * user_data) {
    auto * c = static_cast<kggml_llama_context *>(user_data);
    const int64_t polls = c->abort_polls.fetch_add(1, std::memory_order_relaxed) + 1;
#if defined(KGGML_LLAMA_TEST_HOOKS)
    const int64_t after = c->abort_after_polls.load(std::memory_order_relaxed);
    if (after >= 0 && polls >= after) {
        c->abort.store(true, std::memory_order_release);
    }
#else
    (void) polls;
#endif
    return c->abort.load(std::memory_order_acquire);
}

kggml_status check_context(const kggml_llama_context * context) noexcept {
    if (context == nullptr || context->ctx == nullptr || context->owner == nullptr) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "context is NULL");
    }
    return KGGML_STATUS_OK;
}

kggml_status check_seq(const kggml_llama_context * context, int32_t seq_id) {
    if (seq_id < 0 || static_cast<uint32_t>(seq_id) >= context->n_seq_max) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, ("seq_id " + std::to_string(seq_id) + " out of range [0, " +
                                                    std::to_string(context->n_seq_max) + ")")
                                                       .c_str());
    }
    return KGGML_STATUS_OK;
}

namespace {

kggml_status check_model(const kggml_llama_model * model) noexcept {
    if (model == nullptr || model->model == nullptr || model->vocab == nullptr) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "model is NULL");
    }
    return KGGML_STATUS_OK;
}

kggml_status check_sampler(const kggml_llama_sampler * sampler) noexcept {
    if (sampler == nullptr || sampler->chain == nullptr) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "sampler is NULL");
    }
    return KGGML_STATUS_OK;
}

kggml_status check_token(const kggml_llama_model * model, kggml_llama_token token) {
    const int32_t n_vocab = llama_vocab_n_tokens(model->vocab);
    if (token < 0 || token >= n_vocab) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT,
                    ("token " + std::to_string(token) + " out of range [0, " + std::to_string(n_vocab) + ")").c_str());
    }
    return KGGML_STATUS_OK;
}

kggml_llama_model_params default_model_params() {
    kggml_llama_model_params p{};
    p.backend = KGGML_BACKEND_CPU;
    p.gpu_layers = -1;
    p.gpu_device = 0;
    p.use_mmap = 1;
    p.use_mlock = 0;
    p.vocab_only = 0;
    p.progress_callback = nullptr;
    p.progress_user_data = nullptr;
    return p;
}

kggml_llama_context_params default_context_params() {
    kggml_llama_context_params p{};
    p.n_ctx = 0;
    p.n_batch = 512;
    p.n_ubatch = 0;
    p.n_seq_max = 1;
    p.n_threads = 0;
    p.n_threads_batch = 0;
    p.flash_attn = -1;
    p.embeddings = 0;
    p.pooling = KGGML_LLAMA_POOLING_UNSPECIFIED;
    return p;
}

kggml_llama_sampler_params default_sampler_params() {
    kggml_llama_sampler_params p{};
    p.temperature = 0.8f;
    p.top_k = 40;
    p.top_p = 0.95f;
    p.min_p = 0.05f;
    p.repeat_penalty = 1.0f;
    p.repeat_last_n = 64;
    p.seed = LLAMA_DEFAULT_SEED;
    p.grammar = nullptr;
    p.grammar_root = nullptr;
    return p;
}

const char * backend_name(int32_t backend) {
    switch (backend) {
        case KGGML_BACKEND_CPU: return "CPU";
        case KGGML_BACKEND_METAL: return "Metal";
        case KGGML_BACKEND_VULKAN: return "Vulkan";
        case KGGML_BACKEND_COREML: return "CoreML";
        case KGGML_BACKEND_OPENCL: return "OpenCL";
        default: return "unknown";
    }
}

bool iequals_prefix(const char * s, const char * prefix) {
    if (s == nullptr) return false;
    for (; *prefix != '\0'; ++s, ++prefix) {
        if (*s == '\0' || std::tolower(static_cast<unsigned char>(*s)) != std::tolower(static_cast<unsigned char>(*prefix))) {
            return false;
        }
    }
    return true;
}

/// True if ggml device `dev` belongs to the GPU backend `kind` (Metal registers as "MTL", Vulkan as "Vulkan",
/// OpenCL as "OpenCL").
bool device_matches(ggml_backend_dev_t dev, int32_t kind) {
    const auto type = ggml_backend_dev_type(dev);
    if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * reg_name = reg != nullptr ? ggml_backend_reg_name(reg) : ggml_backend_dev_name(dev);
    switch (kind) {
        case KGGML_BACKEND_METAL: return iequals_prefix(reg_name, "MTL") || iequals_prefix(reg_name, "Metal");
        case KGGML_BACKEND_VULKAN: return iequals_prefix(reg_name, "Vulkan");
        case KGGML_BACKEND_OPENCL: return iequals_prefix(reg_name, "OpenCL");
        default: return false;
    }
}

struct ResolvedDevices {
    std::vector<ggml_backend_dev_t> devices; // NULL-terminated list handed to llama_model_params.devices
    int32_t n_gpu_layers = 0;
};

/// The single place that maps kggml_backend_kind to llama_model_params.devices / n_gpu_layers. CPU hands
/// llama.cpp an EMPTY device list, so neither the model nor its contexts ever create a GPU backend. Requires the
/// runtime to be initialized (backends registered).
kggml_status resolve_devices(const kggml_llama_model_params & p, ResolvedDevices & out) {
    out.devices.clear();
    out.n_gpu_layers = 0;
    if (p.backend == KGGML_BACKEND_CPU) {
        out.devices.push_back(nullptr);
        return KGGML_STATUS_OK;
    }
    if (p.backend == KGGML_BACKEND_COREML) {
        return fail(KGGML_STATUS_BACKEND_UNAVAILABLE, "llama.cpp has no CoreML backend (use Metal or CPU)");
    }
    int32_t available = 0;
    if (kggml_runtime_backend_available(p.backend, &available) != KGGML_STATUS_OK || available != 1) {
        return fail(KGGML_STATUS_BACKEND_UNAVAILABLE,
                    (std::string(backend_name(p.backend)) + " backend is not available in this kggml runtime / on this device")
                        .c_str());
    }
    int32_t seen = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!device_matches(dev, p.backend)) continue;
        if (seen++ == p.gpu_device) {
            out.devices.push_back(dev);
            break;
        }
    }
    if (out.devices.empty()) {
        return fail(KGGML_STATUS_BACKEND_UNAVAILABLE,
                    (std::string(backend_name(p.backend)) + " device " + std::to_string(p.gpu_device) +
                     " not found (" + std::to_string(seen) + " device(s) of that kind)")
                        .c_str());
    }
    out.devices.push_back(nullptr);
    out.n_gpu_layers = p.gpu_layers < 0 ? -1 : p.gpu_layers;
    return KGGML_STATUS_OK;
}

kggml_status validate_model_params(const kggml_llama_model_params & p) {
    if (p.backend < KGGML_BACKEND_CPU || p.backend >= KGGML_BACKEND_KIND_COUNT) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, ("unknown backend kind " + std::to_string(p.backend)).c_str());
    }
    if (p.gpu_device < 0) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "gpu_device must be >= 0");
    }
    if (p.gpu_layers < -1) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "gpu_layers must be >= -1");
    }
    return KGGML_STATUS_OK;
}

struct ProgressBridge {
    kggml_llama_progress_callback callback;
    void * user_data;
    bool cancelled = false;
};

bool progress_trampoline(float progress, void * user_data) {
    auto * bridge = static_cast<ProgressBridge *>(user_data);
    int32_t keep_going = 0;
    try {
        keep_going = bridge->callback(progress, bridge->user_data);
    } catch (...) {
        keep_going = 0; // a C callback must not throw; never let it unwind through llama.cpp
    }
    if (keep_going == 0) {
        bridge->cancelled = true;
        return false;
    }
    return true;
}

/// Copies `data` into a caller buffer with the BUFFER_TOO_SMALL size-query protocol.
template <typename T>
kggml_status copy_out(const T * data, size_t n, T * out, int32_t capacity, int32_t * out_count, const char * what) {
    if (out_count == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_count is NULL");
    if (capacity < 0) return fail(KGGML_STATUS_INVALID_ARGUMENT, "capacity must be >= 0");
    if (n > static_cast<size_t>(INT32_MAX)) return fail(KGGML_STATUS_NATIVE_FAILURE, "output too large");
    *out_count = static_cast<int32_t>(n);
    if (static_cast<int32_t>(n) > capacity || (n > 0 && out == nullptr)) {
        return fail(KGGML_STATUS_BUFFER_TOO_SMALL,
                    (std::string(what) + " buffer too small, need " + std::to_string(n) + " entries").c_str());
    }
    if (n > 0) {
        std::copy(data, data + n, out);
    }
    return KGGML_STATUS_OK;
}

/// False if every candidate logit is -inf (the grammar admits nothing).
bool any_admissible(const llama_token_data_array & arr) {
    for (size_t i = 0; i < arr.size; ++i) {
        if (std::isfinite(arr.data[i].logit)) return true;
    }
    return false;
}

} // namespace
} // namespace kggml::llama

using kggml::llama::fail;
using kggml::llama::guarded;

extern "C" {

// ---------------------------------------------------------------------------------------------------------
// Versions
// ---------------------------------------------------------------------------------------------------------

KGGML_LLAMA_API const char * kggml_llama_version(void) {
    return KGGML_LLAMA_VERSION_STRING;
}

KGGML_LLAMA_API const char * kggml_llama_upstream_version(void) {
    return KGGML_LLAMA_UPSTREAM_VERSION;
}

KGGML_LLAMA_API const char * kggml_llama_upstream_commit(void) {
    return KGGML_LLAMA_UPSTREAM_COMMIT;
}

KGGML_LLAMA_API const char * kggml_llama_system_info(void) {
    try {
        const char * info = llama_print_system_info();
        return info != nullptr ? info : "";
    } catch (...) {
        return "";
    }
}

// ---------------------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------------------

KGGML_LLAMA_API kggml_status kggml_llama_model_default_params(kggml_llama_model_params * params) {
    return guarded(__func__, [&] {
        if (params == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "params is NULL");
        *params = kggml::llama::default_model_params();
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_model_load(const char * model_path, const kggml_llama_model_params * params,
                                                   kggml_llama_model ** out_model) {
    return guarded(__func__, [&] {
        if (out_model == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_model is NULL");
        *out_model = nullptr;
        if (model_path == nullptr || model_path[0] == '\0') {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "model_path is NULL or empty");
        }
        const kggml_llama_model_params p = params != nullptr ? *params : kggml::llama::default_model_params();
        if (const auto s = kggml::llama::validate_model_params(p); s != KGGML_STATUS_OK) return s;

        auto model = std::make_unique<kggml_llama_model>();
        const kggml_status rt = kggml_runtime_init();
        if (rt != KGGML_STATUS_OK) {
            return fail(rt, (std::string("kggml_runtime_init failed: ") + kggml_runtime_last_error()).c_str());
        }
        model->holds_runtime_ref = true;

        kggml::llama::ResolvedDevices resolved;
        if (const auto s = kggml::llama::resolve_devices(p, resolved); s != KGGML_STATUS_OK) return s;

        {
            // Distinguish a missing file from a corrupt one (llama.cpp reports both as NULL).
            std::FILE * f = std::fopen(model_path, "rb");
            if (f == nullptr) {
                return fail(KGGML_STATUS_MODEL_LOAD_FAILED,
                            (std::string("cannot open model file '") + model_path + "'").c_str());
            }
            std::fclose(f);
        }

        llama_model_params mparams = llama_model_default_params();
        mparams.devices = resolved.devices.data(); // decided before the model (and any backend buffer) exists
        mparams.n_gpu_layers = resolved.n_gpu_layers;
        mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
        // Index into `devices`; -1 (CPU) makes llama.cpp drop every device, so nothing is ever offloaded.
        mparams.main_gpu = p.backend == KGGML_BACKEND_CPU ? -1 : 0;
        mparams.use_mmap = p.use_mmap != 0;
        mparams.use_mlock = p.use_mlock != 0;
        mparams.vocab_only = p.vocab_only != 0;

        kggml::llama::ProgressBridge bridge{p.progress_callback, p.progress_user_data};
        if (p.progress_callback != nullptr) {
            mparams.progress_callback = kggml::llama::progress_trampoline;
            mparams.progress_callback_user_data = &bridge;
        }

        model->model = llama_model_load_from_file(model_path, mparams);
        if (model->model == nullptr) {
            if (bridge.cancelled) {
                return fail(KGGML_STATUS_CANCELLED, "model load cancelled by the progress callback");
            }
            return fail(KGGML_STATUS_MODEL_LOAD_FAILED,
                        (std::string("llama.cpp could not load model '") + model_path + "' (invalid or unsupported GGUF)")
                            .c_str());
        }
        model->vocab = llama_model_get_vocab(model->model);
        if (model->vocab == nullptr) {
            return fail(KGGML_STATUS_MODEL_LOAD_FAILED, "model has no vocabulary");
        }
        model->backend = p.backend;
        model->uses_gpu = p.backend != KGGML_BACKEND_CPU;
        *out_model = model.release();
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API void kggml_llama_model_free(kggml_llama_model * model) {
    try {
        delete model;
    } catch (...) {
        // llama_model_free never throws in practice; never let anything escape.
    }
}

KGGML_LLAMA_API kggml_status kggml_llama_model_get_info(const kggml_llama_model * model, kggml_llama_model_info * out_info) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_model(model); s != KGGML_STATUS_OK) return s;
        if (out_info == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_info is NULL");
        kggml_llama_model_info info{};
        info.n_vocab = llama_vocab_n_tokens(model->vocab);
        info.n_ctx_train = llama_model_n_ctx_train(model->model);
        info.n_embd = llama_model_n_embd_out(model->model);
        info.n_layer = llama_model_n_layer(model->model);
        info.has_encoder = llama_model_has_encoder(model->model) ? 1 : 0;
        info.has_decoder = llama_model_has_decoder(model->model) ? 1 : 0;
        info.backend = model->backend;
        info.uses_gpu = model->uses_gpu ? 1 : 0;
        info.size_bytes = llama_model_size(model->model);
        info.n_params = llama_model_n_params(model->model);
        info.bos = llama_vocab_bos(model->vocab);
        info.eos = llama_vocab_eos(model->vocab);
        *out_info = info;
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_model_desc(const kggml_llama_model * model, char * buffer, size_t capacity,
                                                   size_t * out_length) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_model(model); s != KGGML_STATUS_OK) return s;
        if (out_length == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_length is NULL");
        char tmp[256];
        const int32_t n = llama_model_desc(model->model, tmp, sizeof(tmp));
        if (n < 0) return fail(KGGML_STATUS_NATIVE_FAILURE, "llama_model_desc failed");
        const size_t len = std::min(static_cast<size_t>(n), sizeof(tmp) - 1);
        *out_length = len;
        if (buffer != nullptr && capacity > 0) {
            const size_t copied = std::min(len, capacity - 1);
            std::memcpy(buffer, tmp, copied);
            buffer[copied] = '\0';
        }
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API int32_t kggml_llama_token_is_eog(const kggml_llama_model * model, kggml_llama_token token) {
    try {
        if (model == nullptr || model->vocab == nullptr) return 0;
        if (token < 0 || token >= llama_vocab_n_tokens(model->vocab)) return 0;
        return llama_vocab_is_eog(model->vocab, token) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

KGGML_LLAMA_API kggml_status kggml_llama_tokenize(const kggml_llama_model * model, const char * text, size_t text_len,
                                                 int32_t add_special, int32_t parse_special,
                                                 kggml_llama_token * out_tokens, int32_t capacity, int32_t * out_count) {
    return guarded(__func__, [&]() -> kggml_status {
        if (const auto s = kggml::llama::check_model(model); s != KGGML_STATUS_OK) return s;
        if (out_count == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_count is NULL");
        if (capacity < 0) return fail(KGGML_STATUS_INVALID_ARGUMENT, "capacity must be >= 0");
        if (text == nullptr && text_len > 0) return fail(KGGML_STATUS_INVALID_ARGUMENT, "text is NULL");
        if (text_len > static_cast<size_t>(INT32_MAX)) return fail(KGGML_STATUS_INVALID_ARGUMENT, "text too long");
        const char * t = text != nullptr ? text : "";
        // llama_tokenize returns -(required) when the buffer is too small.
        const int32_t n = llama_tokenize(model->vocab, t, static_cast<int32_t>(text_len),
                                         out_tokens, out_tokens != nullptr ? capacity : 0,
                                         add_special != 0, parse_special != 0);
        if (n == INT32_MIN) return fail(KGGML_STATUS_NATIVE_FAILURE, "tokenization overflowed int32");
        if (n < 0) {
            *out_count = -n;
            return fail(KGGML_STATUS_BUFFER_TOO_SMALL,
                        ("token buffer too small, need " + std::to_string(-n) + " entries").c_str());
        }
        *out_count = n;
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_token_to_piece(const kggml_llama_model * model, kggml_llama_token token,
                                                       int32_t special, char * buffer, int32_t capacity,
                                                       int32_t * out_length) {
    return guarded(__func__, [&]() -> kggml_status {
        if (const auto s = kggml::llama::check_model(model); s != KGGML_STATUS_OK) return s;
        if (out_length == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_length is NULL");
        if (capacity < 0) return fail(KGGML_STATUS_INVALID_ARGUMENT, "capacity must be >= 0");
        if (const auto s = kggml::llama::check_token(model, token); s != KGGML_STATUS_OK) return s;
        const int32_t n = llama_token_to_piece(model->vocab, token, buffer, buffer != nullptr ? capacity : 0, 0,
                                               special != 0);
        if (n < 0) {
            *out_length = -n;
            return fail(KGGML_STATUS_BUFFER_TOO_SMALL,
                        ("piece buffer too small, need " + std::to_string(-n) + " bytes").c_str());
        }
        *out_length = n;
        return KGGML_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------------------------------------

KGGML_LLAMA_API kggml_status kggml_llama_context_default_params(kggml_llama_context_params * params) {
    return guarded(__func__, [&] {
        if (params == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "params is NULL");
        *params = kggml::llama::default_context_params();
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_context_create(kggml_llama_model * model,
                                                       const kggml_llama_context_params * params,
                                                       kggml_llama_context ** out_context) {
    return guarded(__func__, [&] {
        if (out_context == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_context is NULL");
        *out_context = nullptr;
        if (const auto s = kggml::llama::check_model(model); s != KGGML_STATUS_OK) return s;
        const kggml_llama_context_params p = params != nullptr ? *params : kggml::llama::default_context_params();
        if (p.n_batch == 0) return fail(KGGML_STATUS_INVALID_ARGUMENT, "n_batch must be > 0");
        if (p.n_seq_max == 0 || p.n_seq_max > llama_max_parallel_sequences()) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT,
                        ("n_seq_max must be in [1, " + std::to_string(llama_max_parallel_sequences()) + "]").c_str());
        }
        if (p.n_threads < 0 || p.n_threads_batch < 0) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "n_threads and n_threads_batch must be >= 0");
        }
        if (p.flash_attn < -1 || p.flash_attn > 1) return fail(KGGML_STATUS_INVALID_ARGUMENT, "flash_attn must be -1, 0 or 1");
        if (p.pooling < KGGML_LLAMA_POOLING_UNSPECIFIED || p.pooling > KGGML_LLAMA_POOLING_LAST) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, ("unknown pooling " + std::to_string(p.pooling)).c_str());
        }
        if (llama_model_n_layer(model->model) == 0 || llama_model_n_embd(model->model) == 0) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "model was loaded with vocab_only = 1 (no weights)");
        }

        auto context = std::make_unique<kggml_llama_context>();
        context->owner = model;
        context->n_seq_max = p.n_seq_max;

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = p.n_ctx;
        cparams.n_batch = p.n_batch;
        cparams.n_ubatch = p.n_ubatch == 0 ? p.n_batch : p.n_ubatch;
        cparams.n_seq_max = p.n_seq_max;
        cparams.n_threads = kggml::llama::resolve_threads(p.n_threads);
        cparams.n_threads_batch = p.n_threads_batch > 0 ? p.n_threads_batch : cparams.n_threads;
        cparams.flash_attn_type = static_cast<llama_flash_attn_type>(p.flash_attn);
        cparams.embeddings = p.embeddings != 0;
        cparams.pooling_type = static_cast<enum llama_pooling_type>(p.pooling);
        cparams.no_perf = true;
        cparams.abort_callback = kggml::llama::abort_callback;
        cparams.abort_callback_data = context.get();
        if (!model->uses_gpu) {
            // CPU model: keep the KV cache and every op on the CPU, never touch a GPU backend.
            cparams.offload_kqv = false;
            cparams.op_offload = false;
        }

        context->ctx = llama_init_from_model(model->model, cparams);
        if (context->ctx == nullptr) {
            return fail(KGGML_STATUS_NATIVE_FAILURE,
                        "llama_init_from_model failed (invalid context parameters or KV cache / compute buffer allocation)");
        }
        // Registered again explicitly so the abort hook does not depend on llama_context_params plumbing.
        llama_set_abort_callback(context->ctx, kggml::llama::abort_callback, context.get());

        context->batch = llama_batch_init(static_cast<int32_t>(llama_n_batch(context->ctx)), 0, 1);
        context->batch_allocated = true;

        *out_context = context.release();
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API void kggml_llama_context_free(kggml_llama_context * context) {
    try {
        delete context;
    } catch (...) {
    }
}

KGGML_LLAMA_API kggml_status kggml_llama_context_n_ctx(const kggml_llama_context * context, uint32_t * out_n_ctx) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_n_ctx == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_n_ctx is NULL");
        *out_n_ctx = llama_n_ctx(context->ctx);
        return KGGML_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------------------
// Abort
// ---------------------------------------------------------------------------------------------------------

KGGML_LLAMA_API void kggml_llama_request_abort(kggml_llama_context * context) {
    if (context != nullptr) {
        context->abort.store(true, std::memory_order_release);
    }
}

KGGML_LLAMA_API void kggml_llama_clear_abort(kggml_llama_context * context) {
    if (context != nullptr) {
        context->abort.store(false, std::memory_order_release);
    }
}

KGGML_LLAMA_API int32_t kggml_llama_abort_requested(const kggml_llama_context * context) {
    return context != nullptr && context->abort.load(std::memory_order_acquire) ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------------------------------------

KGGML_LLAMA_API kggml_status kggml_llama_decode(kggml_llama_context * context, const kggml_llama_token * tokens,
                                               int32_t n_tokens, int32_t pos0, int32_t seq_id, int32_t logits_last) {
    return guarded(__func__, [&]() -> kggml_status {
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (tokens == nullptr || n_tokens <= 0) return fail(KGGML_STATUS_INVALID_ARGUMENT, "tokens is NULL or n_tokens <= 0");
        if (pos0 < 0 || n_tokens > INT32_MAX - pos0) return fail(KGGML_STATUS_INVALID_ARGUMENT, "invalid pos0");
        if (const auto s = kggml::llama::check_seq(context, seq_id); s != KGGML_STATUS_OK) return s;
        for (int32_t i = 0; i < n_tokens; ++i) {
            if (const auto s = kggml::llama::check_token(context->owner, tokens[i]); s != KGGML_STATUS_OK) return s;
        }

        context->logits_valid = false;
        const int32_t n_batch = static_cast<int32_t>(llama_n_batch(context->ctx));
        llama_batch & batch = context->batch;
        for (int32_t start = 0; start < n_tokens; start += n_batch) {
            if (context->abort.load(std::memory_order_acquire)) {
                return fail(KGGML_STATUS_CANCELLED, "decode cancelled");
            }
            const int32_t n = std::min(n_batch, n_tokens - start);
            batch.n_tokens = n;
            for (int32_t i = 0; i < n; ++i) {
                batch.token[i] = tokens[start + i];
                batch.pos[i] = pos0 + start + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq_id;
                batch.logits[i] = (logits_last != 0 && start + i == n_tokens - 1) ? 1 : 0;
            }
            const int32_t ret = llama_decode(context->ctx, batch);
            if (ret == 0) continue;
            if (ret == 2 || context->abort.load(std::memory_order_acquire)) {
                return fail(KGGML_STATUS_CANCELLED, "decode cancelled");
            }
            if (ret == 1) {
                return fail(KGGML_STATUS_BUFFER_TOO_SMALL,
                            "no KV cache slot for the batch (context full: remove tokens or increase n_ctx)");
            }
            if (ret == -1) {
                return fail(KGGML_STATUS_INVALID_ARGUMENT,
                            "llama_decode rejected the batch (invalid positions / sequence, see llama.cpp log)");
            }
            return fail(KGGML_STATUS_NATIVE_FAILURE, ("llama_decode failed with " + std::to_string(ret)).c_str());
        }
        context->logits_valid = logits_last != 0;
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_get_logits(kggml_llama_context * context, const float ** out_logits,
                                                   int32_t * out_n_vocab) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_logits == nullptr || out_n_vocab == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_logits / out_n_vocab is NULL");
        }
        *out_logits = nullptr;
        *out_n_vocab = 0;
        if (!context->logits_valid) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "no logits: the last decode did not request logits_last");
        }
        const float * logits = llama_get_logits_ith(context->ctx, -1);
        if (logits == nullptr) return fail(KGGML_STATUS_NATIVE_FAILURE, "llama_get_logits_ith returned NULL");
        *out_logits = logits;
        *out_n_vocab = llama_vocab_n_tokens(context->owner->vocab);
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_memory_seq_rm(kggml_llama_context * context, int32_t seq_id, int32_t p0,
                                                      int32_t p1) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (seq_id >= 0) {
            if (const auto s = kggml::llama::check_seq(context, seq_id); s != KGGML_STATUS_OK) return s;
        }
        if (p0 < 0) return fail(KGGML_STATUS_INVALID_ARGUMENT, "p0 must be >= 0");
        if (!llama_memory_seq_rm(llama_get_memory(context->ctx), seq_id < 0 ? -1 : seq_id, p0, p1 < 0 ? -1 : p1)) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "partial sequence removal is not supported by this model's memory");
        }
        context->logits_valid = false;
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_memory_clear(kggml_llama_context * context) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        llama_memory_clear(llama_get_memory(context->ctx), true);
        context->logits_valid = false;
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_memory_seq_pos_max(kggml_llama_context * context, int32_t seq_id,
                                                           int32_t * out_pos) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (const auto s = kggml::llama::check_seq(context, seq_id); s != KGGML_STATUS_OK) return s;
        if (out_pos == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_pos is NULL");
        *out_pos = llama_memory_seq_pos_max(llama_get_memory(context->ctx), seq_id);
        return KGGML_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------------------------------------

KGGML_LLAMA_API kggml_status kggml_llama_get_embeddings(kggml_llama_context * context, int32_t seq_id,
                                                       int32_t normalize, float * out, int32_t capacity,
                                                       int32_t * out_count) {
    return guarded(__func__, [&]() -> kggml_status {
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (const auto s = kggml::llama::check_seq(context, seq_id); s != KGGML_STATUS_OK) return s;
        if (!context->logits_valid) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "no embeddings: the last decode did not request logits_last");
        }
        const float * emb = nullptr;
        if (llama_pooling_type(context->ctx) == LLAMA_POOLING_TYPE_NONE) {
            emb = llama_get_embeddings_ith(context->ctx, -1);
        } else {
            emb = llama_get_embeddings_seq(context->ctx, seq_id);
        }
        if (emb == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT,
                        "no embeddings available (create the context with embeddings = 1 and decode the sequence)");
        }
        const auto n = static_cast<size_t>(llama_model_n_embd_out(context->owner->model));
        if (normalize == 0) {
            return kggml::llama::copy_out(emb, n, out, capacity, out_count, "embedding");
        }
        double norm = 0.0;
        for (size_t i = 0; i < n; ++i) norm += static_cast<double>(emb[i]) * emb[i];
        norm = std::sqrt(norm);
        std::vector<float> tmp(emb, emb + n);
        if (norm > 0.0) {
            for (float & v : tmp) v = static_cast<float>(v / norm);
        }
        return kggml::llama::copy_out(tmp.data(), n, out, capacity, out_count, "embedding");
    });
}

// ---------------------------------------------------------------------------------------------------------
// State serialization
// ---------------------------------------------------------------------------------------------------------

KGGML_LLAMA_API kggml_status kggml_llama_state_size(kggml_llama_context * context, int32_t seq_id, size_t * out_size) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_size == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_size is NULL");
        if (seq_id < 0) {
            *out_size = llama_state_get_size(context->ctx);
        } else {
            if (const auto s = kggml::llama::check_seq(context, seq_id); s != KGGML_STATUS_OK) return s;
            *out_size = llama_state_seq_get_size(context->ctx, seq_id);
        }
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_state_save(kggml_llama_context * context, int32_t seq_id, uint8_t * buffer,
                                                   size_t capacity, size_t * out_written) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_written == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_written is NULL");
        if (seq_id >= 0) {
            if (const auto s = kggml::llama::check_seq(context, seq_id); s != KGGML_STATUS_OK) return s;
        }
        const size_t needed = seq_id < 0 ? llama_state_get_size(context->ctx)
                                         : llama_state_seq_get_size(context->ctx, seq_id);
        *out_written = needed;
        if (buffer == nullptr || capacity < needed) {
            return fail(KGGML_STATUS_BUFFER_TOO_SMALL,
                        ("state buffer too small, need " + std::to_string(needed) + " bytes").c_str());
        }
        const size_t written = seq_id < 0 ? llama_state_get_data(context->ctx, buffer, capacity)
                                          : llama_state_seq_get_data(context->ctx, buffer, capacity, seq_id);
        if (written == 0 && needed != 0) return fail(KGGML_STATUS_NATIVE_FAILURE, "state serialization failed");
        *out_written = written;
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_state_load(kggml_llama_context * context, int32_t seq_id,
                                                   const uint8_t * data, size_t size) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (data == nullptr || size == 0) return fail(KGGML_STATUS_INVALID_ARGUMENT, "data is NULL or empty");
        context->logits_valid = false;
        size_t read = 0;
        if (seq_id < 0) {
            read = llama_state_set_data(context->ctx, data, size);
        } else {
            if (const auto s = kggml::llama::check_seq(context, seq_id); s != KGGML_STATUS_OK) return s;
            read = llama_state_seq_set_data(context->ctx, data, size, seq_id);
        }
        if (read == 0) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT,
                        "llama.cpp rejected the state data (corrupt, truncated or from an incompatible model / context)");
        }
        return KGGML_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------------------------------------

KGGML_LLAMA_API kggml_status kggml_llama_sampler_default_params(kggml_llama_sampler_params * params) {
    return guarded(__func__, [&] {
        if (params == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "params is NULL");
        *params = kggml::llama::default_sampler_params();
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_sampler_create(const kggml_llama_model * model,
                                                       const kggml_llama_sampler_params * params,
                                                       kggml_llama_sampler ** out_sampler) {
    return guarded(__func__, [&] {
        if (out_sampler == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_sampler is NULL");
        *out_sampler = nullptr;
        if (const auto s = kggml::llama::check_model(model); s != KGGML_STATUS_OK) return s;
        const kggml_llama_sampler_params p = params != nullptr ? *params : kggml::llama::default_sampler_params();
        if (!std::isfinite(p.temperature) || !std::isfinite(p.top_p) || !std::isfinite(p.min_p) ||
            !std::isfinite(p.repeat_penalty) || p.repeat_penalty <= 0.0f || p.repeat_last_n < -1) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT,
                        "temperature / top_p / min_p must be finite, repeat_penalty > 0, repeat_last_n >= -1");
        }

        auto sampler = std::make_unique<kggml_llama_sampler>();
        sampler->vocab = model->vocab;
        if (p.grammar != nullptr && p.grammar[0] != '\0') {
            const char * root = p.grammar_root != nullptr && p.grammar_root[0] != '\0' ? p.grammar_root : "root";
            sampler->grammar = llama_sampler_init_grammar(model->vocab, p.grammar, root);
            if (sampler->grammar == nullptr) {
                return fail(KGGML_STATUS_INVALID_ARGUMENT,
                            (std::string("invalid GBNF grammar (parse error or undefined root symbol '") + root + "')")
                                .c_str());
            }
        }

        llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
        chain_params.no_perf = true;
        sampler->chain = llama_sampler_chain_init(chain_params);
        if (sampler->chain == nullptr) return fail(KGGML_STATUS_NATIVE_FAILURE, "llama_sampler_chain_init failed");
        if (p.repeat_penalty != 1.0f && p.repeat_last_n != 0) {
            llama_sampler_chain_add(sampler->chain, llama_sampler_init_penalties(p.repeat_last_n, p.repeat_penalty, 0.0f, 0.0f));
        }
        if (p.temperature <= 0.0f) {
            llama_sampler_chain_add(sampler->chain, llama_sampler_init_greedy());
        } else {
            if (p.top_k > 0) llama_sampler_chain_add(sampler->chain, llama_sampler_init_top_k(p.top_k));
            if (p.top_p < 1.0f) llama_sampler_chain_add(sampler->chain, llama_sampler_init_top_p(p.top_p, 1));
            if (p.min_p > 0.0f) llama_sampler_chain_add(sampler->chain, llama_sampler_init_min_p(p.min_p, 1));
            llama_sampler_chain_add(sampler->chain, llama_sampler_init_temp(p.temperature));
            llama_sampler_chain_add(sampler->chain, llama_sampler_init_dist(p.seed));
        }
        sampler->candidates.reserve(static_cast<size_t>(llama_vocab_n_tokens(model->vocab)));
        *out_sampler = sampler.release();
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API void kggml_llama_sampler_free(kggml_llama_sampler * sampler) {
    try {
        delete sampler;
    } catch (...) {
    }
}

KGGML_LLAMA_API kggml_status kggml_llama_sampler_sample(kggml_llama_sampler * sampler, kggml_llama_context * context,
                                                       int32_t idx, kggml_llama_token * out_token) {
    return guarded(__func__, [&]() -> kggml_status {
        if (const auto s = kggml::llama::check_sampler(sampler); s != KGGML_STATUS_OK) return s;
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_token == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_token is NULL");
        if (sampler->vocab != context->owner->vocab) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "sampler and context belong to different models");
        }
        if (!context->logits_valid) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "no logits: the last decode did not request logits_last");
        }
        if (idx != -1) return fail(KGGML_STATUS_INVALID_ARGUMENT, "only idx = -1 (last output) is supported");
        const float * logits = llama_get_logits_ith(context->ctx, idx);
        if (logits == nullptr) return fail(KGGML_STATUS_NATIVE_FAILURE, "llama_get_logits_ith returned NULL");

        const int32_t n_vocab = llama_vocab_n_tokens(sampler->vocab);
        auto & cand = sampler->candidates;
        cand.resize(static_cast<size_t>(n_vocab));
        for (int32_t t = 0; t < n_vocab; ++t) {
            cand[static_cast<size_t>(t)] = llama_token_data{t, logits[t], 0.0f};
        }
        llama_token_data_array arr{cand.data(), cand.size(), -1, false};

        // The grammar runs first and on its own, so "no admissible token" is reported instead of reaching
        // llama.cpp's GGML_ASSERT on an empty selection.
        if (sampler->grammar != nullptr) {
            llama_sampler_apply(sampler->grammar, &arr);
            if (!kggml::llama::any_admissible(arr)) {
                return fail(KGGML_STATUS_NATIVE_FAILURE, "the grammar admits no token in the current state");
            }
        }
        llama_sampler_apply(sampler->chain, &arr);
        if (arr.selected < 0 || static_cast<size_t>(arr.selected) >= arr.size) {
            return fail(KGGML_STATUS_NATIVE_FAILURE, "sampler chain selected no token");
        }
        const llama_token token = arr.data[arr.selected].id;
        if (sampler->grammar != nullptr) {
            llama_sampler_accept(sampler->grammar, token);
        }
        llama_sampler_accept(sampler->chain, token);
        *out_token = token;
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_sampler_accept(kggml_llama_sampler * sampler, kggml_llama_token token,
                                                       int32_t apply_grammar) {
    return guarded(__func__, [&]() -> kggml_status {
        if (const auto s = kggml::llama::check_sampler(sampler); s != KGGML_STATUS_OK) return s;
        const int32_t n_vocab = llama_vocab_n_tokens(sampler->vocab);
        if (token < 0 || token >= n_vocab) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT,
                        ("token " + std::to_string(token) + " out of range [0, " + std::to_string(n_vocab) + ")").c_str());
        }
        if (apply_grammar != 0 && sampler->grammar != nullptr) {
            // Probe first: accepting a token the grammar rejects would throw (or abort for EOG) inside llama.cpp.
            llama_token_data probe{token, 0.0f, 0.0f};
            llama_token_data_array arr{&probe, 1, -1, false};
            llama_sampler_apply(sampler->grammar, &arr);
            if (!std::isfinite(probe.logit)) {
                return fail(KGGML_STATUS_INVALID_ARGUMENT,
                            ("token " + std::to_string(token) + " is not admitted by the grammar").c_str());
            }
            llama_sampler_accept(sampler->grammar, token);
        }
        llama_sampler_accept(sampler->chain, token);
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_sampler_reset(kggml_llama_sampler * sampler) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_sampler(sampler); s != KGGML_STATUS_OK) return s;
        if (sampler->grammar != nullptr) llama_sampler_reset(sampler->grammar);
        llama_sampler_reset(sampler->chain);
        return KGGML_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------------------
// Test hooks
// ---------------------------------------------------------------------------------------------------------

#if defined(KGGML_LLAMA_TEST_HOOKS)

KGGML_LLAMA_API kggml_status kggml_llama_test_poll_abort(kggml_llama_context * context, int32_t * out_abort) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_abort == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_abort is NULL");
        *out_abort = kggml::llama::abort_callback(context) ? 1 : 0;
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_test_abort_after_polls(kggml_llama_context * context, int32_t n) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        context->abort_after_polls.store(n < 0 ? -1 : context->abort_polls.load() + n, std::memory_order_relaxed);
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_test_abort_poll_count(const kggml_llama_context * context,
                                                              int64_t * out_polls) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_polls == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_polls is NULL");
        *out_polls = context->abort_polls.load(std::memory_order_relaxed);
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API int64_t kggml_llama_test_live_objects(void) {
    return kggml::llama::g_live_objects.load(std::memory_order_relaxed);
}

KGGML_LLAMA_API kggml_status kggml_llama_test_resolve_devices(const kggml_llama_model_params * params,
                                                             int32_t * out_n_devices, int32_t * out_n_gpu_layers) {
    return guarded(__func__, [&] {
        if (params == nullptr || out_n_devices == nullptr || out_n_gpu_layers == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "NULL argument");
        }
        if (const auto s = kggml::llama::validate_model_params(*params); s != KGGML_STATUS_OK) return s;
        const kggml_status rt = kggml_runtime_init();
        if (rt != KGGML_STATUS_OK) return fail(rt, "kggml_runtime_init failed");
        kggml::llama::ResolvedDevices resolved;
        const kggml_status s = kggml::llama::resolve_devices(*params, resolved);
        kggml_runtime_shutdown();
        if (s != KGGML_STATUS_OK) return s;
        *out_n_devices = static_cast<int32_t>(resolved.devices.size()) - 1; // minus the NULL terminator
        *out_n_gpu_layers = resolved.n_gpu_layers;
        return KGGML_STATUS_OK;
    });
}

#endif // KGGML_LLAMA_TEST_HOOKS

} // extern "C"
