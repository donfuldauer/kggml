// kggml_whisper.cpp - extern "C" whisper.cpp wrapper: context lifecycle, backend toggle, abort plumbing and
// transcription. Every entry point that can fail runs behind kggml::whisper::guarded().
#include "kggml_whisper.h"

#include "coreml_gate.hpp"
#include "guard.hpp"
#include "resampler.hpp"

#if defined(KGGML_WHISPER_TEST_HOOKS)
#    include "kggml_whisper_testing.h"
#endif

#include "whisper.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <climits>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#ifndef KGGML_WHISPER_VERSION_STRING
#    define KGGML_WHISPER_VERSION_STRING "0.0.0"
#endif
#ifndef KGGML_WHISPER_UPSTREAM_VERSION
#    define KGGML_WHISPER_UPSTREAM_VERSION "unknown"
#endif
#ifndef KGGML_WHISPER_UPSTREAM_COMMIT
#    define KGGML_WHISPER_UPSTREAM_COMMIT "unknown"
#endif

struct kggml_whisper_context {
    whisper_context * ctx = nullptr;
    whisper_state * state = nullptr;
    bool holds_runtime_ref = false;

    int32_t backend = KGGML_BACKEND_CPU; // effective backend
    bool use_gpu = false;

    std::atomic<bool> abort{false};
    std::atomic<int64_t> abort_polls{0};
#if defined(KGGML_WHISPER_TEST_HOOKS)
    std::atomic<int64_t> abort_after_polls{-1};
#endif

    std::vector<float> pad_scratch;
    int32_t n_segments = 0;

    ~kggml_whisper_context() {
        if (state != nullptr) {
            whisper_free_state(state);
        }
        if (ctx != nullptr) {
            whisper_free(ctx);
        }
        if (holds_runtime_ref) {
            kggml_runtime_shutdown();
        }
    }
};

#if defined(KGGML_WHISPER_TEST_HOOKS)
namespace {
std::atomic<int64_t> g_live_states{0};
} // namespace
#endif

struct kggml_whisper_state {
    const kggml_whisper_context * owner = nullptr; // identity check only, never dereferenced
    whisper_state * state = nullptr;
    whisper_token eot = 0;  // first non-text token id of the owner's vocabulary
    int32_t n_vocab = 0;
    int32_t n_segments = 0; // segments visible through the accessors
    std::vector<float> pad_scratch;
#if defined(KGGML_WHISPER_TEST_HOOKS)
    std::vector<int32_t> last_prompt;
#endif

    kggml_whisper_state() {
#if defined(KGGML_WHISPER_TEST_HOOKS)
        g_live_states.fetch_add(1, std::memory_order_relaxed);
#endif
    }
    kggml_whisper_state(const kggml_whisper_state &) = delete;
    kggml_whisper_state & operator=(const kggml_whisper_state &) = delete;

    ~kggml_whisper_state() {
        if (state != nullptr) {
            whisper_free_state(state);
        }
#if defined(KGGML_WHISPER_TEST_HOOKS)
        g_live_states.fetch_sub(1, std::memory_order_relaxed);
#endif
    }
};

namespace kggml::whisper {
namespace {

/// whisper_full_params.abort_callback (ggml_abort_callback). Polled from inside ggml graph compute, possibly
/// on ggml worker threads, so it only touches atomics.
bool abort_callback(void * user_data) {
    auto * c = static_cast<kggml_whisper_context *>(user_data);
    const int64_t polls = c->abort_polls.fetch_add(1, std::memory_order_relaxed) + 1;
#if defined(KGGML_WHISPER_TEST_HOOKS)
    const int64_t after = c->abort_after_polls.load(std::memory_order_relaxed);
    if (after >= 0 && polls >= after) {
        c->abort.store(true, std::memory_order_release);
    }
#else
    (void) polls;
#endif
    return c->abort.load(std::memory_order_acquire);
}

/// whisper_full_params.encoder_begin_callback: returning false stops before the next 30 s window is encoded.
bool encoder_begin_callback(whisper_context *, whisper_state *, void * user_data) {
    return !abort_callback(user_data);
}

bool valid_backend(int32_t backend) {
    return backend >= KGGML_BACKEND_CPU && backend < KGGML_BACKEND_KIND_COUNT;
}

struct ResolvedBackend {
    bool use_gpu;
    bool coreml_allowed;
};

/// The single place that maps kggml_backend_kind to whisper_context_params.use_gpu. CPU_ONLY never creates a
/// GPU backend and never loads the CoreML encoder.
ResolvedBackend resolve_backend(int32_t backend) {
    switch (backend) {
        case KGGML_BACKEND_METAL:
        case KGGML_BACKEND_VULKAN:
        case KGGML_BACKEND_OPENCL: return {true, false};
        case KGGML_BACKEND_COREML: return {true, true};
        case KGGML_BACKEND_CPU:
        default: return {false, false};
    }
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

/// True if ggml GPU device `dev` belongs to backend `kind` (Metal registers as "MTL", Vulkan as "Vulkan", OpenCL as
/// "OpenCL").
bool gpu_device_matches(ggml_backend_dev_t dev, int32_t kind) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * reg_name = reg != nullptr ? ggml_backend_reg_name(reg) : ggml_backend_dev_name(dev);
    switch (kind) {
        case KGGML_BACKEND_METAL:
        case KGGML_BACKEND_COREML: return iequals_prefix(reg_name, "MTL") || iequals_prefix(reg_name, "Metal");
        case KGGML_BACKEND_VULKAN: return iequals_prefix(reg_name, "Vulkan");
        case KGGML_BACKEND_OPENCL: return iequals_prefix(reg_name, "OpenCL");
        default: return false;
    }
}

/// whisper.cpp picks its GPU as the `gpu_device`-th GPU / iGPU device over ALL registered backends. With several GPU
/// backends registered (e.g. both Android add-ons) that index would not say which backend; this maps "device
/// `index_in_backend` of backend `kind`" to whisper's global GPU index. -1 when the backend has no such device.
int whisper_gpu_index(int32_t kind, int32_t index_in_backend) {
    int gpu = 0;
    int in_backend = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        if (gpu_device_matches(dev, kind)) {
            if (in_backend == index_in_backend) {
                return gpu;
            }
            ++in_backend;
        }
        ++gpu;
    }
    return -1;
}

bool runtime_backend_available(int32_t kind) {
    int32_t available = 0;
    return kggml_runtime_backend_available(kind, &available) == KGGML_STATUS_OK && available == 1;
}

kggml_status check_context(const kggml_whisper_context * context) {
    if (context == nullptr || context->ctx == nullptr || context->state == nullptr) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "context is NULL");
    }
    return KGGML_STATUS_OK;
}

kggml_whisper_context_params default_context_params() {
    kggml_whisper_context_params p{};
    p.backend = KGGML_BACKEND_CPU;
    p.gpu_device = 0;
    p.flash_attn = 1;
    return p;
}

kggml_whisper_transcribe_params default_transcribe_params() {
    kggml_whisper_transcribe_params p{};
    p.language = nullptr;
    p.initial_prompt = nullptr;
    p.n_threads = 0;
    p.translate = 0;
    p.no_context = 0;
    p.single_segment = 0;
    p.no_timestamps = 0;
    p.max_tokens = 0;
    p.temperature = 0.0f;
    p.temperature_inc = 0.2f;
    return p;
}

kggml_status check_samples(const float * samples, size_t n_samples) {
    if (samples == nullptr && n_samples > 0) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "samples is NULL");
    }
    if (n_samples > static_cast<size_t>(INT_MAX)) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "too many samples (max INT_MAX)");
    }
    return KGGML_STATUS_OK;
}

/// Validates `params` (NULL = defaults) and fills `wparams` with everything except the callbacks. The language
/// pointer stored in `wparams` is either a static string or params->language, so it lives for the whole call.
kggml_status make_full_params(const kggml_whisper_context * context, const kggml_whisper_transcribe_params * params,
                              whisper_full_params & wparams) {
    const kggml_whisper_transcribe_params p = params != nullptr ? *params : default_transcribe_params();
    if (p.n_threads < 0 || p.max_tokens < 0 || p.temperature < 0.0f || p.temperature_inc < 0.0f) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT,
                    "n_threads, max_tokens, temperature and temperature_inc must be >= 0");
    }

    const char * language = p.language;
    if (language == nullptr || language[0] == '\0') {
        language = whisper_is_multilingual(context->ctx) != 0 ? "auto" : "en";
    }
    if (std::string(language) != "auto" && whisper_lang_id(language) < 0) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, (std::string("unknown language '") + language + "'").c_str());
    }

    int32_t n_threads = p.n_threads;
    if (n_threads == 0 && kggml_runtime_performance_thread_count(&n_threads) != KGGML_STATUS_OK) {
        n_threads = 4;
    }

    wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.n_threads = n_threads;
    wparams.language = language;
    wparams.detect_language = false;
    wparams.translate = p.translate != 0;
    wparams.no_context = p.no_context != 0;
    wparams.single_segment = p.single_segment != 0;
    wparams.no_timestamps = p.no_timestamps != 0;
    wparams.max_tokens = p.max_tokens;
    wparams.temperature = p.temperature;
    wparams.temperature_inc = p.temperature_inc;
    wparams.initial_prompt = p.initial_prompt;
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_special = false;
    wparams.print_timestamps = false;
    return KGGML_STATUS_OK;
}

/// Per-call bookkeeping of kggml_whisper_transcribe_with_state(): routes whisper.cpp's callbacks to the
/// context's abort flag and to the caller's new-segment callback.
struct StreamCall {
    kggml_whisper_context * context;
    kggml_whisper_state * state;
    kggml_whisper_new_segment_callback callback;
    void * user_data;
    // Written on the calling thread, read by abort polls that may run on ggml worker threads.
    std::atomic<int32_t> callback_status{KGGML_STATUS_OK};
};

bool stream_abort_callback(void * user_data) {
    auto * call = static_cast<StreamCall *>(user_data);
    const bool aborted = abort_callback(call->context); // counts the poll, honours test countdowns
    return aborted || call->callback_status.load(std::memory_order_acquire) != KGGML_STATUS_OK;
}

bool stream_encoder_begin_callback(whisper_context *, whisper_state *, void * user_data) {
    return !stream_abort_callback(user_data);
}

/// whisper_full_params.new_segment_callback: runs on the thread inside whisper_full_with_state().
void stream_new_segment_callback(whisper_context *, whisper_state * st, int n_new, void * user_data) {
    auto * call = static_cast<StreamCall *>(user_data);
    if (call->callback == nullptr || n_new <= 0 ||
        call->callback_status.load(std::memory_order_acquire) != KGGML_STATUS_OK) {
        return;
    }
    const int total = whisper_full_n_segments_from_state(st);
    const int first = total > n_new ? total - n_new : 0;
    call->state->n_segments = total; // make the new segments readable from inside the callback
    int32_t status = KGGML_STATUS_UNKNOWN;
    try {
        status = call->callback(call->state, first, total - first, call->user_data);
    } catch (...) {
        status = KGGML_STATUS_UNKNOWN; // a C callback must not throw; never let it unwind through whisper.cpp
    }
    if (status != KGGML_STATUS_OK) {
        call->callback_status.store(status, std::memory_order_release);
    }
}

kggml_status check_state(const kggml_whisper_state * state) {
    if (state == nullptr || state->state == nullptr) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "state is NULL");
    }
    return KGGML_STATUS_OK;
}

kggml_status check_segment_index(int32_t index, int32_t count) {
    if (index < 0 || index >= count) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT,
                    ("segment index " + std::to_string(index) + " out of range [0, " + std::to_string(count) + ")")
                        .c_str());
    }
    return KGGML_STATUS_OK;
}

void read_segment(whisper_state * st, int32_t index, kggml_whisper_segment * out_segment) {
    out_segment->t0_ms = whisper_full_get_segment_t0_from_state(st, index) * 10; // centiseconds -> ms
    out_segment->t1_ms = whisper_full_get_segment_t1_from_state(st, index) * 10;
    out_segment->no_speech_prob = whisper_full_get_segment_no_speech_prob_from_state(st, index);
    const char * text = whisper_full_get_segment_text_from_state(st, index);
    out_segment->text = text != nullptr ? text : "";
}

/// Copies `tokens` into the caller's buffer with the BUFFER_TOO_SMALL size-query protocol.
kggml_status copy_tokens(const std::vector<int32_t> & tokens, int32_t * out_tokens, int32_t capacity,
                         int32_t * out_count) {
    if (out_count == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_count is NULL");
    if (capacity < 0) return fail(KGGML_STATUS_INVALID_ARGUMENT, "capacity must be >= 0");
    if (tokens.size() > static_cast<size_t>(INT32_MAX)) return fail(KGGML_STATUS_NATIVE_FAILURE, "too many tokens");
    const auto n = static_cast<int32_t>(tokens.size());
    *out_count = n;
    if (n > capacity || (n > 0 && out_tokens == nullptr)) {
        return fail(KGGML_STATUS_BUFFER_TOO_SMALL,
                    ("token buffer too small, need " + std::to_string(n) + " entries").c_str());
    }
    std::copy(tokens.begin(), tokens.end(), out_tokens);
    return KGGML_STATUS_OK;
}

} // namespace
} // namespace kggml::whisper

using kggml::whisper::fail;
using kggml::whisper::guarded;

extern "C" {

// ---------------------------------------------------------------------------------------------------------
// Versions & capabilities
// ---------------------------------------------------------------------------------------------------------

KGGML_WHISPER_API const char * kggml_whisper_version(void) {
    return KGGML_WHISPER_VERSION_STRING;
}

KGGML_WHISPER_API const char * kggml_whisper_upstream_version(void) {
    return KGGML_WHISPER_UPSTREAM_VERSION;
}

KGGML_WHISPER_API const char * kggml_whisper_upstream_commit(void) {
    return KGGML_WHISPER_UPSTREAM_COMMIT;
}

KGGML_WHISPER_API const char * kggml_whisper_system_info(void) {
    try {
        return whisper_print_system_info();
    } catch (...) {
        return "";
    }
}

KGGML_WHISPER_API int32_t kggml_whisper_has_coreml(void) {
#if defined(KGGML_WHISPER_COREML)
    return 1;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------------------------------------

KGGML_WHISPER_API kggml_status kggml_whisper_context_default_params(kggml_whisper_context_params * params) {
    return guarded(__func__, [&] {
        if (params == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "params is NULL");
        }
        *params = kggml::whisper::default_context_params();
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_init_from_file(const char * model_path,
                                                           const kggml_whisper_context_params * params,
                                                           kggml_whisper_context ** out_context) {
    return guarded(__func__, [&] {
        if (out_context == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_context is NULL");
        }
        *out_context = nullptr;
        if (model_path == nullptr || model_path[0] == '\0') {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "model_path is NULL or empty");
        }
        const kggml_whisper_context_params p = params != nullptr ? *params : kggml::whisper::default_context_params();
        if (!kggml::whisper::valid_backend(p.backend)) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT,
                        ("unknown backend kind " + std::to_string(p.backend)).c_str());
        }
        if (p.gpu_device < 0) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "gpu_device must be >= 0");
        }

        auto context = std::make_unique<kggml_whisper_context>();
        const kggml_status rt = kggml_runtime_init();
        if (rt != KGGML_STATUS_OK) {
            return fail(rt, (std::string("kggml_runtime_init failed: ") + kggml_runtime_last_error()).c_str());
        }
        context->holds_runtime_ref = true;

        if (p.backend == KGGML_BACKEND_COREML) {
            if (kggml_whisper_has_coreml() == 0) {
                return fail(KGGML_STATUS_BACKEND_UNAVAILABLE,
                            "CoreML backend requested but this kggml_whisper build has no CoreML encoder");
            }
        } else if (p.backend != KGGML_BACKEND_CPU && !kggml::whisper::runtime_backend_available(p.backend)) {
            return fail(KGGML_STATUS_BACKEND_UNAVAILABLE,
                        (std::string(kggml::whisper::backend_name(p.backend)) +
                         " backend is not available in this kggml runtime / on this device")
                            .c_str());
        }

        {
            // Distinguish a missing file from a corrupt one (whisper.cpp reports both as NULL).
            std::FILE * f = std::fopen(model_path, "rb");
            if (f == nullptr) {
                return fail(KGGML_STATUS_MODEL_LOAD_FAILED,
                            (std::string("cannot open model file '") + model_path + "'").c_str());
            }
            std::fclose(f);
        }

        const auto resolved = kggml::whisper::resolve_backend(p.backend);
        whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = resolved.use_gpu; // must be decided before the context (and any backend) exists
        cparams.flash_attn = p.flash_attn != 0;
        cparams.gpu_device = p.gpu_device;
        if (resolved.use_gpu) {
            // gpu_device counts the devices of the requested backend; whisper.cpp counts every GPU device.
            const int index = kggml::whisper::whisper_gpu_index(p.backend, p.gpu_device);
            if (index < 0) {
                return fail(KGGML_STATUS_BACKEND_UNAVAILABLE,
                            (std::string(kggml::whisper::backend_name(p.backend)) + " has no GPU device " +
                             std::to_string(p.gpu_device))
                                .c_str());
            }
            cparams.gpu_device = index;
        }

        context->ctx = whisper_init_from_file_with_params_no_state(model_path, cparams);
        if (context->ctx == nullptr) {
            return fail(KGGML_STATUS_MODEL_LOAD_FAILED,
                        (std::string("whisper.cpp could not load model '") + model_path + "' (invalid or unsupported)")
                            .c_str());
        }

        bool coreml_loaded = false;
        {
            kggml::whisper::coreml::Scope gate(resolved.coreml_allowed);
            context->state = whisper_init_state(context->ctx);
            coreml_loaded = gate.loaded();
        }
        if (context->state == nullptr) {
            return fail(KGGML_STATUS_NATIVE_FAILURE,
                        "whisper_init_state failed (compute backend or KV cache allocation)");
        }

        context->use_gpu = resolved.use_gpu;
        if (p.backend == KGGML_BACKEND_COREML) {
            if (coreml_loaded) {
                context->backend = KGGML_BACKEND_COREML;
            } else if (kggml::whisper::runtime_backend_available(KGGML_BACKEND_METAL)) {
                context->backend = KGGML_BACKEND_METAL; // .mlmodelc missing: whisper.cpp fell back to the GPU encoder
            } else {
                context->backend = KGGML_BACKEND_CPU;
            }
        } else {
            context->backend = p.backend;
        }

        *out_context = context.release();
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API void kggml_whisper_free(kggml_whisper_context * context) {
    try {
        delete context;
    } catch (...) {
        // whisper_free never throws in practice; never let anything escape.
    }
}

KGGML_WHISPER_API kggml_status kggml_whisper_get_backend(const kggml_whisper_context * context, int32_t * out_backend) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_backend == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_backend is NULL");
        *out_backend = context->backend;
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_uses_gpu(const kggml_whisper_context * context, int32_t * out_uses_gpu) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_uses_gpu == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_uses_gpu is NULL");
        *out_uses_gpu = context->use_gpu ? 1 : 0;
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_is_multilingual(const kggml_whisper_context * context, int32_t * out) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out is NULL");
        *out = whisper_is_multilingual(context->ctx) != 0 ? 1 : 0;
        return KGGML_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------------------
// Abort
// ---------------------------------------------------------------------------------------------------------

KGGML_WHISPER_API void kggml_whisper_request_abort(kggml_whisper_context * context) {
    if (context != nullptr) {
        context->abort.store(true, std::memory_order_release);
    }
}

KGGML_WHISPER_API void kggml_whisper_clear_abort(kggml_whisper_context * context) {
    if (context != nullptr) {
        context->abort.store(false, std::memory_order_release);
    }
}

KGGML_WHISPER_API int32_t kggml_whisper_abort_requested(const kggml_whisper_context * context) {
    return context != nullptr && context->abort.load(std::memory_order_acquire) ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------------
// Transcription
// ---------------------------------------------------------------------------------------------------------

KGGML_WHISPER_API kggml_status kggml_whisper_transcribe_default_params(kggml_whisper_transcribe_params * params) {
    return guarded(__func__, [&] {
        if (params == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "params is NULL");
        }
        *params = kggml::whisper::default_transcribe_params();
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_transcribe(kggml_whisper_context * context, const float * samples,
                                                       size_t n_samples,
                                                       const kggml_whisper_transcribe_params * params) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_context(context); s != KGGML_STATUS_OK) return s;
        context->n_segments = 0;
        if (const auto s = kggml::whisper::check_samples(samples, n_samples); s != KGGML_STATUS_OK) return s;
        whisper_full_params wparams{};
        if (const auto s = kggml::whisper::make_full_params(context, params, wparams); s != KGGML_STATUS_OK) {
            return s;
        }

        context->abort_polls.store(0, std::memory_order_relaxed);
        if (context->abort.load(std::memory_order_acquire)) {
            return fail(KGGML_STATUS_CANCELLED, "transcription cancelled before it started");
        }

        wparams.abort_callback = kggml::whisper::abort_callback;
        wparams.abort_callback_user_data = context;
        wparams.encoder_begin_callback = kggml::whisper::encoder_begin_callback;
        wparams.encoder_begin_callback_user_data = context;

        // >= 1000 ms floor: whisper.cpp silently returns no output for < 100 ms and degrades badly below 1 s.
        size_t n_feed = 0;
        const float * feed = kggml::whisper::pad_to_min_length(samples, n_samples, KGGML_WHISPER_MIN_SAMPLES,
                                                               context->pad_scratch, n_feed);

        const int rc = whisper_full_with_state(context->ctx, context->state, wparams, feed, static_cast<int>(n_feed));

        if (context->abort.load(std::memory_order_acquire)) {
            return fail(KGGML_STATUS_CANCELLED, "transcription cancelled");
        }
        if (rc != 0) {
            return fail(KGGML_STATUS_NATIVE_FAILURE,
                        ("whisper_full_with_state failed with code " + std::to_string(rc)).c_str());
        }
        context->n_segments = whisper_full_n_segments_from_state(context->state);
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_segment_count(const kggml_whisper_context * context, int32_t * out_count) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_count == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_count is NULL");
        *out_count = context->n_segments;
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_get_segment(const kggml_whisper_context * context, int32_t index,
                                                        kggml_whisper_segment * out_segment) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_segment == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_segment is NULL");
        if (const auto s = kggml::whisper::check_segment_index(index, context->n_segments); s != KGGML_STATUS_OK) {
            return s;
        }
        kggml::whisper::read_segment(context->state, index, out_segment);
        return KGGML_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------------------
// Streaming states
// ---------------------------------------------------------------------------------------------------------

KGGML_WHISPER_API kggml_status kggml_whisper_state_create(kggml_whisper_context * context,
                                                         kggml_whisper_state ** out_state) {
    return guarded(__func__, [&] {
        if (out_state == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_state is NULL");
        *out_state = nullptr;
        if (const auto s = kggml::whisper::check_context(context); s != KGGML_STATUS_OK) return s;

        auto state = std::make_unique<kggml_whisper_state>();
        state->owner = context;
        state->eot = whisper_token_eot(context->ctx);
        state->n_vocab = whisper_n_vocab(context->ctx);
        {
            // Same encoder choice as the context: CoreML only when the context actually loaded it.
            kggml::whisper::coreml::Scope gate(context->backend == KGGML_BACKEND_COREML);
            state->state = whisper_init_state(context->ctx);
        }
        if (state->state == nullptr) {
            return fail(KGGML_STATUS_OUT_OF_MEMORY,
                        "whisper_init_state failed (compute backend or KV cache allocation)");
        }
        *out_state = state.release();
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API void kggml_whisper_state_free(kggml_whisper_state * state) {
    try {
        delete state;
    } catch (...) {
        // whisper_free_state never throws in practice; never let anything escape.
    }
}

KGGML_WHISPER_API kggml_status kggml_whisper_max_prompt_tokens(const kggml_whisper_context * context,
                                                              int32_t * out_count) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_count == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_count is NULL");
        // whisper_full_with_state: max_prompt_ctx = n_text_ctx / 2, one slot of which is the <|prev|> marker.
        *out_count = std::max(0, whisper_n_text_ctx(context->ctx) / 2 - 1);
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_transcribe_with_state(
    kggml_whisper_context * context, kggml_whisper_state * state, const float * samples, size_t n_samples,
    const kggml_whisper_transcribe_params * params, const int32_t * prompt_tokens, int32_t n_prompt_tokens,
    kggml_whisper_new_segment_callback callback, void * user_data) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_context(context); s != KGGML_STATUS_OK) return s;
        if (const auto s = kggml::whisper::check_state(state); s != KGGML_STATUS_OK) return s;
        if (state->owner != context) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "state was created on a different context");
        }
        state->n_segments = 0;
        if (const auto s = kggml::whisper::check_samples(samples, n_samples); s != KGGML_STATUS_OK) return s;
        if (n_prompt_tokens < 0 || (n_prompt_tokens > 0 && prompt_tokens == nullptr)) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "prompt_tokens is NULL or n_prompt_tokens < 0");
        }
        for (int32_t i = 0; i < n_prompt_tokens; ++i) {
            if (prompt_tokens[i] < 0 || prompt_tokens[i] >= state->n_vocab) {
                return fail(KGGML_STATUS_INVALID_ARGUMENT,
                            ("prompt token " + std::to_string(prompt_tokens[i]) + " at " + std::to_string(i) +
                             " is not a vocabulary id [0, " + std::to_string(state->n_vocab) + ")")
                                .c_str());
            }
        }
        whisper_full_params wparams{};
        if (const auto s = kggml::whisper::make_full_params(context, params, wparams); s != KGGML_STATUS_OK) {
            return s;
        }

#if defined(KGGML_WHISPER_TEST_HOOKS)
        state->last_prompt.assign(prompt_tokens, prompt_tokens + n_prompt_tokens);
#endif
        context->abort_polls.store(0, std::memory_order_relaxed);
        if (context->abort.load(std::memory_order_acquire)) {
            return fail(KGGML_STATUS_CANCELLED, "transcription cancelled before it started");
        }

        kggml::whisper::StreamCall call{context, state, callback, user_data};
        wparams.prompt_tokens = n_prompt_tokens > 0 ? prompt_tokens : nullptr;
        wparams.prompt_n_tokens = n_prompt_tokens;
        wparams.abort_callback = kggml::whisper::stream_abort_callback;
        wparams.abort_callback_user_data = &call;
        wparams.encoder_begin_callback = kggml::whisper::stream_encoder_begin_callback;
        wparams.encoder_begin_callback_user_data = &call;
        wparams.new_segment_callback = kggml::whisper::stream_new_segment_callback;
        wparams.new_segment_callback_user_data = &call;

        size_t n_feed = 0;
        const float * feed = kggml::whisper::pad_to_min_length(samples, n_samples, KGGML_WHISPER_MIN_SAMPLES,
                                                               state->pad_scratch, n_feed);

        const int rc = whisper_full_with_state(context->ctx, state->state, wparams, feed, static_cast<int>(n_feed));
        state->n_segments = 0;

        if (context->abort.load(std::memory_order_acquire)) {
            return fail(KGGML_STATUS_CANCELLED, "transcription cancelled");
        }
        const int32_t callback_status = call.callback_status.load(std::memory_order_acquire);
        if (callback_status != KGGML_STATUS_OK) {
            return fail(static_cast<kggml_status>(callback_status),
                        ("new_segment_callback stopped the transcription with status " +
                         std::to_string(callback_status))
                            .c_str());
        }
        if (rc != 0) {
            return fail(KGGML_STATUS_NATIVE_FAILURE,
                        ("whisper_full_with_state failed with code " + std::to_string(rc)).c_str());
        }
        state->n_segments = whisper_full_n_segments_from_state(state->state);
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_state_segment_count(const kggml_whisper_state * state,
                                                                int32_t * out_count) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_state(state); s != KGGML_STATUS_OK) return s;
        if (out_count == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_count is NULL");
        *out_count = state->n_segments;
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_state_get_segment(const kggml_whisper_state * state, int32_t index,
                                                              kggml_whisper_segment * out_segment) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_state(state); s != KGGML_STATUS_OK) return s;
        if (out_segment == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_segment is NULL");
        if (const auto s = kggml::whisper::check_segment_index(index, state->n_segments); s != KGGML_STATUS_OK) {
            return s;
        }
        kggml::whisper::read_segment(state->state, index, out_segment);
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_state_get_segment_tokens(const kggml_whisper_state * state,
                                                                     int32_t index, int32_t * out_tokens,
                                                                     int32_t capacity, int32_t * out_count) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_state(state); s != KGGML_STATUS_OK) return s;
        if (out_count == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_count is NULL");
        *out_count = 0;
        if (const auto s = kggml::whisper::check_segment_index(index, state->n_segments); s != KGGML_STATUS_OK) {
            return s;
        }
        const int n = whisper_full_n_tokens_from_state(state->state, index);
        std::vector<int32_t> text_tokens;
        text_tokens.reserve(static_cast<size_t>(n > 0 ? n : 0));
        for (int j = 0; j < n; ++j) {
            const whisper_token id = whisper_full_get_token_id_from_state(state->state, index, j);
            if (id >= 0 && id < state->eot) {
                text_tokens.push_back(id); // drop <|endoftext|>, task/language markers and timestamps
            }
        }
        return kggml::whisper::copy_tokens(text_tokens, out_tokens, capacity, out_count);
    });
}

// ---------------------------------------------------------------------------------------------------------
// Test hooks (KGGML_WHISPER_TEST_HOOKS builds only)
// ---------------------------------------------------------------------------------------------------------

#if defined(KGGML_WHISPER_TEST_HOOKS)
KGGML_WHISPER_API kggml_status kggml_whisper_test_poll_callbacks(kggml_whisper_context * context, int32_t * out_abort,
                                                                int32_t * out_encoder_continue) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_abort == nullptr || out_encoder_continue == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "output pointer is NULL");
        }
        *out_abort = kggml::whisper::abort_callback(context) ? 1 : 0;
        *out_encoder_continue = kggml::whisper::encoder_begin_callback(context->ctx, context->state, context) ? 1 : 0;
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_test_abort_after_polls(kggml_whisper_context * context, int32_t n) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_context(context); s != KGGML_STATUS_OK) return s;
        context->abort_after_polls.store(n, std::memory_order_relaxed);
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_test_abort_poll_count(const kggml_whisper_context * context,
                                                                  int64_t * out_polls) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_polls == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_polls is NULL");
        *out_polls = context->abort_polls.load(std::memory_order_relaxed);
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API kggml_status kggml_whisper_test_resolve_use_gpu(const kggml_whisper_context_params * params,
                                                                 int32_t * out_use_gpu, int32_t * out_coreml_allowed) {
    return guarded(__func__, [&] {
        if (params == nullptr || out_use_gpu == nullptr || out_coreml_allowed == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "NULL argument");
        }
        if (!kggml::whisper::valid_backend(params->backend)) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "unknown backend kind");
        }
        const auto r = kggml::whisper::resolve_backend(params->backend);
        *out_use_gpu = r.use_gpu ? 1 : 0;
        *out_coreml_allowed = r.coreml_allowed ? 1 : 0;
        return KGGML_STATUS_OK;
    });
}

KGGML_WHISPER_API int64_t kggml_whisper_test_live_state_count(void) {
    return g_live_states.load(std::memory_order_relaxed);
}

KGGML_WHISPER_API kggml_status kggml_whisper_test_state_last_prompt(const kggml_whisper_state * state,
                                                                   int32_t * out_tokens, int32_t capacity,
                                                                   int32_t * out_count) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::whisper::check_state(state); s != KGGML_STATUS_OK) return s;
        return kggml::whisper::copy_tokens(state->last_prompt, out_tokens, capacity, out_count);
    });
}
#endif

} // extern "C"
