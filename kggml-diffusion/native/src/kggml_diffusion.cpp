// kggml_diffusion.cpp - extern "C" stable-diffusion.cpp wrapper: context lifecycle, backend selection, txt2img
// generation with per-generation progress / abort dispatch, and log routing. Every entry point that can fail runs
// behind kggml::diffusion::guarded().
//
// stable-diffusion.cpp's progress and log hooks are PROCESS-WIDE (sd_set_progress_callback / sd_set_log_callback)
// and it has no abort hook. This file therefore installs one dispatcher for each hook, once per process, and routes
// every invocation to the generation running on the CALLING thread (a thread_local pointer set for the duration of
// kggml_diffusion_generate()). The sampler reports progress on the thread that called generate_image(), so
// concurrent generations on different contexts / threads never see each other's callbacks and no global callback
// is ever re-registered (no lock, no clobbering). An abort is delivered by throwing a private AbortSignal from the
// progress dispatcher: it unwinds stable-diffusion.cpp's sampler (RAII-only C++ frames at the pinned commit) back
// into kggml_diffusion_generate(), which maps it to KGGML_STATUS_CANCELLED.
#include "kggml_diffusion.h"

#include "guard.hpp"
#include "internal.hpp"

#if defined(KGGML_DIFFUSION_TEST_HOOKS)
#    include "kggml_diffusion_testing.h"
#endif

#include "ggml-backend.h"
#include "ggml.h"
#include "stable-diffusion.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#ifndef KGGML_DIFFUSION_VERSION_STRING
#    define KGGML_DIFFUSION_VERSION_STRING "0.0.0"
#endif
#ifndef KGGML_DIFFUSION_UPSTREAM_VERSION
#    define KGGML_DIFFUSION_UPSTREAM_VERSION "unknown"
#endif
#ifndef KGGML_DIFFUSION_UPSTREAM_COMMIT
#    define KGGML_DIFFUSION_UPSTREAM_COMMIT "unknown"
#endif

// stable-diffusion.cpp internals (src/util.h, C++ linkage, compiled into this library). Declared here instead of
// including util.h, which drags in the whole tensor library.
void pretty_progress(int step, int steps, float time);
void log_printf(sd_log_level_t level, const char * file, int line, const char * format, ...);

namespace kggml::diffusion {
#if defined(KGGML_DIFFUSION_TEST_HOOKS)
std::atomic<int64_t> g_live_objects{0};
#endif
} // namespace kggml::diffusion

kggml_diffusion_context::~kggml_diffusion_context() {
    if (sd != nullptr) {
        free_sd_ctx(sd);
    }
    if (holds_runtime_ref) {
        kggml_runtime_shutdown();
    }
}

namespace kggml::diffusion {

namespace {

// ---------------------------------------------------------------------------------------------------------
// Per-thread generation dispatch
// ---------------------------------------------------------------------------------------------------------

struct Generation {
    kggml_diffusion_context * context = nullptr;
    kggml_diffusion_progress_callback progress_cb = nullptr;
    kggml_diffusion_abort_callback abort_cb = nullptr;
    void * user_data = nullptr;
    int32_t last_step = -1;
    int32_t last_total = 0;
#if defined(KGGML_DIFFUSION_TEST_HOOKS)
    int32_t fault_at_step = -1;
    int32_t fault = 0;
#endif

    /// Latched context flag or the per-call abort callback (a throwing callback counts as "abort").
    bool abort_requested() const noexcept {
        if (context != nullptr && context->abort.load(std::memory_order_acquire)) {
            return true;
        }
        if (abort_cb != nullptr) {
            try {
                return abort_cb(user_data) != 0;
            } catch (...) {
                return true;
            }
        }
        return false;
    }
};

thread_local Generation * t_generation = nullptr;

/// Makes `g` the calling thread's active generation for the lifetime of the scope.
class GenerationScope {
public:
    explicit GenerationScope(Generation * g) noexcept : previous_(t_generation) { t_generation = g; }
    ~GenerationScope() { t_generation = previous_; }
    GenerationScope(const GenerationScope &) = delete;
    GenerationScope & operator=(const GenerationScope &) = delete;

private:
    Generation * previous_;
};

/// The process-wide sd_progress_cb_t. Called by upstream's pretty_progress() / pretty_bytes_progress() on the thread
/// that runs the sampler / model loader. Outside kggml_diffusion_generate() (e.g. weight loading inside
/// new_sd_ctx(), where the loader's worker threads are still joinable) there is no active generation and the call
/// is ignored, so nothing is ever thrown there.
void progress_dispatch(int step, int steps, float time, void * /*data*/) {
    Generation * g = t_generation;
    if (g == nullptr) {
        return;
    }
    g->last_step = step;
    g->last_total = steps;
#if defined(KGGML_DIFFUSION_TEST_HOOKS)
    if (g->fault != 0 && step == g->fault_at_step) {
        throw_fault(g->fault);
    }
#endif
    if (g->progress_cb != nullptr) {
        try {
            g->progress_cb(step, steps, time, g->user_data);
        } catch (...) {
            throw AbortSignal{}; // a C callback must not throw; treat it as a request to stop
        }
    }
    if (g->abort_requested()) {
        throw AbortSignal{};
    }
}

/// Optional stderr echo of stable-diffusion.cpp's log (KGGML_DIFFUSION_LOG=1|debug), off by default.
int log_echo_level() noexcept {
    static const int level = [] {
        const char * env = std::getenv("KGGML_DIFFUSION_LOG");
        if (env == nullptr || env[0] == '\0' || std::strcmp(env, "0") == 0) return -1;
        if (std::strcmp(env, "debug") == 0) return static_cast<int>(SD_LOG_DEBUG);
        return static_cast<int>(SD_LOG_INFO);
    }();
    return level;
}

/// The process-wide sd_log_cb_t: silences stable-diffusion.cpp (it would otherwise log nothing, but upstream's
/// default progress printer writes to stdout) and keeps the last ERROR line of the calling thread for last_error.
void log_dispatch(enum sd_log_level_t level, const char * text, void * /*data*/) {
    try {
        if (text == nullptr) return;
        if (level == SD_LOG_ERROR) {
            record_sd_error_line(text);
        }
        const int echo = log_echo_level();
        if (echo >= 0 && static_cast<int>(level) >= echo) {
            std::fputs(text, stderr);
        }
    } catch (...) {
        // never unwind into stable-diffusion.cpp's logger
    }
}

/// Installs both dispatchers exactly once per process. They never change afterwards.
void install_hooks() {
    static std::once_flag once;
    std::call_once(once, [] {
        sd_set_log_callback(log_dispatch, nullptr);
        sd_set_progress_callback(progress_dispatch, nullptr);
    });
}

// ---------------------------------------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------------------------------------

kggml_diffusion_context_params default_context_params() {
    kggml_diffusion_context_params p{};
    p.model_path = nullptr;
    p.diffusion_model_path = nullptr;
    p.vae_path = nullptr;
    p.taesd_path = nullptr;
    p.clip_l_path = nullptr;
    p.clip_g_path = nullptr;
    p.t5xxl_path = nullptr;
    p.llm_path = nullptr;
    p.backend = KGGML_BACKEND_CPU;
    p.gpu_device = 0;
    p.n_threads = 0;
    p.use_mmap = 1;
    p.flash_attn = 0;
    p.offload_params_to_cpu = 0;
    p.keep_clip_on_cpu = 0;
    p.keep_vae_on_cpu = 0;
    p.vae_decode_only = 1;
    p.weight_type = -1;
    return p;
}

kggml_diffusion_generate_params default_generate_params() {
    kggml_diffusion_generate_params p{};
    p.prompt = "";
    p.negative_prompt = nullptr;
    p.width = 512;
    p.height = 512;
    p.sample_steps = 20;
    p.cfg_scale = 7.0f;
    p.distilled_guidance = 3.5f;
    p.sample_method = KGGML_DIFFUSION_DEFAULT;
    p.scheduler = KGGML_DIFFUSION_DEFAULT;
    p.seed = -1;
    p.clip_skip = -1;
    p.vae_tiling = 0;
    return p;
}

bool is_set(const char * s) noexcept {
    return s != nullptr && s[0] != '\0';
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
        if (*s == '\0' ||
            std::tolower(static_cast<unsigned char>(*s)) != std::tolower(static_cast<unsigned char>(*prefix))) {
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

kggml_status validate_context_params(const kggml_diffusion_context_params & p) {
    if (!is_set(p.model_path) && !is_set(p.diffusion_model_path)) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "model_path or diffusion_model_path is required");
    }
    if (p.backend < KGGML_BACKEND_CPU || p.backend >= KGGML_BACKEND_KIND_COUNT) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, ("unknown backend kind " + std::to_string(p.backend)).c_str());
    }
    if (p.gpu_device < 0) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "gpu_device must be >= 0");
    }
    if (p.weight_type < -1 || p.weight_type >= static_cast<int32_t>(SD_TYPE_COUNT)) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT,
                    ("weight_type " + std::to_string(p.weight_type) + " out of range [-1, " +
                     std::to_string(static_cast<int>(SD_TYPE_COUNT)) + ")")
                        .c_str());
    }
    return KGGML_STATUS_OK;
}

/// Distinguishes a missing / unreadable / empty file from a corrupt one (stable-diffusion.cpp reports all as NULL).
kggml_status check_model_file(const char * what, const char * path) {
    if (!is_set(path)) {
        return KGGML_STATUS_OK;
    }
    std::FILE * f = std::fopen(path, "rb");
    if (f == nullptr) {
        return fail(KGGML_STATUS_MODEL_LOAD_FAILED, (std::string("cannot open ") + what + " '" + path + "'").c_str());
    }
    unsigned char probe = 0;
    const size_t n = std::fread(&probe, 1, 1, f);
    std::fclose(f);
    if (n != 1) {
        return fail(KGGML_STATUS_MODEL_LOAD_FAILED, (std::string(what) + " '" + path + "' is empty").c_str());
    }
    return KGGML_STATUS_OK;
}

struct ResolvedBackend {
    std::string spec; // stable-diffusion.cpp backend assignment (applies to every module)
    bool uses_gpu = false;
};

/// The single place that maps kggml_backend_kind to stable-diffusion.cpp's backend spec. CPU pins every module to
/// the ggml "CPU" device (no GPU backend is ever created). Requires the runtime to be initialized.
kggml_status resolve_backend(const kggml_diffusion_context_params & p, ResolvedBackend & out) {
    out = ResolvedBackend{};
    if (p.backend == KGGML_BACKEND_CPU) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu == nullptr) {
            return fail(KGGML_STATUS_BACKEND_UNAVAILABLE, "the kggml runtime has no ggml CPU device");
        }
        out.spec = ggml_backend_dev_name(cpu);
        return KGGML_STATUS_OK;
    }
    if (p.backend == KGGML_BACKEND_COREML) {
        return fail(KGGML_STATUS_BACKEND_UNAVAILABLE, "stable-diffusion.cpp has no CoreML backend (use Metal or CPU)");
    }
    int32_t available = 0;
    if (kggml_runtime_backend_available(p.backend, &available) != KGGML_STATUS_OK || available != 1) {
        return fail(KGGML_STATUS_BACKEND_UNAVAILABLE,
                    (std::string(backend_name(p.backend)) +
                     " backend is not available in this kggml runtime / on this device")
                        .c_str());
    }
    int32_t seen = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!device_matches(dev, p.backend)) continue;
        if (seen++ == p.gpu_device) {
            out.spec = ggml_backend_dev_name(dev);
            out.uses_gpu = true;
            return KGGML_STATUS_OK;
        }
    }
    return fail(KGGML_STATUS_BACKEND_UNAVAILABLE,
                (std::string(backend_name(p.backend)) + " device " + std::to_string(p.gpu_device) + " not found (" +
                 std::to_string(seen) + " device(s) of that kind)")
                    .c_str());
}

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

kggml_status validate_generate_params(const kggml_diffusion_generate_params & p) {
    if (p.prompt == nullptr) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "prompt is NULL");
    }
    if (p.width <= 0 || p.height <= 0 || p.width % 8 != 0 || p.height % 8 != 0) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, ("width / height must be positive multiples of 8, got " +
                                                    std::to_string(p.width) + "x" + std::to_string(p.height))
                                                       .c_str());
    }
    if (p.width > 16384 || p.height > 16384) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "width / height must be <= 16384");
    }
    if (p.sample_steps <= 0 || p.sample_steps > 10000) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT,
                    ("sample_steps must be in [1, 10000], got " + std::to_string(p.sample_steps)).c_str());
    }
    if (p.sample_method < KGGML_DIFFUSION_DEFAULT || p.sample_method >= static_cast<int32_t>(SAMPLE_METHOD_COUNT)) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT,
                    ("sample_method " + std::to_string(p.sample_method) + " out of range").c_str());
    }
    if (p.scheduler < KGGML_DIFFUSION_DEFAULT || p.scheduler >= static_cast<int32_t>(SCHEDULER_COUNT)) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, ("scheduler " + std::to_string(p.scheduler) + " out of range").c_str());
    }
    return KGGML_STATUS_OK;
}

kggml_status check_context(const kggml_diffusion_context * context) noexcept {
    if (context == nullptr) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "context is NULL");
    }
    return KGGML_STATUS_OK;
}

/// Owns the sd_image_t array returned by generate_image() (malloc'd array + malloc'd pixel buffers).
struct SdImages {
    sd_image_t * images = nullptr;
    int count = 0;
    ~SdImages() {
        if (images == nullptr) return;
        for (int i = 0; i < count; ++i) {
            std::free(images[i].data);
        }
        std::free(images);
    }
};

/// Shared by kggml_diffusion_generate() and the simulation test hook: latched-abort / re-entrancy checks, the
/// per-thread dispatch scope, AbortSignal -> CANCELLED, and the hand-over of the first RGB8 image to the caller.
/// `body` returns the generate_image()-style malloc'd array (one image) or NULL on failure.
template <typename Body>
kggml_status run_generation(Generation & g, kggml_diffusion_image * out_image, Body && body) {
    install_hooks();
    if (t_generation != nullptr) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT,
                    "kggml_diffusion_generate() called re-entrantly from a progress / abort callback");
    }
    if (g.context->abort.load(std::memory_order_acquire)) {
        return fail(KGGML_STATUS_CANCELLED,
                    "an abort is latched on this context (call kggml_diffusion_clear_abort() before generating)");
    }
    if (g.abort_requested()) {
        return fail(KGGML_STATUS_CANCELLED, "the abort callback requested an abort before generation started");
    }

    SdImages result;
    try {
        GenerationScope scope(&g);
        result.images = body();
        result.count = result.images != nullptr ? 1 : 0;
    } catch (const AbortSignal &) {
        return fail(KGGML_STATUS_CANCELLED, ("generation aborted at progress step " + std::to_string(g.last_step) +
                                             "/" + std::to_string(g.last_total))
                                                .c_str());
    }

    if (result.images == nullptr) {
        return fail_with_sd_log(KGGML_STATUS_NATIVE_FAILURE, "stable-diffusion.cpp failed to generate an image");
    }
    // An abort that arrived after the last progress point (e.g. during the VAE decode) still cancels the call.
    if (g.context->abort.load(std::memory_order_acquire)) {
        return fail(KGGML_STATUS_CANCELLED, "generation aborted after the last denoising step");
    }
    sd_image_t & img = result.images[0];
    if (img.data == nullptr || img.width == 0 || img.height == 0 || img.channel != 3) {
        return fail(KGGML_STATUS_NATIVE_FAILURE,
                    ("stable-diffusion.cpp returned an unexpected image (" + std::to_string(img.width) + "x" +
                     std::to_string(img.height) + "x" + std::to_string(img.channel) + ")")
                        .c_str());
    }
    out_image->width = img.width;
    out_image->height = img.height;
    out_image->channels = img.channel;
    out_image->data = img.data; // zero-copy hand-over of upstream's malloc'd RGB8 buffer
    img.data = nullptr;
    return KGGML_STATUS_OK;
}

std::mutex & create_mutex() {
    static std::mutex m;
    return m;
}

} // namespace
} // namespace kggml::diffusion

using kggml::diffusion::fail;
using kggml::diffusion::guarded;

extern "C" {

// ---------------------------------------------------------------------------------------------------------
// Versions
// ---------------------------------------------------------------------------------------------------------

KGGML_DIFFUSION_API const char * kggml_diffusion_version(void) {
    return KGGML_DIFFUSION_VERSION_STRING;
}

KGGML_DIFFUSION_API const char * kggml_diffusion_upstream_version(void) {
    return KGGML_DIFFUSION_UPSTREAM_VERSION;
}

KGGML_DIFFUSION_API const char * kggml_diffusion_upstream_commit(void) {
    return KGGML_DIFFUSION_UPSTREAM_COMMIT;
}

KGGML_DIFFUSION_API const char * kggml_diffusion_system_info(void) {
    try {
        const char * info = sd_get_system_info();
        return info != nullptr ? info : "";
    } catch (...) {
        return "";
    }
}

// ---------------------------------------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------------------------------------

KGGML_DIFFUSION_API kggml_status kggml_diffusion_context_default_params(kggml_diffusion_context_params * params) {
    return guarded(__func__, [&] {
        if (params == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "params is NULL");
        *params = kggml::diffusion::default_context_params();
        return KGGML_STATUS_OK;
    });
}

KGGML_DIFFUSION_API kggml_status kggml_diffusion_context_create(const kggml_diffusion_context_params * params,
                                                               kggml_diffusion_context ** out_context) {
    return guarded(__func__, [&] {
        namespace d = kggml::diffusion;
        if (out_context == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_context is NULL");
        *out_context = nullptr;
        if (params == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "params is NULL");
        const kggml_diffusion_context_params p = *params;
        if (const auto s = d::validate_context_params(p); s != KGGML_STATUS_OK) return s;

        // Every file is checked up front so a typo is MODEL_LOAD_FAILED naming the file, not an upstream log line.
        const std::pair<const char *, const char *> files[] = {
            {"model file", p.model_path},       {"diffusion model file", p.diffusion_model_path},
            {"VAE file", p.vae_path},           {"TAESD file", p.taesd_path},
            {"CLIP-L file", p.clip_l_path},     {"CLIP-G file", p.clip_g_path},
            {"T5-XXL file", p.t5xxl_path},      {"LLM text encoder file", p.llm_path},
        };
        for (const auto & f : files) {
            if (const auto s = d::check_model_file(f.first, f.second); s != KGGML_STATUS_OK) return s;
        }

        auto context = std::make_unique<kggml_diffusion_context>();
        const kggml_status rt = kggml_runtime_init();
        if (rt != KGGML_STATUS_OK) {
            return fail(rt, (std::string("kggml_runtime_init failed: ") + kggml_runtime_last_error()).c_str());
        }
        context->holds_runtime_ref = true;

        d::ResolvedBackend resolved;
        if (const auto s = d::resolve_backend(p, resolved); s != KGGML_STATUS_OK) return s;
        context->backend = p.backend;
        context->uses_gpu = resolved.uses_gpu;
        context->backend_spec = resolved.spec;
        context->n_threads = d::resolve_threads(p.n_threads);

        d::install_hooks();

        sd_ctx_params_t sp;
        sd_ctx_params_init(&sp);
        sp.model_path = d::is_set(p.model_path) ? p.model_path : "";
        sp.diffusion_model_path = d::is_set(p.diffusion_model_path) ? p.diffusion_model_path : "";
        sp.vae_path = d::is_set(p.vae_path) ? p.vae_path : "";
        sp.taesd_path = d::is_set(p.taesd_path) ? p.taesd_path : "";
        sp.clip_l_path = d::is_set(p.clip_l_path) ? p.clip_l_path : "";
        sp.clip_g_path = d::is_set(p.clip_g_path) ? p.clip_g_path : "";
        sp.t5xxl_path = d::is_set(p.t5xxl_path) ? p.t5xxl_path : "";
        sp.llm_path = d::is_set(p.llm_path) ? p.llm_path : "";
        sp.vae_decode_only = p.vae_decode_only != 0;
        sp.free_params_immediately = false; // keep the weights resident across generations
        sp.n_threads = context->n_threads;
        sp.wtype = p.weight_type < 0 ? SD_TYPE_COUNT : static_cast<sd_type_t>(p.weight_type);
        sp.enable_mmap = p.use_mmap != 0;
        sp.flash_attn = p.flash_attn != 0;
        sp.diffusion_flash_attn = p.flash_attn != 0;
        sp.offload_params_to_cpu = resolved.uses_gpu && p.offload_params_to_cpu != 0;
        sp.keep_clip_on_cpu = p.keep_clip_on_cpu != 0;
        sp.keep_vae_on_cpu = p.keep_vae_on_cpu != 0;
        sp.backend = context->backend_spec.c_str();
        sp.params_backend = nullptr; // weights live on the runtime backend (or the CPU with offload_params_to_cpu)

        {
            // new_sd_ctx() resets the process-wide ggml logger to ggml's stderr default; restore whatever was
            // installed before (other engines' routing). Serialized so concurrent creates cannot interleave the
            // save / restore.
            std::lock_guard<std::mutex> lock(d::create_mutex());
            ggml_log_callback saved_cb = nullptr;
            void * saved_ud = nullptr;
            ggml_log_get(&saved_cb, &saved_ud);
            try {
                context->sd = new_sd_ctx(&sp);
            } catch (const std::bad_alloc &) {
                ggml_log_set(saved_cb, saved_ud);
                throw;
            } catch (const std::exception & e) {
                // e.g. a malformed safetensors JSON header (nlohmann::json) inside the model loader.
                ggml_log_set(saved_cb, saved_ud);
                return fail(KGGML_STATUS_MODEL_LOAD_FAILED,
                            (std::string("stable-diffusion.cpp could not parse the model: ") + e.what()).c_str());
            }
            ggml_log_set(saved_cb, saved_ud);
        }
        if (context->sd == nullptr) {
            const char * path = d::is_set(p.model_path) ? p.model_path : p.diffusion_model_path;
            return kggml::diffusion::fail_with_sd_log(
                KGGML_STATUS_MODEL_LOAD_FAILED,
                (std::string("stable-diffusion.cpp could not load a diffusion model from '") + path + "'").c_str());
        }
        *out_context = context.release();
        return KGGML_STATUS_OK;
    });
}

KGGML_DIFFUSION_API void kggml_diffusion_context_free(kggml_diffusion_context * context) {
    try {
        delete context;
    } catch (...) {
        // destructors must not throw across the boundary; nothing else can be done here
    }
}

KGGML_DIFFUSION_API kggml_status kggml_diffusion_context_get_info(const kggml_diffusion_context * context,
                                                                 kggml_diffusion_context_info * out_info) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::diffusion::check_context(context); s != KGGML_STATUS_OK) return s;
        if (out_info == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_info is NULL");
        kggml_diffusion_context_info info{};
        info.backend = context->backend;
        info.uses_gpu = context->uses_gpu ? 1 : 0;
        info.n_threads = context->n_threads;
        info.supports_image_generation =
            context->sd != nullptr && sd_ctx_supports_image_generation(context->sd) ? 1 : 0;
        info.default_sample_method =
            context->sd != nullptr ? static_cast<int32_t>(sd_get_default_sample_method(context->sd)) : -1;
        *out_info = info;
        return KGGML_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------------------
// Abort
// ---------------------------------------------------------------------------------------------------------

KGGML_DIFFUSION_API void kggml_diffusion_request_abort(kggml_diffusion_context * context) {
    if (context != nullptr) {
        context->abort.store(true, std::memory_order_release);
    }
}

KGGML_DIFFUSION_API void kggml_diffusion_clear_abort(kggml_diffusion_context * context) {
    if (context != nullptr) {
        context->abort.store(false, std::memory_order_release);
    }
}

KGGML_DIFFUSION_API int32_t kggml_diffusion_abort_requested(const kggml_diffusion_context * context) {
    return context != nullptr && context->abort.load(std::memory_order_acquire) ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------------------------------------

KGGML_DIFFUSION_API kggml_status kggml_diffusion_generate_default_params(kggml_diffusion_generate_params * params) {
    return guarded(__func__, [&] {
        if (params == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "params is NULL");
        *params = kggml::diffusion::default_generate_params();
        return KGGML_STATUS_OK;
    });
}

KGGML_DIFFUSION_API kggml_status kggml_diffusion_generate(kggml_diffusion_context * context,
                                                         const kggml_diffusion_generate_params * params,
                                                         kggml_diffusion_progress_callback progress_cb,
                                                         kggml_diffusion_abort_callback abort_cb, void * user_data,
                                                         kggml_diffusion_image * out_image) {
    return guarded(__func__, [&] {
        namespace d = kggml::diffusion;
        if (out_image == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_image is NULL");
        *out_image = kggml_diffusion_image{};
        if (const auto s = d::check_context(context); s != KGGML_STATUS_OK) return s;
        if (params == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "params is NULL");
        const kggml_diffusion_generate_params p = *params;
        if (const auto s = d::validate_generate_params(p); s != KGGML_STATUS_OK) return s;
        if (context->sd == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "context has no model");

        sd_img_gen_params_t gp;
        sd_img_gen_params_init(&gp);
        gp.prompt = p.prompt;
        gp.negative_prompt = p.negative_prompt != nullptr ? p.negative_prompt : "";
        gp.clip_skip = p.clip_skip > 0 ? p.clip_skip : -1;
        gp.width = p.width;
        gp.height = p.height;
        gp.seed = p.seed;
        gp.batch_count = 1;
        gp.sample_params.sample_steps = p.sample_steps;
        gp.sample_params.guidance.txt_cfg = p.cfg_scale;
        gp.sample_params.guidance.distilled_guidance = p.distilled_guidance;
        gp.sample_params.sample_method =
            p.sample_method < 0 ? SAMPLE_METHOD_COUNT : static_cast<sample_method_t>(p.sample_method);
        gp.sample_params.scheduler = p.scheduler < 0 ? SCHEDULER_COUNT : static_cast<scheduler_t>(p.scheduler);
        gp.vae_tiling_params.enabled = p.vae_tiling != 0;

        d::Generation g;
        g.context = context;
        g.progress_cb = progress_cb;
        g.abort_cb = abort_cb;
        g.user_data = user_data;
        sd_ctx_t * sd = context->sd;
        return d::run_generation(g, out_image, [&]() -> sd_image_t * { return generate_image(sd, &gp); });
    });
}

KGGML_DIFFUSION_API void kggml_diffusion_image_free(kggml_diffusion_image * image) {
    if (image == nullptr) return;
    std::free(image->data); // allocated with malloc by stable-diffusion.cpp inside this library
    *image = kggml_diffusion_image{};
}

// ---------------------------------------------------------------------------------------------------------
// Test hooks (KGGML_DIFFUSION_TEST_HOOKS builds only)
// ---------------------------------------------------------------------------------------------------------

#if defined(KGGML_DIFFUSION_TEST_HOOKS)

KGGML_DIFFUSION_API kggml_status kggml_diffusion_test_context_create_stub(kggml_diffusion_context ** out_context) {
    return guarded(__func__, [&] {
        if (out_context == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_context is NULL");
        auto context = std::make_unique<kggml_diffusion_context>();
        context->backend_spec = "CPU";
        *out_context = context.release();
        return KGGML_STATUS_OK;
    });
}

KGGML_DIFFUSION_API kggml_status kggml_diffusion_test_simulate_generate(
    kggml_diffusion_context * context, int32_t steps, int32_t width, int32_t height,
    kggml_diffusion_progress_callback progress_cb, kggml_diffusion_abort_callback abort_cb, void * user_data,
    int32_t fault_at_step, int32_t fault, kggml_diffusion_image * out_image) {
    return guarded(__func__, [&] {
        namespace d = kggml::diffusion;
        if (out_image == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_image is NULL");
        *out_image = kggml_diffusion_image{};
        if (const auto s = d::check_context(context); s != KGGML_STATUS_OK) return s;
        if (steps <= 0 || width <= 0 || height <= 0) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "steps / width / height must be > 0");
        }
        d::Generation g;
        g.context = context;
        g.progress_cb = progress_cb;
        g.abort_cb = abort_cb;
        g.user_data = user_data;
        g.fault_at_step = fault_at_step;
        g.fault = fault;
        return d::run_generation(g, out_image, [&]() -> sd_image_t * {
            // Same reporting pattern as StableDiffusionGGML::sample(): 0 at the start, then 1..steps.
            for (int32_t step = 0; step <= steps; ++step) {
                pretty_progress(step, steps, 0.01f);
            }
            auto * images = static_cast<sd_image_t *>(std::calloc(1, sizeof(sd_image_t)));
            if (images == nullptr) throw std::bad_alloc();
            const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
            auto * px = static_cast<uint8_t *>(std::malloc(n));
            if (px == nullptr) {
                std::free(images);
                throw std::bad_alloc();
            }
            for (int32_t y = 0; y < height; ++y) {
                for (int32_t x = 0; x < width; ++x) {
                    uint8_t * p = px + (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3u;
                    p[0] = static_cast<uint8_t>(x & 255);
                    p[1] = static_cast<uint8_t>(y & 255);
                    p[2] = static_cast<uint8_t>((x + y) & 255);
                }
            }
            images[0] = sd_image_t{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 3u, px};
            return images;
        });
    });
}

KGGML_DIFFUSION_API void kggml_diffusion_test_emit_sd_log(int32_t level, const char * text) {
    try {
        kggml::diffusion::install_hooks();
        log_printf(static_cast<sd_log_level_t>(level), __FILE__, __LINE__, "%s", text != nullptr ? text : "");
    } catch (...) {
    }
}

KGGML_DIFFUSION_API const char * kggml_diffusion_test_captured_sd_error(void) {
    return kggml::diffusion::sd_error_line();
}

KGGML_DIFFUSION_API int64_t kggml_diffusion_test_live_objects(void) {
    return kggml::diffusion::g_live_objects.load(std::memory_order_relaxed);
}

KGGML_DIFFUSION_API kggml_status kggml_diffusion_test_resolve_backend(const kggml_diffusion_context_params * params,
                                                                     char * buffer, size_t capacity,
                                                                     int32_t * out_uses_gpu) {
    return guarded(__func__, [&] {
        namespace d = kggml::diffusion;
        if (params == nullptr || buffer == nullptr || capacity == 0) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "params / buffer is NULL");
        }
        buffer[0] = '\0';
        kggml_diffusion_context_params p = *params;
        if (!d::is_set(p.model_path) && !d::is_set(p.diffusion_model_path)) p.model_path = "unused";
        if (const auto s = d::validate_context_params(p); s != KGGML_STATUS_OK) return s;
        const kggml_status rt = kggml_runtime_init();
        if (rt != KGGML_STATUS_OK) return fail(rt, "kggml_runtime_init failed");
        struct RuntimeRef {
            ~RuntimeRef() { kggml_runtime_shutdown(); }
        } ref;
        d::ResolvedBackend resolved;
        if (const auto s = d::resolve_backend(p, resolved); s != KGGML_STATUS_OK) return s;
        if (resolved.spec.size() + 1 > capacity) return fail(KGGML_STATUS_BUFFER_TOO_SMALL, "buffer too small");
        std::memcpy(buffer, resolved.spec.c_str(), resolved.spec.size() + 1);
        if (out_uses_gpu != nullptr) *out_uses_gpu = resolved.uses_gpu ? 1 : 0;
        return KGGML_STATUS_OK;
    });
}

#endif // KGGML_DIFFUSION_TEST_HOOKS

} // extern "C"
