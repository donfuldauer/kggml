// coreml_gate.cpp - see coreml_gate.hpp.
#include "coreml_gate.hpp"

#if defined(KGGML_WHISPER_COREML)
#    include "coreml/whisper-encoder.h" // upstream declaration of whisper_coreml_init (not renamed here)
#endif

namespace kggml::whisper::coreml {

namespace {
thread_local bool t_allowed = false;
thread_local bool t_loaded = false;
} // namespace

Scope::Scope(bool allowed) noexcept {
    t_allowed = allowed;
    t_loaded = false;
}

Scope::~Scope() {
    t_allowed = false;
}

bool Scope::loaded() const noexcept {
    return t_loaded;
}

#if defined(KGGML_WHISPER_COREML)
namespace detail {
bool allowed() noexcept { return t_allowed; }
void mark_loaded() noexcept { t_loaded = true; }
} // namespace detail
#endif

} // namespace kggml::whisper::coreml

#if defined(KGGML_WHISPER_COREML)
// The upstream loader, compiled from whisper-encoder.mm with -Dwhisper_coreml_init=kggml_upstream_whisper_coreml_init.
extern "C" struct whisper_coreml_context * kggml_upstream_whisper_coreml_init(const char * path_model);

// Called by whisper.cpp's whisper_init_state(). Returning NULL makes it fall back to the ggml encoder
// (whisper.cpp is compiled with WHISPER_COREML_ALLOW_FALLBACK).
extern "C" struct whisper_coreml_context * whisper_coreml_init(const char * path_model) {
    if (!kggml::whisper::coreml::detail::allowed()) {
        return nullptr;
    }
    struct whisper_coreml_context * ctx = kggml_upstream_whisper_coreml_init(path_model);
    if (ctx != nullptr) {
        kggml::whisper::coreml::detail::mark_loaded();
    }
    return ctx;
}
#endif
