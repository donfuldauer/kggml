// coreml_gate.hpp - INTERNAL runtime gate in front of whisper.cpp's CoreML encoder loader.
//
// whisper.cpp built with WHISPER_COREML always tries to load "<model>-encoder.mlmodelc" from
// whisper_init_state(). Kggml must only do that when the caller asked for KGGML_BACKEND_COREML (CPU_ONLY
// must never touch the ANE/GPU, e.g. in the iOS background). The upstream loader is compiled renamed to
// kggml_upstream_whisper_coreml_init and coreml_gate.cpp provides whisper_coreml_init(), which consults the
// calling thread's gate (set around whisper_init_state()) and records whether the encoder actually loaded.
#pragma once

namespace kggml::whisper::coreml {

/// RAII: allows / forbids CoreML for whisper_init_state() calls made by this thread inside the scope.
class Scope {
public:
    explicit Scope(bool allowed) noexcept;
    ~Scope();
    Scope(const Scope &) = delete;
    Scope & operator=(const Scope &) = delete;

    /// True if whisper.cpp successfully loaded a CoreML encoder inside this scope.
    bool loaded() const noexcept;
};

} // namespace kggml::whisper::coreml
