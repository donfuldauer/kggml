// kggml_runtime.cpp - C ABI implementation of the Kggml shared ggml runtime.
//
// Rule: every extern "C" function either cannot throw (trivial accessors marked noexcept) or runs its body
// through kggml::detail::guarded(), so no C++ exception ever crosses the JNI / Kotlin/Native boundary.

#include "kggml_runtime.h"

#include "cpu_topology.hpp"
#include "guard.hpp"

#if defined(KGGML_RUNTIME_TEST_HOOKS)
#    include "kggml_runtime_testing.h"
#endif

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef KGGML_RUNTIME_VERSION_STRING
#    error "KGGML_RUNTIME_VERSION_STRING must be defined by the build"
#endif
#ifndef KGGML_GGML_PINNED_COMMIT
#    error "KGGML_GGML_PINNED_COMMIT must be defined by the build"
#endif

namespace kggml::detail {

// ---------------------------------------------------------------------------------------------------------
// Last error
// ---------------------------------------------------------------------------------------------------------

std::string & last_error_storage() noexcept {
    static thread_local std::string message;
    return message;
}

void set_last_error(const char * message) noexcept {
    try {
        last_error_storage().assign(message != nullptr ? message : "");
    } catch (...) {
        last_error_storage().clear(); // OOM while recording the error: keep the status code, drop the text
    }
}

void set_last_error(const char * where, const char * prefix, const char * detail) noexcept {
    try {
        std::string & out = last_error_storage();
        out.clear();
        if (where != nullptr) {
            out.append(where).append(": ");
        }
        if (prefix != nullptr) {
            out.append(prefix);
        }
        out.append(detail != nullptr ? detail : "(no detail)");
    } catch (...) {
        last_error_storage().clear();
    }
}

void clear_last_error() noexcept {
    last_error_storage().clear();
}

// ---------------------------------------------------------------------------------------------------------
// Test hooks (compiled only with KGGML_RUNTIME_TEST_HOOKS)
// ---------------------------------------------------------------------------------------------------------

#if defined(KGGML_RUNTIME_TEST_HOOKS)
namespace {
thread_local int32_t g_injected_fault = KGGML_TEST_FAULT_NONE;
} // namespace

void maybe_throw_injected_fault() {
    const int32_t fault = g_injected_fault;
    g_injected_fault = KGGML_TEST_FAULT_NONE;
    switch (fault) {
        case KGGML_TEST_FAULT_BAD_ALLOC:     throw std::bad_alloc();
        case KGGML_TEST_FAULT_BAD_ARRAY_NEW: throw std::bad_array_new_length();
        case KGGML_TEST_FAULT_RUNTIME_ERROR: throw std::runtime_error("injected std::runtime_error");
        case KGGML_TEST_FAULT_LOGIC_ERROR:   throw std::invalid_argument("injected std::invalid_argument");
        case KGGML_TEST_FAULT_NON_STD:       throw 42;
        default:                             return;
    }
}
#endif

// ---------------------------------------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------------------------------------

namespace {

struct runtime_state {
    std::mutex mutex;
    int32_t ref_count = 0;
    std::array<kggml_backend_info, KGGML_BACKEND_KIND_COUNT> backends{};
    std::vector<std::string> loaded_modules; // backend modules registered by kggml_runtime_load_backend
};

runtime_state & state() noexcept {
    static runtime_state s; // function-local static: no static-init-order issues, never destroyed early
    return s;
}

void fill_info(kggml_backend_info & info, kggml_backend_kind kind, const char * name, bool compiled,
               int32_t device_count) noexcept {
    std::memset(&info, 0, sizeof(info));
    info.kind         = kind;
    info.compiled     = compiled ? 1 : 0;
    info.device_count = device_count < 0 ? 0 : device_count;
    info.available    = (compiled && info.device_count > 0) ? 1 : 0;
    const size_t len = std::min(std::strlen(name), static_cast<size_t>(KGGML_BACKEND_NAME_MAX - 1));
    std::memcpy(info.name, name, len); // memset above already NUL-terminated the buffer
}

// Devices registered by the ggml backend registry (libggml) under `reg_name` ("CPU", "MTL", "Vulkan").
// A missing registry entry (e.g. no Vulkan loader / driver at runtime) simply yields 0 devices.
int32_t registry_device_count(const char * reg_name) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name(reg_name);
    return reg == nullptr ? 0 : static_cast<int32_t>(ggml_backend_reg_dev_count(reg));
}

// Probes one optional GPU backend; a failing driver means "unavailable", never an error.
[[maybe_unused]] int32_t probe_devices(const char * reg_name) noexcept {
    try {
        return registry_device_count(reg_name);
    } catch (...) {
        return 0;
    }
}

void probe_backends(std::array<kggml_backend_info, KGGML_BACKEND_KIND_COUNT> & out) {
    fill_info(out[KGGML_BACKEND_CPU], KGGML_BACKEND_CPU, "CPU", true, registry_device_count("CPU"));
    // The CPU backend is mandatory: a runtime without a CPU device is a broken build.
    if (out[KGGML_BACKEND_CPU].device_count < 1) {
        throw std::runtime_error("ggml CPU backend registered no devices");
    }

    // GGML_USE_<BACKEND> is a PUBLIC compile definition of libggml when GGML_BACKEND_DL is OFF.
#if defined(GGML_USE_METAL)
    fill_info(out[KGGML_BACKEND_METAL], KGGML_BACKEND_METAL, "Metal", true, probe_devices("MTL"));
#else
    fill_info(out[KGGML_BACKEND_METAL], KGGML_BACKEND_METAL, "Metal", false, 0);
#endif

    // Vulkan is either linked in (GGML_USE_VULKAN, desktop builds with KGGML_ENABLE_VULKAN) or registered at runtime
    // from a backend module (the Android kggml-runtime-vulkan add-on, kggml_runtime_load_backend): ask the registry.
    const bool vulkan_registered = ggml_backend_reg_by_name("Vulkan") != nullptr;
    fill_info(out[KGGML_BACKEND_VULKAN], KGGML_BACKEND_VULKAN, "Vulkan", vulkan_registered,
              vulkan_registered ? probe_devices("Vulkan") : 0);

    // OpenCL is only ever a registered backend module (the Android kggml-runtime-opencl add-on).
    const bool opencl_registered = ggml_backend_reg_by_name("OpenCL") != nullptr;
    fill_info(out[KGGML_BACKEND_OPENCL], KGGML_BACKEND_OPENCL, "OpenCL", opencl_registered,
              opencl_registered ? probe_devices("OpenCL") : 0);

    // CoreML is a compile-time capability flag; the engine (whisper.cpp encoder) owns the actual model.
#if defined(KGGML_HAS_COREML)
    fill_info(out[KGGML_BACKEND_COREML], KGGML_BACKEND_COREML, "CoreML", true, 0);
    out[KGGML_BACKEND_COREML].available = 1;
#else
    fill_info(out[KGGML_BACKEND_COREML], KGGML_BACKEND_COREML, "CoreML", false, 0);
#endif
}

} // namespace
} // namespace kggml::detail

using namespace kggml::detail;

// =========================================================================================================
// Public C ABI
// =========================================================================================================

extern "C" {

const char * kggml_status_name(kggml_status status) {
    switch (status) {
        case KGGML_STATUS_OK:                  return "KGGML_STATUS_OK";
        case KGGML_STATUS_MODEL_LOAD_FAILED:   return "KGGML_STATUS_MODEL_LOAD_FAILED";
        case KGGML_STATUS_INVALID_ARGUMENT:    return "KGGML_STATUS_INVALID_ARGUMENT";
        case KGGML_STATUS_BACKEND_UNAVAILABLE: return "KGGML_STATUS_BACKEND_UNAVAILABLE";
        case KGGML_STATUS_OUT_OF_MEMORY:       return "KGGML_STATUS_OUT_OF_MEMORY";
        case KGGML_STATUS_CANCELLED:           return "KGGML_STATUS_CANCELLED";
        case KGGML_STATUS_NATIVE_FAILURE:      return "KGGML_STATUS_NATIVE_FAILURE";
        case KGGML_STATUS_NOT_INITIALIZED:     return "KGGML_STATUS_NOT_INITIALIZED";
        case KGGML_STATUS_BUFFER_TOO_SMALL:    return "KGGML_STATUS_BUFFER_TOO_SMALL";
        case KGGML_STATUS_UNKNOWN:             return "KGGML_STATUS_UNKNOWN";
    }
    return "KGGML_STATUS_<invalid>";
}

// ----- lifecycle ------------------------------------------------------------------------------------------

kggml_status kggml_runtime_init(void) {
    return guarded("kggml_runtime_init", []() -> kggml_status {
        runtime_state & s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.ref_count == 0) {
            ggml_cpu_init(); // fp16 lookup tables; idempotent inside ggml
            std::array<kggml_backend_info, KGGML_BACKEND_KIND_COUNT> probed{};
            probe_backends(probed); // may throw: ref count stays 0 on failure
            s.backends = probed;
        }
        if (s.ref_count == INT32_MAX) {
            return fail(KGGML_STATUS_NATIVE_FAILURE, "kggml_runtime_init: reference count overflow");
        }
        ++s.ref_count;
        return KGGML_STATUS_OK;
    });
}

kggml_status kggml_runtime_shutdown(void) {
    return guarded("kggml_runtime_shutdown", []() -> kggml_status {
        runtime_state & s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.ref_count == 0) {
            return KGGML_STATUS_OK; // idempotent
        }
        if (--s.ref_count == 0) {
            s.backends = {}; // ggml's backend registry is process-static; nothing else to release
        }
        return KGGML_STATUS_OK;
    });
}

int32_t kggml_runtime_is_initialized(void) {
    return kggml_runtime_ref_count() > 0 ? 1 : 0;
}

int32_t kggml_runtime_ref_count(void) {
    try {
        runtime_state & s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        return s.ref_count;
    } catch (...) { // std::mutex::lock may throw std::system_error; never let it cross the FFI boundary
        return 0;
    }
}

// ----- backends -------------------------------------------------------------------------------------------

kggml_status kggml_runtime_load_backend(const char * path) {
    return guarded("kggml_runtime_load_backend", [&]() -> kggml_status {
        if (path == nullptr || *path == '\0') {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "kggml_runtime_load_backend: path is NULL or empty");
        }
        runtime_state & s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const std::string module(path);
        if (std::find(s.loaded_modules.begin(), s.loaded_modules.end(), module) != s.loaded_modules.end()) {
            return KGGML_STATUS_OK; // registered once per process; a second registration would duplicate its devices
        }
        if (ggml_backend_load(path) == nullptr) {
            const std::string message = "kggml_runtime_load_backend: could not load ggml backend module '" + module +
                                        "' (missing, wrong ABI, or built against another ggml)";
            return fail(KGGML_STATUS_BACKEND_UNAVAILABLE, message.c_str());
        }
        s.loaded_modules.push_back(module);
        if (s.ref_count > 0) {
            probe_backends(s.backends); // already initialized: report the new backend right away
        }
        return KGGML_STATUS_OK;
    });
}

kggml_status kggml_runtime_get_backends(kggml_backend_info * out, int32_t capacity, int32_t * out_count) {
    return guarded("kggml_runtime_get_backends", [&]() -> kggml_status {
        if (out_count == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "kggml_runtime_get_backends: out_count is NULL");
        }
        if (capacity < 0) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "kggml_runtime_get_backends: capacity is negative");
        }
        if (out == nullptr && capacity > 0) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "kggml_runtime_get_backends: out is NULL but capacity > 0");
        }
        runtime_state & s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.ref_count == 0) {
            return fail(KGGML_STATUS_NOT_INITIALIZED, "kggml_runtime_get_backends: runtime not initialized");
        }
        *out_count = KGGML_BACKEND_KIND_COUNT;
        if (capacity < KGGML_BACKEND_KIND_COUNT) {
            return fail(KGGML_STATUS_BUFFER_TOO_SMALL,
                        "kggml_runtime_get_backends: capacity smaller than KGGML_BACKEND_KIND_COUNT");
        }
        for (int32_t i = 0; i < KGGML_BACKEND_KIND_COUNT; ++i) {
            out[i] = s.backends[static_cast<size_t>(i)];
        }
        return KGGML_STATUS_OK;
    });
}

kggml_status kggml_runtime_backend_available(int32_t kind, int32_t * out_available) {
    return guarded("kggml_runtime_backend_available", [&]() -> kggml_status {
        if (out_available == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "kggml_runtime_backend_available: out_available is NULL");
        }
        if (kind < 0 || kind >= KGGML_BACKEND_KIND_COUNT) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "kggml_runtime_backend_available: unknown backend kind");
        }
        runtime_state & s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (s.ref_count == 0) {
            return fail(KGGML_STATUS_NOT_INITIALIZED, "kggml_runtime_backend_available: runtime not initialized");
        }
        *out_available = s.backends[static_cast<size_t>(kind)].available;
        return KGGML_STATUS_OK;
    });
}

int32_t kggml_runtime_has_coreml(void) {
#if defined(KGGML_HAS_COREML)
    return 1;
#else
    return 0;
#endif
}

// ----- CPU topology ---------------------------------------------------------------------------------------

kggml_status kggml_runtime_performance_thread_count(int32_t * out_count) {
    return guarded("kggml_runtime_performance_thread_count", [&]() -> kggml_status {
        if (out_count == nullptr) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "kggml_runtime_performance_thread_count: out_count is NULL");
        }
        *out_count = static_cast<int32_t>(performance_cores());
        return KGGML_STATUS_OK;
    });
}

int32_t kggml_runtime_hardware_thread_count(void) {
    return static_cast<int32_t>(hardware_threads());
}

// ----- versions -------------------------------------------------------------------------------------------

const char * kggml_runtime_version(void) {
    return KGGML_RUNTIME_VERSION_STRING;
}

const char * kggml_runtime_ggml_version(void) {
    return ggml_version();
}

const char * kggml_runtime_ggml_commit(void) {
    return KGGML_GGML_PINNED_COMMIT;
}

const char * kggml_runtime_ggml_build_commit(void) {
    return ggml_commit();
}

// ----- errors ---------------------------------------------------------------------------------------------

const char * kggml_runtime_last_error(void) {
    return last_error_storage().c_str();
}

kggml_status kggml_runtime_copy_last_error(char * buffer, size_t capacity, size_t * out_length) {
    // Deliberately NOT routed through guarded(): it must not clear or overwrite the message it copies.
    const std::string & message = last_error_storage();
    if (out_length != nullptr) {
        *out_length = message.size();
    }
    if (buffer == nullptr && capacity > 0) {
        return KGGML_STATUS_INVALID_ARGUMENT;
    }
    if (capacity == 0) {
        return KGGML_STATUS_BUFFER_TOO_SMALL; // size query (buffer may be NULL)
    }
    if (capacity <= message.size()) {
        std::memcpy(buffer, message.data(), capacity - 1);
        buffer[capacity - 1] = '\0'; // truncated but always terminated
        return KGGML_STATUS_BUFFER_TOO_SMALL;
    }
    std::memcpy(buffer, message.c_str(), message.size() + 1);
    return KGGML_STATUS_OK;
}

void kggml_runtime_clear_last_error(void) {
    clear_last_error();
}

#if defined(KGGML_RUNTIME_TEST_HOOKS)
void kggml_runtime_test_inject_fault(int32_t fault) {
    g_injected_fault = fault;
}
#endif

} // extern "C"
