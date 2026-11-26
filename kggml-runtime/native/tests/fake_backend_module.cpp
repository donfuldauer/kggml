// fake_backend_module.cpp - a minimal ggml backend module for kggml_runtime_load_backend tests: exports
// ggml_backend_init (GGML_BACKEND_DL) returning a registry entry named KGGML_FAKE_BACKEND_NAME with no devices.
// Built twice: a valid module and one reporting a wrong api_version (KGGML_FAKE_BAD_API_VERSION).
#include "ggml-backend-impl.h"

namespace {

const char * fake_get_name(ggml_backend_reg_t) { return KGGML_FAKE_BACKEND_NAME; }
size_t fake_get_device_count(ggml_backend_reg_t) { return 0; }
ggml_backend_dev_t fake_get_device(ggml_backend_reg_t, size_t) { return nullptr; }
void * fake_get_proc_address(ggml_backend_reg_t, const char *) { return nullptr; }

ggml_backend_reg_t fake_reg() {
    static ggml_backend_reg reg = {
#if defined(KGGML_FAKE_BAD_API_VERSION)
        /* .api_version = */ GGML_BACKEND_API_VERSION + 1000,
#else
        /* .api_version = */ GGML_BACKEND_API_VERSION,
#endif
        /* .iface       = */ {fake_get_name, fake_get_device_count, fake_get_device, fake_get_proc_address},
        /* .context     = */ nullptr,
    };
    return &reg;
}

} // namespace

GGML_BACKEND_DL_IMPL(fake_reg)
