// Which GPU devices the runtime offers for each backend (internal; not part of the C ABI).
#pragma once

namespace kggml::runtime {

/// Environment variable that opts PowerVR GPUs back in for Vulkan ("1" = allow). See vulkan_device_allowed.
inline constexpr const char * kAllowPowerVrEnv = "KGGML_VULKAN_ALLOW_POWERVR";

/// Whether a Vulkan device with this ggml description (the driver's device name, e.g. "Adreno (TM) 750" or
/// "PowerVR D-Series DXT-48-1536") counts as usable.
///
/// Imagination PowerVR GPUs (Google Tensor G5 and later, some MediaTek parts) are excluded: their Vulkan drivers run
/// ggml's compute shaders incorrectly (wrong k-quant results reported upstream) and can hang the device queue, so a
/// model load or a generation on them may never finish. Excluded devices make Vulkan unavailable, AUTO falls back to the
/// CPU, and an explicit GPU_VULKAN falls back (or fails when fallback is off). `allow_env` is the value of
/// kAllowPowerVrEnv (nullptr when unset); "1" keeps them, for testing new drivers.
bool vulkan_device_allowed(const char * description, const char * allow_env) noexcept;

} // namespace kggml::runtime
