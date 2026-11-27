#include "device_policy.hpp"

#include <cctype>
#include <cstring>

namespace kggml::runtime {

namespace {

bool contains_ignore_case(const char * haystack, const char * needle) noexcept {
    const size_t n = std::strlen(needle);
    for (const char * p = haystack; *p != '\0'; ++p) {
        size_t i = 0;
        while (i < n && p[i] != '\0' &&
               std::tolower(static_cast<unsigned char>(p[i])) == std::tolower(static_cast<unsigned char>(needle[i]))) {
            ++i;
        }
        if (i == n) return true;
    }
    return false;
}

} // namespace

bool vulkan_device_allowed(const char * description, const char * allow_env) noexcept {
    if (description == nullptr) return true;
    const bool powervr = contains_ignore_case(description, "PowerVR") || contains_ignore_case(description, "Imagination");
    if (!powervr) return true;
    return allow_env != nullptr && std::strcmp(allow_env, "1") == 0;
}

} // namespace kggml::runtime
