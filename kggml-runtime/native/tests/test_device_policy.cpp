// Vulkan device policy: which GPUs the runtime offers (device_policy.hpp).
#include "device_policy.hpp"

#include <gtest/gtest.h>

using kggml::runtime::vulkan_device_allowed;

namespace {

TEST(DevicePolicyTest, CommonMobileAndDesktopGpusAreAllowed) {
    for (const char * name : {"Adreno (TM) 750", "Mali-G715", "Mali-G925-Immortalis MC12", "Samsung Xclipse 940",
                              "NVIDIA GeForce RTX 4070", "AMD Radeon RX 7800 XT (RADV NAVI32)", "llvmpipe (LLVM 17.0.6, 256 bits)",
                              "Intel(R) UHD Graphics 770"}) {
        EXPECT_TRUE(vulkan_device_allowed(name, nullptr)) << name;
    }
}

TEST(DevicePolicyTest, PowerVrIsExcludedByDefault) {
    for (const char * name : {"PowerVR D-Series DXT-48-1536", "PowerVR B-Series BXM-8-256", "powervr rogue GE8320",
                              "Imagination Technologies PowerVR", "IMAGINATION DXT"}) {
        EXPECT_FALSE(vulkan_device_allowed(name, nullptr)) << name;
        EXPECT_FALSE(vulkan_device_allowed(name, "")) << name;
        EXPECT_FALSE(vulkan_device_allowed(name, "0")) << name;
        EXPECT_FALSE(vulkan_device_allowed(name, "yes")) << name;
    }
}

TEST(DevicePolicyTest, OptInKeepsPowerVr) {
    EXPECT_TRUE(vulkan_device_allowed("PowerVR D-Series DXT-48-1536", "1"));
}

TEST(DevicePolicyTest, MissingDescriptionIsAllowed) {
    EXPECT_TRUE(vulkan_device_allowed(nullptr, nullptr));
    EXPECT_TRUE(vulkan_device_allowed("", nullptr));
}

} // namespace
