// kggml_runtime_load_backend: registering ggml backend modules (the Android GPU add-ons) at runtime.
#include "kggml_runtime.h"

#include "ggml-backend.h"

#include <gtest/gtest.h>

#include <string>

namespace {

std::string last_error() { return kggml_runtime_last_error(); }

TEST(BackendModulesTest, NullOrEmptyPathIsRejected) {
    EXPECT_EQ(kggml_runtime_load_backend(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_runtime_load_backend(""), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(last_error().find("path is NULL or empty"), std::string::npos);
}

TEST(BackendModulesTest, MissingModuleIsUnavailableWithItsPath) {
    const char * path = "/nonexistent/kggml/libggml-vulkan.so";
    EXPECT_EQ(kggml_runtime_load_backend(path), KGGML_STATUS_BACKEND_UNAVAILABLE);
    EXPECT_NE(last_error().find(path), std::string::npos) << last_error();
}

TEST(BackendModulesTest, ModuleWithAnotherApiVersionIsRefused) {
    EXPECT_EQ(kggml_runtime_load_backend(KGGML_TEST_BAD_BACKEND_MODULE), KGGML_STATUS_BACKEND_UNAVAILABLE);
    EXPECT_EQ(ggml_backend_reg_by_name("KggmlFakeBad"), nullptr);
}

TEST(BackendModulesTest, ModuleIsRegisteredOnceAndVisibleToTheRegistry) {
    ASSERT_EQ(ggml_backend_reg_by_name("KggmlFake"), nullptr);
    ASSERT_EQ(kggml_runtime_load_backend(KGGML_TEST_BACKEND_MODULE), KGGML_STATUS_OK) << last_error();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("KggmlFake");
    ASSERT_NE(reg, nullptr);
    EXPECT_EQ(ggml_backend_reg_dev_count(reg), 0u);

    // A second load of the same module is a no-op: no duplicate registry entry.
    const size_t regs = ggml_backend_reg_count();
    EXPECT_EQ(kggml_runtime_load_backend(KGGML_TEST_BACKEND_MODULE), KGGML_STATUS_OK);
    EXPECT_EQ(ggml_backend_reg_count(), regs);
}

TEST(BackendModulesTest, LoadingAfterInitKeepsTheBackendListConsistent) {
    ASSERT_EQ(kggml_runtime_init(), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_runtime_load_backend(KGGML_TEST_BACKEND_MODULE), KGGML_STATUS_OK);
    kggml_backend_info infos[KGGML_BACKEND_KIND_COUNT];
    int32_t count = 0;
    ASSERT_EQ(kggml_runtime_get_backends(infos, KGGML_BACKEND_KIND_COUNT, &count), KGGML_STATUS_OK);
    EXPECT_EQ(infos[KGGML_BACKEND_CPU].available, 1);
    // A module that is not Vulkan never makes Vulkan appear.
    EXPECT_EQ(infos[KGGML_BACKEND_VULKAN].compiled, KGGML_TEST_EXPECT_VULKAN ? 1 : 0);
    EXPECT_EQ(kggml_runtime_shutdown(), KGGML_STATUS_OK);
}

} // namespace
