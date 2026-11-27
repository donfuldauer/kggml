// Backend capability enumeration.
#include "kggml_runtime.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>

namespace {

class BackendsTest : public ::testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(kggml_runtime_init(), KGGML_STATUS_OK); }
    void TearDown() override { EXPECT_EQ(kggml_runtime_shutdown(), KGGML_STATUS_OK); }
};

TEST_F(BackendsTest, EnumeratesEveryKindInOrderWithCpuAlwaysPresent) {
    kggml_backend_info infos[KGGML_BACKEND_KIND_COUNT];
    int32_t count = 0;
    ASSERT_EQ(kggml_runtime_get_backends(infos, KGGML_BACKEND_KIND_COUNT, &count), KGGML_STATUS_OK);
    ASSERT_EQ(count, KGGML_BACKEND_KIND_COUNT);

    for (int32_t i = 0; i < count; ++i) {
        EXPECT_EQ(infos[i].kind, i);
        EXPECT_LT(std::strlen(infos[i].name), static_cast<size_t>(KGGML_BACKEND_NAME_MAX));
        EXPECT_GT(std::strlen(infos[i].name), 0u);
        if (!infos[i].compiled) {
            EXPECT_EQ(infos[i].available, 0) << infos[i].name << " cannot be available if not compiled";
        }
    }

    const kggml_backend_info & cpu = infos[KGGML_BACKEND_CPU];
    EXPECT_STREQ(cpu.name, "CPU");
    EXPECT_EQ(cpu.compiled, 1);
    EXPECT_EQ(cpu.available, 1);
    EXPECT_GE(cpu.device_count, 1);

    EXPECT_STREQ(infos[KGGML_BACKEND_METAL].name, "Metal");
    EXPECT_STREQ(infos[KGGML_BACKEND_VULKAN].name, "Vulkan");
    EXPECT_STREQ(infos[KGGML_BACKEND_COREML].name, "CoreML");
    EXPECT_STREQ(infos[KGGML_BACKEND_OPENCL].name, "OpenCL");
    // OpenCL only exists as a registered add-on module (Android); a plain runtime build never has it.
    EXPECT_EQ(infos[KGGML_BACKEND_OPENCL].compiled, 0);
    EXPECT_EQ(infos[KGGML_BACKEND_OPENCL].available, 0);
}

TEST_F(BackendsTest, CompiledFlagsMatchBuildConfiguration) {
    kggml_backend_info infos[KGGML_BACKEND_KIND_COUNT];
    int32_t count = 0;
    ASSERT_EQ(kggml_runtime_get_backends(infos, KGGML_BACKEND_KIND_COUNT, &count), KGGML_STATUS_OK);
    EXPECT_EQ(infos[KGGML_BACKEND_METAL].compiled, KGGML_TEST_EXPECT_METAL);
    EXPECT_EQ(infos[KGGML_BACKEND_VULKAN].compiled, KGGML_TEST_EXPECT_VULKAN);
    EXPECT_EQ(infos[KGGML_BACKEND_COREML].compiled, KGGML_TEST_EXPECT_COREML);
    EXPECT_EQ(kggml_runtime_has_coreml(), KGGML_TEST_EXPECT_COREML);
    // CoreML is a pure capability flag: no ggml devices, available iff compiled.
    EXPECT_EQ(infos[KGGML_BACKEND_COREML].device_count, 0);
    EXPECT_EQ(infos[KGGML_BACKEND_COREML].available, KGGML_TEST_EXPECT_COREML);
}

TEST_F(BackendsTest, SizeQueryWithNullBufferReportsRequiredCount) {
    int32_t count = -1;
    EXPECT_EQ(kggml_runtime_get_backends(nullptr, 0, &count), KGGML_STATUS_BUFFER_TOO_SMALL);
    EXPECT_EQ(count, KGGML_BACKEND_KIND_COUNT);
}

TEST_F(BackendsTest, InsufficientCapacityWritesNothing) {
    kggml_backend_info infos[KGGML_BACKEND_KIND_COUNT];
    std::memset(infos, 0xAB, sizeof(infos));
    int32_t count = -1;
    EXPECT_EQ(kggml_runtime_get_backends(infos, KGGML_BACKEND_KIND_COUNT - 1, &count),
              KGGML_STATUS_BUFFER_TOO_SMALL);
    EXPECT_EQ(count, KGGML_BACKEND_KIND_COUNT);
    EXPECT_NE(std::string(kggml_runtime_last_error()).find("capacity"), std::string::npos);

    kggml_backend_info sentinel;
    std::memset(&sentinel, 0xAB, sizeof(sentinel));
    for (const auto & info : infos) {
        EXPECT_EQ(std::memcmp(&info, &sentinel, sizeof(info)), 0) << "buffer must be untouched";
    }
}

TEST_F(BackendsTest, LargerCapacityIsAccepted) {
    kggml_backend_info infos[KGGML_BACKEND_KIND_COUNT + 3];
    int32_t count = 0;
    EXPECT_EQ(kggml_runtime_get_backends(infos, KGGML_BACKEND_KIND_COUNT + 3, &count), KGGML_STATUS_OK);
    EXPECT_EQ(count, KGGML_BACKEND_KIND_COUNT);
}

TEST_F(BackendsTest, NullAndNegativeArgumentsAreRejected) {
    kggml_backend_info infos[KGGML_BACKEND_KIND_COUNT];
    int32_t count = 0;
    EXPECT_EQ(kggml_runtime_get_backends(infos, KGGML_BACKEND_KIND_COUNT, nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::string(kggml_runtime_last_error()).find("out_count"), std::string::npos);
    EXPECT_EQ(kggml_runtime_get_backends(nullptr, 4, &count), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_runtime_get_backends(infos, -1, &count), KGGML_STATUS_INVALID_ARGUMENT);
}

TEST_F(BackendsTest, BackendAvailableMatchesEnumeration) {
    kggml_backend_info infos[KGGML_BACKEND_KIND_COUNT];
    int32_t count = 0;
    ASSERT_EQ(kggml_runtime_get_backends(infos, KGGML_BACKEND_KIND_COUNT, &count), KGGML_STATUS_OK);
    for (int32_t kind = 0; kind < KGGML_BACKEND_KIND_COUNT; ++kind) {
        int32_t available = -1;
        ASSERT_EQ(kggml_runtime_backend_available(kind, &available), KGGML_STATUS_OK);
        EXPECT_EQ(available, infos[kind].available) << infos[kind].name;
    }
}

TEST_F(BackendsTest, BackendAvailableRejectsBadArguments) {
    int32_t available = -1;
    EXPECT_EQ(kggml_runtime_backend_available(-1, &available), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_runtime_backend_available(KGGML_BACKEND_KIND_COUNT, &available), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(available, -1);
    EXPECT_EQ(kggml_runtime_backend_available(KGGML_BACKEND_CPU, nullptr), KGGML_STATUS_INVALID_ARGUMENT);
}

} // namespace
