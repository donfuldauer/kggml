// Init / teardown: reference counting, idempotent shutdown, calls before init, concurrency.
#include "kggml_runtime.h"

#include <gtest/gtest.h>

#include <string>
#include <thread>
#include <vector>

namespace {

// Drives the process-wide runtime back to ref count 0 so every test starts from a clean state.
void reset_runtime() {
    while (kggml_runtime_ref_count() > 0) {
        ASSERT_EQ(kggml_runtime_shutdown(), KGGML_STATUS_OK);
    }
}

class LifecycleTest : public ::testing::Test {
protected:
    void SetUp() override { reset_runtime(); }
    void TearDown() override { reset_runtime(); }
};

TEST_F(LifecycleTest, StartsUninitialized) {
    EXPECT_EQ(kggml_runtime_ref_count(), 0);
    EXPECT_EQ(kggml_runtime_is_initialized(), 0);
}

TEST_F(LifecycleTest, InitAndShutdownAreReferenceCounted) {
    ASSERT_EQ(kggml_runtime_init(), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_runtime_ref_count(), 1);
    EXPECT_EQ(kggml_runtime_is_initialized(), 1);

    ASSERT_EQ(kggml_runtime_init(), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_runtime_ref_count(), 2);

    ASSERT_EQ(kggml_runtime_shutdown(), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_runtime_ref_count(), 1);
    EXPECT_EQ(kggml_runtime_is_initialized(), 1) << "one reference is still held";

    ASSERT_EQ(kggml_runtime_shutdown(), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_runtime_ref_count(), 0);
    EXPECT_EQ(kggml_runtime_is_initialized(), 0);
}

TEST_F(LifecycleTest, ShutdownWithoutInitIsIdempotentNoOp) {
    EXPECT_EQ(kggml_runtime_shutdown(), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_runtime_shutdown(), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_runtime_ref_count(), 0) << "ref count must never go negative";
    EXPECT_STREQ(kggml_runtime_last_error(), "");
}

TEST_F(LifecycleTest, ExtraShutdownAfterBalancedPairsStaysAtZero) {
    ASSERT_EQ(kggml_runtime_init(), KGGML_STATUS_OK);
    ASSERT_EQ(kggml_runtime_shutdown(), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_runtime_shutdown(), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_runtime_ref_count(), 0);
    // And the runtime can be brought up again afterwards.
    ASSERT_EQ(kggml_runtime_init(), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_runtime_ref_count(), 1);
}

TEST_F(LifecycleTest, CallsBeforeInitReportNotInitialized) {
    kggml_backend_info infos[KGGML_BACKEND_KIND_COUNT];
    int32_t count = -1;
    EXPECT_EQ(kggml_runtime_get_backends(infos, KGGML_BACKEND_KIND_COUNT, &count), KGGML_STATUS_NOT_INITIALIZED);
    EXPECT_NE(std::string(kggml_runtime_last_error()).find("not initialized"), std::string::npos);

    int32_t available = -1;
    EXPECT_EQ(kggml_runtime_backend_available(KGGML_BACKEND_CPU, &available), KGGML_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(available, -1) << "outputs must be untouched on failure";
}

TEST_F(LifecycleTest, CallsAfterFinalShutdownReportNotInitialized) {
    ASSERT_EQ(kggml_runtime_init(), KGGML_STATUS_OK);
    ASSERT_EQ(kggml_runtime_shutdown(), KGGML_STATUS_OK);
    int32_t available = 0;
    EXPECT_EQ(kggml_runtime_backend_available(KGGML_BACKEND_CPU, &available), KGGML_STATUS_NOT_INITIALIZED);
}

TEST_F(LifecycleTest, VersionAndTopologyQueriesWorkWithoutInit) {
    EXPECT_STRNE(kggml_runtime_version(), "");
    int32_t threads = 0;
    EXPECT_EQ(kggml_runtime_performance_thread_count(&threads), KGGML_STATUS_OK);
    EXPECT_GE(threads, 1);
}

TEST_F(LifecycleTest, ConcurrentBalancedInitShutdownEndsAtZero) {
    constexpr int kThreads = 8;
    constexpr int kIterations = 200;
    std::vector<std::thread> threads;
    std::vector<int> failures(kThreads, 0);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([t, &failures] {
            for (int i = 0; i < kIterations; ++i) {
                if (kggml_runtime_init() != KGGML_STATUS_OK) {
                    ++failures[t];
                }
                int32_t cpu = 0;
                if (kggml_runtime_backend_available(KGGML_BACKEND_CPU, &cpu) != KGGML_STATUS_OK || cpu != 1) {
                    ++failures[t];
                }
                if (kggml_runtime_shutdown() != KGGML_STATUS_OK) {
                    ++failures[t];
                }
            }
        });
    }
    for (auto & th : threads) {
        th.join();
    }
    for (int t = 0; t < kThreads; ++t) {
        EXPECT_EQ(failures[t], 0) << "thread " << t;
    }
    EXPECT_EQ(kggml_runtime_ref_count(), 0);
}

} // namespace
