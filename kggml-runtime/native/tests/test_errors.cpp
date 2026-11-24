// Status codes, thread-local last error, and C++ exception -> kggml_status translation at the FFI boundary.
#include "kggml_runtime.h"

#if defined(KGGML_RUNTIME_TEST_HOOKS)
#    include "kggml_runtime_testing.h"
#endif

#include <gtest/gtest.h>

#include <cstring>
#include <ostream>
#include <string>
#include <thread>

namespace {

// The numeric contract with com.donfuldauer.kggml.core.NativeErrorCode (kggml-core-api).
static_assert(KGGML_STATUS_OK == 0, "NativeErrorCode.OK");
static_assert(KGGML_STATUS_MODEL_LOAD_FAILED == 1, "NativeErrorCode.MODEL_LOAD_FAILED");
static_assert(KGGML_STATUS_INVALID_ARGUMENT == 2, "NativeErrorCode.INVALID_INPUT");
static_assert(KGGML_STATUS_BACKEND_UNAVAILABLE == 3, "NativeErrorCode.BACKEND_UNAVAILABLE");
static_assert(KGGML_STATUS_OUT_OF_MEMORY == 4, "NativeErrorCode.OUT_OF_MEMORY");
static_assert(KGGML_STATUS_CANCELLED == 5, "NativeErrorCode.CANCELLED");
static_assert(KGGML_STATUS_NATIVE_FAILURE == 6, "NativeErrorCode.NATIVE_FAILURE");
// Runtime-only codes must stay clear of the shared 0..99 range (Kotlin maps them to NativeFailureException).
static_assert(KGGML_STATUS_NOT_INITIALIZED >= 100, "runtime-specific range");
static_assert(KGGML_STATUS_BUFFER_TOO_SMALL >= 100, "runtime-specific range");
static_assert(KGGML_STATUS_UNKNOWN >= 100, "runtime-specific range");
static_assert(sizeof(kggml_status) == sizeof(int32_t), "kggml_status must be a 32-bit int on the ABI");

bool contains(const char * haystack, const char * needle) {
    return haystack != nullptr && std::strstr(haystack, needle) != nullptr;
}

TEST(StatusTest, EveryStatusHasAStableName) {
    const struct {
        kggml_status status;
        const char * name;
    } cases[] = {
        {KGGML_STATUS_OK, "KGGML_STATUS_OK"},
        {KGGML_STATUS_MODEL_LOAD_FAILED, "KGGML_STATUS_MODEL_LOAD_FAILED"},
        {KGGML_STATUS_INVALID_ARGUMENT, "KGGML_STATUS_INVALID_ARGUMENT"},
        {KGGML_STATUS_BACKEND_UNAVAILABLE, "KGGML_STATUS_BACKEND_UNAVAILABLE"},
        {KGGML_STATUS_OUT_OF_MEMORY, "KGGML_STATUS_OUT_OF_MEMORY"},
        {KGGML_STATUS_CANCELLED, "KGGML_STATUS_CANCELLED"},
        {KGGML_STATUS_NATIVE_FAILURE, "KGGML_STATUS_NATIVE_FAILURE"},
        {KGGML_STATUS_NOT_INITIALIZED, "KGGML_STATUS_NOT_INITIALIZED"},
        {KGGML_STATUS_BUFFER_TOO_SMALL, "KGGML_STATUS_BUFFER_TOO_SMALL"},
        {KGGML_STATUS_UNKNOWN, "KGGML_STATUS_UNKNOWN"},
    };
    for (const auto & c : cases) {
        EXPECT_STREQ(kggml_status_name(c.status), c.name);
    }
    EXPECT_STREQ(kggml_status_name(static_cast<kggml_status>(-7)), "KGGML_STATUS_<invalid>");
}

// --- last error -------------------------------------------------------------------------------------------

TEST(LastErrorTest, FailureSetsMessageAndSuccessClearsIt) {
    EXPECT_EQ(kggml_runtime_performance_thread_count(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_TRUE(contains(kggml_runtime_last_error(), "kggml_runtime_performance_thread_count"));

    int32_t n = 0;
    EXPECT_EQ(kggml_runtime_performance_thread_count(&n), KGGML_STATUS_OK);
    EXPECT_STREQ(kggml_runtime_last_error(), "") << "a successful guarded call clears the last error";
}

TEST(LastErrorTest, ClearLastError) {
    EXPECT_EQ(kggml_runtime_performance_thread_count(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    ASSERT_STRNE(kggml_runtime_last_error(), "");
    kggml_runtime_clear_last_error();
    EXPECT_STREQ(kggml_runtime_last_error(), "");
}

TEST(LastErrorTest, IsThreadLocal) {
    EXPECT_EQ(kggml_runtime_performance_thread_count(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    const std::string mine = kggml_runtime_last_error();
    ASSERT_FALSE(mine.empty());

    std::string other_before;
    std::string other_after;
    std::thread([&] {
        other_before = kggml_runtime_last_error();
        int32_t available = 0;
        (void) kggml_runtime_backend_available(-5, &available); // INVALID_ARGUMENT on the other thread
        other_after = kggml_runtime_last_error();
    }).join();

    EXPECT_EQ(other_before, "") << "a fresh thread starts without an error";
    EXPECT_TRUE(contains(other_after.c_str(), "kggml_runtime_backend_available"));
    EXPECT_EQ(std::string(kggml_runtime_last_error()), mine) << "other threads must not clobber ours";
}

TEST(LastErrorTest, CopyLastErrorVariants) {
    EXPECT_EQ(kggml_runtime_performance_thread_count(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    const std::string expected = kggml_runtime_last_error();
    ASSERT_FALSE(expected.empty());

    // Size query.
    size_t len = 0;
    EXPECT_EQ(kggml_runtime_copy_last_error(nullptr, 0, &len), KGGML_STATUS_BUFFER_TOO_SMALL);
    EXPECT_EQ(len, expected.size());

    // NULL buffer with a non-zero capacity is a caller bug.
    EXPECT_EQ(kggml_runtime_copy_last_error(nullptr, 16, &len), KGGML_STATUS_INVALID_ARGUMENT);

    // Exact fit (needs room for the terminator).
    std::string buf(expected.size() + 1, 'x');
    EXPECT_EQ(kggml_runtime_copy_last_error(buf.data(), buf.size(), &len), KGGML_STATUS_OK);
    EXPECT_STREQ(buf.c_str(), expected.c_str());

    // One byte short: truncated, still terminated.
    std::string small(expected.size(), 'x');
    EXPECT_EQ(kggml_runtime_copy_last_error(small.data(), small.size(), nullptr), KGGML_STATUS_BUFFER_TOO_SMALL);
    EXPECT_EQ(small[small.size() - 1], '\0');
    EXPECT_EQ(std::string(small.c_str()), expected.substr(0, expected.size() - 1));

    // Copying never clears or overwrites the message.
    EXPECT_EQ(std::string(kggml_runtime_last_error()), expected);
}

TEST(LastErrorTest, CopyEmptyMessage) {
    kggml_runtime_clear_last_error();
    char buf[4] = {'x', 'x', 'x', 'x'};
    size_t len = 99;
    EXPECT_EQ(kggml_runtime_copy_last_error(buf, sizeof(buf), &len), KGGML_STATUS_OK);
    EXPECT_EQ(len, 0u);
    EXPECT_EQ(buf[0], '\0');
}

// --- exception -> status translation ----------------------------------------------------------------------

#if defined(KGGML_RUNTIME_TEST_HOOKS)

struct FaultCase {
    int32_t fault;
    kggml_status expected;
    const char * message_fragment;
};

// Readable ctest names instead of gtest's raw byte dump of the parameter.
void PrintTo(const FaultCase & c, std::ostream * os) {
    *os << "fault=" << c.fault << " expected=" << kggml_status_name(c.expected);
}

class ExceptionTranslationTest : public ::testing::TestWithParam<FaultCase> {
protected:
    void SetUp() override {
        while (kggml_runtime_ref_count() > 0) {
            ASSERT_EQ(kggml_runtime_shutdown(), KGGML_STATUS_OK);
        }
    }
    void TearDown() override {
        kggml_runtime_test_inject_fault(KGGML_TEST_FAULT_NONE);
        while (kggml_runtime_ref_count() > 0) {
            ASSERT_EQ(kggml_runtime_shutdown(), KGGML_STATUS_OK);
        }
    }
};

TEST_P(ExceptionTranslationTest, InitTranslatesAndLeavesRuntimeUninitialized) {
    const FaultCase c = GetParam();
    kggml_runtime_test_inject_fault(c.fault);
    EXPECT_EQ(kggml_runtime_init(), c.expected);
    EXPECT_TRUE(contains(kggml_runtime_last_error(), "kggml_runtime_init")) << kggml_runtime_last_error();
    EXPECT_TRUE(contains(kggml_runtime_last_error(), c.message_fragment)) << kggml_runtime_last_error();
    EXPECT_EQ(kggml_runtime_ref_count(), 0) << "a failed init must not leak a reference";

    // The fault is one-shot: the next call succeeds and clears the error.
    EXPECT_EQ(kggml_runtime_init(), KGGML_STATUS_OK);
    EXPECT_STREQ(kggml_runtime_last_error(), "");
    EXPECT_EQ(kggml_runtime_ref_count(), 1);
}

TEST_P(ExceptionTranslationTest, ShutdownTranslatesAndKeepsReference) {
    const FaultCase c = GetParam();
    ASSERT_EQ(kggml_runtime_init(), KGGML_STATUS_OK);
    kggml_runtime_test_inject_fault(c.fault);
    EXPECT_EQ(kggml_runtime_shutdown(), c.expected);
    EXPECT_TRUE(contains(kggml_runtime_last_error(), "kggml_runtime_shutdown"));
    EXPECT_EQ(kggml_runtime_ref_count(), 1);
}

TEST_P(ExceptionTranslationTest, QueryEntryPointsTranslate) {
    const FaultCase c = GetParam();
    ASSERT_EQ(kggml_runtime_init(), KGGML_STATUS_OK);

    kggml_backend_info infos[KGGML_BACKEND_KIND_COUNT];
    int32_t count = -1;
    kggml_runtime_test_inject_fault(c.fault);
    EXPECT_EQ(kggml_runtime_get_backends(infos, KGGML_BACKEND_KIND_COUNT, &count), c.expected);
    EXPECT_EQ(count, -1);
    EXPECT_TRUE(contains(kggml_runtime_last_error(), "kggml_runtime_get_backends"));

    int32_t available = -1;
    kggml_runtime_test_inject_fault(c.fault);
    EXPECT_EQ(kggml_runtime_backend_available(KGGML_BACKEND_CPU, &available), c.expected);
    EXPECT_EQ(available, -1);

    int32_t threads = -1;
    kggml_runtime_test_inject_fault(c.fault);
    EXPECT_EQ(kggml_runtime_performance_thread_count(&threads), c.expected);
    EXPECT_EQ(threads, -1);
    EXPECT_TRUE(contains(kggml_runtime_last_error(), c.message_fragment));
}

TEST_P(ExceptionTranslationTest, FaultIsThreadLocal) {
    const FaultCase c = GetParam();
    kggml_runtime_test_inject_fault(c.fault);
    int32_t threads = 0;
    kggml_status other = KGGML_STATUS_UNKNOWN;
    std::thread([&] { other = kggml_runtime_performance_thread_count(&threads); }).join();
    EXPECT_EQ(other, KGGML_STATUS_OK) << "a fault armed on one thread must not fire on another";
    EXPECT_EQ(kggml_runtime_performance_thread_count(&threads), c.expected);
}

INSTANTIATE_TEST_SUITE_P(
    AllExceptionKinds, ExceptionTranslationTest,
    ::testing::Values(
        FaultCase{KGGML_TEST_FAULT_BAD_ALLOC, KGGML_STATUS_OUT_OF_MEMORY, "out of memory"},
        FaultCase{KGGML_TEST_FAULT_BAD_ARRAY_NEW, KGGML_STATUS_OUT_OF_MEMORY, "out of memory"},
        FaultCase{KGGML_TEST_FAULT_RUNTIME_ERROR, KGGML_STATUS_NATIVE_FAILURE, "injected std::runtime_error"},
        FaultCase{KGGML_TEST_FAULT_LOGIC_ERROR, KGGML_STATUS_NATIVE_FAILURE, "injected std::invalid_argument"},
        FaultCase{KGGML_TEST_FAULT_NON_STD, KGGML_STATUS_UNKNOWN, "unknown non-standard C++ exception"}),
    [](const ::testing::TestParamInfo<FaultCase> & info) {
        switch (info.param.fault) {
            case KGGML_TEST_FAULT_BAD_ALLOC:     return std::string("BadAlloc");
            case KGGML_TEST_FAULT_BAD_ARRAY_NEW: return std::string("BadArrayNewLength");
            case KGGML_TEST_FAULT_RUNTIME_ERROR: return std::string("RuntimeError");
            case KGGML_TEST_FAULT_LOGIC_ERROR:   return std::string("InvalidArgument");
            default:                             return std::string("NonStdException");
        }
    });

TEST(ExceptionTranslationNoFault, NoneFaultIsHarmless) {
    kggml_runtime_test_inject_fault(KGGML_TEST_FAULT_NONE);
    int32_t threads = 0;
    EXPECT_EQ(kggml_runtime_performance_thread_count(&threads), KGGML_STATUS_OK);
}

#else

TEST(ExceptionTranslationTest, RequiresTestHooks) {
    GTEST_SKIP() << "configure with -DKGGML_RUNTIME_TEST_HOOKS=ON to run exception translation tests";
}

#endif

} // namespace
