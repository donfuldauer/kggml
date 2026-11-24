// Performance (P-) core detection: public query plus the sysfs parser against fake sysfs trees.
#include "kggml_runtime.h"

#include "cpu_topology.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace {

TEST(CpuTopologyTest, PerformanceThreadCountIsWithinHardwareBounds) {
    int32_t perf = 0;
    ASSERT_EQ(kggml_runtime_performance_thread_count(&perf), KGGML_STATUS_OK);
    const int32_t hw = kggml_runtime_hardware_thread_count();
    EXPECT_GE(hw, 1);
    EXPECT_GE(perf, 1);
    EXPECT_LE(perf, hw);
    const unsigned std_hw = std::thread::hardware_concurrency();
    if (std_hw > 0) {
        EXPECT_EQ(hw, static_cast<int32_t>(std_hw));
    }
}

TEST(CpuTopologyTest, PerformanceThreadCountIsStable) {
    int32_t a = 0;
    int32_t b = 0;
    ASSERT_EQ(kggml_runtime_performance_thread_count(&a), KGGML_STATUS_OK);
    ASSERT_EQ(kggml_runtime_performance_thread_count(&b), KGGML_STATUS_OK);
    EXPECT_EQ(a, b);
}

TEST(CpuTopologyTest, NullOutputIsRejected) {
    EXPECT_EQ(kggml_runtime_performance_thread_count(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::string(kggml_runtime_last_error()).find("out_count"), std::string::npos);
}

// --- fake sysfs -------------------------------------------------------------------------------------------

class FakeSysfs : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        root_ = fs::temp_directory_path() / ("kggml-sysfs-" + std::to_string(rd()) + "-" +
                                             ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::create_directories(root_);
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    void write(const std::string & rel, const std::string & content) {
        const fs::path p = root_ / rel;
        fs::create_directories(p.parent_path());
        std::ofstream(p) << content << "\n";
    }
    void cpu(int id, long long max_freq_khz, const std::string & siblings = "") {
        write("cpu" + std::to_string(id) + "/cpufreq/cpuinfo_max_freq", std::to_string(max_freq_khz));
        if (!siblings.empty()) {
            write("cpu" + std::to_string(id) + "/topology/thread_siblings_list", siblings);
        }
    }
    int detect() const { return kggml::detail::performance_cores_from_sysfs(root_.string()); }

    fs::path root_;
};

TEST_F(FakeSysfs, BigLittleCountsOnlyTopFrequencyCores) {
    // Typical phone: 4x little @1.8 GHz, 3x big @2.4 GHz, 1x prime @3.0 GHz.
    write("present", "0-7");
    for (int i = 0; i < 4; ++i) cpu(i, 1800000);
    for (int i = 4; i < 7; ++i) cpu(i, 2400000);
    cpu(7, 3000000);
    EXPECT_EQ(detect(), 1);
}

TEST_F(FakeSysfs, TwoPrimeCores) {
    write("present", "0-5");
    for (int i = 0; i < 4; ++i) cpu(i, 2000000);
    cpu(4, 3200000);
    cpu(5, 3200000);
    EXPECT_EQ(detect(), 2);
}

TEST_F(FakeSysfs, HomogeneousCpuCountsAllCores) {
    write("present", "0-3");
    for (int i = 0; i < 4; ++i) cpu(i, 2500000);
    EXPECT_EQ(detect(), 4);
}

TEST_F(FakeSysfs, SmtSiblingsAreCountedOnce) {
    // Hybrid x86: 2 P-cores with HT (cpu0/1, cpu2/3) + 4 E-cores.
    write("present", "0-7");
    cpu(0, 5000000, "0-1");
    cpu(1, 5000000, "0-1");
    cpu(2, 5000000, "2-3");
    cpu(3, 5000000, "2-3");
    for (int i = 4; i < 8; ++i) cpu(i, 3800000, std::to_string(i));
    EXPECT_EQ(detect(), 2);
}

TEST_F(FakeSysfs, CoreCpusListTakesPrecedence) {
    write("present", "0-1");
    cpu(0, 3000000, "0");
    cpu(1, 3000000, "1");
    write("cpu0/topology/core_cpus_list", "0-1");
    write("cpu1/topology/core_cpus_list", "0-1");
    EXPECT_EQ(detect(), 1);
}

TEST_F(FakeSysfs, SparsePresentListAndUnreadableCores) {
    // cpu3 is listed but has no cpufreq (offline / hot-unplugged): ignored, not fatal.
    write("present", "0-1,3,6-7");
    cpu(0, 1000000);
    cpu(1, 1000000);
    cpu(6, 2000000);
    cpu(7, 2000000);
    EXPECT_EQ(detect(), 2);
}

TEST_F(FakeSysfs, FallsBackToPossibleWhenPresentMissing) {
    write("possible", "0-2");
    cpu(0, 1000000);
    cpu(1, 1500000);
    cpu(2, 1500000);
    EXPECT_EQ(detect(), 2);
}

TEST_F(FakeSysfs, ProbesCpuDirectoriesWhenListsAreMalformed) {
    write("present", "garbage");
    write("possible", "3-1");
    cpu(0, 1000000);
    cpu(1, 2000000);
    EXPECT_EQ(detect(), 1);
}

TEST_F(FakeSysfs, IgnoresNonPositiveAndNonNumericFrequencies) {
    write("present", "0-2");
    cpu(0, 0);
    write("cpu1/cpufreq/cpuinfo_max_freq", "n/a");
    cpu(2, 900000);
    EXPECT_EQ(detect(), 1);
}

TEST_F(FakeSysfs, NoCpufreqDataReturnsZeroSoCallerFallsBack) {
    write("present", "0-3");
    EXPECT_EQ(detect(), 0);
}

TEST_F(FakeSysfs, MissingRootReturnsZero) {
    EXPECT_EQ(kggml::detail::performance_cores_from_sysfs((root_ / "does-not-exist").string()), 0);
}

TEST(CpuTopologyInternal, PlatformDetectionIsClamped) {
    const int perf = kggml::detail::performance_cores();
    EXPECT_GE(perf, 1);
    EXPECT_LE(perf, kggml::detail::hardware_threads());
}

} // namespace
