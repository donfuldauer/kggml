// cpu_topology.hpp - INTERNAL CPU topology probing for kggml_runtime_performance_thread_count().
#pragma once

#include <string>

namespace kggml::detail {

/// std::thread::hardware_concurrency(), at least 1.
int hardware_threads() noexcept;

/// Linux/Android sysfs probe rooted at `cpu_root` (normally "/sys/devices/system/cpu"):
/// number of physical cores (SMT siblings counted once) whose cpufreq/cpuinfo_max_freq equals the highest
/// cpuinfo_max_freq. Returns 0 when nothing usable could be read. Exposed for tests (fake sysfs trees).
int performance_cores_from_sysfs(const std::string & cpu_root);

/// Platform dispatch (sysfs / sysctl / GetLogicalProcessorInformationEx), clamped to [1, hardware_threads()].
int performance_cores();

} // namespace kggml::detail
