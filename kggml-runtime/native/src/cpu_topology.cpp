// cpu_topology.cpp - performance core detection (the default inference thread count).
#include "cpu_topology.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

namespace kggml::detail {

namespace {

bool read_first_line(const std::string & path, std::string & out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::getline(in, out);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
        out.pop_back();
    }
    return !out.empty();
}

bool read_long(const std::string & path, long long & out) {
    std::string line;
    if (!read_first_line(path, line)) {
        return false;
    }
    char * end = nullptr;
    const long long v = std::strtoll(line.c_str(), &end, 10);
    if (end == line.c_str() || v <= 0) {
        return false;
    }
    out = v;
    return true;
}

// Parses sysfs cpu lists such as "0-3,6,8-9". Returns false on malformed input.
bool parse_cpu_list(const std::string & text, std::vector<int> & out) {
    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) {
            continue;
        }
        const auto dash = part.find('-');
        char * end = nullptr;
        const long first = std::strtol(part.c_str(), &end, 10);
        if (end == part.c_str() || first < 0) {
            return false;
        }
        long last = first;
        if (dash != std::string::npos) {
            const std::string tail = part.substr(dash + 1);
            last = std::strtol(tail.c_str(), &end, 10);
            if (end == tail.c_str() || last < first) {
                return false;
            }
        }
        if (last - first > 65536) {
            return false;
        }
        for (long cpu = first; cpu <= last; ++cpu) {
            out.push_back(static_cast<int>(cpu));
        }
    }
    return !out.empty();
}

} // namespace

int hardware_threads() noexcept {
    const unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : static_cast<int>(std::min<unsigned>(n, 1u << 20));
}

int performance_cores_from_sysfs(const std::string & cpu_root) {
    std::vector<int> cpus;
    std::string list;
    if (!(read_first_line(cpu_root + "/present", list) && parse_cpu_list(list, cpus)) &&
        !(read_first_line(cpu_root + "/possible", list) && parse_cpu_list(list, cpus))) {
        cpus.clear();
        for (int i = 0; i < 1024; ++i) {
            cpus.push_back(i);
        }
    }

    struct cpu_info {
        int cpu;
        long long max_freq;  // 0 when unknown
        long long capacity;  // 0 when unknown
    };
    std::vector<cpu_info> infos;
    for (int cpu : cpus) {
        const std::string base = cpu_root + "/cpu" + std::to_string(cpu);
        long long f = 0;
        long long c = 0;
        const bool has_freq = read_long(base + "/cpufreq/cpuinfo_max_freq", f);
        const bool has_capacity = read_long(base + "/cpu_capacity", c);
        if (has_freq || has_capacity) {
            infos.push_back({cpu, has_freq ? f : 0, has_capacity ? c : 0});
        }
    }

    // Preferred signal (ARM / Android): cpu_capacity, the scheduler's per-core performance score normalised to the
    // fastest core (1024). Phones have prime, performance and efficiency clusters; the efficiency cores score far
    // below half of the prime core, the performance cores above it, so "at least half the highest capacity" keeps
    // prime + performance cores. Only used when the capacities differ (homogeneous CPUs report the same value).
    long long max_capacity = 0;
    long long min_capacity = 0;
    int with_capacity = 0;
    for (const auto & ci : infos) {
        if (ci.capacity <= 0) {
            continue;
        }
        max_capacity = std::max(max_capacity, ci.capacity);
        min_capacity = with_capacity == 0 ? ci.capacity : std::min(min_capacity, ci.capacity);
        ++with_capacity;
    }
    const bool use_capacity = with_capacity >= 2 && min_capacity != max_capacity;

    // Fallback (x86 and anything without differing capacities): cores at the highest cpuinfo_max_freq.
    long long highest = 0;
    for (const auto & ci : infos) {
        highest = std::max(highest, ci.max_freq);
    }
    if (!use_capacity && highest == 0) {
        return 0;
    }

    // Count physical cores: SMT siblings share the same core_cpus_list / thread_siblings_list.
    std::set<std::string> cores;
    for (const auto & ci : infos) {
        const bool selected = use_capacity ? ci.capacity > 0 && ci.capacity * 2 >= max_capacity
                                           : ci.max_freq > 0 && ci.max_freq == highest;
        if (!selected) {
            continue;
        }
        const std::string topo = cpu_root + "/cpu" + std::to_string(ci.cpu) + "/topology/";
        std::string siblings;
        if (!read_first_line(topo + "core_cpus_list", siblings) &&
            !read_first_line(topo + "thread_siblings_list", siblings)) {
            siblings = "cpu" + std::to_string(ci.cpu);
        }
        cores.insert(siblings);
    }
    return static_cast<int>(cores.size());
}

int performance_cores() {
    const int hw = hardware_threads();
    int detected = 0;

#if defined(__APPLE__)
    int value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &value, &size, nullptr, 0) == 0 && value > 0) {
        detected = value;
    } else {
        value = 0;
        size = sizeof(value);
        if (sysctlbyname("hw.physicalcpu", &value, &size, nullptr, 0) == 0 && value > 0) {
            detected = value;
        }
    }
#elif defined(_WIN32)
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (length > 0 && GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        std::vector<unsigned char> buffer(length);
        auto * info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length)) {
            int max_class = -1;
            int count = 0;
            for (DWORD offset = 0; offset < length;) {
                auto * item = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
                if (item->Relationship == RelationProcessorCore) {
                    // Higher EfficiencyClass == higher performance; homogeneous systems report 0 everywhere.
                    const int cls = item->Processor.EfficiencyClass;
                    if (cls > max_class) {
                        max_class = cls;
                        count = 1;
                    } else if (cls == max_class) {
                        ++count;
                    }
                }
                if (item->Size == 0) {
                    break;
                }
                offset += item->Size;
            }
            detected = count;
        }
    }
#elif defined(__linux__) || defined(__ANDROID__)
    detected = performance_cores_from_sysfs("/sys/devices/system/cpu");
#endif

    if (detected <= 0) {
        detected = hw;
    }
    return std::clamp(detected, 1, hw);
}

} // namespace kggml::detail
