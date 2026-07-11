#include "cpu_threads.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <utility>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__linux__)
#include <sched.h>
#endif

namespace orbitquant::cpu {
namespace {

int environment_thread_count() {
  char const *value = std::getenv("ORBITQUANT_CPU_THREADS");
  if (value == nullptr) {
    value = std::getenv("OMP_NUM_THREADS");
  }
  if (value == nullptr) {
    return 0;
  }
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0) {
    return 0;
  }
  return static_cast<int>(std::min<long>(parsed, 64));
}

#if defined(__linux__)
int affinity_physical_core_count() {
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0) {
    return 0;
  }

  int logical_cpus = 0;
  std::set<std::pair<int, int>> physical_cores;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &affinity)) {
      continue;
    }
    ++logical_cpus;
    int package = -1;
    int core = -1;
    std::ifstream package_file(
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
        "/topology/physical_package_id");
    std::ifstream core_file(
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
        "/topology/core_id");
    if (package_file >> package && core_file >> core) {
      physical_cores.emplace(package, core);
    }
  }
  return physical_cores.empty()
      ? logical_cpus
      : static_cast<int>(physical_cores.size());
}
#endif

int default_thread_count() {
#if defined(__APPLE__)
  int performance_cores = 0;
  std::size_t size = sizeof(performance_cores);
  if (sysctlbyname(
          "hw.perflevel0.physicalcpu",
          &performance_cores,
          &size,
          nullptr,
          0) == 0 &&
      performance_cores > 0) {
    return std::min(performance_cores, 64);
  }
#endif
#if defined(__linux__)
  const int affinity_cores = affinity_physical_core_count();
  if (affinity_cores > 0) {
    return std::min(affinity_cores, 64);
  }
#endif
  const unsigned hardware = std::thread::hardware_concurrency();
  return static_cast<int>(hardware == 0 ? 1 : std::min<unsigned>(hardware, 64));
}

}  // namespace

int requested_threads() {
  const int environment = environment_thread_count();
  static const int default_threads = default_thread_count();
  return environment > 0 ? environment : default_threads;
}

}  // namespace orbitquant::cpu
