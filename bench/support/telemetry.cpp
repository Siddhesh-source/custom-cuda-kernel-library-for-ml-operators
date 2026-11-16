#include "bench/support/telemetry.h"

#ifdef CCML_BENCH_HAVE_NVML
#include <nvml.h>
#include <vector>
#ifdef _WIN32
#include <process.h>
#define CCML_GETPID _getpid
#else
#include <unistd.h>
#define CCML_GETPID getpid
#endif
#endif

namespace ccml_bench {

bool is_throttled(unsigned long long reasons) {
  constexpr unsigned long long kHwSlowdown        = 0x08ull;
  constexpr unsigned long long kSwThermalSlowdown = 0x20ull;
  constexpr unsigned long long kHwThermalSlowdown = 0x40ull;
  constexpr unsigned long long kHwPowerBrake      = 0x80ull;
  return (reasons & (kHwSlowdown | kSwThermalSlowdown | kHwThermalSlowdown |
                     kHwPowerBrake)) != 0;
}

#ifdef CCML_BENCH_HAVE_NVML

TelemetrySource::TelemetrySource() {
  if (nvmlInit_v2() != NVML_SUCCESS) {
    return;
  }
  nvmlDevice_t device = nullptr;
  if (nvmlDeviceGetHandleByIndex_v2(0, &device) != NVML_SUCCESS) {
    nvmlShutdown();
    return;
  }
  device_ = device;
  ok_ = true;
}

TelemetrySource::~TelemetrySource() {
  if (ok_) {
    nvmlShutdown();
  }
}

Telemetry TelemetrySource::sample() const {
  Telemetry t;
  if (!ok_) {
    return t;
  }
  nvmlDevice_t device = static_cast<nvmlDevice_t>(device_);
  unsigned int value = 0;
  t.available = true;
  if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &value) == NVML_SUCCESS) {
    t.sm_clock_mhz = static_cast<int>(value);
  }
  if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &value) == NVML_SUCCESS) {
    t.mem_clock_mhz = static_cast<int>(value);
  }
  if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &value) == NVML_SUCCESS) {
    t.temp_c = static_cast<int>(value);
  }
  unsigned long long reasons = 0;
  if (nvmlDeviceGetCurrentClocksEventReasons(device, &reasons) == NVML_SUCCESS) {
    t.reasons = reasons;
    t.throttled = is_throttled(reasons);
  }
  return t;
}

std::string TelemetrySource::driver_version() const {
  char buffer[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {0};
  if (ok_ && nvmlSystemGetDriverVersion(buffer, sizeof(buffer)) == NVML_SUCCESS) {
    return buffer;
  }
  return "unknown";
}

int TelemetrySource::max_sm_clock_mhz() const {
  unsigned int value = 0;
  if (ok_ && nvmlDeviceGetMaxClockInfo(static_cast<nvmlDevice_t>(device_), NVML_CLOCK_SM,
                                       &value) == NVML_SUCCESS) {
    return static_cast<int>(value);
  }
  return 0;
}

int TelemetrySource::other_processes() const {
  if (!ok_) {
    return -1;
  }
  nvmlDevice_t device = static_cast<nvmlDevice_t>(device_);
  unsigned int count = 64;
  std::vector<nvmlProcessInfo_t> infos(count);
  if (nvmlDeviceGetComputeRunningProcesses(device, &count, infos.data()) != NVML_SUCCESS) {
    return -1;
  }
  const unsigned int self = static_cast<unsigned int>(CCML_GETPID());
  int others = 0;
  for (unsigned int i = 0; i < count; ++i) {
    if (infos[i].pid != self) {
      ++others;
    }
  }
  return others;
}

#else  /* no NVML: report unavailable, never invent values */

TelemetrySource::TelemetrySource() = default;
TelemetrySource::~TelemetrySource() = default;
Telemetry TelemetrySource::sample() const { return Telemetry{}; }
std::string TelemetrySource::driver_version() const { return "unknown"; }
int TelemetrySource::max_sm_clock_mhz() const { return 0; }
int TelemetrySource::other_processes() const { return -1; }

#endif

}  // namespace ccml_bench
