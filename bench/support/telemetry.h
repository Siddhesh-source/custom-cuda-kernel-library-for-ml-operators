#ifndef CCML_BENCH_SUPPORT_TELEMETRY_H
#define CCML_BENCH_SUPPORT_TELEMETRY_H

#include <string>

namespace ccml_bench {

/*
 * GPU conditions recorded with every CSV row (reference.md, Method: "keep
 * clocks identical; watch for thermal throttling and other processes on the
 * GPU"). Clocks cannot be locked on Kaggle, so they are measured instead.
 *
 * Backed by NVML when the build found it (CCML_BENCH_HAVE_NVML); otherwise
 * every field reports "unavailable" rather than a made-up value.
 */
struct Telemetry {
  bool               available = false;
  int                sm_clock_mhz = -1;
  int                mem_clock_mhz = -1;
  int                temp_c = -1;
  unsigned long long reasons = 0;  /* NVML clock-event reason bitmask */
  /* Thermal or hardware slowdown. Power capping is reported in `reasons` but
     not counted here: a T4 at its 70 W limit is its normal operating state. */
  bool               throttled = false;
};

class TelemetrySource {
 public:
  TelemetrySource();
  ~TelemetrySource();
  TelemetrySource(const TelemetrySource&) = delete;
  TelemetrySource& operator=(const TelemetrySource&) = delete;

  Telemetry sample() const;
  bool available() const { return ok_; }
  std::string driver_version() const;
  /* NVML's maximum SM clock (boost ceiling); 0 if unknown. */
  int max_sm_clock_mhz() const;
  /* Compute processes on the GPU other than this one; -1 if unknown. */
  int other_processes() const;

 private:
  bool  ok_ = false;
  void* device_ = nullptr;  /* nvmlDevice_t, kept opaque to callers */
};

/* Bits of the NVML reason mask that count as throttling (HW slowdown, SW and
   HW thermal slowdown, power-brake). Exposed for tests. */
bool is_throttled(unsigned long long reasons);

}  // namespace ccml_bench

#endif /* CCML_BENCH_SUPPORT_TELEMETRY_H */
