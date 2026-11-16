#include "bench/support/metrics.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace ccml_bench {

double matmul_flops(long long M, long long N, long long K) {
  return 2.0 * static_cast<double>(M) * static_cast<double>(N) *
         static_cast<double>(K);
}

double matmul_bytes(long long M, long long N, long long K, int elem_bytes) {
  return static_cast<double>(M * K + K * N + M * N) * elem_bytes;
}

double layernorm_bytes(long long rows, long long cols, int elem_bytes,
                       bool affine) {
  return static_cast<double>(2 * rows * cols + (affine ? 2 * cols : 0)) *
         elem_bytes;
}

double softmax_bytes(long long rows, long long cols, int elem_bytes) {
  return static_cast<double>(2 * rows * cols) * elem_bytes;
}

int fp32_cores_per_sm(int major, int minor) {
  switch (major) {
    case 6: return minor == 0 ? 64 : 128;   /* P100 64; consumer Pascal 128 */
    case 7: return 64;                      /* Volta, Turing */
    case 8: return minor == 0 ? 64 : 128;   /* A100 64; GA10x and Ada 128 */
    case 9: return 128;                     /* Hopper */
    default: return 0;
  }
}

DevicePeaks theoretical_peaks(const ccml_device_info& info, int max_sm_clock_mhz) {
  DevicePeaks peaks;
  const int lanes = fp32_cores_per_sm(info.compute_major, info.compute_minor);
  double clock_hz = static_cast<double>(info.clock_khz) * 1e3;
  if (max_sm_clock_mhz > 0 && static_cast<double>(max_sm_clock_mhz) * 1e6 > clock_hz) {
    clock_hz = static_cast<double>(max_sm_clock_mhz) * 1e6;
  }
  peaks.fp32_tflops_theory = static_cast<double>(info.multiprocessor_count) *
                             lanes * 2.0 * clock_hz / 1e12;
  peaks.bandwidth_gbps_theory = 2.0 *
                                (static_cast<double>(info.memory_clock_khz) * 1e3) *
                                (static_cast<double>(info.memory_bus_width) / 8.0) /
                                1e9;
  return peaks;
}

namespace {

/* Finds "key": <number> in a flat JSON object. peaks.json is written by
   ccml_bench_peaks in exactly that form, so a full parser is not needed. */
bool find_number(const std::string& text, const std::string& key, double* out) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t at = text.find(needle);
  if (at == std::string::npos) {
    return false;
  }
  const std::size_t colon = text.find(':', at + needle.size());
  if (colon == std::string::npos) {
    return false;
  }
  char* end = nullptr;
  const double value = std::strtod(text.c_str() + colon + 1, &end);
  if (end == text.c_str() + colon + 1) {
    return false;
  }
  *out = value;
  return true;
}

}  // namespace

bool load_measured_peaks(const std::string& path, DevicePeaks* peaks) {
  std::ifstream file(path);
  if (!file) {
    return false;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  const std::string text = buffer.str();
  const bool a = find_number(text, "fp32_tflops_measured", &peaks->fp32_tflops_measured);
  const bool b = find_number(text, "bandwidth_gbps_measured", &peaks->bandwidth_gbps_measured);
  return a && b;
}

}  // namespace ccml_bench
