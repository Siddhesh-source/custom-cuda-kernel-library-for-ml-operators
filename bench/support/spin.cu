#include "bench/support/spin.h"

namespace ccml_bench {

namespace {

__global__ void spin_kernel(long long cycles) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

long long cycles_per_us() {
  static long long cached = 0;
  if (cached == 0) {
    int device = 0, khz = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, device);
    cached = khz > 0 ? khz / 1000 : 1000;
  }
  return cached;
}

}  // namespace

void launch_spin(cudaStream_t stream, double microseconds) {
  if (microseconds <= 0.0) return;
  spin_kernel<<<1, 1, 0, stream>>>(static_cast<long long>(microseconds) * cycles_per_us());
}

}  // namespace ccml_bench
