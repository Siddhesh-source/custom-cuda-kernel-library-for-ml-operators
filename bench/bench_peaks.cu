/* Measured device ceilings (enhancement M5): FP32 FMA throughput and DRAM
   copy bandwidth. "% of peak" is then reported against both the theoretical
   spec-sheet peak and what this GPU actually reaches (reference.md §6).

   Writes a flat JSON object:  ccml_bench_peaks [--out peaks.json]          */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bench/support/metrics.h"
#include "bench/support/stats.h"
#include "bench/support/telemetry.h"
#include "ccml/device.h"
#include "ccml/status.h"

namespace {

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

constexpr int kChains = 8;   /* independent FMA chains per thread hide latency */
constexpr int kFmaIters = 4096;

__global__ void fma_peak(float* sink, float seed) {
  float acc[kChains];
#pragma unroll
  for (int c = 0; c < kChains; ++c) acc[c] = seed + static_cast<float>(c + threadIdx.x);
  const float m = 1.0000001f, a = 1e-7f;
  for (int i = 0; i < kFmaIters; ++i) {
#pragma unroll
    for (int c = 0; c < kChains; ++c) acc[c] = fmaf(acc[c], m, a);
  }
  float total = 0.0f;
#pragma unroll
  for (int c = 0; c < kChains; ++c) total += acc[c];
  /* Never true for these inputs, but the compiler cannot prove it, so the
     arithmetic above cannot be eliminated. */
  if (total == 12345.678f) sink[blockIdx.x * blockDim.x + threadIdx.x] = total;
}

__global__ void copy_peak(float4* __restrict__ dst, const float4* __restrict__ src,
                          std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       i < n; i += stride) {
    dst[i] = src[i];
  }
}

/* Warm up for at least kWarmupMs of GPU time before measuring. An idle GPU
   sits at its lowest clock (300 MHz on a T4) and ramps over hundreds of
   milliseconds; a fixed handful of launches measured the ramp, not the peak
   (first Kaggle run: 3.2 "measured" FP32 TFLOPS while cuBLAS reached 6.1). */
constexpr double kWarmupMs = 1500.0;

template <typename Launch>
double median_ms(Launch launch, int reps) {
  cudaEvent_t a, b;
  check(cudaEventCreate(&a), "cudaEventCreate");
  check(cudaEventCreate(&b), "cudaEventCreate");
  double warmed = 0.0;
  while (warmed < kWarmupMs) {
    check(cudaEventRecord(a), "cudaEventRecord");
    launch();
    check(cudaGetLastError(), "launch");
    check(cudaEventRecord(b), "cudaEventRecord");
    check(cudaEventSynchronize(b), "cudaEventSynchronize");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime");
    warmed += ms;
  }
  std::vector<double> samples;
  for (int i = 0; i < reps; ++i) {
    check(cudaEventRecord(a), "cudaEventRecord");
    launch();
    check(cudaGetLastError(), "launch");
    check(cudaEventRecord(b), "cudaEventRecord");
    check(cudaEventSynchronize(b), "cudaEventSynchronize");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime");
    samples.push_back(ms);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ccml_bench::median_of(samples);
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_path;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--out") == 0) out_path = argv[i + 1];
  }
  try {
    ccml_device_info info;
    if (ccml_get_device_info(0, &info) != CCML_SUCCESS) {
      throw std::runtime_error(ccml_last_error_string());
    }
    ccml_bench::TelemetrySource telemetry;
    const ccml_bench::DevicePeaks theory =
        ccml_bench::theoretical_peaks(info, telemetry.max_sm_clock_mhz());

    /* FP32: enough resident blocks to saturate every SM several times over. */
    const int threads = 256;
    const int blocks = info.multiprocessor_count * 32;
    float* sink = nullptr;
    check(cudaMalloc(&sink, sizeof(float) * blocks * threads), "cudaMalloc sink");
    const double fma_ms = median_ms([&] { fma_peak<<<blocks, threads>>>(sink, 1.0f); }, 20);
    const ccml_bench::Telemetry during_fma = telemetry.sample();
    const double flops = 2.0 * kChains * kFmaIters * static_cast<double>(blocks) * threads;
    const double tflops = flops / (fma_ms * 1e-3) / 1e12;
    cudaFree(sink);

    /* Bandwidth: a copy far larger than L2, sized to the free memory. */
    std::size_t free_bytes = 0, total = 0;
    check(cudaMemGetInfo(&free_bytes, &total), "cudaMemGetInfo");
    std::size_t bytes = std::min<std::size_t>(512u << 20, free_bytes / 4);
    bytes -= bytes % sizeof(float4);
    float4 *src = nullptr, *dst = nullptr;
    check(cudaMalloc(&src, bytes), "cudaMalloc src");
    check(cudaMalloc(&dst, bytes), "cudaMalloc dst");
    check(cudaMemset(src, 1, bytes), "cudaMemset");
    const std::size_t n = bytes / sizeof(float4);
    const int copy_blocks = info.multiprocessor_count * 8;
    const double copy_ms =
        median_ms([&] { copy_peak<<<copy_blocks, 512>>>(dst, src, n); }, 30);
    const ccml_bench::Telemetry during_copy = telemetry.sample();
    const double gbps = 2.0 * static_cast<double>(bytes) / (copy_ms * 1e-3) / 1e9;
    cudaFree(src);
    cudaFree(dst);

    char json[1536];
    std::snprintf(json, sizeof(json),
                  "{\n  \"gpu\": \"%s\",\n  \"cc\": \"%d.%d\",\n"
                  "  \"fp32_tflops_theory\": %.4f,\n  \"fp32_tflops_measured\": %.4f,\n"
                  "  \"bandwidth_gbps_theory\": %.2f,\n  \"bandwidth_gbps_measured\": %.2f,\n"
                  "  \"sm_clock_mhz_during_fma\": %d,\n  \"sm_clock_mhz_during_copy\": %d,\n"
                  "  \"max_sm_clock_mhz\": %d,\n  \"clock_reasons_during_fma\": \"0x%llx\",\n"
                  "  \"copy_bytes\": %zu,\n  \"method\": \"%.0f ms GPU-time warmup, then median "
                  "of event-timed runs; fma: %d chains x %d iters; copy: float4 grid-stride, "
                  "read+write\"\n}\n",
                  info.name, info.compute_major, info.compute_minor,
                  theory.fp32_tflops_theory, tflops, theory.bandwidth_gbps_theory, gbps,
                  during_fma.sm_clock_mhz, during_copy.sm_clock_mhz,
                  telemetry.max_sm_clock_mhz() > 0 ? telemetry.max_sm_clock_mhz()
                                                   : info.clock_khz / 1000,
                  during_fma.reasons, bytes, kWarmupMs, kChains, kFmaIters);
    std::fputs(json, stdout);
    if (!out_path.empty()) {
      std::FILE* f = std::fopen(out_path.c_str(), "w");
      if (f == nullptr) throw std::runtime_error("cannot write " + out_path);
      std::fputs(json, f);
      std::fclose(f);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ccml_bench_peaks: %s\n", e.what());
    return 1;
  }
  return 0;
}
