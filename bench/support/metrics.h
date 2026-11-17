#ifndef CCML_BENCH_SUPPORT_METRICS_H
#define CCML_BENCH_SUPPORT_METRICS_H

#include <string>

#include "ccml/device.h"

namespace ccml_bench {

/*
 * Work and traffic per call, and the device ceilings they are divided by
 * (reference.md §6: TFLOPS + % of peak for matmul, GB/s + % of peak bandwidth
 * for row-wise ops). Byte counts are the MINIMUM traffic the operator must
 * move, so a percentage of peak can never be inflated by counting re-reads.
 */
double matmul_flops(long long M, long long N, long long K);
double matmul_bytes(long long M, long long N, long long K, int elem_bytes);
/* x read + y written + gamma and beta read. */
double layernorm_bytes(long long rows, long long cols, int elem_bytes, bool affine);
/* x read + y written. */
double softmax_bytes(long long rows, long long cols, int elem_bytes);

/* FP32 lanes per SM, by architecture. Returns 0 for an unknown architecture,
   which makes the theoretical peak "unknown" rather than wrong. */
int fp32_cores_per_sm(int major, int minor);

/* Dense Tensor Core FLOPs per clock per SM with 16-bit inputs and FP32
   accumulation (the mode every ccml and cuBLAS 16-bit matmul here uses).
   0 = no Tensor Cores for that input type on that architecture (BF16 needs
   8.0+). By compute capability, so SKUs that share one are ambiguous: 7.5 is
   the T4's rate (GeForce Turing runs FP32 accumulation at half of it); 8.6 and
   8.9 are the GeForce rates (the data-centre A10/L4 run twice as fast). The
   measured peak from ccml_bench_peaks settles it for the GPU actually used. */
int tc_flops_per_clock_per_sm(int major, int minor, bool bf16);

struct DevicePeaks {
  double fp32_tflops_theory  = 0.0;  /* SMs x lanes x 2 x max SM clock */
  double bandwidth_gbps_theory = 0.0; /* 2 (DDR) x memory clock x bus width */
  double fp32_tflops_measured  = 0.0; /* from bench_peaks; 0 = not measured */
  double bandwidth_gbps_measured = 0.0;
  /* Tensor Core ceilings, FP16/BF16 in, FP32 accumulate. 0 = not available. */
  double tc_f16_tflops_theory = 0.0;
  double tc_bf16_tflops_theory = 0.0;
  double tc_f16_tflops_measured = 0.0;
  double tc_bf16_tflops_measured = 0.0;
};

/* The compute ceiling a matmul of `dtype` ("f32", "f16", "bf16") is judged
   against: FP32 -> the CUDA-core FP32 peak; FP16 -> the FP16 Tensor Core peak;
   BF16 -> the BF16 Tensor Core peak where it exists, otherwise the FP32 peak
   (on sm_75 BF16 can only be computed by widening to FP32, so that is the
   hardware's real ceiling for it). Returns 0 when unknown. */
double matmul_peak_tflops(const DevicePeaks& peaks, const std::string& dtype, bool measured);

/* `max_sm_clock_mhz` overrides the CUDA clock attribute when it is HIGHER:
   laptop and consumer GPUs boost above the clock CUDA reports (measured: an
   RTX 3050 Laptop at 2550 MHz against a reported 2205), and a theoretical peak
   below what the hardware does would inflate every percent-of-peak figure.
   Pass NVML's maximum SM clock, or 0 when it is unavailable. */
DevicePeaks theoretical_peaks(const ccml_device_info& info, int max_sm_clock_mhz = 0);

/* Fills the *_measured fields from a peaks.json written by ccml_bench_peaks.
   Returns false (and leaves them 0) when the file is absent or unreadable. */
bool load_measured_peaks(const std::string& path, DevicePeaks* peaks);

}  // namespace ccml_bench

#endif /* CCML_BENCH_SUPPORT_METRICS_H */
