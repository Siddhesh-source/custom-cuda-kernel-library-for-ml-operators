#ifndef CCML_BENCH_SUPPORT_TYPED_H
#define CCML_BENCH_SUPPORT_TYPED_H

/* Storage-type traits and a typed device array for the matmul benches: how to
   round a float to the storage type, call the ccml entry points, and widen the
   f32 tolerance (support/compare.h) for one final 16-bit rounding. */

#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "bench/baselines/cublas.h"
#include "ccml/matmul.h"
#include "ccml/tuning.h"
#include "support/compare.h"

namespace ccml_bench {

/* ------------------------------------------------------- dtype traits -- */

struct F32 {
  using Pub = float;
  static constexpr const char* name = "f32";
  static constexpr BlasType blas = BlasType::F32;
  static constexpr double unit_roundoff = 0.0;
  static Pub from_float(float f) { return f; }
  static float to_float(Pub p) { return p; }
  static ccml_status run(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                         ccml_matmul_algo algo, cudaStream_t s) {
    return ccml_matmul_f32(c, a, b, m, n, k, algo, s);
  }
  static ccml_status tuned(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                           ccml_matmul_algo algo, int index, cudaStream_t s) {
    return ccml_matmul_tuned_f32(c, a, b, m, n, k, algo, index, s);
  }
  static ccml_test::Tolerance widen(const ccml_test::Tolerance& t) { return t; }
};

struct F16 {
  using Pub = ccml_f16;
  static constexpr const char* name = "f16";
  static constexpr BlasType blas = BlasType::F16;
  static constexpr double unit_roundoff = ccml_test::kF16UnitRoundoff;
  static Pub from_float(float f) {
    const __half h = __float2half_rn(f);
    Pub p;
    std::memcpy(&p, &h, sizeof(p));
    return p;
  }
  static float to_float(Pub p) {
    __half h;
    std::memcpy(&h, &p, sizeof(h));
    return __half2float(h);
  }
  static ccml_status run(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                         ccml_matmul_algo algo, cudaStream_t s) {
    return ccml_matmul_f16(c, a, b, m, n, k, algo, s);
  }
  static ccml_status tuned(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                           ccml_matmul_algo algo, int index, cudaStream_t s) {
    return ccml_matmul_tuned_f16(c, a, b, m, n, k, algo, index, s);
  }
  static ccml_test::Tolerance widen(const ccml_test::Tolerance& t) {
    return ccml_test::widen_for_storage(t, ccml_test::kF16UnitRoundoff,
                                        ccml_test::kF16SubnormalStep);
  }
};

struct BF16 {
  using Pub = ccml_bf16;
  static constexpr const char* name = "bf16";
  static constexpr BlasType blas = BlasType::BF16;
  static constexpr double unit_roundoff = ccml_test::kBF16UnitRoundoff;
  static Pub from_float(float f) {
    const __nv_bfloat16 h = __float2bfloat16_rn(f);
    Pub p;
    std::memcpy(&p, &h, sizeof(p));
    return p;
  }
  static float to_float(Pub p) {
    __nv_bfloat16 h;
    std::memcpy(&h, &p, sizeof(h));
    return __bfloat162float(h);
  }
  static ccml_status run(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                         ccml_matmul_algo algo, cudaStream_t s) {
    return ccml_matmul_bf16(c, a, b, m, n, k, algo, s);
  }
  static ccml_status tuned(Pub* c, const Pub* a, const Pub* b, int m, int n, int k,
                           ccml_matmul_algo algo, int index, cudaStream_t s) {
    return ccml_matmul_tuned_bf16(c, a, b, m, n, k, algo, index, s);
  }
  static ccml_test::Tolerance widen(const ccml_test::Tolerance& t) {
    return ccml_test::widen_for_storage(t, ccml_test::kBF16UnitRoundoff,
                                        1.1754943508222875e-38 /* FLT_MIN */);
  }
};

/* Owning device array of any element type. */
template <typename T>
class DeviceArray {
 public:
  explicit DeviceArray(std::size_t n) : n_(n) {
    if (cudaMalloc(&ptr_, n * sizeof(T)) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
  }
  ~DeviceArray() { cudaFree(ptr_); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  T* data() { return static_cast<T*>(ptr_); }
  std::size_t count() const { return n_; }
  void upload(const std::vector<T>& h) {
    if (cudaMemcpy(ptr_, h.data(), n_ * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("upload failed");
    }
  }
  bool download(std::vector<T>& h) const {
    h.resize(n_);
    return cudaMemcpy(h.data(), ptr_, n_ * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
  }

 private:
  void*       ptr_ = nullptr;
  std::size_t n_ = 0;
};

}  // namespace ccml_bench

#endif /* CCML_BENCH_SUPPORT_TYPED_H */
