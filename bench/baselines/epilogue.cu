#include "bench/baselines/epilogue.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ccml_bench {

namespace {

__device__ __forceinline__ float widen(__half v) { return __half2float(v); }
__device__ __forceinline__ float widen(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void narrow(float v, __half* out) { *out = __float2half_rn(v); }
__device__ __forceinline__ void narrow(float v, __nv_bfloat16* out) {
  *out = __float2bfloat16_rn(v);
}

template <typename T>
__global__ void epilogue_kernel(T* C, const T* bias, const T* residual, int M, int N, int act) {
  const long long count = static_cast<long long>(M) * N;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    float v = widen(C[i]);
    if (bias) v += widen(bias[i % N]);
    if (act == 1) v = v > 0.0f ? v : 0.0f;
    if (act == 2) v = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    if (residual) v += widen(residual[i]);
    narrow(v, &C[i]);
  }
}

template <typename T>
bool launch(void* C, const void* bias, const void* residual, int M, int N, int act,
            cudaStream_t stream) {
  const long long count = static_cast<long long>(M) * N;
  const long long blocks = (count + 255) / 256;
  epilogue_kernel<T><<<static_cast<unsigned>(blocks < 4096 ? blocks : 4096), 256, 0, stream>>>(
      static_cast<T*>(C), static_cast<const T*>(bias), static_cast<const T*>(residual), M, N, act);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool apply_epilogue(void* C, const void* bias, const void* residual, int M, int N, int act,
                    BlasType type, cudaStream_t stream) {
  if (type == BlasType::F16) return launch<__half>(C, bias, residual, M, N, act, stream);
  if (type == BlasType::BF16) return launch<__nv_bfloat16>(C, bias, residual, M, N, act, stream);
  return false;
}

}  // namespace ccml_bench
