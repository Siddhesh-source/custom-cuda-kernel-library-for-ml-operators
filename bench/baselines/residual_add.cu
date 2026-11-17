#include "bench/baselines/residual_add.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ccml_bench {

namespace {

__device__ __forceinline__ float widen(float v) { return v; }
__device__ __forceinline__ float widen(__half v) { return __half2float(v); }
__device__ __forceinline__ float widen(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void narrow(float v, float* o) { *o = v; }
__device__ __forceinline__ void narrow(float v, __half* o) { *o = __float2half_rn(v); }
__device__ __forceinline__ void narrow(float v, __nv_bfloat16* o) { *o = __float2bfloat16_rn(v); }

template <typename T>
__global__ void residual_add_kernel(T* s, const T* x, const T* r, const T* b, int rows,
                                    int cols) {
  const long long count = static_cast<long long>(rows) * cols;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    float v = widen(x[i]) + widen(r[i]);
    if (b) v += widen(b[i % cols]);
    narrow(v, &s[i]);
  }
}

template <typename T>
bool launch(void* s, const void* x, const void* r, const void* b, int rows, int cols,
            cudaStream_t stream) {
  const long long count = static_cast<long long>(rows) * cols;
  const long long blocks = (count + 255) / 256;
  residual_add_kernel<T><<<static_cast<unsigned>(blocks < 8192 ? blocks : 8192), 256, 0,
                           stream>>>(static_cast<T*>(s), static_cast<const T*>(x),
                                     static_cast<const T*>(r), static_cast<const T*>(b), rows,
                                     cols);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool residual_add(void* s, const void* x, const void* r, const void* b, int rows, int cols,
                  BlasType type, cudaStream_t stream) {
  if (type == BlasType::F32) return launch<float>(s, x, r, b, rows, cols, stream);
  if (type == BlasType::F16) return launch<__half>(s, x, r, b, rows, cols, stream);
  return launch<__nv_bfloat16>(s, x, r, b, rows, cols, stream);
}

}  // namespace ccml_bench
