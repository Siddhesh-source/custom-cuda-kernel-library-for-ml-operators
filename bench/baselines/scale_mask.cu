#include "bench/baselines/scale_mask.h"

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
__global__ void scale_mask_kernel(T* out, const T* x, const T* mask, int mask_rows, int rows,
                                  int cols, float scale, int causal_q) {
  const long long count = static_cast<long long>(rows) * cols;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int r = static_cast<int>(i / cols), c = static_cast<int>(i % cols);
    float v = widen(x[i]) * scale;
    if (mask) v += widen(mask[static_cast<long long>(r % mask_rows) * cols + c]);
    if (causal_q > 0 && c > r % causal_q + (cols - causal_q)) v = -INFINITY;
    narrow(v, &out[i]);
  }
}

template <typename T>
bool launch(void* out, const void* x, const void* mask, int mask_rows, int rows, int cols,
            float scale, int causal_q, cudaStream_t stream) {
  const long long count = static_cast<long long>(rows) * cols;
  const long long blocks = (count + 255) / 256;
  scale_mask_kernel<T><<<static_cast<unsigned>(blocks < 8192 ? blocks : 8192), 256, 0, stream>>>(
      static_cast<T*>(out), static_cast<const T*>(x), static_cast<const T*>(mask), mask_rows,
      rows, cols, scale, causal_q);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool scale_mask(void* out, const void* x, const void* mask, int mask_rows, int rows, int cols,
                float scale, int causal_q, BlasType type, cudaStream_t stream) {
  if (type == BlasType::F32)
    return launch<float>(out, x, mask, mask_rows, rows, cols, scale, causal_q, stream);
  if (type == BlasType::F16)
    return launch<__half>(out, x, mask, mask_rows, rows, cols, scale, causal_q, stream);
  return launch<__nv_bfloat16>(out, x, mask, mask_rows, rows, cols, scale, causal_q, stream);
}

}  // namespace ccml_bench
