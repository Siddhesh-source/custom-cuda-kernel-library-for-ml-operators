#ifndef CCML_BENCH_BASELINES_EPILOGUE_H
#define CCML_BENCH_BASELINES_EPILOGUE_H

#include <cuda_runtime.h>

#include "bench/baselines/cublas.h"

namespace ccml_bench {

/* The unfused half of "matmul + separate kernel" (enhancement F4): in place,
   C = act(C + bias) + residual over a row-major [M, N] 16-bit C, FP32 math, one
   rounding. act: 0 none, 1 ReLU, 2 GELU (tanh form). bias/residual may be
   null. A plain grid-stride elementwise kernel, each element read and written
   once, which is what any unfused pipeline pays. Returns false if the launch
   failed. */
bool apply_epilogue(void* C, const void* bias, const void* residual, int M, int N, int act,
                    BlasType type, cudaStream_t stream);

}  // namespace ccml_bench

#endif /* CCML_BENCH_BASELINES_EPILOGUE_H */
