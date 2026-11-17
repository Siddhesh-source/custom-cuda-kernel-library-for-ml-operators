#ifndef CCML_BENCH_BASELINES_RESIDUAL_ADD_H
#define CCML_BENCH_BASELINES_RESIDUAL_ADD_H

#include <cuda_runtime.h>

#include "bench/baselines/cublas.h"

namespace ccml_bench {

/* The unfused half of "add + layer norm" (enhancement F4): s = x + r + b over a
   row-major [rows, cols] tensor, FP32 math as (x + r) + b, one rounding to the
   storage type: exactly what ccml_layernorm_residual_* writes to sum_out. b is
   [cols] or null. A plain grid-stride kernel. Returns false if the launch
   failed. */
bool residual_add(void* s, const void* x, const void* r, const void* b, int rows, int cols,
                  BlasType type, cudaStream_t stream);

}  // namespace ccml_bench

#endif /* CCML_BENCH_BASELINES_RESIDUAL_ADD_H */
