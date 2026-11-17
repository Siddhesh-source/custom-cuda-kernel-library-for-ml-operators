#ifndef CCML_BENCH_BASELINES_SCALE_MASK_H
#define CCML_BENCH_BASELINES_SCALE_MASK_H

#include <cuda_runtime.h>

#include "bench/baselines/cublas.h"

namespace ccml_bench {

/* The unfused half of "scale + mask + softmax" (enhancement F4):
   out = x * scale + mask[r % mask_rows][c], causal keys -inf (same
   definition as ccml_softmax_masked_*), rounded to the storage type. mask may
   be null. A plain grid-stride kernel. Returns false if the launch failed. */
bool scale_mask(void* out, const void* x, const void* mask, int mask_rows, int rows, int cols,
                float scale, int causal_q, BlasType type, cudaStream_t stream);

}  // namespace ccml_bench

#endif /* CCML_BENCH_BASELINES_SCALE_MASK_H */
