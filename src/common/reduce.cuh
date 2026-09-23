#ifndef CCML_SRC_COMMON_REDUCE_CUH
#define CCML_SRC_COMMON_REDUCE_CUH

/*
 * Block-wide reduction, shared by the operators.
 *
 * This lives in common/ rather than in one operator's kernel header because
 * layer normalization and softmax both need it, and operators must never
 * include each other. Softmax also needs a maximum as well as a sum, which is
 * why it takes the combining operation as a parameter instead of being two
 * near-identical copies.
 *
 * Everything reduces in double. For sums that is a correctness requirement --
 * see the note in layernorm_kernels.cuh -- and for maxima it costs nothing,
 * since converting a float to double is exact.
 */

#include <cuda_runtime.h>

namespace ccml {
namespace detail {

/* Scratch a block reduction needs: one slot per warp, plus one to broadcast the
   result. Sized for the largest block CUDA allows. */
constexpr int kMaxWarpsPerBlock = 32;
constexpr int kReduceScratch    = kMaxWarpsPerBlock + 1;

struct SumOp {
  __device__ double operator()(double a, double b) const { return a + b; }
};

struct MaxOp {
  __device__ double operator()(double a, double b) const { return a > b ? a : b; }
};

/*
 * Combines `value` across the whole block; every thread receives the result.
 *
 * Two stages: each warp reduces internally with shuffles, needing no shared
 * memory and no barrier, then one warp reduces the per-warp partials. A full
 * shared-memory tree over all threads would also work, but costs a barrier per
 * level where this costs two in total.
 *
 * `identity` is what a thread contributes when it has no data -- 0.0 for a sum,
 * -DBL_MAX for a maximum. Threads holding no elements must pass it, and stage
 * two uses it for lanes beyond the warp count.
 *
 * Requirements:
 *   - every thread in the block must call this, since it contains barriers;
 *   - blockDim.x must be a multiple of the warp size;
 *   - `scratch` must have kReduceScratch elements;
 *   - callers must __syncthreads() between successive calls sharing a buffer,
 *     because the next call writes where the previous one is still reading.
 */
template <typename Op>
__device__ inline double block_reduce(double value, double identity, Op op,
                                      double* scratch) {
  constexpr unsigned kFullMask = 0xffffffffu;

  /* Stage 1: reduce within each warp. */
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value = op(value, __shfl_down_sync(kFullMask, value, offset));
  }

  const int lane       = static_cast<int>(threadIdx.x) % warpSize;
  const int warp       = static_cast<int>(threadIdx.x) / warpSize;
  const int warp_count = (static_cast<int>(blockDim.x) + warpSize - 1) / warpSize;

  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();

  /* Stage 2: the first warp reduces the per-warp partials. */
  if (warp == 0) {
    double partial = (lane < warp_count) ? scratch[lane] : identity;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      partial = op(partial, __shfl_down_sync(kFullMask, partial, offset));
    }
    if (lane == 0) {
      scratch[warp_count] = partial;
    }
  }
  __syncthreads();

  return scratch[warp_count];
}

__device__ inline double block_reduce_sum(double value, double* scratch) {
  return block_reduce(value, 0.0, SumOp{}, scratch);
}

}  // namespace detail
}  // namespace ccml

#endif /* CCML_SRC_COMMON_REDUCE_CUH */
