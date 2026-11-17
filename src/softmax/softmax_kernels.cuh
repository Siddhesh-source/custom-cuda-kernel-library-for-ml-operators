#ifndef CCML_SRC_SOFTMAX_KERNELS_CUH
#define CCML_SRC_SOFTMAX_KERNELS_CUH

/*
 * Device code for softmax. Included by softmax.cu only.
 *
 *   y[r, c] = exp(x[r, c] - max(x[r, :])) / sum_j exp(x[r, j] - max(x[r, :]))
 *
 * One block per row, three passes over the row: the row maximum, the sum of
 * shifted exponentials, then the normalized write. Templated on the storage
 * type T; elements are widened on load and rounded once on store
 * (common/dtype.cuh).
 *
 * Why the maximum is subtracted
 * -----------------------------
 * This is a correctness requirement, not a refinement. expf overflows to
 * infinity above about x = 88.7, so a row containing 1000 would produce inf for
 * every large element and inf/inf = NaN for every output. Subtracting the row
 * maximum makes the largest shifted value exactly 0, so the largest exponential
 * is exactly 1 and nothing can overflow. Terms far below the maximum underflow
 * to zero instead, which is the correct answer to within float precision.
 *
 * It also guarantees the denominator is never zero: the maximum element always
 * contributes exp(0) = 1, so the sum is at least 1 however extreme the input.
 * There is no division-by-zero case to guard.
 *
 * Why three passes over x rather than caching the exponentials
 * -----------------------------------------------------------
 * The alternative is to write the exponentials into the output during pass two
 * and read them back in pass three, which evaluates expf once per element
 * instead of twice. It moves more bytes, not fewer: recomputing costs three
 * reads of x and one write of y, while caching costs two reads of x, a write of
 * y, a read of y and a second write. For a memory-bound kernel the cheaper
 * arithmetic is the wrong thing to optimize, so the exponential is recomputed.
 *
 * Precision
 * ---------
 * The sum accumulates in double, because error there compounds across the whole
 * row. The exponential itself is evaluated in float: expf is accurate to within
 * a couple of ulp, its error is per-element rather than accumulating, and the
 * result is stored as float anyway. Using a double exponential would cost
 * considerably more for a difference the output cannot represent.
 */

#include <cfloat>

#include <cuda_runtime.h>

#include "common/dtype.cuh"
#include "common/reduce.cuh"
#include "common/util.h"

namespace ccml {
namespace kernels {

/* Threads per block. One block owns one row. Not tuned: 256 is a conventional
   starting point that gives eight warps to reduce across. */
constexpr int kSoftmaxBlockSize = 256;

/*
 * grid  = (rows)
 * block = (threads), a multiple of the warp size
 *
 * Rows wider than the block are handled by striding, so any column count works;
 * nothing here assumes cols is a multiple of anything.
 */
template <typename T>
__global__ void softmax_block_per_row(T*       y,
                                      const T* x,
                                      int      rows,
                                      int      cols) {
  using ccml::detail::to_float;
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) {
    return;  /* uniform across the block, so the barriers below stay safe */
  }

  /* 64-bit offset: rows * cols overflows int for large tensors even though each
     dimension fits comfortably. */
  const long long base = static_cast<long long>(row) * cols;
  const T* __restrict__ row_in  = x + base;
  T* __restrict__       row_out = y + base;

  __shared__ double scratch[ccml::detail::kReduceScratch];

  /* Pass 1: the row maximum. Threads holding no elements contribute -DBL_MAX,
     the identity for a maximum. */
  double thread_max = -DBL_MAX;
  for (int c = static_cast<int>(threadIdx.x); c < cols;
       c += static_cast<int>(blockDim.x)) {
    const double value = static_cast<double>(to_float(row_in[c]));
    thread_max = (value > thread_max) ? value : thread_max;
  }
  const double row_max = ccml::detail::block_reduce(
      thread_max, -DBL_MAX, ccml::detail::MaxOp{}, scratch);

  /* The maximum is one of the input floats, so narrowing it back is exact and
     the shift below is a plain float subtraction. */
  const float row_max_f = static_cast<float>(row_max);

  /* The next reduction writes over the scratch this one is still reading. */
  __syncthreads();

  /* Pass 2: the sum of shifted exponentials. Every term is in (0, 1]. */
  double sum = 0.0;
  for (int c = static_cast<int>(threadIdx.x); c < cols;
       c += static_cast<int>(blockDim.x)) {
    sum += static_cast<double>(expf(to_float(row_in[c]) - row_max_f));
  }
  const double total = ccml::detail::block_reduce_sum(sum, scratch);

  /* total >= 1 always, because the maximum element contributes exp(0) = 1. */
  const double inv_total = 1.0 / total;

  /* Pass 3: normalize and write. */
  for (int c = static_cast<int>(threadIdx.x); c < cols;
       c += static_cast<int>(blockDim.x)) {
    const double shifted = static_cast<double>(expf(to_float(row_in[c]) - row_max_f));
    row_out[c] = ccml::detail::from_double<T>(shifted * inv_total);
  }
}

}  // namespace kernels
}  // namespace ccml

#endif /* CCML_SRC_SOFTMAX_KERNELS_CUH */
