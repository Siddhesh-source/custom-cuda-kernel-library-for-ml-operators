#ifndef CCML_SRC_LAYERNORM_KERNELS_CUH
#define CCML_SRC_LAYERNORM_KERNELS_CUH

/*
 * Device code for layer normalization. Included by layernorm.cu only.
 *
 *   mean     = average(row)
 *   variance = average((value - mean)^2)
 *   output   = (value - mean) / sqrt(variance + eps) * gamma + beta
 *
 * One block per row, three passes over the row: mean, variance about that
 * mean, then the normalized write. Templated on the storage type T; elements
 * are widened on load and rounded once on store (common/dtype.cuh).
 *
 * Why two passes and not one
 * --------------------------
 * The single-pass form accumulates sum and sum-of-squares together and recovers
 * the variance as E[x^2] - E[x]^2. That is algebraically correct and
 * numerically terrible: when the mean is large relative to the spread, the two
 * terms are nearly equal and their difference loses most of its significant
 * digits. With mean 1000 and standard deviation 1, E[x^2] and E[x]^2 agree to
 * six digits and float32 has barely seven, so the result is almost entirely
 * rounding error. Computing the variance about an already-known mean has no
 * such cancellation.
 *
 * Welford's method is the numerically sound single-pass alternative and is
 * reserved as a separate algorithm, so that the two can be compared rather than
 * one silently replacing the other.
 *
 * Why the reductions accumulate in double
 * ---------------------------------------
 * This is a correctness requirement, not caution. The normalized output divides
 * by the standard deviation, so any error in the mean is magnified by
 * 1/stddev. A float32 mean carries roughly half an ulp of error: at magnitude
 * 10000 that is about 5e-4, which against a standard deviation of 0.01 is a 5%
 * error in every output element. No amount of care in the summation order fixes
 * that -- the mean simply has to be held to better than float32 precision, and
 * carried in that precision through the subtraction.
 *
 * The cost is fp64 arithmetic, which consumer GPUs run at a fraction of fp32
 * rate. Whether it matters here is unmeasured: layer normalization is expected
 * to be memory-bound, so a handful of fp64 operations per element may well be
 * hidden. That question is deliberately left to measurement rather than
 * guessed at, and the obviously-correct version is what ships first.
 */

#include <cuda_runtime.h>

#include "common/dtype.cuh"
#include "common/reduce.cuh"
#include "common/util.h"

namespace ccml {
namespace kernels {

/* Threads per block. One block owns one row, so this is also the number of
   threads cooperating on a single reduction. Not tuned: 256 is a conventional
   starting point that gives eight warps to reduce across. */
constexpr int kLayerNormBlockSize = 256;

/*
 * grid  = (rows)
 * block = (threads), a multiple of the warp size
 *
 * gamma and beta may each be null, meaning that part of the affine is skipped.
 * Rows wider than the block are handled by striding, so any column count works;
 * nothing here assumes cols is a multiple of anything.
 */
template <typename T>
__global__ void layernorm_two_pass(T*       y,
                                   const T* x,
                                   const T* gamma,
                                   const T* beta,
                                   int      rows,
                                   int      cols,
                                   float    eps) {
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

  /* Pass 1: the mean. */
  double sum = 0.0;
  for (int c = static_cast<int>(threadIdx.x); c < cols;
       c += static_cast<int>(blockDim.x)) {
    sum += static_cast<double>(to_float(row_in[c]));
  }
  const double mean =
      ccml::detail::block_reduce_sum(sum, scratch) / static_cast<double>(cols);

  /* The next reduction writes over the scratch this one is still reading. */
  __syncthreads();

  /* Pass 2: the variance, about the mean just computed. */
  double sum_squares = 0.0;
  for (int c = static_cast<int>(threadIdx.x); c < cols;
       c += static_cast<int>(blockDim.x)) {
    const double deviation = static_cast<double>(to_float(row_in[c])) - mean;
    sum_squares += deviation * deviation;
  }
  const double variance =
      ccml::detail::block_reduce_sum(sum_squares, scratch) /
      static_cast<double>(cols);

  /* eps is inside the square root, matching the definition. It also keeps a
     constant row -- variance exactly zero -- finite rather than dividing by 0. */
  const double inv_stddev = 1.0 / sqrt(variance + static_cast<double>(eps));

  /* Pass 3: normalize, apply the optional affine, write. */
  for (int c = static_cast<int>(threadIdx.x); c < cols;
       c += static_cast<int>(blockDim.x)) {
    double normalized = (static_cast<double>(to_float(row_in[c])) - mean) * inv_stddev;
    if (gamma != nullptr) {
      normalized *= static_cast<double>(to_float(gamma[c]));
    }
    if (beta != nullptr) {
      normalized += static_cast<double>(to_float(beta[c]));
    }
    row_out[c] = ccml::detail::from_double<T>(normalized);
  }
}

}  // namespace kernels
}  // namespace ccml

#endif /* CCML_SRC_LAYERNORM_KERNELS_CUH */
