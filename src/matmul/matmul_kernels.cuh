#ifndef CCML_SRC_MATMUL_KERNELS_CUH
#define CCML_SRC_MATMUL_KERNELS_CUH

/*
 * Device code for matrix multiplication. Included by matmul.cu only.
 *
 *   C[M, N] = A[M, K] * B[K, N], row-major, no transpose, no accumulation.
 *
 * Accumulation is in float32, unlike the row-wise operators which accumulate in
 * double. The reason the others need double does not apply here: layer
 * normalization divides by a standard deviation, so an error in the mean is
 * magnified by 1/stddev. A matrix product has no such amplification -- the
 * error stays proportional to the result -- and float32 accumulation is what
 * cuBLAS and PyTorch do for float32 inputs, so matching them keeps the
 * comparison honest.
 */

#include <cuda_runtime.h>

#include "common/util.h"

namespace ccml {
namespace kernels {

/* Tile edge for the shared-memory variant. Threads per block is the square of
   this, so 16 gives 256 threads. Chosen provisionally; it has not been tuned
   against measurement. */
constexpr int kMatmulTileDim = 16;

/*
 * One thread per output element, accumulating across K in a register.
 *
 * Every thread reads a whole row of A and a whole column of B from global
 * memory, so an element of A is re-read by every thread in its block row. There
 * is no reuse by construction. Kept permanently as the correctness baseline
 * that faster variants are differenced against, not as a stepping stone.
 *
 *   grid  = (div_up(N, kMatmulTileDim), div_up(M, kMatmulTileDim))
 *   block = (kMatmulTileDim, kMatmulTileDim)
 */
__global__ void matmul_naive_f32(float*       C,
                                 const float* A,
                                 const float* B,
                                 int          M,
                                 int          N,
                                 int          K) {
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

  if (row >= M || col >= N) {
    return;
  }

  float accumulator = 0.0f;
  for (int k = 0; k < K; ++k) {
    accumulator += A[static_cast<long long>(row) * K + k] *
                   B[static_cast<long long>(k) * N + col];
  }
  C[static_cast<long long>(row) * N + col] = accumulator;
}

/*
 * Shared-memory tiled.
 *
 * Each block walks K in steps of kMatmulTileDim. At each step it cooperatively
 * stages one tile of A and one of B into shared memory, then every thread
 * accumulates that tile's contribution from shared rather than global memory.
 * Each staged element is therefore read once from global and used
 * kMatmulTileDim times.
 *
 *   grid  = (div_up(N, kMatmulTileDim), div_up(M, kMatmulTileDim))
 *   block = (kMatmulTileDim, kMatmulTileDim)
 *
 * The edges matter more than the tiling. When M, N or K is not a multiple of
 * the tile dimension, out-of-range elements are ZERO-FILLED rather than
 * skipped: every thread in the block has to reach both __syncthreads() calls,
 * so a thread with no data of its own must still participate. Returning early
 * for it would hang the block. Zeros contribute nothing to the sum, so the
 * result is unaffected.
 */
__global__ void matmul_tiled_f32(float*       C,
                                 const float* A,
                                 const float* B,
                                 int          M,
                                 int          N,
                                 int          K) {
  __shared__ float tile_a[kMatmulTileDim][kMatmulTileDim];
  __shared__ float tile_b[kMatmulTileDim][kMatmulTileDim];

  const int local_x = static_cast<int>(threadIdx.x);
  const int local_y = static_cast<int>(threadIdx.y);
  const int col = static_cast<int>(blockIdx.x) * kMatmulTileDim + local_x;
  const int row = static_cast<int>(blockIdx.y) * kMatmulTileDim + local_y;

  float accumulator = 0.0f;

  const int tile_count = ccml::detail::div_up(K, kMatmulTileDim);
  for (int tile = 0; tile < tile_count; ++tile) {
    const int a_col = tile * kMatmulTileDim + local_x;
    const int b_row = tile * kMatmulTileDim + local_y;

    tile_a[local_y][local_x] =
        (row < M && a_col < K)
            ? A[static_cast<long long>(row) * K + a_col]
            : 0.0f;
    tile_b[local_y][local_x] =
        (b_row < K && col < N)
            ? B[static_cast<long long>(b_row) * N + col]
            : 0.0f;

    /* Every thread must arrive: the staging above is cooperative. */
    __syncthreads();

    for (int k = 0; k < kMatmulTileDim; ++k) {
      accumulator += tile_a[local_y][k] * tile_b[k][local_x];
    }

    /* And again before the next iteration overwrites the tiles. */
    __syncthreads();
  }

  if (row < M && col < N) {
    C[static_cast<long long>(row) * N + col] = accumulator;
  }
}

}  // namespace kernels
}  // namespace ccml

#endif /* CCML_SRC_MATMUL_KERNELS_CUH */
