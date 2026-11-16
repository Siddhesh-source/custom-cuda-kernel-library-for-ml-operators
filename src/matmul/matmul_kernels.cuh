#ifndef CCML_SRC_MATMUL_KERNELS_CUH
#define CCML_SRC_MATMUL_KERNELS_CUH

/*
 * Device code for matrix multiplication. Included by matmul.cu only.
 *
 *   C[M, N] = A[M, K] * B[K, N], row-major, no transpose, no accumulation.
 *
 * Templated on the storage type T (float, __half, __nv_bfloat16). Elements are
 * widened to float on load and the result is rounded to T once, on store
 * (common/dtype.cuh). The shared-memory tiles hold the widened floats.
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

#include "common/dtype.cuh"
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
template <typename T>
__global__ void matmul_naive(T*       C,
                             const T* A,
                             const T* B,
                             int      M,
                             int      N,
                             int      K) {
  using ccml::detail::to_float;
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

  if (row >= M || col >= N) {
    return;
  }

  float accumulator = 0.0f;
  for (int k = 0; k < K; ++k) {
    accumulator += to_float(A[static_cast<long long>(row) * K + k]) *
                   to_float(B[static_cast<long long>(k) * N + col]);
  }
  C[static_cast<long long>(row) * N + col] = ccml::detail::from_float<T>(accumulator);
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
template <typename T>
__global__ void matmul_tiled(T*       C,
                             const T* A,
                             const T* B,
                             int      M,
                             int      N,
                             int      K) {
  using ccml::detail::to_float;
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
            ? to_float(A[static_cast<long long>(row) * K + a_col])
            : 0.0f;
    tile_b[local_y][local_x] =
        (b_row < K && col < N)
            ? to_float(B[static_cast<long long>(b_row) * N + col])
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
    C[static_cast<long long>(row) * N + col] = ccml::detail::from_float<T>(accumulator);
  }
}

/*
 * Register-blocked (docs/optimization-log.md, entry 1).
 *
 * A block computes a BM x BN tile of C with (BM/TM) * (BN/TN) threads; each
 * thread owns a TM x TN patch held in registers. Per k-step of BK the block
 * stages a BM x BK slice of A and a BK x BN slice of B in shared memory, and
 * then, for each of the BK columns, every thread loads TM values of A and TN
 * values of B into registers and does TM * TN FMAs from them. Each shared-memory
 * value is therefore reused TM or TN times from a register instead of being
 * re-read per FMA.
 *
 * The A slice is stored transposed (As[k][m]) so that, for a given k, the TM
 * values a thread needs are contiguous. That makes the transposing store
 * strided across banks; it is left as it is here, because removing it is a
 * separate step (vectorized loads, Phase 10) and this step must change one
 * thing only.
 *
 * Edges: out-of-range elements are zero-filled when staged and the store is
 * guarded, so every thread reaches every barrier and no dimension needs to be
 * a multiple of any tile.
 *
 *   grid  = (div_up(N, BN), div_up(M, BM))
 *   block = ((BM / TM) * (BN / TN))
 */
template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
matmul_register_tiled(T* C, const T* A, const T* B, int M, int N, int K) {
  using ccml::detail::to_float;
  constexpr int kCols = BN / TN;             /* threads along N */
  constexpr int kThreads = (BM / TM) * kCols;
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide the block tile");

  __shared__ float As[BK][BM];  /* transposed: As[k][m] */
  __shared__ float Bs[BK][BN];

  const int tid = static_cast<int>(threadIdx.x);
  const int thread_row = tid / kCols;
  const int thread_col = tid % kCols;
  const int block_row = static_cast<int>(blockIdx.y) * BM;
  const int block_col = static_cast<int>(blockIdx.x) * BN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
  }
  float a_reg[TM];
  float b_reg[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    /* Stage A[block_row : +BM, k0 : +BK], transposed. Consecutive threads take
       consecutive k within a row, so global reads are contiguous per row. */
    for (int i = tid; i < BM * BK; i += kThreads) {
      const int r = i / BK;
      const int c = i % BK;
      const int gr = block_row + r;
      const int gc = k0 + c;
      As[c][r] = (gr < M && gc < K) ? to_float(A[static_cast<long long>(gr) * K + gc]) : 0.0f;
    }
    /* Stage B[k0 : +BK, block_col : +BN]. Consecutive threads take consecutive
       columns: coalesced. */
    for (int i = tid; i < BK * BN; i += kThreads) {
      const int r = i / BN;
      const int c = i % BN;
      const int gr = k0 + r;
      const int gc = block_col + c;
      Bs[r][c] = (gr < K && gc < N) ? to_float(B[static_cast<long long>(gr) * N + gc]) : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) a_reg[i] = As[k][thread_row * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][thread_col * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a_reg[i] * b_reg[j];
      }
    }
    /* Before the next iteration overwrites the slices. */
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = block_row + thread_row * TM + i;
    if (r >= M) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int c = block_col + thread_col * TN + j;
      if (c < N) {
        C[static_cast<long long>(r) * N + c] = ccml::detail::from_float<T>(acc[i][j]);
      }
    }
  }
}

/*
 * The tile configurations REGISTER_TILED can run, for the tile-size sweep
 * (reference.md §5) and later the measured dispatch table (enhancement S4).
 * Index 0 is the default. Defaults are NOT tuned yet: the sweep decides.
 */
struct MatmulTileConfig {
  int bm, bn, bk, tm, tn;
};

constexpr MatmulTileConfig kMatmulTileConfigs[] = {
    {128, 128, 8, 8, 8},  /* 0: default, the hypothesis in the optimization log */
    {128, 64, 8, 8, 4},   /* 1 */
    {64, 64, 8, 4, 4},    /* 2 */
    {64, 128, 8, 4, 8},   /* 3 */
    {32, 64, 16, 4, 4},   /* 4: smaller BM for short M */
    {16, 128, 16, 2, 8},  /* 5: decode-sized M */
};
constexpr int kMatmulTileConfigCount =
    static_cast<int>(sizeof(kMatmulTileConfigs) / sizeof(kMatmulTileConfigs[0]));

}  // namespace kernels
}  // namespace ccml

#endif /* CCML_SRC_MATMUL_KERNELS_CUH */
