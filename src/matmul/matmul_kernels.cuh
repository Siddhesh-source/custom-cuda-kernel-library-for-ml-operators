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

#include <type_traits>

#include <cuda_runtime.h>
#include <mma.h>

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

/* Four consecutive elements, aligned so that one load or store moves them all:
   128 bits for float, 64 bits for the 16-bit types. */
template <typename T>
struct alignas(4 * sizeof(T)) Vec4 {
  T v[4];
};

/*
 * Vectorized, bank-conflict-free register blocking (docs/optimization-log.md,
 * entry 2). Same tiling as matmul_register_tiled; only the memory accesses
 * change:
 *
 *  - Global loads of A and B, and the store of C, move 4 elements per
 *    instruction when kVec is true (the host checks that K and N are
 *    multiples of 4 and every pointer is aligned). Otherwise they are scalar,
 *    with the same zero-fill at the edges.
 *  - A is staged transposed into As[k][m] with 4 floats of padding per row,
 *    so the transposed stores of a warp spread over all 32 banks, and row
 *    starts stay 16-byte aligned for vector reads.
 *  - Each thread's TN columns are split into TN/4 chunks of 4, spaced
 *    BN/(TN/4) apart. Consecutive threads then read consecutive 16-byte chunks
 *    of Bs, so a warp's B reads cover distinct banks. With 8 contiguous
 *    columns per thread (entry 1) they collided 4 ways.
 *  - Shared-memory reads are float4: TM/4 + TN/4 loads per k-step instead of
 *    TM + TN.
 *
 * Requires TM, TN and BK to be multiples of 4.
 *
 *   grid  = (div_up(N, BN), div_up(M, BM))
 *   block = ((BM / TM) * (BN / TN))
 */
/* Occupancy floor: at least 2 resident blocks per SM. Without it the vector
   temporaries push the kernel to 155 registers per thread, one block per SM
   and 16.7% of warps active, which cancelled the memory gains (NCU,
   docs/optimization-log.md entry 2). The bound makes the compiler fit 2
   blocks (<= 128 registers per thread at 256 threads). */
template <typename T, int BM, int BN, int BK, int TM, int TN, bool kVec>
__global__ void __launch_bounds__((BM / TM) * (BN / TN), 2)
matmul_vectorized(T* C, const T* A, const T* B, int M, int N, int K) {
  using ccml::detail::to_float;
  constexpr int kCols = BN / TN;
  constexpr int kThreads = (BM / TM) * kCols;
  constexpr int kChunksN = TN / 4;
  constexpr int kChunkStride = BN / kChunksN;  /* == kCols * 4 */
  constexpr int kPadA = 4;
  static_assert(TM % 4 == 0 && TN % 4 == 0 && BK % 4 == 0,
                "vectorized tiles need multiples of 4");
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide the block tile");

  __shared__ __align__(16) float As[BK][BM + kPadA];  /* transposed, padded */
  __shared__ __align__(16) float Bs[BK][BN];

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
    /* Stage A: BM x BK as quads of 4 consecutive k. Two adjacent threads share
       a row, so a warp reads 16 rows x 32 contiguous bytes. */
    for (int q = tid; q < BM * BK / 4; q += kThreads) {
      const int r = q / (BK / 4);
      const int c = (q % (BK / 4)) * 4;
      const int gr = block_row + r;
      const int gc = k0 + c;
      float v[4];
      if (kVec) {
        /* K % 4 == 0 and gc % 4 == 0, so gc < K implies gc + 3 < K. */
        if (gr < M && gc < K) {
          const Vec4<T> x =
              *reinterpret_cast<const Vec4<T>*>(A + static_cast<long long>(gr) * K + gc);
#pragma unroll
          for (int j = 0; j < 4; ++j) v[j] = to_float(x.v[j]);
        } else {
#pragma unroll
          for (int j = 0; j < 4; ++j) v[j] = 0.0f;
        }
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          v[j] = (gr < M && gc + j < K)
                     ? to_float(A[static_cast<long long>(gr) * K + gc + j])
                     : 0.0f;
        }
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) As[c + j][r] = v[j];
    }

    /* Stage B: BK x BN as quads of 4 consecutive columns; consecutive threads
       take consecutive quads, so both the global read and the shared store
       are contiguous. */
    for (int q = tid; q < BK * BN / 4; q += kThreads) {
      const int r = q / (BN / 4);
      const int c = (q % (BN / 4)) * 4;
      const int gr = k0 + r;
      const int gc = block_col + c;
      float v[4];
      if (kVec) {
        if (gr < K && gc < N) {
          const Vec4<T> x =
              *reinterpret_cast<const Vec4<T>*>(B + static_cast<long long>(gr) * N + gc);
#pragma unroll
          for (int j = 0; j < 4; ++j) v[j] = to_float(x.v[j]);
        } else {
#pragma unroll
          for (int j = 0; j < 4; ++j) v[j] = 0.0f;
        }
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          v[j] = (gr < K && gc + j < N)
                     ? to_float(B[static_cast<long long>(gr) * N + gc + j])
                     : 0.0f;
        }
      }
      *reinterpret_cast<float4*>(&Bs[r][c]) = make_float4(v[0], v[1], v[2], v[3]);
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i4 = 0; i4 < TM / 4; ++i4) {
        const float4 a4 = *reinterpret_cast<const float4*>(&As[k][thread_row * TM + i4 * 4]);
        a_reg[i4 * 4 + 0] = a4.x;
        a_reg[i4 * 4 + 1] = a4.y;
        a_reg[i4 * 4 + 2] = a4.z;
        a_reg[i4 * 4 + 3] = a4.w;
      }
#pragma unroll
      for (int c4 = 0; c4 < kChunksN; ++c4) {
        const float4 b4 =
            *reinterpret_cast<const float4*>(&Bs[k][c4 * kChunkStride + thread_col * 4]);
        b_reg[c4 * 4 + 0] = b4.x;
        b_reg[c4 * 4 + 1] = b4.y;
        b_reg[c4 * 4 + 2] = b4.z;
        b_reg[c4 * 4 + 3] = b4.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a_reg[i] * b_reg[j];
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = block_row + thread_row * TM + i;
    if (r >= M) continue;
#pragma unroll
    for (int c4 = 0; c4 < kChunksN; ++c4) {
      const int col0 = block_col + c4 * kChunkStride + thread_col * 4;
      T* out = C + static_cast<long long>(r) * N + col0;
      if (kVec && col0 + 3 < N) {
        Vec4<T> x;
#pragma unroll
        for (int j = 0; j < 4; ++j) x.v[j] = ccml::detail::from_float<T>(acc[i][c4 * 4 + j]);
        *reinterpret_cast<Vec4<T>*>(out) = x;
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (col0 + j < N) out[j] = ccml::detail::from_float<T>(acc[i][c4 * 4 + j]);
        }
      }
    }
  }
}

/* Loads 4 consecutive elements of row `row` of a [rows, cols] matrix starting
   at column `col` (a multiple of 4), widened to float and zero-filled outside
   the matrix. kVec: one vector load; the host guarantees cols % 4 == 0 and
   alignment, so col < cols implies col + 3 < cols. */
template <typename T, bool kVec>
__device__ __forceinline__ float4 load_quad(const T* M_, int rows, int cols, int row, int col) {
  using ccml::detail::to_float;
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (row < rows) {
    const T* p = M_ + static_cast<long long>(row) * cols + col;
    if (kVec) {
      if (col < cols) {
        const Vec4<T> x = *reinterpret_cast<const Vec4<T>*>(p);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = to_float(x.v[j]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (col + j < cols) v[j] = to_float(p[j]);
      }
    }
  }
  return make_float4(v[0], v[1], v[2], v[3]);
}

/*
 * Double-buffered (docs/optimization-log.md, entry 3): matmul_vectorized's
 * tiling and conflict-free layout, plus a second shared-memory buffer.
 *
 *   prologue:  fetch tile 0 -> stash in buffer 0 -> barrier
 *   step k:    issue global loads of tile k+1 into registers   (in flight...)
 *              compute tile k from buffer k%2                  (...during this)
 *              stash the registers into buffer (k+1)%2
 *              barrier
 *
 * One barrier per step is enough. The stash at step k writes the buffer that
 * step k-1 read, and every thread passed step k-1's barrier only after
 * finishing that read. The barrier at the end of step k then publishes the
 * stash before step k+1 reads it.
 *
 * This is the register-prefetch form, which runs on sm_75. The cp.async form
 * (sm_80+) is docs/local.md item L2.
 */
template <typename T, int BM, int BN, int BK, int TM, int TN, bool kVec>
__global__ void __launch_bounds__((BM / TM) * (BN / TN), 2)
matmul_double_buffered(T* C, const T* A, const T* B, int M, int N, int K) {
  constexpr int kCols = BN / TN;
  constexpr int kThreads = (BM / TM) * kCols;
  constexpr int kChunksN = TN / 4;
  constexpr int kChunkStride = BN / kChunksN;
  constexpr int kPadA = 4;
  constexpr int kQuadsA = BM * BK / 4;
  constexpr int kQuadsB = BK * BN / 4;
  constexpr int kPerThreadA = (kQuadsA + kThreads - 1) / kThreads;
  constexpr int kPerThreadB = (kQuadsB + kThreads - 1) / kThreads;
  static_assert(TM % 4 == 0 && TN % 4 == 0 && BK % 4 == 0,
                "double-buffered tiles need multiples of 4");

  __shared__ __align__(16) float As[2][BK][BM + kPadA];
  __shared__ __align__(16) float Bs[2][BK][BN];

  const int tid = static_cast<int>(threadIdx.x);
  const int thread_row = tid / kCols;
  const int thread_col = tid % kCols;
  const int block_row = static_cast<int>(blockIdx.y) * BM;
  const int block_col = static_cast<int>(blockIdx.x) * BN;

  float4 ra[kPerThreadA];
  float4 rb[kPerThreadB];

  /* Global -> registers for the slice starting at k0. */
  auto fetch = [&](int k0) {
#pragma unroll
    for (int p = 0; p < kPerThreadA; ++p) {
      const int q = tid + p * kThreads;
      if (q < kQuadsA) {
        ra[p] = load_quad<T, kVec>(A, M, K, block_row + q / (BK / 4), k0 + (q % (BK / 4)) * 4);
      }
    }
#pragma unroll
    for (int p = 0; p < kPerThreadB; ++p) {
      const int q = tid + p * kThreads;
      if (q < kQuadsB) {
        rb[p] = load_quad<T, kVec>(B, K, N, k0 + q / (BN / 4), block_col + (q % (BN / 4)) * 4);
      }
    }
  };
  /* Registers -> shared buffer `buf`: A transposed and padded, B as float4. */
  auto stash = [&](int buf) {
#pragma unroll
    for (int p = 0; p < kPerThreadA; ++p) {
      const int q = tid + p * kThreads;
      if (q < kQuadsA) {
        const int r = q / (BK / 4);
        const int c = (q % (BK / 4)) * 4;
        As[buf][c + 0][r] = ra[p].x;
        As[buf][c + 1][r] = ra[p].y;
        As[buf][c + 2][r] = ra[p].z;
        As[buf][c + 3][r] = ra[p].w;
      }
    }
#pragma unroll
    for (int p = 0; p < kPerThreadB; ++p) {
      const int q = tid + p * kThreads;
      if (q < kQuadsB) {
        *reinterpret_cast<float4*>(&Bs[buf][q / (BN / 4)][(q % (BN / 4)) * 4]) = rb[p];
      }
    }
  };

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
  }
  float a_reg[TM];
  float b_reg[TN];

  fetch(0);
  stash(0);
  __syncthreads();

  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const bool has_next = k0 + BK < K;
    if (has_next) fetch(k0 + BK);  /* in flight during the math below */

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i4 = 0; i4 < TM / 4; ++i4) {
        const float4 a4 =
            *reinterpret_cast<const float4*>(&As[buf][k][thread_row * TM + i4 * 4]);
        a_reg[i4 * 4 + 0] = a4.x;
        a_reg[i4 * 4 + 1] = a4.y;
        a_reg[i4 * 4 + 2] = a4.z;
        a_reg[i4 * 4 + 3] = a4.w;
      }
#pragma unroll
      for (int c4 = 0; c4 < kChunksN; ++c4) {
        const float4 b4 = *reinterpret_cast<const float4*>(
            &Bs[buf][k][c4 * kChunkStride + thread_col * 4]);
        b_reg[c4 * 4 + 0] = b4.x;
        b_reg[c4 * 4 + 1] = b4.y;
        b_reg[c4 * 4 + 2] = b4.z;
        b_reg[c4 * 4 + 3] = b4.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a_reg[i] * b_reg[j];
      }
    }

    if (has_next) stash(buf ^ 1);
    __syncthreads();
    buf ^= 1;
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = block_row + thread_row * TM + i;
    if (r >= M) continue;
#pragma unroll
    for (int c4 = 0; c4 < kChunksN; ++c4) {
      const int col0 = block_col + c4 * kChunkStride + thread_col * 4;
      T* out = C + static_cast<long long>(r) * N + col0;
      if (kVec && col0 + 3 < N) {
        Vec4<T> x;
#pragma unroll
        for (int j = 0; j < 4; ++j) x.v[j] = ccml::detail::from_float<T>(acc[i][c4 * 4 + j]);
        *reinterpret_cast<Vec4<T>*>(out) = x;
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (col0 + j < N) out[j] = ccml::detail::from_float<T>(acc[i][c4 * 4 + j]);
        }
      }
    }
  }
}

/* ============================================================ Tensor Cores ==
 *
 * The Tensor Core kernels take 16-bit operands (FP16 everywhere from sm_70,
 * BF16 from sm_80) and accumulate in FP32. They share one block-tile
 * description and one global->shared loader that moves raw 16-bit words, so
 * the same code serves both storage types.
 */

/* Block tile BM x BN x BK computed by (BM/WM) x (BN/WN) warps, each owning a
   WM x WN warp tile. */
struct MatmulTcConfig {
  int bm, bn, bk, wm, wn;
};

constexpr MatmulTcConfig kMatmulTcConfigs[] = {
    {128, 128, 32, 64, 32}, /* 0: default, 8 warps (optimization log entry 4) */
    {64, 64, 32, 32, 32},   /* 1: 4 warps, for mid-sized grids */
    {16, 128, 32, 16, 32},  /* 2: one 16-row fragment band, decode-sized M */
};
constexpr int kMatmulTcConfigCount =
    static_cast<int>(sizeof(kMatmulTcConfigs) / sizeof(kMatmulTcConfigs[0]));

/* True when this compilation pass can emit Tensor Core instructions for T.
   Evaluated in device passes only; the host pass never runs kernel bodies. */
template <typename T>
struct TcAvailable : std::false_type {};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
template <>
struct TcAvailable<__half> : std::true_type {};
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
struct TcAvailable<__nv_bfloat16> : std::true_type {};
#endif

/* Eight 16-bit elements: one 128-bit access. */
struct alignas(16) Raw8 {
  unsigned short v[8];
};

/* Loads 8 consecutive 16-bit words of row `row` of a [rows, cols] matrix,
   starting at column `col` (a multiple of 8), zero-filled outside the matrix.
   Zero bits are +0 in FP16 and BF16 alike. kVec: one 128-bit load; the host
   guarantees cols % 8 == 0 and 16-byte alignment, so col < cols implies
   col + 7 < cols. */
template <bool kVec>
__device__ __forceinline__ Raw8 load_raw8(const unsigned short* M_, int rows, int cols, int row,
                                          int col) {
  Raw8 r;
  if (kVec) {
    if (row < rows && col < cols) {
      r = *reinterpret_cast<const Raw8*>(M_ + static_cast<long long>(row) * cols + col);
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) r.v[j] = 0;
    }
  } else {
    const unsigned short* p = M_ + static_cast<long long>(row) * cols + col;
#pragma unroll
    for (int j = 0; j < 8; ++j) r.v[j] = (row < rows && col + j < cols) ? p[j] : 0;
  }
  return r;
}

/* Writes a 16 x 16 FP32 tile (row-major, in shared memory `tile`) to C at
   (row0, col0), rounded to T, bounds-checked. One warp; each lane writes
   half a row (8 elements), with one 128-bit store when kVec and the run is
   inside the matrix. */
template <typename T, bool kVec>
__device__ __forceinline__ void store_tile16(T* C, int M, int N, int row0, int col0,
                                             const float* tile, int lane) {
  const int r = lane >> 1;
  const int c = (lane & 1) * 8;
  const int gr = row0 + r;
  const int gc = col0 + c;
  if (gr >= M) return;
  alignas(16) T out[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) out[j] = ccml::detail::from_float<T>(tile[r * 16 + c + j]);
  T* dst = C + static_cast<long long>(gr) * N + gc;
  if (kVec && gc + 7 < N) {
    *reinterpret_cast<Raw8*>(dst) = *reinterpret_cast<const Raw8*>(out);
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (gc + j < N) dst[j] = out[j];
    }
  }
}

/*
 * WMMA (docs/optimization-log.md, entry 4): entry 3's register-prefetch
 * double buffering, with the inner product on Tensor Cores.
 *
 *  - Shared tiles hold the raw 16-bit operands: As[2][BM][BK+8] and
 *    Bs[2][BK][BN+8], both row-major. The 8-element pad shifts each row by 16
 *    bytes so the rows a fragment load touches start in different banks, and
 *    keeps every fragment pointer 32-byte aligned (load_matrix_sync's rule):
 *    a 16-row step is 16*(BK+8)*2 bytes, a 16-column step 32 bytes.
 *  - Per k-step of BK: BK/16 rounds of mma_sync(16x16x16) per accumulator
 *    fragment, FP32 accumulate.
 *  - Epilogue: each accumulator fragment goes through a per-warp 16x16 FP32
 *    scratch tile (the fragment's register layout is unspecified), aliased
 *    onto the operand buffers after the main loop.
 *
 *   grid  = (div_up(N, BN), div_up(M, BM))
 *   block = (BM/WM) * (BN/WN) warps
 *
 * T must be __half (sm_70+) or __nv_bfloat16 (sm_80+). For a type the
 * compilation target lacks Tensor Cores for, the body is empty; the host
 * never launches it (matmul.cu checks the compiled architecture).
 */
/* Occupancy floor of 2 blocks per SM (<= 128 registers at 256 threads): the
   BF16 instantiation of configuration 0 otherwise compiles to 132 registers on
   sm_89, one block per SM, and runs 1.75x slower than FP16 (optimization log
   entry 4). On the T4 configuration 0's 37.9 KB of shared memory already
   limits it to one block per SM; the bound then only caps registers. */
template <typename T, int BM, int BN, int BK, int WM, int WN, bool kVec>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32, 2)
matmul_wmma(T* C, const T* A, const T* B, int M, int N, int K) {
  if constexpr (TcAvailable<T>::value) {
    using namespace nvcuda;
    constexpr int kWarpsN = BN / WN;
    constexpr int kWarps = (BM / WM) * kWarpsN;
    constexpr int kThreads = kWarps * 32;
    constexpr int kFragM = WM / 16;
    constexpr int kFragN = WN / 16;
    constexpr int kLdA = BK + 8;
    constexpr int kLdB = BN + 8;
    constexpr int kStageA = BM * kLdA;
    constexpr int kStageB = BK * kLdB;
    constexpr int kChunksA = BM * BK / 8;
    constexpr int kChunksB = BK * BN / 8;
    constexpr int kPerThreadA = (kChunksA + kThreads - 1) / kThreads;
    constexpr int kPerThreadB = (kChunksB + kThreads - 1) / kThreads;
    constexpr int kOperandWords = 2 * (kStageA + kStageB);
    constexpr int kScratchWords = kWarps * 256 * 2; /* 16x16 floats per warp */
    constexpr int kSmemWords = kOperandWords > kScratchWords ? kOperandWords : kScratchWords;
    static_assert(BM % WM == 0 && BN % WN == 0 && WM % 16 == 0 && WN % 16 == 0 && BK % 16 == 0,
                  "Tensor Core tiles are multiples of 16");

    __shared__ __align__(32) unsigned short smem[kSmemWords];
    unsigned short* const As = smem;                  /* [2][BM][kLdA] */
    unsigned short* const Bs = smem + 2 * kStageA;    /* [2][BK][kLdB] */

    const unsigned short* const Ar = reinterpret_cast<const unsigned short*>(A);
    const unsigned short* const Br = reinterpret_cast<const unsigned short*>(B);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_row = (warp / kWarpsN) * WM;
    const int warp_col = (warp % kWarpsN) * WN;
    const int block_row = static_cast<int>(blockIdx.y) * BM;
    const int block_col = static_cast<int>(blockIdx.x) * BN;

    Raw8 ra[kPerThreadA];
    Raw8 rb[kPerThreadB];
    auto fetch = [&](int k0) {
#pragma unroll
      for (int p = 0; p < kPerThreadA; ++p) {
        const int q = tid + p * kThreads;
        if (q < kChunksA) {
          ra[p] = load_raw8<kVec>(Ar, M, K, block_row + q / (BK / 8), k0 + (q % (BK / 8)) * 8);
        }
      }
#pragma unroll
      for (int p = 0; p < kPerThreadB; ++p) {
        const int q = tid + p * kThreads;
        if (q < kChunksB) {
          rb[p] = load_raw8<kVec>(Br, K, N, k0 + q / (BN / 8), block_col + (q % (BN / 8)) * 8);
        }
      }
    };
    auto stash = [&](int buf) {
#pragma unroll
      for (int p = 0; p < kPerThreadA; ++p) {
        const int q = tid + p * kThreads;
        if (q < kChunksA) {
          *reinterpret_cast<Raw8*>(As + buf * kStageA + (q / (BK / 8)) * kLdA +
                                   (q % (BK / 8)) * 8) = ra[p];
        }
      }
#pragma unroll
      for (int p = 0; p < kPerThreadB; ++p) {
        const int q = tid + p * kThreads;
        if (q < kChunksB) {
          *reinterpret_cast<Raw8*>(Bs + buf * kStageB + (q / (BN / 8)) * kLdB +
                                   (q % (BN / 8)) * 8) = rb[p];
        }
      }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kFragM][kFragN];
#pragma unroll
    for (int i = 0; i < kFragM; ++i) {
#pragma unroll
      for (int j = 0; j < kFragN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    }

    fetch(0);
    stash(0);
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
      const bool has_next = k0 + BK < K;
      if (has_next) fetch(k0 + BK); /* in flight during the math below */

      const T* const as = reinterpret_cast<const T*>(As + buf * kStageA);
      const T* const bs = reinterpret_cast<const T*>(Bs + buf * kStageB);
#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag[kFragN];
#pragma unroll
        for (int j = 0; j < kFragN; ++j) {
          wmma::load_matrix_sync(b_frag[j], bs + kk * kLdB + warp_col + j * 16, kLdB);
        }
#pragma unroll
        for (int i = 0; i < kFragM; ++i) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
          wmma::load_matrix_sync(a_frag, as + (warp_row + i * 16) * kLdA + kk, kLdA);
#pragma unroll
          for (int j = 0; j < kFragN; ++j) wmma::mma_sync(acc[i][j], a_frag, b_frag[j], acc[i][j]);
        }
      }

      if (has_next) stash(buf ^ 1);
      __syncthreads();
      buf ^= 1;
    }

    /* The loop ended on a barrier, so the operand buffers are free. */
    float* const scratch = reinterpret_cast<float*>(smem) + warp * 256;
#pragma unroll
    for (int i = 0; i < kFragM; ++i) {
#pragma unroll
      for (int j = 0; j < kFragN; ++j) {
        wmma::store_matrix_sync(scratch, acc[i][j], 16, wmma::mem_row_major);
        __syncwarp();
        store_tile16<T, kVec>(C, M, N, block_row + warp_row + i * 16,
                              block_col + warp_col + j * 16, scratch, lane);
        __syncwarp();
      }
    }
  }
}

}  // namespace kernels
}  // namespace ccml

#endif /* CCML_SRC_MATMUL_KERNELS_CUH */
