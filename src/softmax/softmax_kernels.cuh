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

/* ------------------------------------------------------------- online ---
 *
 * docs/optimization-log.md entry 13 (Milakov & Gimelshein, 2018). A running
 * maximum m and the sum s of exp(x - m) over the values seen so far, in FP32;
 * when m grows, s is rescaled by exp(m_old - m_new). Every term is in (0, 1]
 * and positive terms do not cancel, so FP32 suffices.
 *
 * -inf needs care: exp(-inf - (-inf)) is NaN, so a -inf value (or a partial
 * result whose sum is 0) contributes nothing instead of being exponentiated.
 * A NaN input still poisons the row (NaN > m is false, and exp(NaN) is NaN),
 * and a row that is entirely -inf ends with m = -inf, s = 0 and writes
 * exp(NaN) / 0 = NaN, the documented policy (docs/correctness.md).
 */
struct OnlineMaxSum {
  float m; /* running maximum */
  float s; /* sum of exp(x - m) */
};

__device__ __forceinline__ void online_add(OnlineMaxSum& a, float v) {
  if (v > a.m) {
    a.s = a.s * expf(a.m - v) + 1.0f; /* a.s == 0 when a.m == -inf: 0 * 0 */
    a.m = v;
  } else if (v != -INFINITY) {
    a.s += expf(v - a.m);
  }
}

__device__ __forceinline__ OnlineMaxSum online_merge(OnlineMaxSum a, OnlineMaxSum b) {
  const float m = a.m > b.m ? a.m : b.m;
  const float sa = a.s == 0.0f ? 0.0f : a.s * expf(a.m - m);
  const float sb = b.s == 0.0f ? 0.0f : b.s * expf(b.m - m);
  return OnlineMaxSum{m, sa + sb};
}

__device__ __forceinline__ OnlineMaxSum online_warp_reduce(OnlineMaxSum v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    OnlineMaxSum o;
    o.m = __shfl_xor_sync(0xffffffffu, v.m, offset);
    o.s = __shfl_xor_sync(0xffffffffu, v.s, offset);
    v = online_merge(v, o);
  }
  return v;
}

/* Block-wide merge; every thread receives the result. Contains barriers. */
__device__ __forceinline__ OnlineMaxSum online_block_reduce(OnlineMaxSum v,
                                                            OnlineMaxSum* scratch) {
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int warps = (static_cast<int>(blockDim.x) + 31) / 32;
  v = online_warp_reduce(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  OnlineMaxSum t = lane < warps ? scratch[lane] : OnlineMaxSum{-INFINITY, 0.0f};
  return online_warp_reduce(t);
}

/*
 * Online softmax, block per row: pass 1 the running (max, sum) statistics,
 * pass 2 the write. Two reads of the row instead of three, FP32 only.
 *
 *   grid = (rows), block = (threads), a multiple of 32
 */
template <typename T>
__global__ void softmax_online(T* y, const T* x, int rows, int cols) {
  using ccml::detail::to_float;
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) return;
  const long long base = static_cast<long long>(row) * cols;
  const T* __restrict__ row_in = x + base;
  T* __restrict__ row_out = y + base;
  __shared__ OnlineMaxSum scratch[32];

  OnlineMaxSum st{-INFINITY, 0.0f};
  for (int c = static_cast<int>(threadIdx.x); c < cols; c += static_cast<int>(blockDim.x)) {
    online_add(st, to_float(row_in[c]));
  }
  st = online_block_reduce(st, scratch);
  const float inv = 1.0f / st.s;
  for (int c = static_cast<int>(threadIdx.x); c < cols; c += static_cast<int>(blockDim.x)) {
    row_out[c] = ccml::detail::from_float<T>(expf(to_float(row_in[c]) - st.m) * inv);
  }
}

/* ------------------------------------------------------- warp per row ---
 *
 * docs/optimization-log.md entry 14. One warp per row (cols <= 1024), 4 rows
 * per block, no barriers. The row is read once into registers in chunks of kV
 * elements: kV = 16 / sizeof(T) (one 128-bit access) when the host found cols a
 * chunk multiple and the pointers aligned, else kV = 1. Lane l owns chunks
 * l, l + 32, ...; exp(x - max) is computed once and kept.
 *
 *   grid = (ceil(rows / kSoftmaxRowsPerBlock)), block = (32 * kSoftmaxRowsPerBlock)
 */
constexpr int kSoftmaxRowsPerBlock = 4;

template <typename T, int kV>
struct alignas(sizeof(T) * kV) SoftmaxChunk {
  T v[kV];
};

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, v, o);
    v = other > v ? other : v;
  }
  return v;
}

__device__ __forceinline__ float warp_sum_f(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <typename T, int kV, int kChunks>
__global__ void __launch_bounds__(32 * kSoftmaxRowsPerBlock)
softmax_warp_per_row(T* y, const T* x, int rows, int cols) {
  using ccml::detail::to_float;
  using Chunk = SoftmaxChunk<T, kV>;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int row = static_cast<int>(blockIdx.x) * kSoftmaxRowsPerBlock +
                  static_cast<int>(threadIdx.x) / 32;
  if (row >= rows) return;
  const long long base = static_cast<long long>(row) * cols;
  const T* __restrict__ row_in = x + base;
  T* __restrict__ row_out = y + base;

  float v[kChunks][kV];
  float m = -INFINITY;
  bool nan = false;
#pragma unroll
  for (int j = 0; j < kChunks; ++j) {
    const int c0 = (lane + 32 * j) * kV;
    if (c0 < cols) {
      const Chunk in = *reinterpret_cast<const Chunk*>(row_in + c0);
#pragma unroll
      for (int e = 0; e < kV; ++e) {
        v[j][e] = to_float(in.v[e]);
        m = v[j][e] > m ? v[j][e] : m;
        nan |= v[j][e] != v[j][e];
      }
    } else {
#pragma unroll
      for (int e = 0; e < kV; ++e) v[j][e] = -INFINITY;
    }
  }
  m = warp_max(m);
  /* NaN anywhere poisons the row (the max reduction alone would drop it). */
  if (__any_sync(0xffffffffu, nan)) m = NAN;
  float s = 0.0f;
#pragma unroll
  for (int j = 0; j < kChunks; ++j) {
#pragma unroll
    for (int e = 0; e < kV; ++e) {
      /* -inf terms (including the padding) contribute exactly 0, unless the
         row holds a NaN (m is then NaN and every term NaN); a row that is
         entirely -inf keeps m = -inf and ends 0 / 0 = NaN, the policy. */
      v[j][e] = (v[j][e] == -INFINITY && m == m) ? 0.0f : expf(v[j][e] - m);
      s += v[j][e];
    }
  }
  const float inv = 1.0f / warp_sum_f(s);
#pragma unroll
  for (int j = 0; j < kChunks; ++j) {
    const int c0 = (lane + 32 * j) * kV;
    if (c0 < cols) {
      Chunk out;
#pragma unroll
      for (int e = 0; e < kV; ++e) out.v[e] = ccml::detail::from_float<T>(v[j][e] * inv);
      *reinterpret_cast<Chunk*>(row_out + c0) = out;
    }
  }
}

/* --------------------------------------------------------- split row ---
 *
 * docs/optimization-log.md entry 15. Row r is cut into gridDim.x segments of
 * `seg` columns. Kernel 1 writes each segment's (max, sum) to
 * partial[r * S + s]; kernel 2 merges the S pairs of its row in fixed order
 * (deterministic) and writes its segment. The NaN / -inf policy is ONLINE's.
 */
template <typename T>
__global__ void softmax_split_stats(OnlineMaxSum* partial, const T* x, int rows, int cols,
                                    int seg) {
  using ccml::detail::to_float;
  const int row = static_cast<int>(blockIdx.y);
  const int s = static_cast<int>(blockIdx.x);
  if (row >= rows) return;
  __shared__ OnlineMaxSum scratch[32];
  const T* row_in = x + static_cast<long long>(row) * cols;
  const int begin = s * seg;
  const int end = min(cols, begin + seg);
  OnlineMaxSum st{-INFINITY, 0.0f};
  for (int c = begin + static_cast<int>(threadIdx.x); c < end; c += static_cast<int>(blockDim.x)) {
    online_add(st, to_float(row_in[c]));
  }
  st = online_block_reduce(st, scratch);
  if (threadIdx.x == 0) partial[static_cast<long long>(row) * gridDim.x + s] = st;
}

template <typename T>
__global__ void softmax_split_write(T* y, const OnlineMaxSum* partial, const T* x, int rows,
                                    int cols, int seg) {
  using ccml::detail::to_float;
  const int row = static_cast<int>(blockIdx.y);
  const int s = static_cast<int>(blockIdx.x);
  if (row >= rows) return;
  /* Every thread merges the row's S pairs in the same fixed order: S is small
     (<= a few dozen) and the pairs are an L2-resident broadcast. */
  const OnlineMaxSum* p = partial + static_cast<long long>(row) * gridDim.x;
  OnlineMaxSum st = p[0];
  for (unsigned i = 1; i < gridDim.x; ++i) st = online_merge(st, p[i]);
  const float inv = 1.0f / st.s;
  const long long base = static_cast<long long>(row) * cols;
  const int begin = s * seg;
  const int end = min(cols, begin + seg);
  for (int c = begin + static_cast<int>(threadIdx.x); c < end; c += static_cast<int>(blockDim.x)) {
    y[base + c] = ccml::detail::from_float<T>(expf(to_float(x[base + c]) - st.m) * inv);
  }
}

/* ------------------------------------------------- fused scale + mask ---
 *
 * docs/optimization-log.md entry 16 (enhancement F3).
 *   v = x * scale + mask[row % mask_rows][c]          (mask may be null)
 *   causal_q > 0: q = row % causal_q; key c is masked (v = -inf) when
 *                 c > q + (cols - causal_q)
 * A fully masked row is written as zeros (not NaN); a NaN in x still makes its
 * row NaN.
 */
struct SoftmaxMask {
  const void* mask;
  int mask_rows;
  float scale;
  int causal_q;
};

template <typename T>
__device__ __forceinline__ float masked_value(float xv, const T* mask_row, int c, int causal_limit,
                                              const SoftmaxMask& m) {
  using ccml::detail::to_float;
  float v = xv * m.scale;
  if (mask_row != nullptr) v += to_float(mask_row[c]);
  if (c > causal_limit) v = -INFINITY;
  return v;
}

/* Causal limit of `row`: the largest unmasked key index (INT_MAX: no causal). */
__device__ __forceinline__ int causal_limit_of(int row, int cols, const SoftmaxMask& m) {
  if (m.causal_q <= 0) return 0x7fffffff;
  return row % m.causal_q + (cols - m.causal_q);
}

/* Warp per row with the mask fused into the load (entry 14's structure). */
template <typename T, int kV, int kChunks>
__global__ void __launch_bounds__(32 * kSoftmaxRowsPerBlock)
softmax_masked_warp_per_row(T* y, const T* x, int rows, int cols, SoftmaxMask msk) {
  using ccml::detail::to_float;
  using Chunk = SoftmaxChunk<T, kV>;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int row = static_cast<int>(blockIdx.x) * kSoftmaxRowsPerBlock +
                  static_cast<int>(threadIdx.x) / 32;
  if (row >= rows) return;
  const long long base = static_cast<long long>(row) * cols;
  const T* mask_row = msk.mask == nullptr
                          ? nullptr
                          : static_cast<const T*>(msk.mask) +
                                static_cast<long long>(row % msk.mask_rows) * cols;
  const int limit = causal_limit_of(row, cols, msk);

  float v[kChunks][kV];
  float m = -INFINITY;
  bool nan = false;
#pragma unroll
  for (int j = 0; j < kChunks; ++j) {
    const int c0 = (lane + 32 * j) * kV;
    if (c0 < cols) {
      const Chunk in = *reinterpret_cast<const Chunk*>(x + base + c0);
#pragma unroll
      for (int e = 0; e < kV; ++e) {
        const float xv = to_float(in.v[e]);
        nan |= xv != xv;
        v[j][e] = masked_value<T>(xv, mask_row, c0 + e, limit, msk);
        m = v[j][e] > m ? v[j][e] : m;
      }
    } else {
#pragma unroll
      for (int e = 0; e < kV; ++e) v[j][e] = -INFINITY;
    }
  }
  m = warp_max(m);
  if (__any_sync(0xffffffffu, nan)) m = NAN;
  const bool all_masked = m == -INFINITY; /* NaN compares false: stays NaN */
  float s = 0.0f;
#pragma unroll
  for (int j = 0; j < kChunks; ++j) {
#pragma unroll
    for (int e = 0; e < kV; ++e) {
      v[j][e] = (v[j][e] == -INFINITY && m == m) ? 0.0f : expf(v[j][e] - m);
      s += v[j][e];
    }
  }
  const float inv = all_masked ? 0.0f : 1.0f / warp_sum_f(s);
#pragma unroll
  for (int j = 0; j < kChunks; ++j) {
    const int c0 = (lane + 32 * j) * kV;
    if (c0 < cols) {
      Chunk out;
#pragma unroll
      for (int e = 0; e < kV; ++e) out.v[e] = ccml::detail::from_float<T>(v[j][e] * inv);
      *reinterpret_cast<Chunk*>(y + base + c0) = out;
    }
  }
}

/* Block per row (ONLINE) with the mask fused; any width. */
template <typename T>
__global__ void softmax_masked_online(T* y, const T* x, int rows, int cols, SoftmaxMask msk) {
  using ccml::detail::to_float;
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) return;
  const long long base = static_cast<long long>(row) * cols;
  const T* mask_row = msk.mask == nullptr
                          ? nullptr
                          : static_cast<const T*>(msk.mask) +
                                static_cast<long long>(row % msk.mask_rows) * cols;
  const int limit = causal_limit_of(row, cols, msk);
  __shared__ OnlineMaxSum scratch[32];
  OnlineMaxSum st{-INFINITY, 0.0f};
  for (int c = static_cast<int>(threadIdx.x); c < cols; c += static_cast<int>(blockDim.x)) {
    online_add(st, masked_value<T>(to_float(x[base + c]), mask_row, c, limit, msk));
  }
  st = online_block_reduce(st, scratch);
  /* Fully masked: m = -inf and s = 0 exactly (a NaN would have made s NaN). */
  const bool all_masked = st.m == -INFINITY && st.s == 0.0f;
  const float inv = all_masked ? 0.0f : 1.0f / st.s;
  for (int c = static_cast<int>(threadIdx.x); c < cols; c += static_cast<int>(blockDim.x)) {
    const float v = masked_value<T>(to_float(x[base + c]), mask_row, c, limit, msk);
    const float e = (all_masked || v == -INFINITY) && st.s == st.s ? 0.0f : expf(v - st.m);
    y[base + c] = ccml::detail::from_float<T>(e * inv);
  }
}

}  // namespace kernels
}  // namespace ccml

#endif /* CCML_SRC_SOFTMAX_KERNELS_CUH */
