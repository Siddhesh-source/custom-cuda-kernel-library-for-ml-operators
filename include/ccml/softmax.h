#ifndef CCML_SOFTMAX_H
#define CCML_SOFTMAX_H

#include "ccml/export.h"
#include "ccml/status.h"
#include "ccml/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ccml_softmax_algo {
  CCML_SOFTMAX_AUTO           = 0, /* measured per dtype and width (softmax.cu resolve_auto) */
  CCML_SOFTMAX_BLOCK_PER_ROW  = 1, /* one thread block reduces one row */
  CCML_SOFTMAX_WARP_PER_ROW   = 2, /* one warp per row, row in registers, 128-bit when
                                     aligned; cols <= 1024 (else UNSUPPORTED_SHAPE) */
  CCML_SOFTMAX_ONLINE         = 3, /* online (running max, sum) in FP32, block per row, 2 reads */
  /* A long row split across blocks (few, long rows: vocabulary softmax); two
     kernels and a small workspace from cudaMallocAsync on `stream`;
     deterministic. Short rows run ONLINE. */
  CCML_SOFTMAX_SPLIT_ROW      = 4,
  /* Block per row with the row in registers, 128-bit when aligned (long rows);
     cols <= 8192 FP32 / 16384 FP16, BF16 aligned, 2048 unaligned (else
     UNSUPPORTED_SHAPE). */
  CCML_SOFTMAX_BLOCK_RESIDENT = 5
} ccml_softmax_algo;

CCML_API const char* ccml_softmax_algo_name(ccml_softmax_algo algo);

/*
 * Softmax over the last dimension.
 *
 *   x, y : [rows, cols]   row-major
 *
 *   y[r, c] = exp(x[r, c] - max(x[r, :])) / sum_j exp(x[r, j] - max(x[r, :]))
 *
 * The max subtraction is mandatory, not an optimization: it is what keeps the
 * operator finite for large inputs. BLOCK_PER_ROW reduces in double; ONLINE
 * in FP32 (positive terms after the shift do not cancel; same tolerance).
 * NaN anywhere in a row gives a NaN row; -inf entries give exact zeros; a
 * row entirely -inf gives NaN (0/0), as torch.softmax does.
 */
CCML_API ccml_status ccml_softmax_f32(float*       y,
                                      const float* x,
                                      int          rows,
                                      int          cols,
                                      ccml_softmax_algo algo,
                                      cudaStream_t stream);

/* FP16 and BF16 storage, same contract as ccml_softmax_f32: max-subtracted,
   exponentials in float, sum in double, y rounded once on store. */
CCML_API ccml_status ccml_softmax_f16(ccml_f16*       y,
                                      const ccml_f16* x,
                                      int             rows,
                                      int             cols,
                                      ccml_softmax_algo algo,
                                      cudaStream_t    stream);

CCML_API ccml_status ccml_softmax_bf16(ccml_bf16*       y,
                                       const ccml_bf16* x,
                                       int              rows,
                                       int              cols,
                                       ccml_softmax_algo algo,
                                       cudaStream_t     stream);

/*
 * Fused scale + mask softmax (attention scores):
 *
 *   v[r, c] = x[r, c] * scale + mask[r % mask_rows, c]       (mask may be NULL)
 *   causal_q > 0: with q = r % causal_q, key c is masked (v = -inf) when
 *                 c > q + (cols - causal_q)   (bottom-right aligned; causal_q =
 *                 cols for self-attention)
 *   y[r, :] = softmax(v[r, :])
 *
 * mask: [mask_rows, cols] additive (0 or -inf for a padding mask), same storage
 * type as x. A FULLY MASKED row is written as zeros, not NaN (the attention
 * convention: a padded query has no valid key). A NaN in x still gives a NaN
 * row. algo: WARP_PER_ROW (cols <= 1024), ONLINE (any width) or AUTO; others
 * return CCML_ERROR_NOT_IMPLEMENTED.
 */
CCML_API ccml_status ccml_softmax_masked_f32(float* y, const float* x, const float* mask,
                                             int mask_rows, int rows, int cols, float scale,
                                             int causal_q, ccml_softmax_algo algo,
                                             cudaStream_t stream);
CCML_API ccml_status ccml_softmax_masked_f16(ccml_f16* y, const ccml_f16* x,
                                             const ccml_f16* mask, int mask_rows, int rows,
                                             int cols, float scale, int causal_q,
                                             ccml_softmax_algo algo, cudaStream_t stream);
CCML_API ccml_status ccml_softmax_masked_bf16(ccml_bf16* y, const ccml_bf16* x,
                                              const ccml_bf16* mask, int mask_rows, int rows,
                                              int cols, float scale, int causal_q,
                                              ccml_softmax_algo algo, cudaStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CCML_SOFTMAX_H */
