#ifndef CCML_SOFTMAX_H
#define CCML_SOFTMAX_H

#include "ccml/export.h"
#include "ccml/status.h"
#include "ccml/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ccml_softmax_algo {
  CCML_SOFTMAX_AUTO           = 0, /* library picks; currently BLOCK_PER_ROW */
  CCML_SOFTMAX_BLOCK_PER_ROW  = 1, /* one thread block reduces one row */
  CCML_SOFTMAX_WARP_PER_ROW   = 2, /* one warp per row, row in registers, 128-bit when
                                     aligned; cols <= 1024 (else UNSUPPORTED_SHAPE) */
  CCML_SOFTMAX_ONLINE         = 3  /* online (running max, sum) in FP32, block per row, 2 reads */
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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CCML_SOFTMAX_H */
