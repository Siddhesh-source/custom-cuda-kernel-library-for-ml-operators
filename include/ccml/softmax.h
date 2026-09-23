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
  CCML_SOFTMAX_WARP_PER_ROW   = 2  /* one warp per row, for narrow rows (reserved) */
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
 * operator finite for large inputs. Reductions accumulate in float32.
 */
CCML_API ccml_status ccml_softmax_f32(float*       y,
                                      const float* x,
                                      int          rows,
                                      int          cols,
                                      ccml_softmax_algo algo,
                                      cudaStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CCML_SOFTMAX_H */
