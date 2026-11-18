#ifndef CCML_LAYERNORM_H
#define CCML_LAYERNORM_H

#include "ccml/export.h"
#include "ccml/status.h"
#include "ccml/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ccml_layernorm_algo {
  CCML_LAYERNORM_AUTO     = 0, /* measured: WARP_PER_ROW <= 256 cols, VECTORIZED <= 1024, else WELFORD */
  CCML_LAYERNORM_TWO_PASS = 1, /* separate mean and variance reductions */
  CCML_LAYERNORM_WELFORD  = 2, /* single-pass FP32 Welford on pivot-shifted data, block per row */
  /* One warp per row, the row held in registers (cols <= 1024; wider rows
     return CCML_ERROR_UNSUPPORTED_SHAPE). */
  CCML_LAYERNORM_WARP_PER_ROW = 3,
  /* WARP_PER_ROW with 128-bit loads and stores (cols a multiple of 4 for f32,
     8 for f16/bf16, 16-byte-aligned pointers; otherwise WARP_PER_ROW's scalar
     kernel runs). cols <= 1024. */
  CCML_LAYERNORM_VECTORIZED = 4
} ccml_layernorm_algo;

CCML_API const char* ccml_layernorm_algo_name(ccml_layernorm_algo algo);

/*
 * Layer normalization over the last dimension.
 *
 *   x, y : [rows, cols]   row-major
 *   gamma, beta : [cols], or NULL to skip the affine transform
 *
 *   y[r, c] = (x[r, c] - mean(x[r, :])) / sqrt(var(x[r, :]) + eps)
 *             * gamma[c] + beta[c]
 *
 * var is the biased (1/cols) estimator, matching torch.nn.functional.layer_norm.
 * eps must be > 0.
 *
 * gamma and beta are independently optional: either may be NULL.
 *
 * Any tensor whose last dimension is the normalized one is supported: collapse
 * every leading dimension into rows. A [batch, seq, hidden] tensor normalized
 * over hidden is rows = batch * seq, cols = hidden.
 *
 * Precision. The output divides by the standard deviation, so error in the
 * mean is magnified by 1/stddev: a float32 mean accumulated at magnitude 10000
 * carries about 5e-4 of error, which against a standard deviation of 0.01 is a
 * 5% error in every output element. TWO_PASS avoids it by reducing in DOUBLE.
 * WELFORD reduces in FP32 but first subtracts a pivot (the row's first
 * element), which removes the large intermediate instead; both meet the same
 * FP32 tolerance (docs/optimization-log.md entry 9).
 */
CCML_API ccml_status ccml_layernorm_f32(float*       y,
                                        const float* x,
                                        const float* gamma,
                                        const float* beta,
                                        int          rows,
                                        int          cols,
                                        float        eps,
                                        ccml_layernorm_algo algo,
                                        cudaStream_t stream);

/* FP16 and BF16 storage, same contract as ccml_layernorm_f32. x, y, gamma and
   beta all use the one storage type (as torch.nn.functional.layer_norm
   requires); eps stays float. Reductions are in double exactly as for f32, and
   y is rounded once on store. */
CCML_API ccml_status ccml_layernorm_f16(ccml_f16*       y,
                                        const ccml_f16* x,
                                        const ccml_f16* gamma,
                                        const ccml_f16* beta,
                                        int             rows,
                                        int             cols,
                                        float           eps,
                                        ccml_layernorm_algo algo,
                                        cudaStream_t    stream);

CCML_API ccml_status ccml_layernorm_bf16(ccml_bf16*       y,
                                         const ccml_bf16* x,
                                         const ccml_bf16* gamma,
                                         const ccml_bf16* beta,
                                         int              rows,
                                         int              cols,
                                         float            eps,
                                         ccml_layernorm_algo algo,
                                         cudaStream_t     stream);

/*
 * Fused residual + bias + layer norm (pre-LN transformer block):
 *
 *   s = round_to_T(x + residual + bias)       (FP32 math, one rounding)
 *   y = layernorm(s) * gamma + beta
 *
 * x, residual, y, sum_out : [rows, cols]; bias, gamma, beta : [cols] or NULL;
 * sum_out NULL skips writing s. The norm is of the rounded s, so y equals a
 * separate add kernel followed by ccml_layernorm. sum_out must not alias x or
 * residual. algo: VECTORIZED (cols <= 1024, 128-bit; other widths or
 * alignments run the WELFORD kernel), WELFORD (any width), or AUTO; others
 * return CCML_ERROR_NOT_IMPLEMENTED.
 */
CCML_API ccml_status ccml_layernorm_residual_f32(float* y, float* sum_out, const float* x,
                                                 const float* residual, const float* bias,
                                                 const float* gamma, const float* beta,
                                                 int rows, int cols, float eps,
                                                 ccml_layernorm_algo algo, cudaStream_t stream);
CCML_API ccml_status ccml_layernorm_residual_f16(ccml_f16* y, ccml_f16* sum_out,
                                                 const ccml_f16* x, const ccml_f16* residual,
                                                 const ccml_f16* bias, const ccml_f16* gamma,
                                                 const ccml_f16* beta, int rows, int cols,
                                                 float eps, ccml_layernorm_algo algo,
                                                 cudaStream_t stream);
CCML_API ccml_status ccml_layernorm_residual_bf16(ccml_bf16* y, ccml_bf16* sum_out,
                                                  const ccml_bf16* x, const ccml_bf16* residual,
                                                  const ccml_bf16* bias, const ccml_bf16* gamma,
                                                  const ccml_bf16* beta, int rows, int cols,
                                                  float eps, ccml_layernorm_algo algo,
                                                  cudaStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CCML_LAYERNORM_H */
