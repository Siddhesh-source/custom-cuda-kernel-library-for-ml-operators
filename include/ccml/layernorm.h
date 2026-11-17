#ifndef CCML_LAYERNORM_H
#define CCML_LAYERNORM_H

#include "ccml/export.h"
#include "ccml/status.h"
#include "ccml/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ccml_layernorm_algo {
  CCML_LAYERNORM_AUTO     = 0, /* library picks; currently resolves to TWO_PASS */
  CCML_LAYERNORM_TWO_PASS = 1, /* separate mean and variance reductions */
  CCML_LAYERNORM_WELFORD  = 2  /* single-pass Welford (reserved; not yet built) */
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
 * Reductions accumulate in DOUBLE, unlike the float32 default stated in
 * types.h. The output divides by the standard deviation, so error in the mean
 * is magnified by 1/stddev: a float32 mean at magnitude 10000 carries about
 * 5e-4 of error, which against a standard deviation of 0.01 is a 5% error in
 * every output element. See src/layernorm/layernorm_kernels.cuh.
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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CCML_LAYERNORM_H */
