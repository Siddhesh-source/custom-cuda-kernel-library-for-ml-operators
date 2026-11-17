#ifndef CCML_TUNING_H
#define CCML_TUNING_H

/*
 * Tuning interface: run a tiled matmul variant with an explicit tile
 * configuration.
 *
 * Exists for the tile-size sweep (reference.md §5) and for building the
 * measured dispatch table behind AUTO (enhancement S4). Normal callers never
 * need it: pass an algo enum to ccml_matmul_<dtype> instead.
 *
 * Deliberately not part of ccml.h.
 */

#include "ccml/export.h"
#include "ccml/matmul.h"
#include "ccml/status.h"
#include "ccml/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ccml_matmul_tile_config {
  int bm, bn, bk;  /* block tile */
  int tm, tn;      /* per-thread register tile */
  int threads;     /* (bm / tm) * (bn / tn) */
} ccml_matmul_tile_config;

/* Number of tile configurations. Index 0 is what the algo enum uses. */
CCML_API int ccml_matmul_tile_config_count(void);

CCML_API ccml_status ccml_matmul_tile_config_get(int index, ccml_matmul_tile_config* out);

/* Tensor Core tile configurations: a BM x BN x BK block tile computed by
   (bm/wm) x (bn/wn) warps, each owning a wm x wn warp tile. Index 0 is what
   the algo enum uses. */
typedef struct ccml_matmul_tc_config {
  int bm, bn, bk;  /* block tile */
  int wm, wn;      /* warp tile */
  int warps;       /* (bm / wm) * (bn / wn) */
} ccml_matmul_tc_config;

CCML_API int ccml_matmul_tc_config_count(void);

CCML_API ccml_status ccml_matmul_tc_config_get(int index, ccml_matmul_tc_config* out);

/* `algo` with tile configuration `index`. For the CUDA-core tiled family the
   index is into the tile configurations above: CCML_MATMUL_REGISTER_TILED
   (every configuration), CCML_MATMUL_VECTORIZED and
   CCML_MATMUL_DOUBLE_BUFFERED (configurations whose TM, TN and BK are
   multiples of 4; others return CCML_ERROR_INVALID_ARGUMENT). For the Tensor
   Core family (CCML_MATMUL_WMMA) it is into the Tensor Core configurations.
   Same contract as ccml_matmul_<dtype>. */
CCML_API ccml_status ccml_matmul_tuned_f32(float* C, const float* A, const float* B, int M,
                                           int N, int K, ccml_matmul_algo algo, int index,
                                           cudaStream_t stream);
CCML_API ccml_status ccml_matmul_tuned_f16(ccml_f16* C, const ccml_f16* A, const ccml_f16* B,
                                           int M, int N, int K, ccml_matmul_algo algo,
                                           int index, cudaStream_t stream);
CCML_API ccml_status ccml_matmul_tuned_bf16(ccml_bf16* C, const ccml_bf16* A,
                                            const ccml_bf16* B, int M, int N, int K,
                                            ccml_matmul_algo algo, int index,
                                            cudaStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CCML_TUNING_H */
