#ifndef CCML_MATMUL_H
#define CCML_MATMUL_H

#include "ccml/export.h"
#include "ccml/status.h"
#include "ccml/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Algorithm selection.
 *
 * This enum is the library's extension point for optimization work: a new
 * kernel variant is added here and wired into the dispatch switch in
 * src/matmul/matmul.cu. No public signature changes, so tests and benchmarks
 * can compare variants against each other without any other edits. */
typedef enum ccml_matmul_algo {
  CCML_MATMUL_AUTO  = 0, /* library picks; currently resolves to TILED */
  CCML_MATMUL_NAIVE = 1, /* one thread per output element; correctness baseline */
  CCML_MATMUL_TILED = 2  /* shared-memory tiled */
} ccml_matmul_algo;

CCML_API const char* ccml_matmul_algo_name(ccml_matmul_algo algo);

/*
 * C = A * B, single precision, row-major, device pointers.
 *
 *   A : [M, K]
 *   B : [K, N]
 *   C : [M, N]   (written, not accumulated -- no beta term)
 *
 * Accumulation is in float32. M, N, K must all be > 0; arbitrary values are
 * accepted, including sizes that are not multiples of any tile dimension.
 *
 * Asynchronous with respect to the host on the given stream.
 */
CCML_API ccml_status ccml_matmul_f32(float*       C,
                                     const float* A,
                                     const float* B,
                                     int          M,
                                     int          N,
                                     int          K,
                                     ccml_matmul_algo algo,
                                     cudaStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CCML_MATMUL_H */
