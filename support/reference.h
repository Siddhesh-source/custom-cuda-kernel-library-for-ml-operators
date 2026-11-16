#ifndef CCML_SUPPORT_REFERENCE_H
#define CCML_SUPPORT_REFERENCE_H

#include <vector>

namespace ccml_test {

/*
 * Double-precision CPU oracles.
 *
 * These are the trusted implementations every GPU kernel is differenced
 * against. They are written for obviousness, not speed: straightforward loops,
 * double accumulation, no blocking, no vectorization. If one of these is ever
 * hard to read, it is wrong.
 *
 * Each returns false while unimplemented. Callers MUST check the return value
 * and skip, never treat a zero-filled output as a passing comparison. That is
 * the only thing standing between a skeleton and a false green build.
 *
 * A second, independent check against PyTorch CPU lives in python/tests/.
 */

/* out[M, N] = a[M, K] * b[K, N], row-major. */
bool matmul_reference(std::vector<double>&      out,
                      const std::vector<float>& a,
                      const std::vector<float>& b,
                      int M, int N, int K);

/* Layer normalization over the last dimension, biased (1/cols) variance.
   gamma and beta may be empty, meaning that part of the affine is skipped. */
bool layernorm_reference(std::vector<double>&      out,
                         const std::vector<float>& x,
                         const std::vector<float>& gamma,
                         const std::vector<float>& beta,
                         int rows, int cols, double eps);

/* Softmax over the last dimension, with max subtraction. */
bool softmax_reference(std::vector<double>&      out,
                       const std::vector<float>& x,
                       int rows, int cols);

}  // namespace ccml_test

#endif /* CCML_SUPPORT_REFERENCE_H */
