#include "support/reference.h"

#include <cmath>

/*
 * Double-precision CPU oracles.
 *
 * Written for obviousness, not speed: straightforward loops, double
 * accumulation, no blocking and no vectorization. If one of these is ever hard
 * to read, it is wrong. They are the trusted implementations every GPU kernel
 * is differenced against, and a clever reference is a reference nobody can
 * check.
 *
 * Functions still returning false have no oracle yet; callers must skip rather
 * than treat a zero-filled output as a passing comparison.
 */

namespace ccml_test {

bool matmul_reference(std::vector<double>& out,
                      const std::vector<float>& a,
                      const std::vector<float>& b,
                      int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) {
    return false;
  }
  if (a.size() != static_cast<std::size_t>(M) * K ||
      b.size() != static_cast<std::size_t>(K) * N) {
    return false;
  }

  out.assign(static_cast<std::size_t>(M) * N, 0.0);

  /* The obvious triple loop, accumulating in double. Slow and that is fine: an
     oracle nobody can check by eye is not an oracle. */
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += static_cast<double>(a[static_cast<std::size_t>(row) * K + k]) *
               static_cast<double>(b[static_cast<std::size_t>(k) * N + col]);
      }
      out[static_cast<std::size_t>(row) * N + col] = sum;
    }
  }

  return true;
}

bool layernorm_reference(std::vector<double>& out,
                         const std::vector<float>& x,
                         const std::vector<float>& gamma,
                         const std::vector<float>& beta,
                         int rows, int cols, double eps) {
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  const std::size_t elements = static_cast<std::size_t>(rows) * cols;
  if (x.size() != elements) {
    return false;
  }
  const bool use_gamma = !gamma.empty();
  const bool use_beta  = !beta.empty();
  if ((use_gamma && gamma.size() != static_cast<std::size_t>(cols)) ||
      (use_beta && beta.size() != static_cast<std::size_t>(cols))) {
    return false;
  }

  out.assign(elements, 0.0);

  for (int row = 0; row < rows; ++row) {
    const std::size_t base = static_cast<std::size_t>(row) * cols;

    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      sum += static_cast<double>(x[base + c]);
    }
    const double mean = sum / cols;

    /* Variance about the mean just computed, never E[x^2] - E[x]^2. The oracle
       must not share the failure mode it exists to detect. */
    double sum_squares = 0.0;
    for (int c = 0; c < cols; ++c) {
      const double deviation = static_cast<double>(x[base + c]) - mean;
      sum_squares += deviation * deviation;
    }
    const double variance = sum_squares / cols;

    const double inv_stddev = 1.0 / std::sqrt(variance + eps);

    for (int c = 0; c < cols; ++c) {
      double value = (static_cast<double>(x[base + c]) - mean) * inv_stddev;
      if (use_gamma) {
        value *= static_cast<double>(gamma[c]);
      }
      if (use_beta) {
        value += static_cast<double>(beta[c]);
      }
      out[base + c] = value;
    }
  }

  return true;
}

bool softmax_reference(std::vector<double>& out,
                       const std::vector<float>& x,
                       int rows, int cols) {
  if (rows <= 0 || cols <= 0) {
    return false;
  }
  const std::size_t elements = static_cast<std::size_t>(rows) * cols;
  if (x.size() != elements) {
    return false;
  }

  out.assign(elements, 0.0);

  for (int row = 0; row < rows; ++row) {
    const std::size_t base = static_cast<std::size_t>(row) * cols;

    /* The maximum is subtracted here too. The oracle is computed in double, so
       it would survive inputs that overflow float -- but it must model the same
       algorithm the kernel implements, not a more forgiving one, or it would
       hide exactly the failure it exists to detect. */
    double row_max = static_cast<double>(x[base]);
    for (int c = 1; c < cols; ++c) {
      const double value = static_cast<double>(x[base + c]);
      if (value > row_max) {
        row_max = value;
      }
    }

    double sum = 0.0;
    for (int c = 0; c < cols; ++c) {
      const double shifted = std::exp(static_cast<double>(x[base + c]) - row_max);
      out[base + c] = shifted;
      sum += shifted;
    }

    /* sum >= 1 always: the maximum element contributes exp(0) = 1. */
    for (int c = 0; c < cols; ++c) {
      out[base + c] /= sum;
    }
  }

  return true;
}

}  // namespace ccml_test
