#include "support/compare.h"

#include <cmath>
#include <cstdio>
#include <limits>

namespace ccml_test {

std::string Tolerance::to_string() const {
  char buffer[128];
  std::snprintf(buffer, sizeof(buffer), "abs_tol=%.3e rel_tol=%.3e",
                abs_tol, rel_tol);
  return std::string(buffer);
}

std::string ErrorStats::to_string() const {
  char buffer[256];
  std::snprintf(buffer, sizeof(buffer),
                "n=%zu max_abs=%.3e max_rel=%.3e normalized=%.3f "
                "worst_index=%zu nonfinite=%zu",
                count, max_abs_error, max_rel_error, max_normalized_error,
                worst_index, nonfinite_count);
  return std::string(buffer);
}

Tolerance tolerance_for_f32_reduction(int terms) {
  constexpr double kFloat32Eps = 1.1920928955078125e-07;
  const double growth = std::sqrt(static_cast<double>(terms < 1 ? 1 : terms));

  Tolerance tolerance;
  tolerance.abs_tol = 8.0 * kFloat32Eps * growth;
  tolerance.rel_tol = 8.0 * kFloat32Eps * growth;
  return tolerance;
}

Tolerance tolerance_for_softmax_f32(int cols) {
  Tolerance tolerance;
  tolerance.abs_tol = 1.1754943508222875e-38;  /* FLT_MIN */
  tolerance.rel_tol = tolerance_for_f32_reduction(cols).rel_tol;
  return tolerance;
}

Tolerance widen_for_storage(const Tolerance& f32, double unit_roundoff,
                            double abs_floor) {
  Tolerance tolerance;
  tolerance.abs_tol = f32.abs_tol + abs_floor;
  tolerance.rel_tol = f32.rel_tol + 2.0 * unit_roundoff;
  return tolerance;
}

Tolerance tolerance_for_f32_dot_product(int terms, double scale) {
  const Tolerance base = tolerance_for_f32_reduction(terms);

  Tolerance tolerance;
  /* Absolute part carries the scale; relative part is kept so that large
     elements are still held to a relative standard. */
  tolerance.abs_tol = base.abs_tol * (scale > 0.0 ? scale : 1.0);
  tolerance.rel_tol = base.rel_tol;
  return tolerance;
}

double rms(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  double sum_squares = 0.0;
  for (double value : values) {
    sum_squares += value * value;
  }
  return std::sqrt(sum_squares / static_cast<double>(values.size()));
}

ErrorStats compare(const std::vector<float>& got,
                   const std::vector<double>& reference,
                   const Tolerance& tolerance) {
  return compare(got, reference, tolerance, std::vector<double>());
}

ErrorStats compare(const std::vector<float>& got, const std::vector<double>& reference,
                   const Tolerance& tolerance, const std::vector<double>& extra_abs) {
  ErrorStats stats;
  stats.count = got.size();

  if (got.size() != reference.size()) {
    stats.size_mismatch = true;
    stats.max_normalized_error = std::numeric_limits<double>::infinity();
    return stats;
  }

  for (std::size_t i = 0; i < got.size(); ++i) {
    const double value = static_cast<double>(got[i]);

    if (!std::isfinite(value)) {
      if (stats.nonfinite_count == 0) {
        stats.worst_index = i;
      }
      ++stats.nonfinite_count;
      continue;
    }

    const double expected  = reference[i];
    const double abs_error = std::fabs(value - expected);
    const double magnitude = std::fabs(expected);

    const double rel_error = (magnitude > 0.0) ? (abs_error / magnitude) : 0.0;
    const double bound = tolerance.abs_tol + tolerance.rel_tol * magnitude +
                         (extra_abs.empty() ? 0.0 : extra_abs[i]);
    const double normalized = (bound > 0.0)
                                  ? (abs_error / bound)
                                  : (abs_error > 0.0
                                         ? std::numeric_limits<double>::infinity()
                                         : 0.0);

    if (abs_error > stats.max_abs_error) {
      stats.max_abs_error = abs_error;
    }
    if (rel_error > stats.max_rel_error) {
      stats.max_rel_error = rel_error;
    }
    if (normalized > stats.max_normalized_error) {
      stats.max_normalized_error = normalized;
      if (stats.nonfinite_count == 0) {
        stats.worst_index = i;
      }
    }
  }

  return stats;
}

bool within(const ErrorStats& stats) {
  return !stats.size_mismatch && stats.nonfinite_count == 0 &&
         stats.max_normalized_error <= 1.0;
}

}  // namespace ccml_test
