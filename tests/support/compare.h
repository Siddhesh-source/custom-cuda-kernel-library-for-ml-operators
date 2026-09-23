#ifndef CCML_TESTS_SUPPORT_COMPARE_H
#define CCML_TESTS_SUPPORT_COMPARE_H

#include <cstddef>
#include <string>
#include <vector>

namespace ccml_test {

/* Tolerance applied ELEMENT-WISE as:
 *
 *     |got[i] - ref[i]|  <=  abs_tol + rel_tol * |ref[i]|
 *
 * The combined form is deliberate. A pure relative bound is meaningless where
 * the reference is near zero (softmax tails, normalized values), and a pure
 * absolute bound is meaningless where it is large (matmul over a long K). */
struct Tolerance {
  double abs_tol = 0.0;
  double rel_tol = 0.0;

  std::string to_string() const;
};

/* Suggested tolerance for a float32 result produced by summing `terms` values.
 *
 * Rationale: float32 has ~1.19e-7 unit roundoff, and error in a sequential sum
 * grows roughly with sqrt(terms) for uncorrelated rounding. The margin below is
 * a STARTING POINT for the correctness work, not a validated policy --
 * docs/correctness.md must record the tolerance actually adopted and the
 * measured errors that justify it. */
Tolerance tolerance_for_f32_reduction(int terms);

/* Tolerance for a value obtained by summing `terms` products, where the natural
 * scale of the result is `scale` rather than the element's own magnitude.
 *
 * This exists because tolerance_for_f32_reduction is the wrong model for a
 * matrix product. It judges each element against its own size, which works for
 * softmax and layer normalization -- their outputs have a bounded, meaningful
 * magnitude. A matrix product element is a sum of `terms` signed products, so
 * cancellation can leave it arbitrarily close to zero even though the
 * computation that produced it involved terms of magnitude one. Judged
 * relatively, such an element shows an enormous error for no reason; a K=768
 * case measured a relative error of 0.23 on an element that had cancelled to
 * near zero, while its absolute error was ordinary.
 *
 * The meaningful scale is the size of the quantity being accumulated, for which
 * the RMS of the reference output is a good stand-in. Pass it as `scale`. */
Tolerance tolerance_for_f32_dot_product(int terms, double scale);

/* Root mean square of a reference, for use as the `scale` above. */
double rms(const std::vector<double>& values);

/* What a comparison against the double-precision reference actually found.
   Reported whether the test passes or fails, so docs/correctness.md can quote
   measured numbers rather than assertions. */
struct ErrorStats {
  double      max_abs_error = 0.0;
  double      max_rel_error = 0.0;

  /* max over i of |got[i] - ref[i]| / (abs_tol + rel_tol * |ref[i]|).
     <= 1.0 means every element satisfied the element-wise bound. This is the
     value the pass/fail decision is made on. */
  double      max_normalized_error = 0.0;

  std::size_t worst_index     = 0;  /* index attaining max_normalized_error */
  std::size_t nonfinite_count = 0;  /* NaN or inf in the GPU result */
  std::size_t count           = 0;
  bool        size_mismatch   = false;

  std::string to_string() const;
};

/* got is the GPU result; reference is the double-precision CPU oracle. */
ErrorStats compare(const std::vector<float>&  got,
                   const std::vector<double>& reference,
                   const Tolerance&           tolerance);

/* True only if every element met the bound and nothing was NaN or inf. */
bool within(const ErrorStats& stats);

}  // namespace ccml_test

#endif /* CCML_TESTS_SUPPORT_COMPARE_H */
