#ifndef CCML_SUPPORT_COMPARE_H
#define CCML_SUPPORT_COMPARE_H

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

/* Softmax outputs are probabilities whose size is set by the row, not by the
 * number of terms: in a 32000-wide row a typical output is ~3e-5. An absolute
 * tolerance scaled by sqrt(cols) (tolerance_for_f32_reduction) would therefore
 * accept a kernel that wrote zeros. Every source of error in softmax --
 * expf, the double sum, the division -- is RELATIVE to each output, so the
 * tolerance is relative, with an absolute floor only at float32's smallest
 * normal value, below which expf itself underflows. */
Tolerance tolerance_for_softmax_f32(int cols);

/* Widens an f32 tolerance for a 16-bit storage type (FP16, BF16).
 *
 * The kernels compute in float32 or wider and round to the storage type once,
 * on store; the reference is computed in double from the SAME rounded inputs.
 * So the only extra error is that final rounding: at most half a unit in the
 * last place, i.e. unit_roundoff * |y| relative. Twice that is allowed, for the
 * f32 result itself sitting up to its own tolerance away from the reference
 * before it is rounded. `abs_floor` covers outputs that round into the
 * subnormal range, where the spacing is absolute (FP16: 2^-24).
 *
 *   FP16: unit_roundoff 2^-11, abs_floor 2^-24
 *   BF16: unit_roundoff 2^-8,  abs_floor FLT_MIN (same exponent range as f32) */
Tolerance widen_for_storage(const Tolerance& f32, double unit_roundoff,
                            double abs_floor);

constexpr double kF16UnitRoundoff  = 1.0 / 2048.0;       /* 2^-11 */
constexpr double kBF16UnitRoundoff = 1.0 / 256.0;        /* 2^-8  */
constexpr double kF16SubnormalStep = 5.9604644775390625e-08;  /* 2^-24 */

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

/* As above, with an additional per-element absolute allowance: element i is
   within bound when |got - ref| <= abs_tol + rel_tol * |ref| + extra_abs[i]
   (an empty vector adds nothing). For pipelines that round an intermediate to
   a 16-bit type before a second kernel (a matmul followed by a separate
   epilogue), the extra error of that rounding is element-specific:
   unit_roundoff * |intermediate_i|. */
ErrorStats compare(const std::vector<float>&  got,
                   const std::vector<double>& reference,
                   const Tolerance&           tolerance,
                   const std::vector<double>& extra_abs);

/* True only if every element met the bound and nothing was NaN or inf. */
bool within(const ErrorStats& stats);

}  // namespace ccml_test

#endif /* CCML_SUPPORT_COMPARE_H */
