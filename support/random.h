#ifndef CCML_SUPPORT_RANDOM_H
#define CCML_SUPPORT_RANDOM_H

#include <cstddef>
#include <cstdint>

namespace ccml_support {

/* Deterministic generator for test and benchmark inputs.
 *
 * Values are derived from the raw 64-bit engine output by explicit arithmetic
 * rather than through std::uniform_real_distribution, because the standard does
 * not specify the distribution's output sequence. Two different standard
 * libraries would otherwise disagree, and reproducibility is a stated goal of
 * this project. */
class Rng {
 public:
  explicit Rng(std::uint64_t seed);

  /* Fill with values uniform in [low, high). */
  void uniform(float* out, std::size_t count, float low, float high);

  /* Fill with approximately standard-normal values (Box-Muller). */
  void normal(float* out, std::size_t count, float mean, float stddev);

 private:
  std::uint64_t next_u64();
  double next_unit();  /* [0, 1) */

  std::uint64_t state_;
};

}  // namespace ccml_support

#endif /* CCML_SUPPORT_RANDOM_H */
