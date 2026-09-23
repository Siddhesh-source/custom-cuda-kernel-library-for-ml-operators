#include "support/random.h"

#include <cmath>

namespace ccml_support {

namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
}  // namespace

/* splitmix64: tiny, well-distributed, and identical on every platform. */
Rng::Rng(std::uint64_t seed) : state_(seed) {}

std::uint64_t Rng::next_u64() {
  state_ += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state_;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

double Rng::next_unit() {
  /* Top 53 bits scaled into [0, 1); exactly representable in a double. */
  return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
}

void Rng::uniform(float* out, std::size_t count, float low, float high) {
  const double span = static_cast<double>(high) - static_cast<double>(low);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = static_cast<float>(static_cast<double>(low) + next_unit() * span);
  }
}

void Rng::normal(float* out, std::size_t count, float mean, float stddev) {
  for (std::size_t i = 0; i < count; i += 2) {
    /* Guard against log(0); next_unit() can return exactly 0. */
    const double u1 = 1.0 - next_unit();
    const double u2 = next_unit();
    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double z0 = radius * std::cos(kTwoPi * u2);
    const double z1 = radius * std::sin(kTwoPi * u2);

    out[i] = static_cast<float>(mean + stddev * z0);
    if (i + 1 < count) {
      out[i + 1] = static_cast<float>(mean + stddev * z1);
    }
  }
}

}  // namespace ccml_support
