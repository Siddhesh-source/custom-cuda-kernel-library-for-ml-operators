#include "bench/support/stats.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ccml_bench {

double percentile_sorted(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) {
    return 0.0;
  }
  q = std::min(1.0, std::max(0.0, q));
  const double position = q * static_cast<double>(sorted.size() - 1);
  const std::size_t below = static_cast<std::size_t>(std::floor(position));
  const std::size_t above = std::min(below + 1, sorted.size() - 1);
  const double fraction = position - static_cast<double>(below);
  return sorted[below] + (sorted[above] - sorted[below]) * fraction;
}

double median_of(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return percentile_sorted(values, 0.5);
}

namespace {

/* splitmix64: tiny, well-distributed, and identical on every platform, unlike
   std::uniform_int_distribution whose output is implementation-defined. */
std::uint64_t next_random(std::uint64_t* state) {
  std::uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

}  // namespace

void bootstrap_median_ci(const std::vector<double>& samples, int resamples,
                         std::uint64_t seed, double* lo, double* hi) {
  *lo = *hi = 0.0;
  if (samples.empty() || resamples <= 0) {
    return;
  }
  const std::size_t n = samples.size();
  std::uint64_t state = seed;
  std::vector<double> medians(static_cast<std::size_t>(resamples));
  std::vector<double> draw(n);

  for (int r = 0; r < resamples; ++r) {
    for (std::size_t i = 0; i < n; ++i) {
      draw[i] = samples[static_cast<std::size_t>(next_random(&state) % n)];
    }
    medians[static_cast<std::size_t>(r)] = median_of(draw);
  }
  std::sort(medians.begin(), medians.end());
  *lo = percentile_sorted(medians, 0.025);
  *hi = percentile_sorted(medians, 0.975);
}

Summary summarize(const std::vector<double>& samples, int resamples,
                  std::uint64_t seed) {
  Summary s;
  if (samples.empty()) {
    return s;
  }
  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());

  s.n      = static_cast<int>(sorted.size());
  s.median = percentile_sorted(sorted, 0.50);
  s.p10    = percentile_sorted(sorted, 0.10);
  s.p90    = percentile_sorted(sorted, 0.90);
  s.iqr    = percentile_sorted(sorted, 0.75) - percentile_sorted(sorted, 0.25);
  s.min    = sorted.front();
  s.max    = sorted.back();
  s.mean   = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
             static_cast<double>(sorted.size());
  bootstrap_median_ci(samples, resamples, seed, &s.ci_lo, &s.ci_hi);
  s.valid = true;
  return s;
}

}  // namespace ccml_bench
