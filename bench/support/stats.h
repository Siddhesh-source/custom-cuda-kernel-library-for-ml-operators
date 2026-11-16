#ifndef CCML_BENCH_SUPPORT_STATS_H
#define CCML_BENCH_SUPPORT_STATS_H

#include <cstdint>
#include <vector>

namespace ccml_bench {

/*
 * Distribution summary of per-iteration timings (reference.md, Method: "report
 * the median and spread of 100+ iterations"). Pure host code, no CUDA, so it
 * is unit-tested directly in tests/test_bench_stats.cpp.
 */
struct Summary {
  int    n      = 0;
  double median = 0.0;
  double p10    = 0.0;
  double p90    = 0.0;
  double iqr    = 0.0;  /* p75 - p25 */
  double min    = 0.0;
  double max    = 0.0;
  double mean   = 0.0;
  double ci_lo  = 0.0;  /* bootstrap 95% CI on the median */
  double ci_hi  = 0.0;
  bool   valid  = false;
};

/* Linear-interpolated percentile, q in [0, 1], of an ascending-sorted vector
   (the same definition as numpy's default "linear" method). */
double percentile_sorted(const std::vector<double>& sorted, double q);

double median_of(std::vector<double> values);

/* Percentile bootstrap of the median. Deterministic for a given seed, so a
   rerun on the same samples reproduces the same interval. */
void bootstrap_median_ci(const std::vector<double>& samples, int resamples,
                         std::uint64_t seed, double* lo, double* hi);

Summary summarize(const std::vector<double>& samples, int resamples = 2000,
                  std::uint64_t seed = 0x5eed);

}  // namespace ccml_bench

#endif /* CCML_BENCH_SUPPORT_STATS_H */
