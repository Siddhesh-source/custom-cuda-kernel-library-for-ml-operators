#ifndef CCML_BENCH_SUPPORT_TIMER_H
#define CCML_BENCH_SUPPORT_TIMER_H

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bench/support/stats.h"

namespace ccml_bench {

/*
 * CUDA-event timing (reference.md, Method).
 *
 *  - Warmup launches are discarded (module load, JIT, clock ramp-up).
 *  - Every timed iteration is bracketed by its own event pair, so the median
 *    and spread come from >= 100 real samples.
 *  - The spec's batched pattern (one event pair around ITERS launches) is also
 *    run and reported as batched_mean_us, as a cross-check on the per-iteration
 *    numbers.
 *  - Candidates are interleaved in rounds (A,B,C,A,B,C,...) so slow drift in
 *    clocks or temperature hits every candidate equally.
 *  - Cold mode overwrites a buffer larger than L2 before each iteration, outside
 *    the timed region, so small shapes cannot be timed out of a warm L2.
 */

/* `launch` enqueues one call on the stream and returns false if it could not
   (the call is then reported as not run, never as a fast time). */
struct Candidate {
  std::string           impl;   /* "ccml", "cublas", "cublaslt", "cudnn" */
  std::string           algo;   /* variant name within impl */
  std::function<bool()> launch;
};

struct TimingConfig {
  int    warmup = 20;
  int    max_iters = 200;
  int    min_iters = 100;          /* the spec's floor; never go below it */
  double budget_ms = 3000.0;       /* per candidate per cache mode */
  int    rounds = 4;
  bool   cold_l2 = false;
};

struct Measured {
  Summary     summary;             /* microseconds */
  double      batched_mean_us = 0.0;
  bool        valid = false;
  std::string error;
};

/* Owns a scratch buffer of 2 x L2 size, written to evict L2 before a cold run. */
class L2Flusher {
 public:
  explicit L2Flusher(std::size_t l2_bytes);
  ~L2Flusher();
  L2Flusher(const L2Flusher&) = delete;
  L2Flusher& operator=(const L2Flusher&) = delete;
  void flush(cudaStream_t stream);

 private:
  void*       buffer_ = nullptr;
  std::size_t bytes_ = 0;
};

std::vector<Measured> time_interleaved(const std::vector<Candidate>& candidates,
                                       const TimingConfig& config,
                                       cudaStream_t stream,
                                       L2Flusher* flusher);

/* Iteration count for an estimated per-call time: as many as fit in the
   budget, clamped to [min_iters, max_iters]. Exposed for tests. */
int choose_iterations(double estimate_ms, const TimingConfig& config);

}  // namespace ccml_bench

#endif /* CCML_BENCH_SUPPORT_TIMER_H */
