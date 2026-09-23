#ifndef CCML_BENCH_SUPPORT_TIMER_H
#define CCML_BENCH_SUPPORT_TIMER_H

#include <functional>
#include <string>

#include <cuda_runtime.h>

namespace ccml_bench {

/*
 * CUDA-event timing.
 *
 * Measured on the device, not with a host clock. On the Windows WDDM driver
 * model the OS batches command submission, so a host-side wall clock around a
 * launch can report almost anything; cudaEvent timestamps are recorded in the
 * stream itself and do not have that problem.
 *
 * Warmup iterations are discarded. They are not a formality: the first launch
 * of a kernel pays module load and JIT, and the GPU clocks are still ramping.
 */

struct TimingResult {
  double mean_ms   = 0.0;
  double min_ms    = 0.0;
  double max_ms    = 0.0;
  double stddev_ms = 0.0;
  int    samples   = 0;
  bool   valid     = false;  /* false if the work under test did not run */
};

/* Runs `launch` warmup times, then samples it `repeats` times.
 *
 * `launch` must return true if the work was actually submitted. Returning false
 * (an unimplemented operator, say) produces an invalid result rather than a
 * timing of nothing, which would otherwise look like an extremely fast kernel. */
TimingResult time_launch(const std::function<bool()>& launch,
                         int warmup,
                         int repeats,
                         cudaStream_t stream);

}  // namespace ccml_bench

#endif /* CCML_BENCH_SUPPORT_TIMER_H */
