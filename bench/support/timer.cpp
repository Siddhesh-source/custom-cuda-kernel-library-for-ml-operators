#include "bench/support/timer.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "support/cuda_event.h"

namespace ccml_bench {

TimingResult time_launch(const std::function<bool()>& launch,
                         int warmup,
                         int repeats,
                         cudaStream_t stream) {
  TimingResult result;

  /* Warmup is discarded, and not as a formality: the first launch of a kernel
     pays module load and any JIT, and the GPU clocks are still ramping up from
     idle. Including those samples would report a kernel as several times slower
     than it is. */
  for (int i = 0; i < warmup; ++i) {
    if (!launch()) {
      return result;  /* invalid: nothing ran */
    }
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return result;
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repeats));

  try {
    ccml_support::EventTimer timer;

    for (int i = 0; i < repeats; ++i) {
      timer.start(stream);
      if (!launch()) {
        return result;  /* invalid: the work stopped being submittable */
      }
      samples.push_back(timer.stop_ms(stream));
    }
  } catch (const std::exception&) {
    /* A CUDA failure during timing invalidates the measurement. Reporting an
       invalid result is right: a partial set of samples would be published as
       though it were a complete one. */
    return result;
  }

  if (samples.empty()) {
    return result;
  }

  double sum = 0.0;
  for (double sample : samples) {
    sum += sample;
  }
  const double mean = sum / static_cast<double>(samples.size());

  double variance = 0.0;
  for (double sample : samples) {
    variance += (sample - mean) * (sample - mean);
  }
  variance /= static_cast<double>(samples.size());

  result.mean_ms   = mean;
  result.min_ms    = *std::min_element(samples.begin(), samples.end());
  result.max_ms    = *std::max_element(samples.begin(), samples.end());
  result.stddev_ms = std::sqrt(variance);
  result.samples   = static_cast<int>(samples.size());
  result.valid     = true;
  return result;
}

}  // namespace ccml_bench
